# SPDX-License-Identifier: Apache-2.0
"""GB10 P2 stage-1 kernel-level stochastic correctness gate (R-031).

Executes the astra closure-alignment gate spec (task-mv8oapbk, R-031 in
RESULTS.md): known conditional target/proposal distributions through the PINNED
image's GPU block verifier, against an independent fp64 oracle, with recorded
RNG inputs, invariants, and sensitivity controls that the gate must REJECT.

Invocation path exercised (pinned image qwen38-flash-dgx:iter7f-20261015,
sampler files sha-identical in iter7d-20261014):

  GPUModelRunner._sample (vllm/v1/worker/gpu/model_runner.py:1419)
    -> RejectionSampler.__call__(logits, input_batch, speculator.draft_logits)
       (vllm/v1/worker/gpu/spec_decode/rejection_sampler.py:252)
       -> _verify_in_chunks -> _verify
          -> rejection_sample(..., use_block_verification=True)
             (vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py:922)
             -> _compute_local_logits_stats_kernel     (:193)
             -> _compute_cumulative_log_p_kernel       (:306)   [block only]
             -> _compute_local_residual_mass_kernel    (:387)   [block only]
             -> _rejection_kernel                      (:484)   acceptance
             -> _resample_kernel                       (:696)   residual
             -> _insert_resampled_kernel               (:856)   commit

The gate drives `rejection_sample` directly with the exact tensor contract
`_verify` provides (target logits POST-sampling-params i.e. post-temperature
fp32; draft_logits = the proposal-logit cache speculator.draft_logits
[max_num_reqs, K, V] PRE-temperature), and generates proposals through the
REAL cache-writing drafter path (Speculator.sample_draft -> gumbel_sample with
logits_cache / logits_cache_col), so the "cached q == the distribution that
produced that proposal" invariant is exercised on the actual write/read pair.

Row/window layout (per request r, state row s, K steps, base position b):
  rows start..start+K (K+1 rows); draft_sampled[row] = token FED at that row
  (t_0..t_K; -1 pads invalid tail rows); pos[row] = b + row-index;
  expanded_idx_mapping[row] = s; expanded_local_pos[row] = row-index (0..K);
  cu_num_logits[r+1] - cu_num_logits[r] = K + 1.

RNG semantics (probed bit-exact against triton 3.7.1 in the pinned image):
  acceptance draw u_i = tl_rand32(seed_state, pos[row_i]) (fp32, clamped > 0)
      = uint_to_uniform(philox(seed=sext64(seed), off=pos).c0)
  proposal/resample gumbel seed g = tl.randint(seed_state, pos[row]) = philox
      c0 as an UNSIGNED 32-bit pattern; the per-vocab draw is
      philox(seed=ZEXT64(g), off=x) (an int32 seed would sign-extend; probed).
  fp32 gumbel noise = -log(-log1p(-u)); fp64: tl_rand64 r=(c1<<32)|c0,
      u=r*2^-64, noise=-log(-log(u)).

Block-verification semantics implemented by the pinned kernel (= arXiv:2403.10444
Algorithm 2 with the STEPWISE-capped running product p_i = min(p_{i-1}*r_i, 1);
verified empirically to reproduce the target distribution exactly, including
cap-event branches where a proposal ratio exceeds 1):
  log r_i = log p_i(t_{i+1}) - log q_i(t_{i+1})
  clp[start+i] = fp32(min(clp_prev + log r_i, 0))   (stepwise cap, -1 carried)
  tau_j = exp(clp[start+j-1]);  Z_j = sum_x max(tau_j*p_j(x) - q_j(x), 0)
  h_i = Z_{i+1}/(Z_{i+1} + 1 - tau_{i+1})   if t_{i+2} >= 0   (interior)
      = tau_{i+1}                           if t_{i+2} < 0    (last valid)
  accepted_length = 1 + max{i : u_i <= h_i} (argmax rule; a -1 draft ends
  verifying; greedy instead stores the target argmax at the first mismatch).
  Resample at row start+L (L = accepted_length): bonus (L==K) or rejected
  draft < 0 -> sample the target row directly; else gumbel-max over
  log(max(tau_L*p_L - q_L, 0)) in log1p form. num_sampled = L + 1.

Run inside a throwaway --gpus all container from the pinned image:
  python3 -I tests/test_blockverifier_gate.py [--json-out PATH] [--quick]
Also pytest-compatible: test_blockverifier_gate_quick runs everything in
quick mode with assert-guards.
"""

import argparse
import hashlib
import itertools
import json
import math
import os
import sys
import time

import numpy as np
import torch

M32 = 0xFFFFFFFF
M64 = (1 << 64) - 1
_PHILOX_A = 0xD2511F53
_PHILOX_B = 0xCD9E8D57
_PHILOX_KA = 0x9E3779B9
_PHILOX_KB = 0xBB67AE85
_TL_RAND_MIN = 4.6566127342e-10

RESULTS = {"meta": {}, "verdict": {}}
_CURRENT = None


def _record(name, ok, detail):
    RESULTS[_CURRENT][name] = {"ok": bool(ok), "detail": detail}
    return bool(ok)


def _section(name):
    global _CURRENT
    _CURRENT = name
    RESULTS[name] = {}


def _finish_section(name, t0):
    n_ok = sum(1 for k, v in RESULTS[name].items()
               if k != "_elapsed_s" and v["ok"])
    n_all = sum(1 for k in RESULTS[name] if k != "_elapsed_s")
    RESULTS[name]["_elapsed_s"] = round(time.time() - t0, 3)
    print(f"== section {name}: {n_ok}/{n_all} ok "
          f"({RESULTS[name]['_elapsed_s']}s)", flush=True)


# ===========================================================================
# philox replication (exact; validated in section rng against triton kernels)
# ===========================================================================
def philox_4x32(seed_u64, off_u64, n_rounds=10):
    c0 = off_u64 & M32
    c1 = (off_u64 >> 32) & M32
    c2 = 0
    c3 = 0
    k0 = seed_u64 & M32
    k1 = (seed_u64 >> 32) & M32
    for _ in range(n_rounds):
        _c0, _c2 = c0, c2
        c0 = ((((_PHILOX_B * _c2) >> 32) & M32) ^ c1 ^ k0) & M32
        c2 = ((((_PHILOX_A * _c0) >> 32) & M32) ^ c3 ^ k1) & M32
        c1 = (_PHILOX_B * _c2) & M32
        c3 = (_PHILOX_A * _c0) & M32
        k0 = (k0 + _PHILOX_KA) & M32
        k1 = (k1 + _PHILOX_KB) & M32
    return c0, c1, c2, c3


def _uint_to_uniform_u32(x_u32):
    x = x_u32 if x_u32 < (1 << 31) else x_u32 - (1 << 32)
    if x < 0:
        x = -x - 1
    return float(np.float32(np.float32(x) * np.float32(4.6566127342e-10)))


def acceptance_uniform(seed_i64, pos_i64):
    u = _uint_to_uniform_u32(philox_4x32(seed_i64 & M64, pos_i64 & M64)[0])
    return max(u, _TL_RAND_MIN)


def gumbel_seed_of(seed_i64, pos_i64):
    return philox_4x32(seed_i64 & M64, pos_i64 & M64)[0]


def gumbel_u32(gumbel_seed_u32, x):
    u = _uint_to_uniform_u32(philox_4x32(gumbel_seed_u32, x & M32)[0])
    return max(u, _TL_RAND_MIN)


def gumbel_u64(gumbel_seed_u32, x):
    c0, c1, _, _ = philox_4x32(gumbel_seed_u32, x & M32)
    r = ((c1 << 32) | c0) & M64
    u = float(np.float64(r) * np.float64(5.421010862427522170037e-20))
    return max(u, 2.2250738585072014e-308)


def gumbel_noise_fp32(gumbel_seed_u32, x):
    u = np.float32(gumbel_u32(gumbel_seed_u32, x))
    return float(-np.log(-np.log1p(-u)))


def gumbel_noise_fp64(gumbel_seed_u32, x):
    u = gumbel_u64(gumbel_seed_u32, x)
    return float(-np.log(-np.log(u)))


# ===========================================================================
# fp64 oracle: rejection_sample(..., use_block_verification=True)
# ===========================================================================
class BlockVerifierOracle:
    """fp64 reference; semantics in the module docstring."""

    def __init__(self, target_logits, draft_logits, draft_sampled,
                 cu_num_logits, pos, idx_mapping, expanded_idx_mapping,
                 expanded_local_pos, temperature, seeds, num_speculative_steps,
                 use_fp64=False):
        self.tl = np.asarray(target_logits, dtype=np.float64)
        self.dl = np.asarray(draft_logits, dtype=np.float64)
        self.ds = np.asarray(draft_sampled)
        self.cu = np.asarray(cu_num_logits)
        self.pos = np.asarray(pos)
        self.idx = np.asarray(idx_mapping)
        self.eidx = np.asarray(expanded_idx_mapping)
        self.elocal = np.asarray(expanded_local_pos)
        self.temp = np.asarray(temperature, dtype=np.float64)
        self.seeds = np.asarray(seeds)
        self.K = int(num_speculative_steps)
        self.use_fp64 = use_fp64
        self.num_reqs = len(self.idx)
        self.V = self.tl.shape[1]

    def _row_lse(self, logits):
        m = logits.max()
        return m + np.log(np.exp(logits - m).sum())

    def run(self):
        sampled = np.zeros((self.num_reqs, self.K + 1), dtype=np.int64)
        num_sampled = np.zeros(self.num_reqs, dtype=np.int32)
        details = []
        for req in range(self.num_reqs):
            start = int(self.cu[req])
            end = int(self.cu[req + 1])
            num_draft = end - start - 1
            state = int(self.idx[req])
            temp = self.temp[state]
            seed = int(self.seeds[state])
            d = self._run_request(start, end, num_draft, state, temp, seed,
                                  temp == 0.0)
            details.append(d)
            L = d["accepted_length"]
            for i in range(L):
                sampled[req, i] = self.ds[start + i + 1]
            sampled[req, L] = d["resampled"]
            num_sampled[req] = L + 1
        return sampled, num_sampled, details

    def _run_request(self, start, end, num_draft, state, temp, seed,
                     is_greedy):
        greedy_argmax = self.tl[start:start + num_draft + 1].argmax(axis=1)

        clp = np.zeros(num_draft, dtype=np.float64)
        if not is_greedy:
            log_p = 0.0
            for i in range(num_draft):
                tok = int(self.ds[start + i + 1])
                if tok >= 0:
                    t_row = self.tl[start + i]
                    tlp = t_row - self._row_lse(t_row)
                    d_row = self.dl[state, i] / temp
                    dlp = d_row - self._row_lse(d_row)
                    log_p = min(log_p + (tlp[tok] - dlp[tok]), 0.0)
                clp[i] = float(np.float32(log_p))  # kernel stores fp32

        resid_mass = {}
        if not is_greedy:
            for j in range(1, num_draft):
                if int(self.ds[start + j + 1]) < 0:
                    continue
                # the kernel's residual-mass kernel maps the row through
                # (expanded_idx_mapping, expanded_local_pos) and skips rows
                # whose mapped step is 0 or >= K (honest layouts satisfy
                # eidx[row] == idx[req], elocal[row] == j -> never skipped)
                r_state = int(self.eidx[start + j])
                r_step = int(self.elocal[start + j])
                if r_step == 0 or r_step >= self.K:
                    raise AssertionError(
                        "misaligned row mapping leaves the residual mass "
                        "uninitialized (undefined kernel behavior); the gate "
                        "does not construct such inputs")
                if self.temp[r_state] == 0.0:
                    continue
                tau_j = float(np.exp(clp[j - 1]))
                t_row = self.tl[start + j]
                p_row = np.exp(t_row - self._row_lse(t_row))
                d_row = self.dl[r_state, r_step] / self.temp[r_state]
                q_row = np.exp(d_row - self._row_lse(d_row))
                resid_mass[j] = float(np.maximum(tau_j * p_row - q_row,
                                                 0.0).sum())

        verifying = True
        accepted_length = 0
        accept_trace = []
        for i in range(num_draft):
            tok = int(self.ds[start + i + 1])
            is_valid = tok >= 0
            if is_greedy:
                if verifying:
                    u = acceptance_uniform(seed, int(self.pos[start + i]))
                    target_argmax = int(greedy_argmax[i])
                    accepted = (target_argmax == max(tok, 0)) and is_valid
                    verifying = accepted
                    if accepted:
                        accepted_length = i + 1
                    accept_trace.append((i, u, accepted))
            else:
                verifying &= is_valid
                if verifying:
                    u = acceptance_uniform(seed, int(self.pos[start + i]))
                    tau_next = float(np.exp(clp[i]))
                    next_tok = (int(self.ds[start + i + 2])
                                if i < num_draft - 1 else -1)
                    if next_tok >= 0:
                        Z = resid_mass[i + 1]
                        denom = Z + 1.0 - tau_next
                        h = Z / denom if denom > 0 else 1.0
                    else:
                        h = tau_next
                    if u <= h:
                        accepted_length = i + 1
                    accept_trace.append((i, u, h, bool(u <= h)))

        L = accepted_length
        resample_row = start + L
        is_bonus = L == num_draft
        state_at_row = int(self.eidx[resample_row])
        temp_at_row = self.temp[state_at_row]
        rejected_draft = None if is_bonus else int(self.ds[resample_row + 1])
        margin = None
        if is_greedy and not is_bonus:
            resampled = int(greedy_argmax[L])
        else:
            if is_bonus or rejected_draft < 0:
                residual_logits = self.tl[resample_row].copy()
            else:
                log_p_tau = 0.0 if L == 0 else clp[L - 1]
                t_row = self.tl[resample_row]
                # target_lse: per-row target stats (no request mapping)
                target_lp = (t_row - self._row_lse(t_row)) + log_p_tau
                # the residual's draft DISTRIBUTION is read at
                # [eidx[resample_row], L] with temp[eidx[resample_row]]
                d_row = self.dl[state_at_row, L] / temp_at_row
                # ...while draft_lse comes from the rejected row's local
                # stats, which the stats kernel computed through the row's
                # own (eidx, elocal) mapping - these differ exactly when the
                # row mapping is misaligned (the defect fingerprint).
                lse_state = int(self.eidx[resample_row])
                lse_step = int(self.elocal[resample_row])
                d_stats = self.dl[lse_state, lse_step] / self.temp[lse_state]
                draft_lp = d_row - self._row_lse(d_stats)
                with np.errstate(all="ignore"):
                    ratio = np.exp(draft_lp - target_lp)
                    residual_logits = np.where(
                        ratio < 1.0, target_lp + np.log1p(-ratio), -np.inf)
                residual_logits = np.nan_to_num(residual_logits, nan=-np.inf,
                                                posinf=-np.inf)
            if temp_at_row == 0.0:
                vals = residual_logits
                resampled = int(np.argmax(vals))
            else:
                # the resample gumbel draw uses seed/temp of the row's
                # mapped state (eidx[resample_row])
                g = gumbel_seed_of(int(self.seeds[state_at_row]),
                                   int(self.pos[resample_row]))
                if self.use_fp64:
                    noise = np.array([gumbel_noise_fp64(g, x)
                                      for x in range(self.V)])
                    vals = residual_logits + noise
                else:
                    noise = np.array([gumbel_noise_fp32(g, x)
                                      for x in range(self.V)],
                                     dtype=np.float32)
                    with np.errstate(invalid="ignore", over="ignore"):
                        vals = residual_logits.astype(np.float32) + noise
                    vals = np.where(np.isnan(vals), -np.inf, vals)
                resampled = int(np.argmax(vals))
                finite = vals[np.isfinite(vals)]
                if finite.size >= 2:
                    srt = np.sort(finite)[::-1]
                    margin = float(srt[0] - srt[1])
        return {"accepted_length": L, "resampled": resampled,
                "accept_trace": accept_trace, "clp": clp,
                "resid_mass": resid_mass, "is_bonus": is_bonus,
                "resample_margin": margin, "greedy_argmax": greedy_argmax}


# ===========================================================================
# harness
# ===========================================================================
DEV = torch.device("cuda")


class GateCase:
    def __init__(self, R, K, V, max_reqs=None):
        self.R = R
        self.K = K
        self.V = V
        self.max_reqs = max_reqs or (R + 4)
        self.num_logits = R * (K + 1)

    def alloc(self):
        R, K, V, mr = self.R, self.K, self.V, self.max_reqs
        self.target = torch.zeros(self.num_logits, V, dtype=torch.float32,
                                  device=DEV)
        self.draft_logits = torch.zeros(mr, K, V, dtype=torch.float32,
                                        device=DEV)
        self.temperature = torch.ones(mr, dtype=torch.float32, device=DEV)
        self.seeds = torch.zeros(mr, dtype=torch.int64, device=DEV)
        self.input_ids = torch.zeros(self.num_logits, dtype=torch.int32,
                                     device=DEV)
        self.pos = torch.zeros(self.num_logits, dtype=torch.int64, device=DEV)
        self.expanded_idx_mapping = torch.zeros(self.num_logits,
                                                dtype=torch.int32, device=DEV)
        self.expanded_local_pos = torch.zeros(self.num_logits,
                                              dtype=torch.int32, device=DEV)
        self.cu = torch.zeros(R + 1, dtype=torch.int32, device=DEV)
        self.idx_mapping = torch.zeros(R, dtype=torch.int32, device=DEV)
        self.expected_valid = [K] * R
        return self


def run_kernel(case, use_fp64=False):
    from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import (
        rejection_sample,
    )
    sampled, num_sampled = rejection_sample(
        case.target, case.draft_logits, case.input_ids, case.cu, case.pos,
        case.idx_mapping, case.expanded_idx_mapping, case.expanded_local_pos,
        case.temperature, case.seeds, case.K, None, use_fp64=use_fp64,
        use_block_verification=True)
    torch.cuda.synchronize()
    return sampled.cpu().numpy(), num_sampled.cpu().numpy()


def run_oracle(case, use_fp64=False):
    v = case.numpy_view()
    oracle = BlockVerifierOracle(
        v["target"], v["draft"], v["ids"], v["cu"], v["pos"], v["idx"],
        v["eidx"], v["elocal"], v["temp"], v["seeds"], case.K,
        use_fp64=use_fp64)
    return oracle.run()


GateCase.numpy_view = lambda self: dict(
    target=self.target.cpu().numpy(),
    draft=self.draft_logits.cpu().numpy(),
    temp=self.temperature.cpu().numpy(),
    seeds=self.seeds.cpu().numpy(),
    ids=self.input_ids.cpu().numpy(),
    pos=self.pos.cpu().numpy(),
    eidx=self.expanded_idx_mapping.cpu().numpy(),
    elocal=self.expanded_local_pos.cpu().numpy(),
    cu=self.cu.cpu().numpy(),
    idx=self.idx_mapping.cpu().numpy(),
)


def run_kernel_num_rejected(case, num_sampled, seq_lens, prefill_len=None):
    from vllm.v1.worker.gpu.input_batch import get_num_sampled_and_rejected
    if prefill_len is None:
        # prefill_len <= seq_len -> not chunked prefilling
        prefill_len = torch.zeros(case.max_reqs, dtype=torch.int32,
                                  device=DEV)
    else:
        prefill_len = torch.as_tensor(prefill_len, dtype=torch.int32,
                                      device=DEV)
    ns = torch.as_tensor(num_sampled, dtype=torch.int32,
                         device=DEV).clone()
    sl = torch.as_tensor(seq_lens, dtype=torch.int32, device=DEV)
    ns2, nrej = get_num_sampled_and_rejected(
        ns, sl, case.cu, case.idx_mapping, prefill_len)
    torch.cuda.synchronize()
    return ns2.cpu().numpy(), nrej.cpu().numpy()


def sample_proposals_via_cache(case, rows, q_logits, base_positions,
                               pos_offset=1, use_fp64=False):
    """Sample proposals through the REAL cache-writing drafter path.

    Mirrors Speculator.sample_draft (speculator.py:336): gumbel_sample over the
    pre-temperature q rows with logits_cache=case.draft_logits and
    logits_cache_col=step, positions+1. The cache write IS the production
    proposal-logit cache the verifier later reads.
    """
    from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample
    K = case.K
    rows_t = torch.as_tensor(np.asarray(rows), dtype=torch.int32, device=DEV)
    draft_tokens = torch.zeros(case.max_reqs, K, dtype=torch.int64,
                               device=DEV)
    for step in range(K):
        pos = torch.as_tensor(np.asarray(base_positions) + step,
                              dtype=torch.int64, device=DEV)
        tok = gumbel_sample(
            q_logits[rows_t, step], rows_t, case.temperature, case.seeds,
            pos + pos_offset, apply_temperature=True,
            logits_cache=case.draft_logits,
            logits_cache_col=torch.tensor(step, dtype=torch.int64, device=DEV),
            use_fp64=use_fp64)
        draft_tokens[rows_t, step] = tok.to(torch.int64)
    torch.cuda.synchronize()
    return draft_tokens.cpu().numpy()


def fill_window(case, req, draft_tokens_row, base_pos, valid_len=None,
                first_token=7, state_row=None):
    K = case.K
    start = req * (K + 1)
    valid_len = K if valid_len is None else valid_len
    state_row = req if state_row is None else state_row
    case.expected_valid[req] = valid_len
    case.input_ids[start] = first_token
    for j in range(1, K + 1):
        case.input_ids[start + j] = (int(draft_tokens_row[j - 1])
                                     if j <= valid_len else -1)
    for j in range(K + 1):
        case.pos[start + j] = base_pos + j
        case.expanded_idx_mapping[start + j] = state_row
        case.expanded_local_pos[start + j] = j
    case.cu[req + 1] = case.cu[req] + (K + 1)
    case.idx_mapping[req] = state_row


def validate_window_layout(case):
    """Contract: -1 pads form a suffix and the valid draft count equals
    the window length the case spec (expected_valid) declares."""
    v = case.numpy_view()
    K = case.K
    for r in range(case.R):
        start = r * (K + 1)
        ids = v["ids"][start:start + K + 1]
        seen_invalid = False
        for j in range(1, K + 1):
            tok = ids[j]
            if tok < 0:
                seen_invalid = True
            elif seen_invalid:
                return False, (r, j, "valid token after -1 pad")
        n_valid = int((ids[1:] >= 0).sum())
        if n_valid != case.expected_valid[r]:
            return False, (r, n_valid,
                           f"valid draft count {n_valid} != expected_valid "
                           f"{case.expected_valid[r]}")
    return True, None


def compare_kernel_oracle(case, tag, use_fp64=False):
    s_k, n_k = run_kernel(case, use_fp64=use_fp64)
    s_o, n_o, details = run_oracle(case, use_fp64=use_fp64)
    mism = []
    margins = []
    for r in range(case.R):
        nk, no = int(n_k[r]), int(n_o[r])
        if nk != no:
            mism.append((r, "num_sampled", nk, no))
            continue
        if s_k[r, :nk].tolist() != s_o[r, :nk].tolist():
            mism.append((r, "tokens", s_k[r, :nk].tolist(),
                         s_o[r, :nk].tolist()))
        m = details[r]["resample_margin"]
        if m is not None:
            margins.append(m)
    ok = not mism
    _record(tag, ok, {
        "mismatches": mism[:4], "num_sampled": n_k[:case.R].tolist(),
        "min_resample_gumbel_margin":
            float(min(margins)) if margins else None})
    return ok, s_k, n_k, details


# ===========================================================================
# statistical machinery (predetermined criteria)
# ===========================================================================
def binomial_max_dev(emp, p, n):
    if n == 0 or p <= 0.0 or p >= 1.0:
        return 0.0 if emp == p else float("inf")
    return abs(emp - p) / np.sqrt(p * (1.0 - p) / n)


def chi_square_sf(stat, df):
    from math import lgamma
    if stat <= 0:
        return 1.0
    s, x = df / 2.0, stat / 2.0
    if x < s + 1.0:
        term = 1.0 / s
        total, k = term, s
        for n_i in range(1, 300):
            k += 1.0
            term *= x / k
            total += term
            if term < total * 1e-15:
                break
        p = total * np.exp(-x + s * np.log(x) - lgamma(s))
        return float(max(0.0, min(1.0, 1.0 - p)))
    b = x + 1.0 - s
    c = 1e300
    d = 1.0 / b
    h = d
    for i in range(1, 300):
        an = -i * (i - s)
        b += 2.0
        d = an * d + b
        if abs(d) < 1e-300:
            d = 1e-300
        c = b + an / c
        if abs(c) < 1e-300:
            c = 1e-300
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-15:
            break
    q = np.exp(-x + s * np.log(x) - lgamma(s)) * h
    return float(max(0.0, min(1.0, q)))


# ===========================================================================
# case builders
# ===========================================================================
def _random_case(R, K, V, seed, valid_lens=None, temps=None, cap_stress=0.0,
                 rows=None):
    case = GateCase(R, K, V).alloc()
    torch.manual_seed(seed)
    tgt = torch.randn(case.num_logits, V, dtype=torch.float32,
                      device=DEV) * 2.0
    tgt = tgt + torch.arange(V, dtype=torch.float32, device=DEV) * 0.013
    case.target.copy_(tgt)
    dl = torch.randn(case.max_reqs, K, V, dtype=torch.float32,
                     device=DEV) * 2.0
    dl = dl + torch.arange(V, dtype=torch.float32, device=DEV) * 0.011
    if cap_stress:
        # Non-uniform draft under-confidence on even vocab entries: raises
        # p/q there so the stepwise running product genuinely binds on those
        # branches. (A uniform shift is invariant under both softmax and
        # gumbel-argmax and would be a vacuous duplicate of the base case.)
        dl[..., ::2] = dl[..., ::2] - cap_stress
    case.draft_logits.copy_(dl)
    temps = temps or [1.0]
    temps = (temps * ((R // len(temps)) + 1))[:R]
    for r, t in enumerate(temps):
        case.temperature[r] = t
    case.seeds[:] = torch.randint(1, 2**40, (case.max_reqs,),
                                  dtype=torch.int64, device=DEV)
    rows = list(range(R)) if rows is None else list(rows)
    rows = (rows * ((R // len(rows)) + 1))[:R]
    dt = sample_proposals_via_cache(case, rows, case.draft_logits,
                                    np.arange(R) * 1000)
    for r in range(R):
        vl = (valid_lens[r % len(valid_lens)]
              if valid_lens is not None else K)
        fill_window(case, r, dt[r], r * 1000, valid_len=vl,
                    state_row=rows[r])
    return case


def _forced_case(R, K, V, force, seed):
    """Explicit small-support distributions forcing the outcome.

    reject_all: target support excludes every draft token -> every ratio ~ 0
    -> tau ~ 0 -> h ~ 0 -> nothing accepted; committed = single resampled
    (bonus) token. accept_all: q == p -> every h = 1 -> full block + bonus.
    """
    case = GateCase(R, K, V).alloc()
    eps = 1e-4
    g = np.random.default_rng(seed)
    for r in range(R):
        base = r * 1000
        for step in range(K):
            p = np.full(V, eps)
            p[(2 * step) % V] = 1.0 - eps * (V - 1)
            if force == "reject_all":
                q = np.full(V, eps)
                q[(2 * step + 1) % V] = 1.0 - eps * (V - 1)
                draft_tok = (2 * step + 1) % V
            else:
                q = p.copy()
                draft_tok = (2 * step) % V
            row = r * (K + 1) + step
            case.target[row] = torch.tensor(np.log(p), dtype=torch.float32,
                                            device=DEV)
            case.draft_logits[r, step] = torch.tensor(
                np.log(q), dtype=torch.float32, device=DEV)
        # bonus row target: a distinct delta so the bonus token is checkable
        p_bonus = np.full(V, eps)
        p_bonus[(2 * K) % V] = 1.0 - eps * (V - 1)
        case.target[r * (K + 1) + K] = torch.tensor(
            np.log(p_bonus), dtype=torch.float32, device=DEV)
        case.temperature[r] = 1.0
        case.seeds[r] = int(g.integers(1, 2**40))
        fill_window(case, r, [(2 * s + 1) % V if force == "reject_all"
                              else (2 * s) % V for s in range(K)], base)
    return case


# ===========================================================================
# sections
# ===========================================================================
def _triton_rng_probe(seeds, poss, V):
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice as tldevice
    from vllm.v1.worker.gpu.sample.gumbel import tl_rand32, tl_rand64

    @triton.jit
    def _probe(seeds_ptr, poss_ptr, out_acc, out_g32, out_g64,
               VV: tl.constexpr):
        i = tl.program_id(0)
        seed = tl.load(seeds_ptr + i)
        pos = tl.load(poss_ptr + i)
        u_acc = tl_rand32(seed, pos, includes_zero=False)
        tl.store(out_acc + i, u_acc)
        x = tl.arange(0, VV)
        g = tl.randint(seed, pos)
        u32 = tl_rand32(g, x, includes_zero=False)
        tl.store(out_g32 + i * VV + x, -tl.log(-tldevice.log1p(-u32)))
        u64 = tl_rand64(g, x, includes_zero=False)
        tl.store(out_g64 + i * VV + x, -tl.log(-tl.log(u64)))

    n = len(seeds)
    st = torch.tensor(seeds, dtype=torch.int64, device=DEV)
    pt = torch.tensor(poss, dtype=torch.int64, device=DEV)
    o_acc = torch.empty(n, dtype=torch.float32, device=DEV)
    o_g32 = torch.empty(n, V, dtype=torch.float32, device=DEV)
    o_g64 = torch.empty(n, V, dtype=torch.float64, device=DEV)
    _probe[(n,)](st, pt, o_acc, o_g32, o_g64, VV=V)
    torch.cuda.synchronize()
    return o_acc.cpu().numpy(), o_g32.cpu().numpy(), o_g64.cpu().numpy()


def section_rng(quick=False):
    _section("rng")
    t0 = time.time()
    rng = np.random.default_rng(1015)
    n = 32 if quick else 257
    V = 64
    seeds = rng.integers(-(2**63), 2**63 - 1, size=n, dtype=np.int64)
    poss = rng.integers(0, 2**33, size=n, dtype=np.int64)
    k_acc, k_g32, k_g64 = _triton_rng_probe(seeds, poss, V)
    bad_acc = bad32 = bad64 = 0
    for i in range(n):
        if acceptance_uniform(int(seeds[i]), int(poss[i])) != float(k_acc[i]):
            bad_acc += 1
        g = gumbel_seed_of(int(seeds[i]), int(poss[i]))
        for x in range(V):
            o32 = gumbel_noise_fp32(g, x)
            if abs(o32 - float(k_g32[i, x])) > 2e-6 * max(1.0, abs(o32)):
                bad32 += 1
            o64 = gumbel_noise_fp64(g, x)
            if abs(o64 - float(k_g64[i, x])) > 1e-9 * max(1.0, abs(o64)):
                bad64 += 1
    _record("acceptance_draw_bitexact", bad_acc == 0,
            {"mismatches": bad_acc, "samples": n})
    _record("gumbel_fp32_noise", bad32 == 0,
            {"mismatches": bad32, "samples": n * V})
    _record("gumbel_fp64_noise", bad64 == 0,
            {"mismatches": bad64, "samples": n * V})
    _finish_section("rng", t0)


def section_exact(quick=False):
    _section("exact")
    t0 = time.time()
    R = 3 if quick else 8
    cells = []

    def cell(tag, case, **kw):
        ok, _, _, _ = compare_kernel_oracle(case, tag, **kw)
        cells.append((tag, ok))

    cell("K3_basic_T1", _random_case(R, 3, 64, seed=1))
    cell("K4_basic_T1", _random_case(R, 4, 64, seed=2))
    cell("K4_mixed_temps", _random_case(R, 4, 64, seed=3,
                                        temps=[1.0, 0.7, 0.0, 1.0]))
    cell("K4_cap_stress", _random_case(R, 4, 64, seed=4, cap_stress=3.0))
    cell("K4_partial_windows",
         _random_case(R, 4, 64, seed=5, valid_lens=[2, 0, 4, 1]))
    cell("K3_fp64_gumbel", _random_case(3, 3, 64, seed=6), use_fp64=True)
    cell("V2500_multiblock", _random_case(2, 3, 2500, seed=7))
    cell("rows_reordered", _random_case(R, 3, 64, seed=8,
                                        rows=[2, 0, 1, 3][:R]))
    cell("rows_reused", _random_case(R, 3, 64, seed=9,
                                     rows=[0, 0, 1, 1][:R]))
    # compare_kernel_oracle already recorded each cell with its mismatch/
    # margin details; re-recording here with empty details would clobber
    # them (review-found regression: 9 cells lost their diagnostics).
    for name, _ in cells:
        assert name in RESULTS["exact"], name

    # 4 -> 3 -> 4 width transition: separate calls, shared request seeds.
    trans = []
    for K in (4, 3, 4):
        case = GateCase(4, K, 64).alloc()
        torch.manual_seed(55)
        tgt = torch.randn(case.num_logits, 64, dtype=torch.float32,
                          device=DEV) * 2.0
        tgt = tgt + torch.arange(64, dtype=torch.float32,
                                 device=DEV) * 0.013
        case.target.copy_(tgt)
        dl = torch.randn(case.max_reqs, K, 64, dtype=torch.float32,
                         device=DEV) * 2.0
        case.draft_logits.copy_(dl)
        case.seeds[:] = ((torch.arange(case.max_reqs, dtype=torch.int64,
                                       device=DEV) + 77) * 104729)
        dt = sample_proposals_via_cache(case, list(range(4)),
                                        case.draft_logits, np.arange(4) * 1000)
        vls = {4: [4, 3, 4, 2], 3: [3, 3, 3, 2]}[K]
        for r in range(4):
            fill_window(case, r, dt[r], r * 1000, valid_len=vls[r])
        ok, _, n_k, _ = compare_kernel_oracle(case, f"transition_K{K}")
        trans.append({"K": K, "ok": ok, "num_sampled": n_k[:4].tolist()})
    _record("transition_4_3_4", all(t["ok"] for t in trans),
            {"calls": trans,
             "note": "separate rejection_sample calls at K=4, K=3, K=4 with "
                     "the same per-request seed stream and fresh windows"})

    # forced outcomes
    case = _forced_case(3, 3, 16, "reject_all", seed=11)
    ok, _, n_k, _ = compare_kernel_oracle(case, "reject_every_position")
    _record("reject_every_position_L0",
            ok and bool((np.asarray(n_k) == 1).all()),
            {"num_sampled": np.asarray(n_k).tolist(),
             "note": "forced rejection at every position -> committed = the "
                     "single resampled token drawn from row-0 residual"})
    case = _forced_case(3, 3, 16, "accept_all", seed=12)
    ok, _, n_k, _ = compare_kernel_oracle(case, "full_accept_bonus")
    _record("full_accept_bonus_LK",
            ok and bool((np.asarray(n_k) == 4).all()),
            {"num_sampled": np.asarray(n_k).tolist(),
             "note": "q == p -> full block accepted; bonus from the K-row"})
    _finish_section("exact", t0)


def section_invariants(quick=False):
    _section("invariants")
    t0 = time.time()
    R, K, V = (2 if quick else 6), 4, 64

    # (1) cached q == the distribution that produced the proposal.
    case = GateCase(R, K, V).alloc()
    torch.manual_seed(21)
    q_logits = (torch.randn(case.max_reqs, K, V, dtype=torch.float32,
                            device=DEV) * 2.0
                + torch.arange(V, dtype=torch.float32, device=DEV) * 0.011)
    case.target.copy_(torch.randn(case.num_logits, V, dtype=torch.float32,
                                  device=DEV) * 2.0)
    case.temperature.fill_(0.8)
    case.seeds[:] = (torch.arange(case.max_reqs, dtype=torch.int64,
                                  device=DEV) * 31337 + 5)
    q_np = q_logits.cpu().numpy()
    dt = sample_proposals_via_cache(case, list(range(R)), q_logits,
                                    np.arange(R) * 1000)
    v = case.numpy_view()
    cache_ok = True
    reprod_ok = True
    for r in range(R):
        for step in range(K):
            if not np.array_equal(case.draft_logits[r, step].cpu().numpy(),
                                  q_np[r, step]):
                cache_ok = False
            g = gumbel_seed_of(int(v["seeds"][r]), int(r * 1000 + step + 1))
            scaled = (q_np[r, step] / float(v["temp"][r])).astype(np.float32)
            noise = np.array([gumbel_noise_fp32(g, x) for x in range(V)],
                             dtype=np.float32)
            if int(np.argmax(scaled + noise)) != int(dt[r, step]):
                reprod_ok = False
    _record("cached_q_equals_proposal_dist", cache_ok,
            {"note": "cache rows bitwise-equal the q logits the proposals "
                     "were sampled from (production write/read pair)"})
    _record("proposal_tokens_reproduced_from_cache", reprod_ok,
            {"note": "every proposal token re-derived from the cached row via "
                     "the recorded (seed, pos) gumbel draw"})

    # (2) invalid -1 suffix never contributes: K=4 window with 2 valid drafts
    #     + -1 tail commits exactly what a K=2 window commits.
    case_a = GateCase(2, 4, 32).alloc()
    case_b = GateCase(2, 2, 32).alloc()
    torch.manual_seed(22)
    tgt_a = torch.randn(case_a.num_logits, 32, dtype=torch.float32,
                        device=DEV) * 2.0
    case_a.target.copy_(tgt_a)
    case_b.target.copy_(tgt_a.view(2, 5, 32)[:, :3].reshape(-1, 32))
    dl_a = torch.randn(case_a.max_reqs, 4, 32, dtype=torch.float32,
                       device=DEV) * 2.0
    case_a.draft_logits.copy_(dl_a)
    case_b.draft_logits.copy_(dl_a[:, :2].contiguous())
    case_a.temperature.fill_(1.0)
    case_b.temperature.fill_(1.0)
    case_a.seeds[:] = torch.arange(case_a.max_reqs, dtype=torch.int64,
                                   device=DEV) * 677 + 1
    case_b.seeds[:] = case_a.seeds
    dt_a = sample_proposals_via_cache(case_a, [0, 1], case_a.draft_logits,
                                      np.arange(2) * 1000)
    dt_b = sample_proposals_via_cache(case_b, [0, 1], case_b.draft_logits,
                                      np.arange(2) * 1000)
    for r in range(2):
        fill_window(case_a, r, dt_a[r], r * 1000, valid_len=2)
        fill_window(case_b, r, dt_b[r], r * 1000, valid_len=2)
    sk_a, nk_a = run_kernel(case_a)
    sk_b, nk_b = run_kernel(case_b)
    s_o, n_o, _ = run_oracle(case_a)
    # compare COMMITTED prefixes only (slots >= num_sampled are uninitialized
    # kernel buffer memory by design; the consumer never reads them)
    committed_a = [sk_a[r, :int(nk_a[r])].tolist() for r in range(2)]
    committed_b = [sk_b[r, :int(nk_b[r])].tolist() for r in range(2)]
    committed_o = [s_o[r, :int(n_o[r])].tolist() for r in range(2)]
    equiv = committed_a == committed_b
    oracle_ok = committed_a == committed_o
    _record("minus1_tail_equivalence", equiv and oracle_ok,
            {"K4_2valid_plus_tail": committed_a,
             "num_a": nk_a[:2].tolist(),
             "K2_window": committed_b,
             "num_b": nk_b[:2].tolist(),
             "oracle_matches_K4": oracle_ok,
             "note": "invalid suffix contributes nothing: the K=4 window "
                     "with a 2-valid prefix commits exactly the K=2 window's "
                     "output (same seeds/positions/distributions); only "
                     "committed prefixes are compared (later slots are "
                     "uninitialized buffer memory by kernel design)"})

    # (3) committed counts & rollback observable.
    case = _random_case(4, 3, 64, seed=23)
    s_k, n_k = run_kernel(case)
    ns2, nrej = run_kernel_num_rejected(case, n_k,
                                        np.array([10, 10, 10, 10],
                                                 dtype=np.int32))
    counts_ok = bool((ns2 == n_k).all()
                     and (nrej == (case.K + 1) - n_k).all())
    ns3, nrej3 = run_kernel_num_rejected(
        case, n_k, np.array([3, 10, 10, 10], dtype=np.int32),
        prefill_len=np.full(case.max_reqs, 5, dtype=np.int32))
    chunked_ok = bool(ns3[0] == 0 and nrej3[0] == 0
                      and (ns3[1:] == ns2[1:]).all())
    _record("committed_counts_and_rollback", counts_ok and chunked_ok,
            {"num_sampled": ns2.tolist(), "num_rejected": nrej.tolist(),
             "chunked_prefill_req0": {"num_sampled": int(ns3[0]),
                                      "num_rejected": int(nrej3[0])},
             "note": "committed tokens = accepted prefix + resample; "
                     "num_rejected = window suffix to roll back; chunked "
                     "prefill zeroes the commit"})

    # (4) greedy independence: fixed greedy target, three proposal sets.
    case = GateCase(3, 3, 16).alloc()
    chain = [3, 9, 1, 12]
    for j in range(4):
        row = np.full(16, -6.0)
        row[chain[j]] = 6.0
        for r in range(3):
            case.target[r * 4 + j] = torch.tensor(row, dtype=torch.float32,
                                                  device=DEV)
    case.temperature.fill_(0.0)
    case.seeds[:] = torch.arange(case.max_reqs, dtype=torch.int64,
                                 device=DEV) + 91
    sets = [[chain[0], chain[1], chain[2]],
            [chain[0], (chain[1] + 1) % 16, chain[2]],
            [chain[0], chain[1], -1]]
    outs = []
    for toks in sets:
        for r in range(3):
            fill_window(case, r, toks, r * 100)
        s_k, n_k = run_kernel(case)
        outs.append([(s_k[r, :int(n_k[r])].tolist()) for r in range(3)])
    prefixes = all(o == [chain[:len(o_r)] for o_r in o] for o in outs)
    _record("greedy_independence", prefixes,
            {"committed": outs,
             "note": "with a fixed greedy target the committed continuation "
                     "is the same argmax chain regardless of proposals "
                     "(full-accept, mismatch-at-1, -1-tail proposal sets)"})
    _finish_section("invariants", t0)


def _stat_cap_event(n_runs, per_call, seed):
    """K=2 cap-event case (chained through fixed rows):
    p=(.5,.5); q0=(.05,.95) [r0=(10,.526)]; q1=(.95,.05) [r1=(.526,10)].
    Branch out1=0 caps the running product (r0*r1 = 5.26 > 1).
    Predetermined criteria: P(out1)=.5; P(out2|out1=a, L>=1) = .5 for all
    (a,b); extended joint = .25; z <= 5 per cell; TV <= 0.015.
    """
    V = 2
    p0 = np.array([0.5, 0.5])
    q0 = np.array([0.05, 0.95])
    p1 = np.array([0.5, 0.5])
    q1 = np.array([0.95, 0.05])
    rng = np.random.default_rng(seed)
    counts = np.zeros((2, 2), dtype=np.int64)
    counts_single = np.zeros(2, dtype=np.int64)
    lengths = np.zeros(4, dtype=np.int64)
    logp, logq0, logq1 = np.log(p0), np.log(q0), np.log(q1)
    q0_cdf, q1_cdf = np.cumsum(q0), np.cumsum(q1)
    dev_t = torch.tensor(np.stack([logp, logp, logp]),
                         dtype=torch.float32, device=DEV)
    q0_t = torch.tensor(np.stack([logq0, logq0]), dtype=torch.float32,
                        device=DEV)
    q1_t = torch.tensor(np.stack([logq1, logq1]), dtype=torch.float32,
                        device=DEV)
    for c_i in range(n_runs // per_call):
        case = GateCase(per_call, 2, V).alloc()
        case.target.copy_(dev_t.repeat(per_call, 1))
        case.draft_logits[:per_call, 0].copy_(q0_t[0])
        case.draft_logits[:per_call, 1].copy_(q1_t[0])
        case.temperature.fill_(1.0)
        case.seeds[:per_call] = torch.arange(500000 + c_i * per_call,
                                             500000 + (c_i + 1) * per_call,
                                             dtype=torch.int64, device=DEV)
        t1 = np.searchsorted(q0_cdf, rng.random(per_call)).clip(0, 1)
        t2 = np.searchsorted(q1_cdf, rng.random(per_call)).clip(0, 1)
        case.input_ids.copy_(torch.tensor(
            np.stack([np.full(per_call, 7), t1, t2], axis=1).reshape(-1),
            dtype=torch.int32, device=DEV))
        pos = (torch.arange(per_call, device=DEV, dtype=torch.int64)
               * 1000).unsqueeze(1) + torch.arange(
            3, device=DEV, dtype=torch.int64).unsqueeze(0)
        case.pos.copy_(pos.reshape(-1))
        case.expanded_idx_mapping.copy_(
            torch.arange(per_call, device=DEV, dtype=torch.int32)
            .repeat_interleave(3))
        case.expanded_local_pos.copy_(
            torch.arange(3, device=DEV, dtype=torch.int32).repeat(per_call))
        case.cu.copy_(torch.arange(per_call + 1, device=DEV,
                                   dtype=torch.int32) * 3)
        case.idx_mapping.copy_(torch.arange(per_call, device=DEV,
                                            dtype=torch.int32))
        s_k, n_k = run_kernel(case)
        for r in range(per_call):
            n = int(n_k[r])
            lengths[n] += 1
            if n >= 2:
                counts[int(s_k[r, 0]), int(s_k[r, 1])] += 1
            else:
                counts_single[int(s_k[r, 0])] += 1
    N = int(counts.sum() + counts_single.sum())
    pair = counts / N
    single = counts_single / N
    marg1 = pair.sum(axis=1) + single
    ext = pair + np.outer(single, p1)
    cond = counts / np.maximum(counts.sum(axis=1, keepdims=True), 1)
    z1 = max(binomial_max_dev(marg1[a], p0[a], N) for a in range(2))
    zc = []
    for a in range(2):
        na = int(counts[a].sum())
        for b in range(2):
            zc.append(binomial_max_dev(cond[a, b], 0.5, na))
    ze = max(binomial_max_dev(ext[a, b], 0.25, N)
             for a in range(2) for b in range(2))
    return {"N": N, "length_hist": (lengths[1:] / N).round(6).tolist(),
            "P_out1": marg1.round(6).tolist(),
            "P_out2_given_out1_Lge1": cond.round(6).tolist(),
            "extended_joint": ext.round(6).tolist(),
            "z_marginal": float(z1), "z_conditional_max": float(max(zc)),
            "z_extended_max": float(ze),
            "tv_extended": float(np.abs(ext - 0.25).sum() / 2.0),
            "counts": counts.tolist()}


def _exact_joint_v2(p_rows, q_rows):
    """Exact committed-sequence distribution for V=2, all-valid K window,
    replicating BlockVerifierOracle semantics mechanically in fp64 (the
    K>=3 chained statistical reference; closes the review-found gap that
    the shipped statistical coverage was a single K=2 cell).

    p_rows: K+1 target distributions (rows 0..K-1 draft positions, row K
    the bonus/resample row target).  q_rows: K draft distributions.

    Per proposal path (2^K), the h vector is path-determined (clp is
    stepwise-capped fp32 as the kernel stores it; Z_j from tau_j =
    exp(clp[j-1])); the argmax rule makes the length distribution
      P(L=0) = prod(1-h_j);  P(L=m) = h_{m-1} * prod_{j>=m}(1-h_j);
      P(L=K) = h_{K-1}
    (u draws are independent uniforms; rows before the max-accepting row
    do not constrain the max).  Committed = path[:L] + resample, where the
    resample is the normalized residual max(tau_L*p_L - q_L, 0) for a
    mid-rejection (all-zero residual -> numpy argmax picks token 0, the
    kernel's actual -inf+noise behavior) and a direct categorical from the
    resample row's target for the bonus (L==K).

    Returns (joint: dict tuple->prob, hist: list P(num_committed=k)
    for k = 1..K+1, i.e. hist[k-1] = P(L=k-1)).
    """
    K = len(q_rows)
    joint = {}
    hist = [0.0] * (K + 1)
    for path in itertools.product((0, 1), repeat=K):
        path_prob = 1.0
        for i, t in enumerate(path):
            path_prob *= q_rows[i][t]
        if path_prob <= 0.0:
            continue
        clp = []
        log_p = 0.0
        for i, t in enumerate(path):
            r = p_rows[i][t] / q_rows[i][t]
            log_p = min(log_p + math.log(r), 0.0)
            clp.append(float(np.float32(log_p)))  # kernel stores fp32
        Z = {}
        for j in range(1, K):
            tau_j = math.exp(clp[j - 1])
            Z[j] = sum(max(tau_j * p_rows[j][x] - q_rows[j][x], 0.0)
                       for x in (0, 1))
        h = []
        for i in range(K):
            tau_next = math.exp(clp[i])
            if i < K - 1:
                denom = Z[i + 1] + 1.0 - tau_next
                h.append(Z[i + 1] / denom if denom > 0 else 1.0)
            else:
                h.append(tau_next)
        len_prob = {0: 1.0}
        for j in range(K):
            len_prob[0] *= (1.0 - h[j])
        for m in range(1, K):
            pr = h[m - 1]
            for j in range(m, K):
                pr *= (1.0 - h[j])
            len_prob[m] = pr
        len_prob[K] = h[K - 1]
        for L, pL in len_prob.items():
            if pL <= 0.0:
                continue
            hist[L] += path_prob * pL  # hist indexed by L; +1 on export
            if L == K:
                for x in (0, 1):
                    if p_rows[K][x] > 0.0:
                        key = tuple(path) + (x,)
                        joint[key] = joint.get(key, 0.0) + \
                            path_prob * pL * p_rows[K][x]
            else:
                log_p_tau = 0.0 if L == 0 else clp[L - 1]
                tau_L = math.exp(log_p_tau)
                res = [max(tau_L * p_rows[L][x] - q_rows[L][x], 0.0)
                       for x in (0, 1)]
                Zr = res[0] + res[1]
                for x in (0, 1):
                    if Zr > 0.0 and res[x] > 0.0:
                        key = tuple(path[:L]) + (x,)
                        joint[key] = joint.get(key, 0.0) + \
                            path_prob * pL * res[x] / Zr
                    elif Zr <= 0.0 and x == 0:
                        # all-zero residual: gumbel over -inf+noise ->
                        # numpy argmax picks token 0 (oracle-mirrored)
                        key = tuple(path[:L]) + (0,)
                        joint[key] = joint.get(key, 0.0) + path_prob * pL
    return joint, list(hist)


def _stat_chained(tag, p_rows, q_rows, n_runs, per_call, seed,
                  quick=False):
    """Chained cap-event distributional cells at K>=3/K=4 through the REAL
    kernel vs the exact enumeration.

    Predetermined criteria (stated before running, per the astra gate spec):
      - marginal P(out1) vs p_rows[0]: z <= 5 per token
      - scheme-free cascade conditionals P(out_k = x | out_1..out_{k-1},
        L >= k) vs target row p_rows[k-1]: z <= 5 for prefixes with
        >= 1000 samples (the criteria the independent review proved have
        power against prod-cap / tau-off-by-one / stale-q' defect classes)
      - length histogram vs enumeration: max abs deviation <= 0.006
      - full committed-sequence TV(empirical, enumeration) <= 0.04
      - per-sequence z <= 5 for cells with analytic P >= 1e-4
    """
    K = len(q_rows)
    joint, hist_L = _exact_joint_v2(p_rows, q_rows)
    # hist_L is indexed by L (= num_committed - 1)
    ana_len = hist_L
    rng = np.random.default_rng(seed)
    rows_per = K + 1
    seq_counts = {}
    len_hist = np.zeros(K + 2, dtype=np.int64)
    cond = [{} for _ in range(K + 2)]  # cond[k][prefix] -> [c0, c1]; k up to K+1
    p_log = np.log(np.asarray(p_rows))     # (K+1, V)
    q_log = np.log(np.asarray(q_rows))     # (K, V)
    q_cdf = [np.cumsum(q) for q in q_rows]
    for c_i in range(n_runs // per_call):
        case = GateCase(per_call, K, 2).alloc()
        case.target.copy_(torch.tensor(
            np.tile(p_log, (per_call, 1)), dtype=torch.float32,
            device=DEV))
        for i in range(K):
            case.draft_logits[:per_call, i].copy_(torch.tensor(
                q_log[i], dtype=torch.float32, device=DEV))
        case.temperature.fill_(1.0)
        case.seeds[:per_call] = torch.arange(
            600_000 + c_i * per_call, 600_000 + (c_i + 1) * per_call,
            dtype=torch.int64, device=DEV)
        toks = np.stack(
            [np.searchsorted(q_cdf[i], rng.random(per_call)).clip(0, 1)
             for i in range(K)], axis=1)
        case.input_ids.copy_(torch.tensor(
            np.concatenate([np.full((per_call, 1), 7, dtype=np.int64),
                            toks.astype(np.int64)], axis=1).reshape(-1),
            dtype=torch.int32, device=DEV))
        pos = (torch.arange(per_call, device=DEV, dtype=torch.int64)
               .unsqueeze(1) * 1000
               + torch.arange(rows_per, device=DEV,
                              dtype=torch.int64).unsqueeze(0))
        case.pos.copy_(pos.reshape(-1))
        case.expanded_idx_mapping.copy_(
            torch.arange(per_call, device=DEV, dtype=torch.int32)
            .repeat_interleave(rows_per))
        case.expanded_local_pos.copy_(
            torch.arange(rows_per, device=DEV, dtype=torch.int32)
            .repeat(per_call))
        case.cu.copy_(torch.arange(per_call + 1, device=DEV,
                                   dtype=torch.int32) * rows_per)
        case.idx_mapping.copy_(torch.arange(per_call, device=DEV,
                                            dtype=torch.int32))
        s_k, n_k = run_kernel(case)
        for r in range(per_call):
            n = int(n_k[r])
            len_hist[n] += 1
            seq = tuple(int(x) for x in s_k[r, :n])
            seq_counts[seq] = seq_counts.get(seq, 0) + 1
            for k in range(1, n + 1):
                prefix = seq[:k - 1]
                slot = cond[k].setdefault(prefix, [0, 0])
                slot[seq[k - 1]] += 1
    N = int(len_hist.sum())
    emp_len = (len_hist[1:K + 2].astype(np.float64) / N)
    z_marg = max(
        binomial_max_dev(
            sum(v for kk, v in seq_counts.items() if kk[0] == a) / N,
            p_rows[0][a], N)
        for a in (0, 1))
    z_cond = []
    for k in range(1, K + 1):
        for prefix, cc in cond[k].items():
            npre = cc[0] + cc[1]
            if npre < 1000:
                continue
            for x in (0, 1):
                z_cond.append(binomial_max_dev(
                    cc[x] / npre, p_rows[k - 1][x], npre))
    emp_seq = {k: v / N for k, v in seq_counts.items()}
    tv = 0.0
    for key in set(emp_seq) | set(joint):
        tv += abs(emp_seq.get(key, 0.0) - joint.get(key, 0.0))
    tv /= 2.0
    z_seq = []
    for key, p in joint.items():
        if p >= 1e-4:
            z_seq.append(binomial_max_dev(emp_seq.get(key, 0.0), p, N))
    len_dev = float(np.abs(emp_len - np.asarray(ana_len)).max())
    res = {"N": N, "K": K,
           "z_marginal": float(z_marg),
           "z_conditional_max": float(max(z_cond)) if z_cond else 0.0,
           "n_cond_cells": len(z_cond),
           "tv_sequence": float(tv),
           "z_sequence_max": float(max(z_seq)) if z_seq else 0.0,
           "len_hist_emp": emp_len.round(6).tolist(),
           "len_hist_exact": np.asarray(ana_len).round(6).tolist(),
           "len_max_dev": len_dev}
    ok = (res["z_marginal"] <= 5.0 and res["z_conditional_max"] <= 5.0
          and res["tv_sequence"] <= 0.04 and res["z_sequence_max"] <= 5.0
          and len_dev <= 0.006)
    _record(tag, ok, res)
    return ok


def _oracle_mc_joint(p_rows, q_rows, n_runs, seed):
    """CPU Monte-Carlo of BlockVerifierOracle itself over the config;
    cross-checks the enumeration against the oracle (three-way:
    published analytic <-> enumeration <-> oracle MC; the kernel cells
    then close the loop to the real GPU stack)."""
    K = len(q_rows)
    rows_per = K + 1
    rng = np.random.default_rng(seed)
    counts = {}
    len_hist = np.zeros(K + 2, dtype=np.int64)
    tlog = np.log(np.asarray(p_rows))
    dlog = np.log(np.asarray(q_rows))
    q_cdf = [np.cumsum(q) for q in q_rows]
    done = 0
    while done < n_runs:
        r = min(200, n_runs - done)
        tl = np.tile(tlog, (r, 1))
        dl = np.tile(dlog, (r, 1, 1))
        seeds = np.arange(9_000_000 + done, 9_000_000 + done + r,
                          dtype=np.int64)
        toks = np.stack(
            [np.searchsorted(q_cdf[i], rng.random(r)).clip(0, 1)
             for i in range(K)], axis=1)
        ds = np.zeros(r * rows_per, dtype=np.int64)
        ds += 7
        for rr in range(r):
            ds[rr * rows_per + 1:rr * rows_per + K + 1] = toks[rr]
        pos = (np.arange(r, dtype=np.int64)[:, None] * 1000
               + np.arange(rows_per, dtype=np.int64)[None, :]).reshape(-1)
        cu = np.arange(r + 1, dtype=np.int64) * rows_per
        idx = np.arange(r, dtype=np.int64)
        eidx = np.repeat(idx, rows_per)
        elocal = np.tile(np.arange(rows_per, dtype=np.int64), r)
        orc = BlockVerifierOracle(tl, dl, ds, cu, pos, idx, eidx, elocal,
                                  np.ones(r), seeds, K, use_fp64=False)
        s_k, n_k, _ = orc.run()
        for rr in range(r):
            n = int(n_k[rr])
            len_hist[n] += 1
            seq = tuple(int(x) for x in s_k[rr, :n])
            counts[seq] = counts.get(seq, 0) + 1
        done += r
    N = int(len_hist.sum())
    emp = {k: v / N for k, v in counts.items()}
    return emp, (len_hist[1:K + 2] / N).tolist(), N


def section_statistical(quick=False):
    _section("statistical")
    t0 = time.time()
    n_cap = 40_000 if quick else 120_000
    res = _stat_cap_event(n_cap, 250, seed=777)
    ok = (res["z_marginal"] <= 5.0 and res["z_conditional_max"] <= 5.0
          and res["z_extended_max"] <= 5.0 and res["tv_extended"] <= 0.015)
    _record("cap_event_distribution_exact", ok, res)

    # ---- chained K>=3/K=4 statistical cells (review-found coverage gap).
    # The enumeration is self-tested against the INDEPENDENTLY derived K=2
    # cap-event analytic numbers (reviewer re-derivation, kernel-validated
    # in the recorded cell above) before it is used as a reference.
    half = 20_000 if quick else 120_000
    half_pc = 250
    half_p = [np.array([0.5, 0.5])] * 4
    half_q2 = [np.array([0.05, 0.95]), np.array([0.95, 0.05])]
    j2, h2 = _exact_joint_v2(half_p[:3], half_q2)
    pub_hist = [0.4501, 0.225, 0.325]
    # the reviewer's published joint is (out1, out2) restricted to n >= 2
    # (its mass sums to P(n>=2) = 1 - P(n=1)); verified to machine precision
    rest = [0.0, 0.0, 0.0, 0.0]
    for seq, p in j2.items():
        if len(seq) >= 2:
            rest[seq[0] * 2 + seq[1]] += p
    pub_ext = [0.025, 0.025, 0.25, 0.25]
    dev_hist = float(np.abs(np.asarray(h2) - np.asarray(pub_hist)).max())
    dev_ext = float(np.abs(np.asarray(rest) - np.asarray(pub_ext)).max())
    _record("enumeration_matches_published", dev_hist <= 2e-3 and
            dev_ext <= 2e-3,
            {"hist_dev": dev_hist, "ext_dev": dev_ext,
             "hist": np.asarray(h2).round(6).tolist(),
             "joint_nge2": np.asarray(rest).round(6).tolist()})
    enum_ok = dev_hist <= 2e-3 and dev_ext <= 2e-3

    # oracle MC cross-check of the enumeration on each config (CPU)
    configs = [
        ("stat_cascade_A_K3", [0.5, 0.5], 3,
         [[0.05, 0.95], [0.95, 0.05], [0.05, 0.95]]),
        ("stat_cascade_B_K3_capdrop", [0.5, 0.5], 3,
         [[0.05, 0.95], [0.5, 0.5], [0.95, 0.05]]),
        ("stat_cascade_C_K4", [0.5, 0.5], 4,
         [[0.05, 0.95], [0.95, 0.05], [0.05, 0.95], [0.95, 0.05]]),
    ]
    n_mc = 8_000 if quick else 20_000
    for tag, _, _, qcfg in configs:
        p_rows = [np.array([0.5, 0.5])] * (len(qcfg) + 1)
        q_rows = [np.array(q) for q in qcfg]
        emp, mc_hist, N_mc = _oracle_mc_joint(p_rows, q_rows, n_mc, 31)
        joint, hist_L = _exact_joint_v2(p_rows, q_rows)
        ana_len = hist_L
        zmax = 0.0
        for key, p in joint.items():
            if p >= 2e-3:
                zmax = max(zmax, binomial_max_dev(emp.get(key, 0.0), p,
                                                  N_mc))
        mc_dev = float(np.abs(np.asarray(mc_hist)
                              - np.asarray(ana_len)).max())
        _record("oracle_mc_" + tag, zmax <= 5.0 and mc_dev <= 0.01,
                {"N": N_mc, "z_max": float(zmax), "len_dev": mc_dev})
        enum_ok = enum_ok and zmax <= 5.0 and mc_dev <= 0.01

    if enum_ok:
        for tag, _, _, qcfg in configs:
            p_rows = [np.array([0.5, 0.5])] * (len(qcfg) + 1)
            q_rows = [np.array(q) for q in qcfg]
            if not _stat_chained(tag, p_rows, q_rows, half, half_pc,
                                 seed=88 + len(tag), quick=quick):
                enum_ok = False
    else:
        _record("stat_cascade_skipped_enum_mismatch", False,
                {"reason": "enumeration failed its self-test; refusing "
                           "to use it as a kernel reference"})
    _finish_section("statistical", t0)


def section_controls(quick=False):
    """Sensitivity controls: the gate must REJECT each injected defect."""
    _section("controls")
    t0 = time.time()
    N = 8_000 if quick else 40_000

    # ---- (a) D-007 stale cached q' (proposals from q=(0.9,0.1), target
    #         p=(0.5,0.5), cache q'=(0.5,0.5)).
    def d007_run(stale_cache, n_runs):
        rng = np.random.default_rng(4242)
        per_call = 200
        counts = np.zeros(2, dtype=np.int64)
        cache_flagged = 0
        for c_i in range(n_runs // per_call):
            case = GateCase(per_call, 1, 2).alloc()
            logp = np.log([0.5, 0.5])
            logq = np.log([0.9, 0.1])
            logq_stale = np.log([0.5, 0.5])
            tgt_t = torch.tensor(np.stack([logp, logp]),
                                 dtype=torch.float32, device=DEV)
            cache_t = torch.tensor(
                np.stack([logq_stale if stale_cache else logq,
                          logq_stale if stale_cache else logq]),
                dtype=torch.float32, device=DEV)
            case.target.copy_(tgt_t.repeat(per_call, 1))
            case.draft_logits[:per_call, 0].copy_(cache_t[0])
            case.temperature.fill_(1.0)
            case.seeds[:per_call] = torch.arange(
                900000 + c_i * per_call, 900000 + (c_i + 1) * per_call,
                dtype=torch.int64, device=DEV)
            # proposals from q=(0.9,0.1): token 0 w.p. 0.9 (inverse CDF)
            t1 = (rng.random(per_call) >= 0.9).astype(np.int64)
            case.input_ids.copy_(torch.tensor(
                np.stack([np.full(per_call, 7), t1], axis=1).reshape(-1),
                dtype=torch.int32, device=DEV))
            pos = (torch.arange(per_call, device=DEV, dtype=torch.int64)
                   * 1000).unsqueeze(1) + torch.arange(
                2, device=DEV, dtype=torch.int64).unsqueeze(0)
            case.pos.copy_(pos.reshape(-1))
            case.expanded_idx_mapping.copy_(
                torch.arange(per_call, device=DEV, dtype=torch.int32)
                .repeat_interleave(2))
            case.expanded_local_pos.copy_(
                torch.arange(2, device=DEV, dtype=torch.int32)
                .repeat(per_call))
            case.cu.copy_(torch.arange(per_call + 1, device=DEV,
                                       dtype=torch.int32) * 2)
            case.idx_mapping.copy_(torch.arange(per_call, device=DEV,
                                                dtype=torch.int32))
            cached = case.draft_logits[:per_call, 0].cpu().numpy()
            # invariant: the cache must equal the proposal distribution q
            if not np.allclose(cached, logq, atol=1e-6):
                cache_flagged += 1
            s_k, _ = run_kernel(case)
            counts += np.bincount(s_k[:per_call, 0], minlength=2)
        emp = counts / counts.sum()
        return {"N": int(counts.sum()), "P_out": emp.round(5).tolist(),
                "z_vs_target": float(binomial_max_dev(emp[0], 0.5,
                                                      int(counts.sum()))),
                "cache_invariant_flagged": cache_flagged > 0}

    honest = d007_run(False, N)
    stale = d007_run(True, N)
    _record("D007_honest_passes",
            honest["z_vs_target"] <= 5.0
            and not honest["cache_invariant_flagged"], honest)
    _record("D007_stale_rejected",
            stale["z_vs_target"] > 5.0 and stale["cache_invariant_flagged"],
            stale)

    # ---- (b) poisoned -1 tails. q == p at every step so every valid
    #         position accepts deterministically: the clean window (2 valid
    #         drafts + -1 tail) commits 3 tokens; a poisoned tail (valid-looking
    #         ids in the pad slots) would commit 5.
    case_clean = GateCase(2, 4, 32).alloc()
    torch.manual_seed(31)
    tgt = (torch.randn(case_clean.num_logits, 32, dtype=torch.float32,
                       device=DEV) * 2.0
           + torch.arange(32, dtype=torch.float32, device=DEV) * 0.013)
    case_clean.target.copy_(tgt)
    # cache rows == the target rows of the corresponding step (q == p)
    for r in range(2):
        for step in range(4):
            case_clean.draft_logits[r, step] = tgt[r * 5 + step]
    case_clean.temperature.fill_(1.0)
    case_clean.seeds[:] = (torch.arange(case_clean.max_reqs,
                                        dtype=torch.int64,
                                        device=DEV) * 31 + 1)
    dt = sample_proposals_via_cache(case_clean, [0, 1],
                                    case_clean.draft_logits,
                                    np.arange(2) * 1000)
    for r in range(2):
        fill_window(case_clean, r, dt[r], r * 1000, valid_len=2)
    ok_clean, _ = validate_window_layout(case_clean)
    s_clean, n_clean = run_kernel(case_clean)
    case_poison = GateCase(2, 4, 32).alloc()
    case_poison.target.copy_(tgt)
    case_poison.draft_logits.copy_(case_clean.draft_logits)
    case_poison.temperature.fill_(1.0)
    case_poison.seeds[:] = case_clean.seeds
    for r in range(2):
        fill_window(case_poison, r, dt[r], r * 1000, valid_len=4)
    case_poison.expected_valid = [2, 2]  # the case SPEC says 2 valid drafts
    ok_poison, where = validate_window_layout(case_poison)
    s_poison, n_poison = run_kernel(case_poison)
    v = case_clean.numpy_view()
    sim = BlockVerifierOracle(
        v["target"], v["draft"],
        np.where(v["ids"] < 0, 0, v["ids"]),  # defect: -1 treated as token 0
        v["cu"], v["pos"], v["idx"], v["eidx"], v["elocal"], v["temp"],
        v["seeds"], 4)
    s_sim, n_sim, _ = sim.run()
    differs = any(
        s_clean[r, :int(n_clean[r])].tolist()
        != s_poison[r, :int(n_poison[r])].tolist()
        or int(n_clean[r]) != int(n_poison[r]) for r in range(2))
    sim_differs = any(
        s_sim[r, :int(n_sim[r])].tolist()
        != s_clean[r, :int(n_clean[r])].tolist()
        or int(n_sim[r]) != int(n_clean[r]) for r in range(2))
    _record("poisoned_tails_rejected",
            ok_clean and (not ok_poison) and differs and sim_differs,
            {"clean_layout_valid": ok_clean,
             "poisoned_layout_flag": ok_poison,
             "poisoned_flag_detail": where,
             "poisoned_output_differs_from_clean": differs,
             "defect_simulator_output_differs": sim_differs,
             "clean": {"num_sampled": n_clean[:2].tolist()},
             "poisoned": {"num_sampled": n_poison[:2].tolist()},
             "note": "gate rejects valid-looking ids replacing the -1 pad: "
                     "layout validator flags the spec violation; the outputs "
                     "differ; and a defect simulator (treating -1 as valid) "
                     "is distinguished from the pinned kernel"})

    # ---- (c) misaligned width transition: a K=4 window whose row start+3 is
    #         mapped at a stale/foreign cache row (the 4->3 transition
    #         defect). Fully deterministic design:
    #           p0=p1=delta_g, q0=q1=delta_g   -> positions 0,1 accept (h=1)
    #           p2 = (.5 g, .5 h); q2 = (.8 g, .2 h) -> r2 = .625
    #                                          -> clp2 = log .625 -> tau3<1
    #           p3 = delta_g (zero mass at the realized t4=b); q3 = delta_b
    #                                          -> r3 = 0 -> tau4 = 0 -> h3 = 0
    #           honest    h2 = Z3/(Z3+1-tau3), Z3 = tau3*p3(g) = .625
    #                                 -> h2 = .625 (accepts for u2 <= .625)
    #           defected row start+3 reads a STALE row holding p3
    #                                 -> Z3' = max(.625*p3 - p3, 0) = 0
    #                                 -> h2' = 0 (never accepts)
    #           seed scan picks u2 in (0, .625] -> honest commits L=3,
    #           defected commits L=2 with a single-point residual (token h):
    #           every request's committed output differs.
    n_mis = 8 if quick else 24
    V_mis = 64
    g_tok, h_tok, b_tok = 3, 17, 11
    case = GateCase(n_mis, 4, V_mis).alloc()
    case.temperature.fill_(1.0)
    for r in range(n_mis):
        start = r * 5
        rows = {
            0: {g_tok: 6.0},                        # p0 = delta_g
            1: {g_tok: 6.0},                        # p1 = delta_g
            2: {g_tok: 6.0, h_tok: 6.0},            # p2 = (.5, .5)
            3: {g_tok: 6.0, b_tok: -1e4},           # p3 = delta_g, 0 at b
        }
        drafts = {
            0: {g_tok: 6.0},                        # q0 = delta_g (r0 = 1)
            1: {g_tok: 6.0},                        # q1 = delta_g (r1 = 1)
            2: {g_tok: 6.0, h_tok: 6.0 - np.log(4.0)},   # q2 = (.8, .2)
            3: {b_tok: 6.0},                        # q3 = delta_b (r3 = 0)
        }
        for step in range(4):
            p_row = np.full(V_mis, -1e4)
            for tok, val in rows[step].items():
                p_row[tok] = val
            case.target[start + step] = torch.tensor(
                p_row, dtype=torch.float32, device=DEV)
            q_row = np.full(V_mis, -1e4)
            for tok, val in drafts[step].items():
                q_row[tok] = val
            case.draft_logits[r, step] = torch.tensor(
                q_row, dtype=torch.float32, device=DEV)
        fill_window(case, r, [g_tok, g_tok, g_tok, b_tok], r * 1000,
                    valid_len=4)
    # stale decoy rows: step 1 holds a copy of p3 (what the defected mapping
    # reads instead of the step-3 proposal distribution)
    decoy_rows = list(range(n_mis, n_mis + 4))
    for d_r in decoy_rows:
        case.draft_logits[d_r, 1] = case.target[g_tok + 3]  # p3 is identical
        for step in (0, 2, 3):
            case.draft_logits[d_r, step] = case.target[g_tok + 3]
    # provisional seeds; extract the seed-independent h2 thresholds from the
    # oracle accept traces, then scan recorded seeds for the deterministic
    # flip (honest accepts position 2, defected rejects it)
    case.seeds[:n_mis] = torch.arange(700000, 700000 + n_mis,
                                      dtype=torch.int64, device=DEV)
    _, _, det_h = run_oracle(case)
    case_mis_prov = GateCase(n_mis, 4, V_mis).alloc()
    case_mis_prov.target.copy_(case.target)
    case_mis_prov.draft_logits.copy_(case.draft_logits)
    case_mis_prov.temperature.copy_(case.temperature)
    case_mis_prov.seeds[:] = case.seeds
    case_mis_prov.expected_valid = [4] * n_mis
    for r in range(n_mis):
        fill_window(case_mis_prov, r, [g_tok, g_tok, g_tok, b_tok],
                    r * 1000, valid_len=4)
        start = r * 5
        case_mis_prov.expanded_idx_mapping[start + 3] = decoy_rows[r % 4]
        case_mis_prov.expanded_local_pos[start + 3] = 1
    _, _, det_m = run_oracle(case_mis_prov)
    h2_honest = det_h[0]["accept_trace"][2][2]
    h2_defect = det_m[0]["accept_trace"][2][2]
    picks = []
    for r in range(n_mis):
        pos2 = int(r * 1000 + 2)
        s = 700000 + r * 131
        for _ in range(100000):
            if acceptance_uniform(s, pos2) <= h2_honest and \
                    not (acceptance_uniform(s, pos2) <= h2_defect):
                picks.append(s)
                break
            s += 7919
        else:
            raise AssertionError("seed scan failed for request %d" % r)
    case.seeds[:n_mis] = torch.tensor(picks, dtype=torch.int64, device=DEV)
    s_h, n_h, _ = run_oracle(case)
    s_k, n_k = run_kernel(case)
    honest_match = all(
        s_k[r, :int(n_k[r])].tolist() == s_h[r, :int(n_k[r])].tolist()
        and int(n_k[r]) == int(n_h[r]) for r in range(n_mis))
    case_mis = case_mis_prov
    case_mis.seeds[:] = case.seeds
    s_mis, n_mis_c, _ = run_oracle(case_mis)
    sk_mis, nk_mis = run_kernel(case_mis)
    cache_flagged = 0
    for r in range(n_mis):
        read_q = case.draft_logits[decoy_rows[r % 4], 1].cpu().numpy()
        true_q = case.draft_logits[r, 3].cpu().numpy()
        if not np.allclose(read_q, true_q, atol=1e-6):
            cache_flagged += 1
    diverged = sum(
        1 for r in range(n_mis)
        if sk_mis[r, :int(nk_mis[r])].tolist()
        != s_k[r, :int(n_k[r])].tolist()
        or int(nk_mis[r]) != int(n_k[r]))
    oracle_reproduces = all(
        sk_mis[r, :int(nk_mis[r])].tolist()
        == s_mis[r, :int(nk_mis[r])].tolist()
        and int(nk_mis[r]) == int(n_mis_c[r]) for r in range(n_mis))
    _record("misaligned_width_rejected",
            cache_flagged == n_mis and honest_match and oracle_reproduces
            and diverged == n_mis,
            {"requests": n_mis, "h2_honest": float(h2_honest),
             "h2_defected": float(h2_defect),
             "cache_invariant_flagged": cache_flagged,
             "honest_kernel_matches_oracle": honest_match,
             "oracle_reproduces_defected_kernel": oracle_reproduces,
             "outputs_diverged": diverged,
             "honest_num_sampled": n_k[:n_mis].tolist(),
             "defected_num_sampled": nk_mis[:n_mis].tolist(),
             "note": "row start+3 remapped to a stale cache row: the "
                     "cache-vs-proposal invariant fires for every request, "
                     "the exact oracle reproduces the defected kernel, and "
                     "every request's committed output flips (honest L=3 "
                     "commits [g,g,g,g]; defected L=2 commits [g,g,h])"})

    # ---- (d) swapped row mappings.
    case = GateCase(2, 2, 32).alloc()
    torch.manual_seed(33)
    tgt = torch.randn(case.num_logits, 32, dtype=torch.float32,
                      device=DEV) * 2.0
    case.target.copy_(tgt)
    q2 = torch.randn(2, 2, 32, dtype=torch.float32, device=DEV) * 2.0
    case.draft_logits[:2].copy_(q2)
    case.temperature.fill_(1.0)
    case.seeds[:2] = torch.tensor([111, 222], dtype=torch.int64, device=DEV)
    dt = sample_proposals_via_cache(case, [0, 1], q2, np.array([0, 1000]))
    for r in range(2):
        fill_window(case, r, dt[r], r * 1000, valid_len=2)
    s_h, n_h, _ = run_oracle(case)
    s_k, n_k = run_kernel(case)
    honest_match = all(
        s_k[r, :int(n_k[r])].tolist() == s_h[r, :int(n_k[r])].tolist()
        and int(n_k[r]) == int(n_h[r]) for r in range(2))
    case_sw = GateCase(2, 2, 32).alloc()
    case_sw.target.copy_(tgt)
    case_sw.draft_logits[:2].copy_(q2)
    case_sw.temperature.fill_(1.0)
    case_sw.seeds[:] = case.seeds
    for r in range(2):
        fill_window(case_sw, r, dt[r], r * 1000, valid_len=2,
                    state_row=1 - r)  # defect: swapped mapping
    s_sw, n_sw, _ = run_oracle(case_sw)
    sk_sw, nk_sw = run_kernel(case_sw)
    swapped_differs = any(
        sk_sw[r, :int(nk_sw[r])].tolist() != s_k[r, :int(nk_sw[r])].tolist()
        or int(nk_sw[r]) != int(n_k[r]) for r in range(2))
    oracle_reproduces = all(
        sk_sw[r, :int(nk_sw[r])].tolist()
        == s_sw[r, :int(nk_sw[r])].tolist()
        and int(nk_sw[r]) == int(n_sw[r]) for r in range(2))
    _record("swapped_rows_rejected",
            swapped_differs and honest_match and oracle_reproduces,
            {"honest_kernel_matches_oracle": honest_match,
             "swapped_output_differs": swapped_differs,
             "oracle_reproduces_swapped_kernel": oracle_reproduces,
             "note": "swapped expanded_idx_mapping: each request reads the "
                     "other request's cache row/seed/temperature; the exact "
                     "oracle and the cache invariant both flag it"})
    _finish_section("controls", t0)


# ===========================================================================
# main
# ===========================================================================
def env_facts():
    import triton
    import vllm
    facts = {"torch": torch.__version__, "triton": triton.__version__,
             "vllm": vllm.__version__,
             "gpu": torch.cuda.get_device_name(0),
             "host_python": sys.version.split()[0],
             "image": os.environ.get("GB10_GATE_IMAGE", "unrecorded")}
    base = "/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu/"
    for rel in ("spec_decode/rejection_sampler.py",
                "spec_decode/rejection_sampler_utils.py"):
        try:
            with open(base + rel, "rb") as f:
                facts["sha256:" + rel.split("/")[-1]] = hashlib.sha256(
                    f.read()).hexdigest()
        except OSError:
            facts["sha256:" + rel.split("/")[-1]] = "MISSING"
    return facts


def run_gate(quick=False, sections=("rng", "exact", "invariants",
                                    "statistical", "controls")):
    assert torch.cuda.is_available(), "GPU required"
    RESULTS["meta"] = env_facts()
    RESULTS["meta"]["gate_spec"] = ("R-031 kernel-level stochastic gate "
                                    "(astra task-mv8oapbk)")
    if "rng" in sections:
        section_rng(quick)
    if "exact" in sections:
        section_exact(quick)
    if "invariants" in sections:
        section_invariants(quick)
    if "statistical" in sections:
        section_statistical(quick)
    if "controls" in sections:
        section_controls(quick)
    failed = [f"{sec}/{k}" for sec in ("rng", "exact", "invariants",
                                       "statistical")
              for k, v in RESULTS.get(sec, {}).items()
              if k != "_elapsed_s" and not v["ok"]]
    ctrl_bad = [k for k, v in RESULTS.get("controls", {}).items()
                if k != "_elapsed_s" and not v["ok"]]
    RESULTS["verdict"] = {"gate_passed": not failed and not ctrl_bad,
                          "failed_checks": failed,
                          "controls_not_rejected": ctrl_bad}
    return RESULTS["verdict"]["gate_passed"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--sections",
                    default="rng,exact,invariants,statistical,controls")
    args = ap.parse_args()
    ok = run_gate(args.quick, tuple(args.sections.split(",")))
    out = json.dumps(RESULTS, indent=2, default=float)
    print(out)
    if args.json_out:
        with open(args.json_out, "w") as f:
            f.write(out)
    return 0 if ok else 1


def test_blockverifier_gate_quick():
    """pytest entry point: full gate in quick mode, assert-guarded."""
    RESULTS.clear()
    RESULTS.update({"meta": {}, "verdict": {}})
    ok = run_gate(quick=True)
    assert ok, json.dumps(RESULTS["verdict"], indent=2)


if __name__ == "__main__":
    sys.exit(main())
