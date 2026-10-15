# SPDX-License-Identifier: Apache-2.0
"""AWFP boot fingerprints for the P2 stage-1 closure (D-014 condition 1).

Runtime BEHAVIORAL fingerprints -- beyond env hashes -- proving on every
boot that (a) the moe-align stable-sort shim is ACTIVE (a probe call
through the patched path with a known input asserts the canonical
within-group output order), (b) atomic_add=0 is in effect for the Marlin
MoE GEMMs (the call-site flag is hard-set false at the two
moe_wna16_marlin_gemm calls -- a load-time source check), and (c) the
cache/kernel-selection policy is RECORDED (the triton autotune selections
-- fla/GDN kernels --, the force-first-config pinning state, the inductor
compile-cache entry count and the cudagraph capture sizes are hashed at
boot into one policy_sha), so cross-boot drift (the autotune boot-lottery
that D1/D3 exposed, R-029) is DETECTABLE: a mismatch against the previous
boot's policy_sha is reported LOUDLY as an AWFP-DRIFT line (and the
harness compares the policy_sha of every boot it manages itself).

Every line carries the fixed tag ``AWFP`` followed by one JSON object with
a ``kind`` field:

  CODE     the marlin module's own load-time code-state check (emitted at
           import of marlin_moe.py, once per process): atomic_add call
           sites == 2 (both hard-set false), the shim marker/call site
           present, the shim env-gate state.
  CONFIG   boot facts at model-load end: versions, the env gates that
           steer kernel selection, cudagraph mode + configured capture
           sizes, spec-decode config, and the marlin code-state.
  PROBE    the behavioral verdict: the shim canonical-order probe (direct
           helper + the real moe_align_block_size composition) and the
           atomic_add call-site assertion. passed=false or an absent
           PROBE line means the boot is NOT proven -- the harness fails
           it as MISSING_INSTRUMENT.
  POLICY   the settled cache/kernel-selection policy hash (emitted at the
           first scheduled step and re-emitted once at step 32 so late
           cudagraph captures land in the last record): policy_sha =
           sha256 over the canonical JSON of {versions, env gates,
           force-first-config state, triton autotuner selections,
           cudagraph configured+captured sizes}. The inductor compile-
           cache ENTRY COUNT is recorded informationally, NOT hashed (the
           cache mount persists across boots and only grows -- a growing
           count is not a per-boot selection change; the runtime
           selections the compile cache produced are covered by the
           triton autotuner records and the captured cudagraph sizes).
  DRIFT    this boot's policy_sha differs from the previous boot's
           persisted record (best-effort, /root/.cache/q38-awfp/) -- the
           boot-lottery fired; reported loudly, never silently.
  FAIL     an internal fingerprint failure (stage named). The fingerprint
           must never break the boot: every stage is guarded.

Pure stdlib at import; torch/vllm imports are lazy and guarded. The
cross-boot drift comparison needs a PERSISTENT cache mount (CACHE_MOUNT
on /root/.cache); without one the in-image comparison is skipped (stated
in the record) and the harness-side comparison across the boots it
manages remains authoritative.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time

TAG = "AWFP"
ENV_AWFP = "VLLM_AWFP"
STATE_DIR = "/root/.cache/q38-awfp"
STATE_FILE = "last_policy.json"

# The env gates recorded in CONFIG and hashed into POLICY (kernel-selection
# steering knobs on this stack).
ENV_KEYS = (
    "VLLM_MOE_ALIGN_STABLESORT",
    "VLLM_MARLIN_USE_ATOMIC_ADD",
    "VLLM_SPECDBG_EVIDENCE",
    "VLLM_TRITON_FORCE_FIRST_CONFIG",
    "VLLM_QSA_DET_TOPK",
    "VLLM_VERIFY_TOPK_TRITON",
    "QWEN38NEXT_LOW_LATENCY_GEMM",
    "QWEN38NEXT_LLG_PDL",
    "VLLM_FP8_HYBRID",
    "VLLM_USE_DEEP_GEMM",
    "VLLM_USE_FLASHINFER_SAMPLER",
    "VLLM_MTP_DRAFT_VOCAB",
    "VLLM_PLE_MMAP_FAST_PATH",
    "VLLM_PLE_MMAP_FAST_ROWS",
    "VLLM_PLE_MMAP_CHUNK",
    "CUBLAS_WORKSPACE_CONFIG",
    "DETERM_BOOTSTRAP",
)

# Module-level one-shot state (per process).
_installed = False
_policy_steps = 0
_policy_emitted_at: set[int] = set()
_POLICY_EMIT_STEPS = (1, 32)
_last_policy_sha: str | None = None

# The autotuner-bearing module filter (bounded walk; fla/GDN kernels are the
# known autotuned population on this stack, plus mamba/SSD ops and any
# inductor-generated module).
_AUTOTUNER_MOD_RE = (
    "flash_linear_attention", "fla", "mamba", "gdn", "qsa", "ple",
    "triton_kernels", "inductor",
)


def awfp_enabled() -> bool:
    """Opt-out env (default ON: the fingerprint is the boot's proof)."""
    return os.environ.get(ENV_AWFP, "1").strip() not in ("", "0")


def _emit(kind: str, payload: dict) -> None:
    record = {"kind": kind, "t": int(time.time() * 1e6), **payload}
    print(f"{TAG} {json.dumps(record, default=str)}", flush=True)


def _env_snapshot() -> dict:
    return {k: os.environ.get(k) for k in ENV_KEYS}


def _marlin_code_state() -> dict | None:
    """The marlin module's own code-state record (import-time check)."""
    try:
        from vllm.model_executor.layers.fused_moe import (  # noqa: E501
            experts as _experts,
        )
        mm = _experts.marlin_moe
        state = getattr(mm, "_Q38_AWFP_CODE", None)
        if state is None:
            state = mm._q38_awfp_code_state()
        return dict(state)
    except Exception as exc:  # noqa: BLE001
        return {"error": repr(exc)}


def _versions() -> dict:
    out = {"python": sys.version.split()[0]}
    for name in ("torch", "triton", "vllm"):
        try:
            mod = __import__(name)
            out[name] = getattr(mod, "__version__", "unknown")
        except Exception as exc:  # noqa: BLE001
            out[name] = f"error:{exc!r}"
    return out


def _ffc_state() -> dict:
    """force-first-config pinning state (the autotune boot-lottery killer)."""
    try:
        from vllm.triton_utils import force_first_config as ffc
        st = {"installed": bool(ffc.is_installed())}
        if st["installed"]:
            try:
                st["n_picked"] = len(ffc._picked_cache)  # noqa: SLF001
            except Exception:  # noqa: BLE001
                st["n_picked"] = None
        return st
    except Exception as exc:  # noqa: BLE001
        return {"error": repr(exc)}


def _autotuner_selections() -> dict:
    """Canonical triton autotune selections (the fla/GDN kernel choices).

    With force-first-config installed the selections live in its module
    cache ((kernel-id, key) -> picked index); the kernel ids are
    process-local, so they are mapped to module-qualified names first.
    Without it, each Autotuner.cache maps key -> chosen Config (the
    benchmarked winner -- the boot-lottery surface itself).
    Bounded: modules whose name matches _AUTOTUNER_MOD_RE, max 64 entries
    per kernel, max 4096 total.
    """
    out: dict = {"n_autotuners": 0, "kernels": {}}
    try:
        import triton as _triton  # noqa: PLC0415
        from triton.runtime.autotuner import Autotuner as _Autotuner
    except Exception as exc:  # noqa: BLE001
        out["error"] = repr(exc)
        return out
    id_to_name: dict[int, str] = {}
    instances: list = []
    for mname, mod in list(sys.modules.items()):
        if mod is None or not isinstance(getattr(mod, "__name__", ""), str):
            continue
        if not any(pat in mname for pat in _AUTOTUNER_MOD_RE):
            continue
        try:
            for attr, obj in vars(mod).items():
                if isinstance(obj, _Autotuner):
                    instances.append(obj)
                    id_to_name[id(obj)] = f"{mname}.{attr}"
        except Exception:  # noqa: BLE001
            continue
    out["n_autotuners"] = len(instances)
    kernels: dict = out["kernels"]
    try:
        from vllm.triton_utils import force_first_config as ffc
        picked = getattr(ffc, "_picked_cache", None)
    except Exception:  # noqa: BLE001
        picked = None
    if picked:
        per_kernel: dict = {}
        for (kid, key), idx in list(picked.items())[:4096]:
            name = id_to_name.get(kid, f"unresolved#{kid}")
            kmap = per_kernel.setdefault(name, {})
            if len(kmap) < 64:
                kmap[str(key)] = int(idx)
        out["mode"] = "force_first_config"
        kernels.update(per_kernel)
    else:
        per_kernel = {}
        for obj in instances:
            name = id_to_name.get(id(obj), "unknown")
            cache = getattr(obj, "cache", None)
            if not isinstance(cache, dict):
                continue
            kmap = per_kernel.setdefault(name, {})
            for key, cfg in list(cache.items())[:64]:
                if len(kmap) >= 64:
                    break
                try:
                    kmap[str(key)] = {
                        "num_warps": getattr(cfg, "num_warps", None),
                        "num_stages": getattr(cfg, "num_stages", None),
                        "kwargs": sorted(
                            (k, str(v)) for k, v in
                            getattr(cfg, "kwargs", {}).items()
                        ),
                    }
                except Exception:  # noqa: BLE001
                    kmap[str(key)] = "unrenderable"
        out["mode"] = "autotuner_cache"
        kernels.update(per_kernel)
        if per_kernel and all(not v for v in per_kernel.values()):
            # honesty note: on this engine's triton (3.7.1) the un-pinned
            # Autotuner.cache dicts are EMPTY at policy time even after the
            # boot-time benchmarking, so the un-pinned selections are NOT
            # observable here -- mode=force_first_config (the pin) is the
            # mechanism that both removes the boot-lottery surface AND
            # records the picks (ffc._picked_cache).
            out["note"] = ("autotuner caches empty at policy time; the "
                           "un-pinned selections are not observable in "
                           "this triton build -- pin "
                           "VLLM_TRITON_FORCE_FIRST_CONFIG to record and "
                           "fix the selections")
    return out


def _cudagraph_state(runner) -> dict:
    out: dict = {}
    try:
        cc = runner.compilation_config
        out["mode"] = str(getattr(cc, "cudagraph_mode", None))
        sizes = getattr(cc, "cudagraph_capture_sizes", None)
        out["configured_sizes"] = sorted(int(s) for s in sizes) if sizes else None
    except Exception as exc:  # noqa: BLE001
        out["configured_error"] = repr(exc)
    try:
        mgr = runner.cudagraph_manager
        counts = mgr.captured_token_counts() if mgr is not None else None
        out["captured_sizes"] = sorted(int(s) for s in counts) if counts else None
    except Exception as exc:  # noqa: BLE001
        out["captured_error"] = repr(exc)
    return out


def _inductor_cache_state() -> dict:
    """Informational, NOT hashed (the cache mount persists and grows)."""
    for cand in (
        os.environ.get("TORCHINDUCTOR_CACHE_DIR"),
        "/root/.cache/vllm/torch_compile_cache",
    ):
        if cand and os.path.isdir(cand):
            try:
                n = sum(len(files) for _, _, files in os.walk(cand))
                return {"dir": cand, "entries": n}
            except Exception as exc:  # noqa: BLE001
                return {"dir": cand, "error": repr(exc)}
    return {"dir": None, "entries": 0}


def _spec_state(runner) -> dict:
    out: dict = {}
    try:
        sc = runner.speculative_config
        for f in ("method", "draft_sample_method", "rejection_sample_method"):
            out[f] = getattr(sc, f, None) if sc is not None else None
    except Exception as exc:  # noqa: BLE001
        out["error"] = repr(exc)
    try:
        out["num_speculative_steps"] = int(runner.num_speculative_steps)
    except Exception:  # noqa: BLE001
        pass
    return out


def policy_snapshot(runner) -> dict:
    """The cache/kernel-selection policy record (hashed into policy_sha).

    Hashed: versions, the kernel-selection env gates, the ffc pinning state
    (+ pick count), the autotune selections and the CONFIGURED cudagraph
    sizes and the spec config -- the things two same-shape boots must agree
    on for a byte-identity comparison to be meaningful.

    Recorded but NOT hashed (both are execution/progress artifacts, not
    policy): cudagraph CAPTURED sizes -- captures land progressively during
    the boot's early steps, so a record collected at step ~32 can carry
    null where an end-of-run record carries the full list (the first live
    matched pair mismatched on exactly this); and the inductor compile-cache
    entry count (the module contract already marked it informational)."""
    snap = {
        "versions": _versions(),
        "env": _env_snapshot(),
        "ffc": _ffc_state(),
        "autotune": _autotuner_selections(),
        "cudagraph": _cudagraph_state(runner),
        "spec": _spec_state(runner),
    }
    inductor = _inductor_cache_state()
    snap["inductor_entries"] = inductor.get("entries")
    snap["_inductor_info"] = inductor  # informational only, excluded below
    cg = dict(snap["cudagraph"]) if isinstance(snap["cudagraph"], dict) else {}
    cg.pop("captured_sizes", None)
    cg.pop("captured_error", None)
    canon = {k: v for k, v in snap.items() if k != "_inductor_info"}
    canon["cudagraph"] = cg
    canon.pop("inductor_entries", None)
    blob = json.dumps(canon, sort_keys=True, default=str).encode()
    snap["policy_sha"] = hashlib.sha256(blob).hexdigest()
    return snap


def _persist_and_compare(policy_sha: str) -> dict:
    """Best-effort cross-boot drift check against the persisted record."""
    result: dict = {"state_dir": STATE_DIR, "drift": None}
    try:
        os.makedirs(STATE_DIR, exist_ok=True)
        path = os.path.join(STATE_DIR, STATE_FILE)
        prev = None
        if os.path.isfile(path):
            with open(path) as fh:
                prev = json.load(fh).get("policy_sha")
        result["prev_sha"] = prev
        if prev is not None and prev != policy_sha:
            result["drift"] = True
        with open(path, "w") as fh:
            json.dump({"policy_sha": policy_sha, "t": time.time()}, fh)
    except Exception as exc:  # noqa: BLE001
        result["error"] = repr(exc)
        result["note"] = (
            "no persistent cache mount: the in-image cross-boot comparison "
            "is unavailable; the harness compares the boots it manages"
        )
    return result


def maybe_install(runner) -> None:
    """Boot-time fingerprints at model-load end (never raises).

    Emits AWFP-CONFIG then runs the behavioral shim/atomic_add probe and
    emits AWFP-PROBE. Guarded one-shot per process; every failure becomes
    an AWFP-FAIL line, never a boot break.
    """
    global _installed
    if _installed or not awfp_enabled():
        return
    _installed = True
    try:
        config = {
            "versions": _versions(),
            "env": _env_snapshot(),
            "spec": _spec_state(runner),
            "cudagraph": _cudagraph_state(runner),
            "marlin_code": _marlin_code_state(),
        }
        _emit("CONFIG", config)
    except Exception as exc:  # noqa: BLE001
        _emit("FAIL", {"stage": "config", "error": repr(exc)})
    try:
        from vllm.model_executor.layers.fused_moe import experts as _experts
        mm = _experts.marlin_moe
        probe = mm._q38_awfp_align_probe(runner.device)
        code = _marlin_code_state() or {}
        passed = (
            probe.get("direct") == "ok"
            and probe.get("composed") == "ok"
            and code.get("atomic_add_sites") == 2
            and code.get("shim_marker") == 1
            and code.get("shim_call_site") == 1
            and bool(code.get("shim_env_on"))
        )
        _emit("PROBE", {
            "passed": bool(passed),
            "probe": probe,
            "code": code,
            "device": str(runner.device),
        })
    except Exception as exc:  # noqa: BLE001
        _emit("FAIL", {"stage": "probe", "error": repr(exc)})


def maybe_emit_policy(runner, requested_k: int | None = None) -> None:
    """Settled policy fingerprint at the first scheduled steps (never raises).

    Called from the runner's draft path with the step's requested K (the
    num_spec_tokens_to_schedule the C2 worker emitter gates on): a step
    with NO speculative verification scheduled -- the JIT-kernel-warmup
    steps carry 0, the dummy/profile runs None -- returns BEFORE the step
    counter advances, so _policy_steps counts SCHEDULED steps only and
    AWFP-POLICY emits at scheduled steps 1 and 32 (late cudagraph captures
    and post-warmup autotune selections land in the LAST record of the
    boot). The gate is producer-side ON PURPOSE: the boot-time policy hash
    stays stable-by-construction, and the emit schedule cannot drift from
    the scheduled steps even if the call site moves.
    """
    global _policy_steps, _last_policy_sha
    if not awfp_enabled():
        return
    if requested_k is None or requested_k <= 0:
        return
    _policy_steps += 1
    if _policy_steps not in _POLICY_EMIT_STEPS:
        return
    if _policy_steps in _policy_emitted_at:
        return
    _policy_emitted_at.add(_policy_steps)
    try:
        snap = policy_snapshot(runner)
        sha = snap.pop("policy_sha")
        drift = _persist_and_compare(sha)
        _last_policy_sha = sha
        _emit("POLICY", {
            "policy_sha": sha,
            "step": _policy_steps,
            # the full components ride the line so a DRIFT can be
            # diagnosed from the boot log alone (which kernel selection
            # moved); bounded by the walker's per-kernel/total caps
            "ffc": snap.get("ffc"),
            "autotune": snap.get("autotune"),
            "cudagraph": snap.get("cudagraph"),
            "env": snap.get("env"),
            "versions": snap.get("versions"),
            "inductor_entries": snap.get("inductor_entries"),
            "drift": drift,
        })
        if drift.get("drift"):
            _emit("DRIFT", {
                "prev_sha": drift.get("prev_sha"),
                "this_sha": sha,
                "note": "boot-lottery fired: the cache/kernel-selection "
                        "policy differs from the previous boot's record "
                        "(see the POLICY record's autotune/ffc components)",
            })
    except Exception as exc:  # noqa: BLE001
        _emit("FAIL", {"stage": "policy", "error": repr(exc)})
