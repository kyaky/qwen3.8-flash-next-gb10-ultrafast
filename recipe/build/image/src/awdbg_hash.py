# SPDX-License-Identifier: Apache-2.0
"""DETHASH: env-gated per-layer output hashing for the D-011 determinism
localization (2026-10-14).

The D-011 bisect (runs/2026-10-14-det-bisect) showed the tie-flip source is
run-to-run numeric variation of the target forward itself, injected
independently at every forward.  This module pins down WHERE inside the
forward the variation first appears: it registers torch forward hooks on
every decoder layer of the target language model (plus the embedding, the
final HC mixer, the model output and the language-model logits output) and
logs one line per layer per forward:

    DETLAYR {"t":..,"f":fwd,"n":ntok,"r":nreq,"L":layer,"i":[..in-hashes..],"o":[..out-hashes..]}

and, in drill mode, one line per sub-module of the drilled layers:

    DETSUB  {...,"L":layer,"sub":"mlp.experts","i":[..],"o":[..]}

Each hash is sha256 of the tensor's bytes after an exact dtype cast
(floating -> float32, integer -> int64; bf16 -> fp32 is exact so hash
(ine)quality reflects the raw bf16 content, NaN payloads included).

Env (read once at install; all default OFF = zero overhead when unset):
  DET_HASH_LAYERS=1        install the hooks at all
  DET_HASH_FILE=<path>     jsonl log path (default /tmp/det_hash.jsonl)
  DET_HASH_DEBUG=1         also log the module inventory as DETHASH-META
  DET_HASH_DRILL="5,7"     additionally hook the sub-modules of these layers
  DET_HASH_DEPTH=2         child depth under each drilled layer (default 2)
  DET_HASH_MIN_LAYERS=8    only install when the largest decoder-layer group
                           has at least this many layers (skips MTP drafters)

Inert unless DET_HASH_LAYERS=1.  Install is called from the patched
GPUModelRunner.load_model (patch_layer_hashes.py) and never raises into
model loading: maybe_install swallows its own errors after logging them,
so a broken hook can degrade the evidence, never the server.
"""
import hashlib
import json
import os
import time

import torch

TENSOR_CAP = 4          # max tensors hashed per module input/output
_LINE_BUDGET = 1 << 22  # hard cap on log size per boot (~4M chars) as a fuse

_state = {
    "file": None,
    "n": 0,            # lines written
    "bytes": 0,
    "fwd": 0,
    "ntok": 0,
    "nreq": 0,
    "dummy": 0,
    "profile": 0,
    "install_id": 0,
    "names": {},       # submodule path -> short id (str) for META
    "errors": 0,
    "drill": set(),
    "depth": 2,
}


def _log(tag, obj):
    st = _state
    if st["n"] > _LINE_BUDGET:
        return
    f = st["file"]
    if f is None:
        path = os.environ.get("DET_HASH_FILE", "/tmp/det_hash.jsonl")
        try:
            f = st["file"] = open(path, "a", buffering=1)
        except OSError as e:
            st["errors"] += 1
            if st["errors"] == 1:
                print(f"DETHASH-FATAL cannot open {path}: {e}", flush=True)
            return
    line = tag + " " + json.dumps(obj, separators=(",", ":")) + "\n"
    try:
        f.write(line)
        st["n"] += 1
        st["bytes"] += len(line)
    except Exception:
        st["errors"] += 1
        f = None  # reopen on next call
        st["file"] = None


def _hash_tensor(t):
    """sha256 of exact-cast bytes; deterministic for identical content."""
    if not isinstance(t, torch.Tensor) or t.numel() == 0:
        return None, None
    if t.dtype.is_floating_point:
        d = t.detach().to(torch.float32)
    elif t.dtype.is_complex:
        return None, None
    else:
        d = t.detach().to(torch.int64)
    d = d.contiguous().cpu()
    try:
        b = d.numpy().tobytes()
    except Exception:
        b = d.view(-1).numpy().tobytes()
    h = hashlib.sha256(b).hexdigest()[:16]
    return h, list(t.shape)


def _pick(obj):
    """Flatten a module input/output into up to TENSOR_CAP (name, tensor)."""
    out = []
    if isinstance(obj, torch.Tensor):
        return [("", obj)]
    if isinstance(obj, dict):
        items = list(obj.items())
    elif isinstance(obj, (tuple, list)):
        items = [(f"{i}", v) for i, v in enumerate(obj)]
    else:
        vals = getattr(obj, "values", None)
        if callable(vals):
            try:
                return _pick(vals())
            except Exception:
                return []
        return []
    for k, v in items:
        if isinstance(v, torch.Tensor):
            out.append((str(k), v))
        elif isinstance(v, (tuple, list, dict)):
            out.extend((f"{k}.{k2}", v2) for k2, v2 in _pick(v))
        if len(out) >= TENSOR_CAP:
            break
    return out[:TENSOR_CAP]


def _hashes(obj):
    res = []
    for k, t in _pick(obj):
        h, s = _hash_tensor(t)
        if h is not None:
            res.append([k, s, h])
    return res


def _pre_hook(layer_id, sub):
    def fn(module, args, kwargs):
        try:
            src = kwargs if kwargs else {}
            if not src:
                src = {f"{i}": v for i, v in enumerate(args)}
            tag = "DETSUB" if sub is not None else "DETLAYR"
            _log(tag, {"t": round(time.time(), 4), "f": _state["fwd"],
                       "n": _state["ntok"], "r": _state["nreq"],
                       "d": _state["dummy"] | (_state["profile"] << 1),
                       "L": layer_id, "sub": sub, "ph": "in",
                       "o": _hashes(src)})
        except Exception:
            _state["errors"] += 1
    return fn


def _post_hook(layer_id, sub):
    def fn(module, args, kwargs, output):
        try:
            tag = "DETSUB" if sub is not None else "DETLAYR"
            _log(tag, {"t": round(time.time(), 4), "f": _state["fwd"],
                       "n": _state["ntok"], "r": _state["nreq"],
                       "d": _state["dummy"] | (_state["profile"] << 1),
                       "L": layer_id, "sub": sub, "ph": "out",
                       "o": _hashes(output)})
        except Exception:
            _state["errors"] += 1
    return fn


_EXCLUDE = ("draft", "mtp", "visual", "vision", "audio", "mmproj",
            "projector", "multi_modal")


def _excluded(name):
    low = name.lower()
    return any(s in low for s in _EXCLUDE)


def _layer_groups(model):
    """Return {parent_path: {int_idx: module}} for names ending layers.N."""
    import re
    groups = {}
    pat = re.compile(r"^(.*)layers\.(\d+)$")
    for name, mod in model.named_modules():
        if name == "":
            continue
        m = pat.match(name)
        if m and not _excluded(name):
            parent, idx = m.group(1).rstrip("."), int(m.group(2))
            groups.setdefault(parent, {})[idx] = mod
    return groups


def _install_group(model, parent, layers):
    st = _state
    st["install_id"] += 1
    ids = sorted(layers)
    handles = []
    inventory = {}
    for idx in ids:
        mod = layers[idx]
        inventory[idx] = sorted(c for c, _ in mod.named_children())
        handles.append(mod.register_forward_pre_hook(_pre_hook(idx, None),
                                                     with_kwargs=True))
        handles.append(mod.register_forward_hook(_post_hook(idx, None),
                                                 with_kwargs=True))
    # Embedding / final mixer / model output / language-model output.
    def _get(path):
        if path.startswith("."):
            return getattr(model, path[1:], None)
        try:
            return model.get_submodule(path)
        except Exception:
            return None

    for suffix, lid in ((".embed_tokens", -1),
                        (".hyper_connection_mixer", -2)):
        path = parent + suffix
        mod = _get(path)
        if mod is None:
            continue
        inventory[lid] = [path]
        handles.append(mod.register_forward_hook(_post_hook(lid, None),
                                                 with_kwargs=True))
    for path, lid in ((parent, -3),  # Qwen3_8FlashNextModel output
                      (parent.rsplit(".", 1)[0], -4)):  # ForCausalLM output
        mod = _get(path)
        if mod is None:
            continue
        inventory[lid] = [path]
        handles.append(mod.register_forward_pre_hook(_pre_hook(lid, None),
                                                     with_kwargs=True))
        handles.append(mod.register_forward_hook(_post_hook(lid, None),
                                                 with_kwargs=True))
    _log("DETHASH-META", {"install": st["install_id"], "parent": parent,
                          "n_layers": len(ids),
                          "layer_idx": ids,
                          "children": inventory,
                          "drill": sorted(st["drill"]), "depth": st["depth"]})
    return handles


def _install_drill(layers):
    depth = _state["depth"]
    for idx in sorted(_state["drill"]):
        mod = layers.get(idx)
        if mod is None:
            continue
        for name, child in mod.named_modules():
            if name == "":
                continue
            rel = name
            if rel.count(".") + 1 > depth:
                continue
            child.register_forward_pre_hook(_pre_hook(idx, rel),
                                            with_kwargs=True)
            child.register_forward_hook(_post_hook(idx, rel),
                                        with_kwargs=True)
        if os.environ.get("DET_HASH_MOE_PROBE", "") == "1":
            _install_moe_probe(idx, mod)


def _install_moe_probe(idx, layer_mod):
    """Wrap runner.router.select_experts on the drilled layer's MoE block so
    the routing result (topk_weights/topk_ids) is hashed separately from the
    routed-expert GEMM.  Splits 'routing kernel is nondeterministic' from
    'Marlin GEMM is nondeterministic' given a bit-identical router output."""
    mlp = getattr(layer_mod, "mlp", None)
    experts = getattr(mlp, "experts", None)
    router = getattr(experts, "router", None)
    if router is None or not hasattr(router, "select_experts"):
        _log("DETRMETA", {"L": idx, "note": "no router/select_experts"})
        return
    if getattr(router, "_awdbg_route_wrapped", False):
        return
    orig = router.select_experts

    def wrapped(*a, **k):
        out = orig(*a, **k)
        try:
            src = k if k else {f"{i}": v for i, v in enumerate(a)}
            rl = src.get("router_logits")
            if rl is None and len(a) >= 2:
                rl = a[1]
            _log("DETR", {"t": round(time.time(), 4), "f": _state["fwd"],
                          "n": _state["ntok"], "r": _state["nreq"],
                          "d": _state["dummy"] | (_state["profile"] << 1),
                          "L": idx,
                          "i": _hashes(rl), "o": _hashes(out)})
        except Exception:
            _state["errors"] += 1
        return out

    router.select_experts = wrapped
    router._awdbg_route_wrapped = True
    routed = getattr(experts, "routed_experts", None)
    print(f"DETHASH-MOEPROBE layer={idx} router={type(router).__name__} "
          f"routed={type(routed).__name__ if routed is not None else None} "
          f"experts_runner={type(experts).__name__}", flush=True)


def _install_gemm_probe():
    """Wrap ops.moe_wna16_marlin_gemm (the Marlin W4A16 fused-MoE GEMM) to
    hash each GEMM's output during the gated forwards.  DET_HASH_GEMM_NTOK
    restricts hashing to forwards with that scheduled-token count (the
    prefill), keeping the probe cheap; -1 hashes every forward."""
    import vllm._custom_ops as _ops
    if getattr(_ops, "_awdbg_gemm_wrapped", False):
        return
    st = _state
    st["gemm_ntok"] = int(os.environ.get("DET_HASH_GEMM_NTOK", "-1"))
    st["gemm_ci"] = 0
    st["gemm_fwd"] = -1
    orig = _ops.moe_wna16_marlin_gemm

    def wrapped(*a, **k):
        out = orig(*a, **k)
        try:
            if st["fwd"] != st["gemm_fwd"]:
                st["gemm_fwd"] = st["fwd"]
                st["gemm_ci"] = 0
            ci = st["gemm_ci"]
            st["gemm_ci"] += 1
            if st["gemm_ntok"] < 0 or st["ntok"] == st["gemm_ntok"]:
                h, shp = _hash_tensor(out)
                _log("DETGEMM", {"t": round(time.time(), 4),
                                 "f": st["fwd"], "n": st["ntok"],
                                 "r": st["nreq"],
                                 "d": st["dummy"] | (st["profile"] << 1),
                                 "ci": ci, "sm": k.get("size_m"),
                                 "sn": k.get("size_n"), "sk": k.get("size_k"),
                                 "tk": k.get("top_k"), "s": shp, "h": h})
        except Exception:
            st["errors"] += 1
        return out

    _ops.moe_wna16_marlin_gemm = wrapped
    _ops._awdbg_gemm_wrapped = True
    print("DETHASH-GEMMPROBE installed on vllm._custom_ops."
          "moe_wna16_marlin_gemm", flush=True)


def maybe_install(model, runner=None):
    """Called from the patched load_model. No-op unless DET_HASH_LAYERS=1."""
    if os.environ.get("DET_HASH_LAYERS", "") != "1":
        return
    if getattr(model, "_awdbg_hash_installed", False):
        return
    model._awdbg_hash_installed = True
    st = _state
    st["fwd"] = 0
    st["drill"] = {int(x) for x in
                   os.environ.get("DET_HASH_DRILL", "").split(",")
                   if x.strip() != ""}
    st["depth"] = int(os.environ.get("DET_HASH_DEPTH", "2"))
    min_layers = int(os.environ.get("DET_HASH_MIN_LAYERS", "8"))
    groups = _layer_groups(model)
    if not groups:
        print("DETHASH-FATAL no decoder layer group found", flush=True)
        return
    parent, layers = max(groups.items(), key=lambda kv: len(kv[1]))
    if len(layers) < min_layers:
        print(f"DETHASH-FATAL largest layer group {parent!r} has only "
              f"{len(layers)} layers (< {min_layers}); refusing to hook "
              f"(draft model?)", flush=True)
        return
    if _excluded(parent):
        print(f"DETHASH-FATAL parent {parent!r} looks like a draft/vision "
              f"stack; refusing", flush=True)
        return
    _install_group(model, parent, layers)
    if st["drill"]:
        _install_drill(layers)
    if os.environ.get("DET_HASH_GEMM_PROBE", "") == "1":
        _install_gemm_probe()
    if runner is not None and getattr(runner, "_awdbg_hash_wrapped", None) is None:
        orig = runner.execute_model

        def wrapped(*a, **k):
            so = a[0] if a else k.get("scheduler_output")
            ntok = getattr(so, "total_num_scheduled_tokens", -1)
            nreq = len(getattr(so, "num_scheduled_tokens", {}) or {})
            st["fwd"] += 1
            st["ntok"] = ntok
            st["nreq"] = nreq
            st["dummy"] = int(bool(k.get("dummy_run", False))) | \
                int(bool(a[2] if len(a) > 2 else False))
            st["profile"] = int(bool(k.get("is_profile", False))) | \
                int(bool(a[4] if len(a) > 4 else False))
            _log("DETFWD", {"t": round(time.time(), 4), "f": st["fwd"],
                            "n": ntok, "r": nreq,
                            "d": st["dummy"] | (st["profile"] << 1)})
            try:
                return orig(*a, **k)
            finally:
                _log("DETFWDEND", {"t": round(time.time(), 4),
                                   "f": st["fwd"]})
        try:
            runner.execute_model = wrapped
            runner._awdbg_hash_wrapped = True
        except Exception as e:
            print(f"DETHASH-WARN execute_model wrap failed: {e}", flush=True)
    print(f"DETHASH-INSTALLED layers={len(layers)} parent={parent!r} "
          f"drill={sorted(st['drill'])} depth={st['depth']} "
          f"file={os.environ.get('DET_HASH_FILE', '/tmp/det_hash.jsonl')}",
          flush=True)


def selftest():
    """Offline CPU gate: hooks log well-formed JSON; identical inputs give
    identical hashes; a one-ULP perturbation changes the hash; the execute-
    model wrapper increments the forward counter."""
    import tempfile
    import torch.nn as nn

    class Blk(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(8, 8)
            self.ln = nn.LayerNorm(8)

        def forward(self, x):
            return self.ln(x + self.lin(x))

    class Toy(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_tokens = nn.Embedding(16, 8)
            self.layers = nn.ModuleList([Blk() for _ in range(4)])
            self.hyper_connection_mixer = nn.LayerNorm(8)

        def forward(self, ids):
            h = self.embed_tokens(ids)
            for l in self.layers:
                h = l(h)
            return self.hyper_connection_mixer(h)

    class SO:
        total_num_scheduled_tokens = 5
        num_scheduled_tokens = {"r0": 5}

    class Runner:
        def execute_model(self, scheduler_output, dummy_run=False):
            return "ok"

    os.environ.update({"DET_HASH_LAYERS": "1", "DET_HASH_DEBUG": "1",
                       "DET_HASH_DRILL": "1", "DET_HASH_DEPTH": "2",
                       "DET_HASH_MIN_LAYERS": "2"})
    os.environ["DET_HASH_FILE"] = tempfile.mktemp(suffix=".jsonl")
    toy = Toy().eval()
    run = Runner()
    maybe_install(toy, runner=run)
    assert toy._awdbg_hash_installed
    ids = torch.arange(5) % 16
    torch.manual_seed(0)
    _state["fwd"] = 1
    out1 = toy(ids)
    _state["fwd"] = 2
    out2 = toy(ids)
    assert torch.equal(out1, out2)
    _state["fwd"] = 3
    out3 = toy(ids)
    assert torch.equal(out1, out3)
    with torch.no_grad():
        toy.layers[2].lin.weight[0, 0] += 1e-3
    _state["fwd"] = 4
    out4 = toy(ids)
    assert not torch.equal(out1, out4)
    assert run.execute_model(SO()) == "ok"
    assert run.execute_model(SO()) == "ok"
    lines = open(os.environ["DET_HASH_FILE"]).read().splitlines()
    kinds = {}
    for raw in open(os.environ["DET_HASH_FILE"]).read().splitlines():
        if not raw.strip():
            continue
        tag, js = raw.split(" ", 1)
        kinds.setdefault(tag, 0)
        kinds[tag] += 1
        json.loads(js)
    assert kinds.get("DETLAYR", 0) >= 4 * 3 * 2, kinds
    assert kinds.get("DETSUB", 0) > 0, kinds
    assert kinds.get("DETFWD", 0) == 2, kinds
    assert kinds.get("DETFWDEND", 0) == 2, kinds
    assert kinds.get("DETHASH-META", 0) == 1, kinds
    # identical forwards -> identical hashes for every layer
    lay = {}
    for raw in open(os.environ["DET_HASH_FILE"]).read().splitlines():
        tag, js = raw.split(" ", 1)
        if tag != "DETLAYR":
            continue
        d = json.loads(js)
        lay.setdefault((d["f"], d["L"], d["ph"]), []).append(
            tuple(o[2] for o in d["o"]))
    fwds = sorted({k[0] for k in lay})
    assert fwds == [1, 2, 3, 4], fwds
    for f in fwds:
        for L in range(4):
            a = lay[(f, L, "out")]
            assert len(a) == 1, (f, L, a)
    # fwds 1..3 identical layer by layer; fwd 4 differs only from L2 on
    sig = {f: tuple(lay[(f, L, "out")][0] for L in range(4)) for f in fwds}
    assert sig[1] == sig[2] == sig[3]
    assert sig[4][2] != sig[1][2] and sig[4][3] != sig[1][3], "drift spread"
    assert sig[4][0] == sig[1][0] and sig[4][1] == sig[1][1]
    # the two execute_model calls logged DETFWD with the scheduler counts
    fw = [json.loads(raw.split(" ", 1)[1])
          for raw in lines if raw.startswith("DETFWD ")]
    assert [w["f"] for w in fw] == [5, 6] and all(w["n"] == 5 for w in fw), fw
    print(f"selftest OK: {kinds}")
    return 0


if __name__ == "__main__":
    raise SystemExit(selftest())
