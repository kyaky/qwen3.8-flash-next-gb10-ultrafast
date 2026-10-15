#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""CPU check for the CF3 anchor-chain repair (round-8 astra hold, CF3).

The round-8 hold verdict found the patch_c2_fields CF3 anchor ending
immediately after the ``_update_request_with_output`` if-block, so the
inserted evidence ``if`` captured the original pooling/encoder ``elif``
arms: with VLLM_SPECDBG_EVIDENCE=1 those stop checks were dead (a behavior
defect, not hygiene). This check proves the repaired patcher WITHOUT docker
and WITHOUT vllm:

  1. the CF3 anchor now spans the WHOLE stop-check if/elif chain (the
     _update_request_with_output if + the pooling + encoder-only elif
     arms). The embedded chain fixture is byte-identical to the region in
     the patcher's real input state (pristine iter7a scheduler.py lines
     2043-2062 == the round-5/iter7d state = the .orig-c2fp input;
     canonical diff: recipe/build/image/verify/iter7a-to-stage1b-round5.diff);
  2. the replacement reproduces the chain bytes exactly and appends the
     capture AFTER the complete chain as an independent sibling statement
     -- never between the if block and its elif arms -- and BEFORE the
     structured-output if that follows in the real scheduler;
  3. the capture's awdbg_commit record carries ``"stopped": bool(stopped)``
     (the stop-trim harness's termination proof) and the emitter helper
     carries the field into the sched row;
  4. the patcher's OWN assert_c2_chain_preserved (extracted from its
     source, so the live code is tested) REJECTS the defective layout
     (elif arms re-chained onto the inserted if) and ACCEPTS the fixed one;
  5. when the pinned patch-input scheduler is available (the
     2026-10-14-p2-stage1b-verify runs copy -- the .orig-c2fp state -- or
     GB10_C2FP_INPUT_DIR), the REAL patcher runs on a temp copy of it with
     SYNTHETIC aux fixtures in the speculator/runner argv slots (built from
     the patcher's own anchors: the pinned iter7d runner/spec bytes live
     only inside the image, and the full-trio byte-reproduction proof is
     gate3's in-image job): every scheduler anchor matches exactly once,
     the chain assertion passes on the real file bytes, the patched
     scheduler keeps the elif arms with the capture as a sibling, and a
     second apply FAILS loudly (the idempotence discipline).

Usage: python3 test_c2_anchor_chain_cpu.py
"""
from __future__ import annotations

import ast
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "src" if HERE.name == "tests" else HERE
PATCHER = SRC / "patch_c2_fields.py"
EMITTER = SRC / "awdbg1b_evidence.py"

FAILURES: list[str] = []


def check(name, ok, detail=""):
    tag = "PASS" if ok else "FAIL"
    print(f"  [{tag}] {name}" + (f": {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


# The stop-check chain EXACTLY as it stands in the patcher's input state
# (12-space indent stripped; byte-identical to pristine scheduler.py lines
# 2043-2062 AND to the round-5/iter7d patched state -- verified 2026-10-15;
# the verify/ round-5 diff shows no hunk touches the region).
CHAIN = """\
# Check for stop and update request status.
if new_token_ids:
    new_token_ids, stopped = self._update_request_with_output(
        request, new_token_ids, is_stale=output_is_stale
    )
elif request.pooling_params and pooler_output is not None:
    # Pooling stops as soon as there is output.
    request.status = RequestStatus.FINISHED_STOPPED
    stopped = True
elif (
    self.is_encoder_only
    and request.num_computed_tokens >= request.num_prompt_tokens
):
    # An encoder instance runs the encoder and publishes the
    # embeddings instead of sampling, so it stops as soon as the
    # whole prompt is consumed. Encoder inputs are never scheduled
    # past a multi-modal item the encoder cache could not admit, so
    # a consumed prompt also means every item in it was encoded.
    request.status = RequestStatus.FINISHED_STOPPED
    stopped = True
"""

# The statement that follows the chain in the real scheduler (the
# structured-output advance) -- proves the capture lands after the chain
# and before it, as a sibling in the same body.
FOLLOWER = """\
if new_token_ids and self.structured_output_manager.should_advance(
    request, new_token_ids=new_token_ids
):
    struct_output_request = request.structured_output_request
    assert struct_output_request is not None
"""

# the assertion helpers the patcher must carry (extracted by name)
PATCHER_ASSERT_FUNCS = (
    "_stop_check_chain_heads",
    "_statement_after",
    "_is_capture_if",
    "_capture_records_stopped",
    "assert_c2_chain_preserved",
)


def _chain_heads(tree):
    """The stop-check chain heads in one parsed tree (same narrow
    discriminator the patcher's own helper uses, re-derived here so the
    fixture leg does not depend on the patcher's code being importable)."""
    heads = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        if not (isinstance(node.test, ast.Name)
                and node.test.id == "new_token_ids"):
            continue
        first = node.body[0] if node.body else None
        if not (isinstance(first, ast.Assign)
                and isinstance(first.value, ast.Call)):
            continue
        fn = first.value.func
        if isinstance(fn, ast.Attribute) \
                and fn.attr == "_update_request_with_output":
            heads.append(node)
    return heads


def _extract_cf3_pair():
    """Pull the LIVE CF3 (anchor, replacement) out of patch_c2_fields.py.

    The pair is read out of the patcher's own AST (the third edit of the
    SCHED_PATH patch call), so this check always exercises the anchor the
    patcher actually applies -- never a drifted copy.
    """
    tree = ast.parse(PATCHER.read_text())
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "patch"):
            continue
        if not (node.args and isinstance(node.args[0], ast.Name)
                and node.args[0].id == "SCHED_PATH"):
            continue
        edits = next((kw.value for kw in node.keywords if kw.arg == "edits"),
                     None)
        if not (isinstance(edits, ast.List) and len(edits.elts) >= 3):
            raise AssertionError("SCHED_PATH patch call has < 3 edits")
        pair = edits.elts[2]
        if not (isinstance(pair, ast.Tuple) and len(pair.elts) == 2
                and all(isinstance(e, ast.Constant)
                        and isinstance(e.value, str) for e in pair.elts)):
            raise AssertionError("CF3 edit is not a (str, str) literal pair")
        return pair.elts[0].value, pair.elts[1].value
    raise AssertionError("no patch(SCHED_PATH, ...) call found in the patcher")


def _patcher_assert_namespace():
    """Exec ONLY the patcher's chain-assertion function defs (no argv side
    effects -- the module-level patch calls are skipped)."""
    tree = ast.parse(PATCHER.read_text())
    funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef)
             and n.name in PATCHER_ASSERT_FUNCS]
    if len(funcs) != len(PATCHER_ASSERT_FUNCS):
        raise AssertionError(
            "patcher assertion helpers missing: "
            f"{sorted(set(PATCHER_ASSERT_FUNCS) - {f.name for f in funcs})}")
    ns = {"ast": ast}
    exec(compile(ast.Module(body=funcs, type_ignores=[]),
                 str(PATCHER), "exec"), ns)
    return ns


def _chain_arms(head):
    """The elif arms of a chain head (one If per elif, in order)."""
    arms = []
    node = head
    while (node.orelse and len(node.orelse) == 1
           and isinstance(node.orelse[0], ast.If)):
        node = node.orelse[0]
        arms.append(node)
    return arms


def _statement_after(tree, node):
    """The sibling statement following `node` in its body (or None)."""
    for parent in ast.walk(tree):
        for field in ("body", "orelse", "finalbody"):
            seq = getattr(parent, field, None)
            if isinstance(seq, list):
                for i, child in enumerate(seq):
                    if child is node:
                        return seq[i + 1] if i + 1 < len(seq) else None
    return None


def _is_capture_if(node):
    """True for the commit-capture If: ``if awdbg_ev_enabled():``."""
    return (isinstance(node, ast.If)
            and isinstance(node.test, ast.Call)
            and isinstance(node.test.func, ast.Name)
            and node.test.func.id == "awdbg_ev_enabled"
            and not node.test.args and not node.test.keywords)


def _stopped_recorded(node):
    """True when the capture If's awdbg_commit record carries the exact
    ``"stopped": bool(stopped)`` entry."""
    for sub in ast.walk(node):
        if not isinstance(sub, ast.Assign):
            continue
        if not any(
            isinstance(t, ast.Subscript)
            and isinstance(t.value, ast.Name)
            and t.value.id == "awdbg_commit"
            for t in sub.targets
        ):
            continue
        if not isinstance(sub.value, ast.Dict):
            continue
        for k, v in zip(sub.value.keys, sub.value.values):
            if (isinstance(k, ast.Constant) and k.value == "stopped"
                    and isinstance(v, ast.Call)
                    and isinstance(v.func, ast.Name) and v.func.id == "bool"
                    and len(v.args) == 1
                    and isinstance(v.args[0], ast.Name)
                    and v.args[0].id == "stopped"):
                return True
    return False


def _locate_input_scheduler():
    """The pinned patch-input scheduler (.orig-c2fp state), when available."""
    env = os.environ.get("GB10_C2FP_INPUT_DIR")
    if env:
        p = Path(env) / "scheduler.py"
        return p if p.is_file() else None
    probe = SRC
    for _ in range(8):
        cand = (probe / "agents" / "glm-5.3-flash-integrator" / "runs"
                / "2026-10-14-p2-stage1b-verify" / "patched"
                / "scheduler.py")
        if cand.is_file():
            return cand
        probe = probe.parent
    return None


def _synth_aux_source(slot_name):
    """A minimal synthetic input file for one aux argv slot (SPEC_PATH /
    RUNNER_PATH): the patcher's OWN anchor for that slot, nested deep
    enough to parse, plus its must_contain markers as comments. Built from
    the live patcher AST, so the scaffolding always matches the patcher by
    construction (a future anchor change regenerates the fixture)."""
    tree = ast.parse(PATCHER.read_text())
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "patch"):
            continue
        if not (node.args and isinstance(node.args[0], ast.Name)
                and node.args[0].id == slot_name):
            continue
        kw = {k.arg: k.value for k in node.keywords if k.arg}
        anchor = ast.literal_eval(kw["edits"].elts[0].elts[0])
        markers = [ast.literal_eval(e) for e in kw["must_contain"].elts]
        nots = [ast.literal_eval(e) for e in kw["must_not_contain"].elts]
        for bad in nots:
            if any(bad in m for m in markers) or bad in anchor:
                raise AssertionError(
                    f"{slot_name}: synthetic fixture would carry "
                    f"must_not_contain {bad!r}")
        indent = len(anchor) - len(anchor.lstrip(" "))
        depth = indent // 4
        wrapper = "class _C2Aux:\n"
        for level in range(1, depth):
            wrapper += "    " * level + "if True:\n"
        marker_lines = "".join(f"# required marker: {m}\n" for m in markers)
        return wrapper + marker_lines + anchor
    raise AssertionError(f"no patch({slot_name}, ...) call in the patcher")


def main():
    print("CF3 anchor-chain check (CPU only, no docker, no vllm)")

    # ---- 0. the live CF3 pair, extracted from the patcher's own AST ----
    try:
        anchor, replacement = _extract_cf3_pair()
    except Exception as exc:
        check("cf3.pair_extracted", False, repr(exc))
        return 1
    check("cf3.pair_extracted", True,
          f"anchor {len(anchor)}B, replacement {len(replacement)}B")

    anchor_d = textwrap.dedent(anchor)
    repl_d = textwrap.dedent(replacement)
    slices_ok = replacement.startswith(anchor) and repl_d.startswith(anchor_d)
    pre_mod = anchor_d + "\n\n" + FOLLOWER
    fixed_mod = repl_d + "\n\n" + FOLLOWER

    # ---- 1. the anchor spans the WHOLE chain (the old defect ended after
    # the if-block alone, leaving ZERO elif arms in the anchor) ----
    check("cf3.anchor_is_full_chain_region", anchor_d == CHAIN,
          "dedented anchor must equal the pinned 20-line chain region")
    heads: list = []
    try:
        pre_tree = ast.parse(pre_mod)
        heads = _chain_heads(pre_tree)
        check("cf3.anchor_parses_and_has_one_chain_head",
              len(heads) == 1, f"found {len(heads)} chain head(s)")
    except SyntaxError as exc:
        pre_tree = None
        check("cf3.anchor_parses_and_has_one_chain_head", False,
              f"line {exc.lineno}: {exc.msg}")
    arms: list = []
    if len(heads) == 1:
        arms = _chain_arms(heads[0])
        arm_tests = [ast.dump(a.test) for a in arms]
        check("cf3.anchor_spans_both_elif_arms", len(arms) == 2,
              f"{len(arms)} arm(s)")
        check("cf3.arms_are_pooling_and_encoder",
              any("pooling_params" in t for t in arm_tests)
              and any("is_encoder_only" in t for t in arm_tests),
              "; ".join(t[:60] for t in arm_tests))

    # ---- 2. the replacement reproduces the chain bytes and appends the
    # capture AFTER the complete chain, BEFORE the structured-output if ----
    check("cf3.replacement_keeps_chain_bytes", replacement.startswith(anchor),
          "the replacement must reproduce the anchor bytes exactly")
    try:
        post_tree = ast.parse(fixed_mod)
        post_heads = _chain_heads(post_tree)
        check("cf3.patched_parses_and_has_one_chain_head",
              len(post_heads) == 1, f"found {len(post_heads)} chain head(s)")
    except SyntaxError as exc:
        post_tree, post_heads = None, []
        check("cf3.patched_parses_and_has_one_chain_head", False,
              f"line {exc.lineno}: {exc.msg}")
    if len(heads) == 1 and len(post_heads) == 1:
        # the chain subtree must be structurally IDENTICAL pre vs post
        # (same head body, same elif sequence -- nothing inserted inside
        # it; the capture must be a sibling OUTSIDE the chain)
        check("cf3.chain_structure_identical",
              ast.dump(heads[0]) == ast.dump(post_heads[0]),
              "ast.dump(pre head) == ast.dump(post head)")
        capture = _statement_after(post_tree, post_heads[0])
        check("cf3.capture_is_sibling_after_chain", _is_capture_if(capture),
              "if awdbg_ev_enabled(): right after the complete chain")
        follower = _statement_after(post_tree, capture) \
            if _is_capture_if(capture) else None
        check("cf3.capture_before_structured_output_if",
              isinstance(follower, ast.If)
              and "should_advance" in ast.dump(follower.test),
              "the structured-output if still follows the capture")
        check("cf3.commit_record_has_stopped_bool",
              _stopped_recorded(capture),
              '"stopped": bool(stopped) in the awdbg_commit record')
    if len(heads) == 1:
        # pre sanity: no capture sat after the chain before the patch
        pre_next = _statement_after(pre_tree, heads[0])
        check("cf3.pre_had_no_capture", not _is_capture_if(pre_next),
              "the pre-patch chain is followed by the structured-output if")

    # ---- 3. the emitter helper carries stopped into the sched row ----
    emitter_src = EMITTER.read_text()
    emitter_fn = emitter_src.split("def build_sched_record")[1] \
        if "def build_sched_record" in emitter_src else ""
    check("emitter.carries_stopped",
          'rq["stopped"] = bool(c["stopped"])' in emitter_src
          and '"stopped"' in emitter_fn.split("def ")[0],
          "build_sched_record maps commit.stopped -> row.stopped")

    # ---- 4. the patcher's OWN assertion: rejects the defective layout,
    # accepts the fixed one ----
    if slices_ok:
        try:
            ns = _patcher_assert_namespace()
            check("cf3.assert_helpers_extracted", True,
                  ", ".join(PATCHER_ASSERT_FUNCS))
        except Exception as exc:
            ns = None
            check("cf3.assert_helpers_extracted", False, repr(exc))
        if ns is not None:
            lines = anchor_d.splitlines(keepends=True)
            head5, tail = "".join(lines[:5]), "".join(lines[5:])
            insert = repl_d[len(anchor_d):]
            # the OLD defect: the capture between the if-block and the
            # elif arms, so the arms re-chain onto the inserted if
            defective_mod = head5 + insert + tail + "\n\n" + FOLLOWER
            raised, detail = False, ""
            try:
                ns["assert_c2_chain_preserved"](pre_mod, defective_mod)
            except AssertionError as exc:
                raised, detail = True, str(exc)[:120]
            check("cf3.assertion_rejects_defective_rechain", raised, detail)
            accepted, detail = True, ""
            try:
                ns["assert_c2_chain_preserved"](pre_mod, fixed_mod)
            except AssertionError as exc:
                accepted, detail = False, str(exc)[:120]
            check("cf3.assertion_accepts_fixed_layout", accepted, detail)
    else:
        check("cf3.replacement_slice_consistency", False,
              "replacement does not extend the anchor; legs 2+4 void")

    # ---- 5. the REAL patcher on the REAL pinned scheduler (.orig-c2fp
    # state), with synthetic aux fixtures in the speculator/runner argv
    # slots (the pinned iter7d runner/spec bytes live only inside the
    # image; the full-trio byte-reproduction proof is gate3's in-image job)
    # ----
    sched_input = _locate_input_scheduler()
    if sched_input is None:
        check("cf3.real_input_run", True,
              "skipped: pinned patch-input scheduler not found "
              "(set GB10_C2FP_INPUT_DIR to enable)")
    else:
        with tempfile.TemporaryDirectory() as td:
            dst_sched = Path(td) / "scheduler.py"
            shutil.copyfile(sched_input, dst_sched)
            dst_spec = Path(td) / "speculator.py"
            dst_run = Path(td) / "gpu_model_runner.py"
            try:
                dst_spec.write_text(_synth_aux_source("SPEC_PATH"))
                dst_run.write_text(_synth_aux_source("RUNNER_PATH"))
                aux_ok = True
            except Exception as exc:
                aux_ok = False
                check("cf3.real_input_run", False,
                      f"synthetic aux fixture build failed: {exc!r:.160}")
            if aux_ok:
                before = dst_sched.read_text()
                cmd = [sys.executable, str(PATCHER), str(dst_sched),
                       str(dst_run), str(dst_spec)]
                r = subprocess.run(cmd, capture_output=True, text=True)
                check("cf3.real_input_patch_ok", r.returncode == 0,
                      (r.stdout + r.stderr).strip().replace("\n", " | ")
                      [:220])
                after = dst_sched.read_text()
                try:
                    ns2 = _patcher_assert_namespace()
                    ns2["assert_c2_chain_preserved"](before, after)
                    check("cf3.real_input_chain_kept", True,
                          "patcher assertion passes on the real file bytes")
                except AssertionError as exc:
                    check("cf3.real_input_chain_kept", False, str(exc)[:160])
                real_tree = ast.parse(after)
                real_heads = _chain_heads(real_tree)
                if len(real_heads) == 1:
                    real_capture = _statement_after(real_tree,
                                                    real_heads[0])
                    check("cf3.real_input_capture_sibling",
                          _is_capture_if(real_capture)
                          and _stopped_recorded(real_capture),
                          "capture sits after the chain with the stopped "
                          "field, on the real patched scheduler")
                    if len(heads) == 1:
                        check("cf3.real_input_matches_fixture",
                              ast.dump(real_heads[0]) == ast.dump(heads[0]),
                              "real chain == embedded fixture (no drift)")
                else:
                    check("cf3.real_input_capture_sibling", False,
                          f"{len(real_heads)} chain head(s) after patch")
                r2 = subprocess.run(cmd, capture_output=True, text=True)
                check("cf3.real_input_second_apply_fails", r2.returncode != 0,
                      "second apply refuses loudly (idempotence)")

    print()
    if FAILURES:
        print(f"FAILED: {FAILURES}")
        return 1
    print("PASS: the CF3 anchor spans the whole stop-check chain, the "
          "capture is a sibling AFTER it with the stopped field, and the "
          "patcher's chain assertion rejects the defective layout")
    return 0


if __name__ == "__main__":
    sys.exit(main())
