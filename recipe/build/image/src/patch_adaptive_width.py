#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Stage-1a shadow wiring for the adaptive draft-width controller (P2).

Shadow semantics: the controller observes per-request accepted-draft counts at
the scheduler's make_spec_decoding_stats call site and logs k/ema/would-switch
events, but k_for_step() always returns the configured K, ZERO behavior change.
Runtime width switching (armed mode) is stage-1c, gated on the padding-disable
interaction audit (D-007); stage-1b measures W4 vs W8 as config variants.

Anchors (all verified in-image on vllm 0.1.dev20073+g8e685d198):
  scheduler.py:283      self.num_spec_tokens = vllm_config.num_speculative_tokens
  scheduler.py:~1995    make_spec_decoding_stats(... num_draft_tokens=..., num_accepted_tokens=..., request_id=...)
Usage: python3 patch_adaptive_width.py <scheduler.py> [off]
"""
import sys

TARGET = sys.argv[1]
s = open(TARGET).read()

# 1. instantiate the controller next to num_spec_tokens
anchor1 = "        self.num_spec_tokens = vllm_config.num_speculative_tokens\n"
new1 = (
    "        self.num_spec_tokens = vllm_config.num_speculative_tokens\n"
    "        # ---- GB10 adaptive width (P2 stage-1a shadow) ----\n"
    "        self._adaptive_width = None\n"
    "        import os as _aw_os\n"
    "        _aw_mode = _aw_os.environ.get(\"VLLM_ADAPTIVE_WIDTH_MODE\", \"off\")\n"
    "        if _aw_mode != \"off\" and self.num_spec_tokens > 0:\n"
    "            try:\n"
    "                from vllm_adaptive_width import install as _aw_install\n"
    "                self._adaptive_width = _aw_install(\n"
    "                    _aw_mode,\n"
    "                    _aw_os.environ.get(\"VLLM_ADAPTIVE_WIDTH_FORCE\"),\n"
    "                    batch_size_fn=lambda: len(self.running),\n"
    "                )\n"
    "                self._aw_log_acc: list[int] = []\n"
    "            except Exception as _aw_err:\n"
    "                logger.warning(\"adaptive width install failed: %s\", _aw_err)\n"
    "                self._adaptive_width = None\n"
    "        # ---- end GB10 adaptive width ----\n"
)
assert s.count(anchor1) == 1, f"anchor1 count {s.count(anchor1)}"
s = s.replace(anchor1, new1)

# 2. feed acceptance at the spec-decoding stats call site
anchor2 = (
    "                spec_decoding_stats = self.make_spec_decoding_stats(\n"
    "                    spec_decoding_stats,\n"
    "                    num_draft_tokens=num_draft_tokens,\n"
    "                    num_accepted_tokens=num_accepted,\n"
    "                    num_invalid_spec_tokens=scheduler_output.num_invalid_spec_tokens,\n"
    "                    request_id=req_id,\n"
    "                )\n"
)
new2 = (
    "                spec_decoding_stats = self.make_spec_decoding_stats(\n"
    "                    spec_decoding_stats,\n"
    "                    num_draft_tokens=num_draft_tokens,\n"
    "                    num_accepted_tokens=num_accepted,\n"
    "                    num_invalid_spec_tokens=scheduler_output.num_invalid_spec_tokens,\n"
    "                    request_id=req_id,\n"
    "                )\n"
    "                # ---- GB10 adaptive width: shadow observation ----\n"
    "                if self._adaptive_width is not None:\n"
    "                    # num_accepted counts accepted draft tokens for this request;\n"
    "                    # correct drafts = accepted (the +1 bonus is not a draft).\n"
    "                    self._aw_log_acc.append(max(0, int(num_accepted)))\n"
    "                    # flush per scheduler step is handled by the batched update\n"
    "                    # below via the last-request marker; see _aw_flush.\n"
    "                # ---- end GB10 adaptive width ----\n"
)
assert s.count(anchor2) == 1, f"anchor2 count {s.count(anchor2)}"
s = s.replace(anchor2, new2)

# 3. append the flush logic at module end: per scheduler step, once all requests'
#    acceptance is collected, feed the controller a batched update.
s += '''

# --- GB10 adaptive width: per-step flush (stage-1a shadow) ---
# The stats call site is per-request; a step's batched update must happen once
# after the last request of the step. We hook the end of update_requests by
# wrapping Scheduler.update_requests at import time of this module's consumer.
# Simpler and robust: flush from update_requests via a monkeypatch installed
# alongside the controller below.


def _gb10_aw_install_flush(scheduler_cls):
    """Flush the per-step acceptance batch into the controller after each
    update_from_output (the per-request stats site lives inside it)."""
    if getattr(scheduler_cls, "_gb10_aw_flush_installed", False):
        return
    orig = scheduler_cls.update_from_output

    def update_from_output(self, *a, **k):
        result = orig(self, *a, **k)
        aw = getattr(self, "_adaptive_width", None)
        acc = getattr(self, "_aw_log_acc", None)
        if aw is not None and acc:
            try:
                aw.observe(len(self.running), acc)
            except Exception:
                pass
            acc.clear()
        return result

    scheduler_cls.update_from_output = update_from_output
    scheduler_cls._gb10_aw_flush_installed = True


def gb10_aw_bind(scheduler_cls):
    _gb10_aw_install_flush(scheduler_cls)
'''
open(TARGET, "w").write(s)
import ast
ast.parse(open(TARGET).read())
print("adaptive width shadow wiring applied OK")


# --- append the class-level flush bind at the end of scheduler.py ---
bind_block = """

# --- GB10 adaptive width: bind the per-step acceptance flush (P2 stage-1a) ---
try:
    from vllm_adaptive_width import gb10_aw_bind as _gb10_aw_bind  # noqa: E402
    _gb10_aw_bind(Scheduler)
except Exception:  # pragma: no cover - inert when the module is absent
    pass
"""
open(TARGET, "a").write(bind_block)
ast.parse(open(TARGET).read())
print("adaptive width bind appended OK")
