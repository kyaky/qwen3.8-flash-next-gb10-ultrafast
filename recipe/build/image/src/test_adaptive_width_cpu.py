#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""CPU gate for vllm_adaptive_width.py: parity against the Pennyroyal donor
semantics (hysteresis boundaries, warmup, interval, zero-step probe) + the
mode matrix (off/shadow/force/oscillate) + the D-009 trap gates (5c/5d:
launch-at-max-width coordination, tier overrides included; 5e: the armed
force contract; 5f: D-009 round-2 config-shape validation, tier keys,
candidate element types, zero-width FORCE refusal, strict FORCE parsing,
launch_k required for driving modes, mode whitelist; 5g: D-009 round-3 -
oscillate never drives the zero width (a 0 candidate is refused in
oscillating ladders for the driving modes), tiers container/value shape,
int-only launch_k, and the zero-step nospec candidate's donor parity).
Run inside the iter7a image and offline; no vllm imports."""
import json
import logging
import sys
import tempfile

sys.path.insert(0, "/selftest/src")
from vllm_adaptive_width import AdaptiveStepSlot, AdaptiveWidthController, load_config  # noqa: E402

fails = []


def check(cond, msg):
    print(("PASS " if cond else "FAIL ") + msg)
    if not cond:
        fails.append(msg)


cfg = {"candidate_steps": [3, 7], "ema_alpha": 0.2, "update_interval": 5,
       "warmup_batches": 10, "down_hysteresis": -0.25, "up_hysteresis": 0.0,
       "ceiling_coeff": 0}

# 1. neutral init + warmup suppression
slot = AdaptiveStepSlot(3, cfg)
check(slot.current_steps == 3 and abs(slot.ema_accept_len - 2.0) < 1e-9, "init at steps=3, ema=2")
for _ in range(10):
    changed = slot.update([2] * 8)  # perfectly neutral stream
check(slot.current_steps == 3, "warmup: no switch during 10 warmup batches")

# 2/3/4. switches only at interval boundaries (warmup=10, interval=5):
# feed exactly 5 updates past warmup to reach one recompute point.
def drive(slot, avg, intervals):
    # warmup consumes 10 updates; each interval adds 5 more; a recompute lands
    # only at counts 15, 20, 25, ...
    for _ in range(10 + intervals * 5):
        slot.update([avg] * 8)

# up-switch: ema -> 3.0 > 2.5 (= 3 - 0.5 + up_hys 0)
slot = AdaptiveStepSlot(3, cfg)
drive(slot, 3.0, 2)  # 2 intervals after warmup
check(slot.current_steps == 7, f"up-switch 3->7 at ema 3.0 (got {slot.current_steps})")

# down-switch from 7: ema 2.0 <= 2.25 (= 3 - 0.5 + down_hys -0.25)
slot = AdaptiveStepSlot(7, cfg)
drive(slot, 7.0, 2)  # stay at 7 while acceptance is full
check(slot.current_steps == 7, "stays at 7 when acceptance is full")
slot.ema_accept_len = 2.0  # force ema below the drop threshold (implementation attribute)
drive(slot, 0.0, 1)         # one interval -> recompute
check(slot.current_steps == 3, f"down-switch 7->3 at ema 2.0 (got {slot.current_steps})")

# hysteresis dead-band: ema exactly 2.5 must NOT move up from 3 (>2.5 required)
slot = AdaptiveStepSlot(3, cfg)
drive(slot, 3.0, 2)
slot.ema_accept_len = 2.5
drive(slot, 0.0, 1)
check(slot.current_steps == 3, "dead-band: ema == 2.5 does not move up")

# 5. mode matrix
ctrl = AdaptiveWidthController.load = None  # placeholder guard against API drift
from vllm_adaptive_width import install  # noqa: E402
import os  # noqa: E402

os.environ.pop("VLLM_ADAPTIVE_WIDTH_CONFIG", None)
c = install("shadow", None, batch_size_fn=lambda: 1)
check(c.k_for_step(1, 3) == 3 and c.mode == "shadow", "shadow returns config K")
c = install("force", "7", batch_size_fn=lambda: 1, launch_k=7)
check(c.k_for_step(1, 3) == 7, "force=7 returns 7")
c = install("force", "oscillate", batch_size_fn=lambda: 1, launch_k=7)
a = c.k_for_step(1, 3); a2 = c.k_for_step(1, 3)
check(a == a2 == 7, f"oscillate stable within one schedule (got {a},{a2})")
c.observe(1, [0] * 8)  # one schedule boundary -> flip
b = c.k_for_step(1, 3)
check(b == 3, f"oscillate flips at the schedule boundary (got {b})")
c = install("off", None, batch_size_fn=lambda: 1)
check(c.k_for_step(1, 3) == 3, "off returns config K")

# 5b. armed mode: EMA-driven, plus the forced-oscillate G2 harness path
# (driving installs require launch_k since the D-009 round-2 contract)
c = install("armed", None, batch_size_fn=lambda: 1, launch_k=7)
# install() seeds the slot at max(candidate_steps) = launch width (committed
# stage-1 behavior; the donor sketch seeds candidates[0], flagged for G5).
check(c.k_for_step(1, 7) == 7, "armed starts at the seeded slot width (max)")
c.slot_for(1).current_steps = 3  # simulate an EMA down-switch
check(c.k_for_step(1, 7) == 3, "armed returns the slot's current width")
c = install("armed", "oscillate", batch_size_fn=lambda: 1, launch_k=7)
a = c.k_for_step(1, 7); a2 = c.k_for_step(1, 7)
check(a == a2 == 7, f"armed+oscillate stable within one schedule (got {a},{a2})")
c.observe(1, [0] * 8)  # one schedule boundary -> flip
b = c.k_for_step(1, 7)
check(b == 3, f"armed+oscillate flips at the schedule boundary (got {b})")
c.observe(1, [0] * 8)
check(c.k_for_step(1, 7) == 7, "armed+oscillate keeps alternating")

# 5c. launch-at-max-width invariant: a driving mode never installs a ladder
# above the launch K. Since the D-009 trap-1 fix, armed COORDINATES (clamps
# the ladder with a logged warning) instead of refusing; an unsatisfiable
# numeric FORCE is still refused. off/shadow stay installable at any launch K.
records: list[str] = []


class _CaptureHandler(logging.Handler):
    def emit(self, record):
        records.append(record.getMessage())


logging.getLogger("vllm.adaptive_width").addHandler(_CaptureHandler())
logging.getLogger("vllm.adaptive_width").setLevel(logging.DEBUG)

c = install("armed", None, batch_size_fn=lambda: 1, launch_k=3)
check(c is not None and c.slot_for(1).candidate_steps == [3],
      "armed at launch_k=3 clamps [3,7] -> [3] (coordination, not refusal)")
check(c.k_for_step(1, 3) == 3, "armed at launch_k=3 drives 3, never above launch")
try:
    install("force", "7", batch_size_fn=lambda: 1, launch_k=3)
    check(False, "force=7 refuses launch_k below the pinned width")
except ValueError:
    check(True, "force=7 refuses launch_k below the pinned width")
c = install("armed", None, batch_size_fn=lambda: 1, launch_k=7)
check(c is not None and c.slot_for(1).candidate_steps == [3, 7],
      "armed accepts launch_k == max candidate (ladder untouched)")
c = install("shadow", None, batch_size_fn=lambda: 1, launch_k=3)
check(c is not None and c.k_for_step(1, 3) == 3,
      "shadow installs at launch_k=3 (never drives K)")

# 5d. D-009 trap 1 gates: install(launch_k=...) coordinates the candidate
# ladder, clamp (with a logged warning) when candidates survive at or below
# the launch K, derive [launch_k] when none do, tier overrides included; a
# numeric FORCE above the launch K is refused (unsatisfiable pin).
c = install("armed", None, batch_size_fn=lambda: 1, launch_k=4)
check(c.slot_for(1).candidate_steps == [3], "launch_k=4 clamps [3,7] -> [3]")
check(c.k_for_step(1, 4) == 3, "armed at launch_k=4 drives 3 (below launch)")
check(any("clamped to [3]" in m for m in records),
      "clamp logs a warning naming the coercion")
c = install("armed", None, batch_size_fn=lambda: 1, launch_k=2)
check(c.slot_for(1).candidate_steps == [2],
      "launch_k=2 below all candidates derives [2]")
check(c.k_for_step(1, 2) == 2, "derived ladder holds the launch width")
check(any("derived single-candidate ladder [2]" in m for m in records),
      "derive logs a warning")
c = install("force", "3", batch_size_fn=lambda: 1, launch_k=4)
check(c.k_for_step(1, 4) == 3, "force=3 at launch_k=4 pins 3 (ladder clamped alongside)")
try:
    install("force", "5", batch_size_fn=lambda: 1, launch_k=4)
    check(False, "force=5 refuses launch_k=4 (unsatisfiable pin)")
except ValueError:
    check(True, "force=5 refuses launch_k=4 (unsatisfiable pin)")

# tier overrides are coerced too (the trap: install only checked the default)
_tier_fd, tier_cfg_path = tempfile.mkstemp(suffix=".json")
os.write(_tier_fd, json.dumps({
    "candidate_steps": [3, 7],
    "tiers": {"1": {"candidate_steps": [2, 9]}, "2": {"candidate_steps": [7]}},
}).encode())
os.close(_tier_fd)
os.environ["VLLM_ADAPTIVE_WIDTH_CONFIG"] = tier_cfg_path
c = install("armed", None, batch_size_fn=lambda: 1, launch_k=4)
check(c._cfg_for_bs(1)["candidate_steps"] == [2],
      "tier 1 [2,9] clamps to [2] at launch_k=4")
check(c._cfg_for_bs(8)["candidate_steps"] == [3],
      "tier 2 [7] unusable at launch_k=4 -> inherits default [3]")
check(c.slot_for(1).candidate_steps == [2] and c.slot_for(8).candidate_steps == [3],
      "coerced tiers seed slots at or below the launch K")
check(any("tier inherits" in m for m in records),
      "unusable tier override logs a warning")
os.environ.pop("VLLM_ADAPTIVE_WIDTH_CONFIG", None)
os.unlink(tier_cfg_path)

# defense in depth: a post-install config mutation cannot push a slot above
# the launch width
c = install("armed", None, batch_size_fn=lambda: 1, launch_k=4)
c.tiers["1"]["candidate_steps"] = [7]
check(c.slot_for(1).candidate_steps == [4], "slot_for clamps post-install mutation")
check(any("post-install config mutation" in m for m in records),
      "slot clamp logs a warning")

# 5e. D-009 trap 2 gates: armed stays EMA-driven pure, numeric pinning
# belongs to mode=force only, enforced at the env-parse boundary (install).
for bad in ("3", "7", "abc"):
    try:
        install("armed", bad, batch_size_fn=lambda: 1, launch_k=7)
        check(False, f"armed refuses FORCE={bad!r} (numeric pinning is force-only)")
    except ValueError:
        check(True, f"armed refuses FORCE={bad!r} (numeric pinning is force-only)")
try:
    install("force", None, batch_size_fn=lambda: 1, launch_k=4)
    check(False, "mode=force refuses a missing FORCE")
except ValueError:
    check(True, "mode=force refuses a missing FORCE")
try:
    install("force", "abc", batch_size_fn=lambda: 1, launch_k=4)
    check(False, "mode=force refuses a non-numeric, non-oscillate FORCE")
except ValueError:
    check(True, "mode=force refuses a non-numeric, non-oscillate FORCE")
try:
    install("force", "-1", batch_size_fn=lambda: 1, launch_k=4)
    check(False, "mode=force refuses a negative FORCE")
except ValueError:
    check(True, "mode=force refuses a negative FORCE")
c = install("shadow", "7", batch_size_fn=lambda: 1)
check(c.k_for_step(1, 3) == 3, "shadow ignores FORCE (returns config K)")
check(any("mode='shadow' never drives K" in m for m in records),
      "shadow+FORCE logs a pointing warning")
c = install("armed", "oscillate", batch_size_fn=lambda: 1, launch_k=7)
check(c.k_for_step(1, 7) == 7, "armed keeps the oscillate harness path")
c = install("force", "oscillate", batch_size_fn=lambda: 1, launch_k=2)
check(c.k_for_step(1, 2) == 2, "oscillate at launch_k=2 degenerates to the derived constant")
check(any("constant width" in m for m in records),
      "degenerate oscillate logs a warning")

# 5f. D-009 round-2 gates: install() is the validation boundary for the
# config SHAPE too, the deployed armed branch calls k_for_step UNGUARDED
# inside Scheduler.update_from_output, so anything install() lets through
# crashes the scheduler loop at the first driving step.

# 5f-1. non-numeric tier keys are refused at install (they would crash
# _cfg_for_bs's int(key) sort on the first k_for_step call)
def _install_with_cfg(cfg_obj, mode, force, launch_k):
    fd, path = tempfile.mkstemp(suffix=".json")
    os.write(fd, json.dumps(cfg_obj).encode())
    os.close(fd)
    os.environ["VLLM_ADAPTIVE_WIDTH_CONFIG"] = path
    try:
        return install(mode, force, batch_size_fn=lambda: 1, launch_k=launch_k), path
    finally:
        os.environ.pop("VLLM_ADAPTIVE_WIDTH_CONFIG", None)
        os.unlink(path)

for bad_tiers, label in (
        ({"high": {"candidate_steps": [3]}}, "non-numeric tier key"),
        ({"0": {"candidate_steps": [3]}}, "zero tier floor"),
        ({" 1": {"candidate_steps": [3]}}, "tier key with whitespace")):
    try:
        _install_with_cfg({"candidate_steps": [3, 7], "tiers": bad_tiers},
                          "armed", None, 4)
        check(False, f"install refuses {label}")
    except ValueError:
        check(True, f"install refuses {label}")
try:
    _install_with_cfg({"candidate_steps": [3, 7], "tiers": {"high": {"candidate_steps": [3]}}},
                      "armed", None, 4)
    check(False, "non-numeric tier key refusal names the key")
except ValueError as e:
    check("high" in str(e), "non-numeric tier key refusal names the key")

# 5f-2. non-integer candidate elements are refused at install (they would
# TypeError the tier sort / clamp into controller=None)
for bad_cands, where in (([3, "7"], "top-level"), (["3"], "top-level"),
                         ([3.5], "top-level"), ([True], "top-level")):
    try:
        _install_with_cfg({"candidate_steps": bad_cands}, "armed", None, 4)
        check(False, f"install refuses non-integer candidate {bad_cands!r}")
    except ValueError:
        check(True, f"install refuses non-integer candidate {bad_cands!r}")
try:
    _install_with_cfg({"candidate_steps": [3, 7],
                       "tiers": {"1": {"candidate_steps": [2, "9"]}}},
                      "armed", None, 4)
    check(False, "install refuses a non-integer candidate inside a tier")
except ValueError:
    check(True, "install refuses a non-integer candidate inside a tier")
try:
    _install_with_cfg({"candidate_steps": []}, "armed", None, 4)
    check(False, "install refuses an empty candidate ladder")
except ValueError:
    check(True, "install refuses an empty candidate ladder")
# the donor's zero-step nospec candidate stays legal (parity)
c, _ = _install_with_cfg({"candidate_steps": [0, 3],
                          "tiers": {"1": {"candidate_steps": [0, 3]}}},
                         "shadow", None, None)
check(c is not None and c._cfg_for_bs(1)["candidate_steps"] == [0, 3],
      "candidate 0 (donor nospec interval) still installs")

# 5f-3. zero/negative numeric FORCE is a config error on this stack (armed
# maps a zero slot to config_k; the launch pins K >= 1): refused at install
for bad_force in ("0", "-0", " -1 "):
    try:
        install("force", bad_force, batch_size_fn=lambda: 1, launch_k=4)
        check(False, f"mode=force refuses FORCE={bad_force!r} (zero-width is a config error)")
    except ValueError:
        check(True, f"mode=force refuses FORCE={bad_force!r} (zero-width is a config error)")
try:
    install("force", "0", batch_size_fn=lambda: 1, launch_k=4)
    check(False, "zero-width refusal explains the stack semantics")
except ValueError as e:
    check("config error" in str(e), "zero-width refusal explains the stack semantics")

# 5f-4. FORCE parsing is strict: strip, then ASCII digits only, int()'s
# leniency (' 3 ', '+3') is not accepted at the env boundary
c = install("force", " 3 ", batch_size_fn=lambda: 1, launch_k=4)
check(c.k_for_step(1, 4) == 3 and c.force == 3,
      "FORCE=' 3 ' strips to 3 and installs as the int 3")
for bad_force in ("+3", "3.0", "three", "3 3", "-3"):
    try:
        install("force", bad_force, batch_size_fn=lambda: 1, launch_k=4)
        check(False, f"mode=force refuses non-plain-digits FORCE={bad_force!r}")
    except ValueError:
        check(True, f"mode=force refuses non-plain-digits FORCE={bad_force!r}")

# 5f-5. launch_k is REQUIRED for the driving modes (armed/force): without it
# no ladder coordination is possible. off/shadow may omit it.
try:
    install("armed", None, batch_size_fn=lambda: 1)
    check(False, "armed refuses to install without launch_k")
except ValueError:
    check(True, "armed refuses to install without launch_k")
try:
    install("force", "3", batch_size_fn=lambda: 1)
    check(False, "force refuses to install without launch_k")
except ValueError:
    check(True, "force refuses to install without launch_k")
c = install("shadow", None, batch_size_fn=lambda: 1)
check(c is not None and c.k_for_step(1, 3) == 3,
      "shadow still installs without launch_k (never drives)")
c = install("off", None, batch_size_fn=lambda: 1)
check(c is not None and c.k_for_step(1, 3) == 3,
      "off still installs without launch_k (never drives)")

# 5f-6. unknown MODE strings are refused (a typo would otherwise fall through
# to the armed drive branch of k_for_step)
for bad_mode in ("armd", "Armed", "FORCE", ""):
    try:
        install(bad_mode, None, batch_size_fn=lambda: 1, launch_k=4)
        check(False, f"install refuses unknown mode {bad_mode!r}")
    except ValueError:
        check(True, f"install refuses unknown mode {bad_mode!r}")

# 5g. D-009 round-3 gates: the shapes round 2 left reachable that would crash
# later or contradict the driving contract, plus the zero-step nospec
# candidate's donor parity (the real assertions that replace the harness
# selftest's tautological 'zero-width support is the empty set' case).

# 5g-1. the oscillate square wave must never drive the zero width (the wave
# drives the ladder's low extreme; a 0 there is the same config error the
# force contract refuses as FORCE=0). install() refuses the combination for
# the driving modes - default scope AND every tier.
for mode in ("armed", "force"):
    try:
        _install_with_cfg({"candidate_steps": [0, 3]}, mode, "oscillate", 4)
        check(False, f"{mode}+oscillate refuses a 0 candidate in the ladder")
    except ValueError as e:
        check("oscillate" in str(e) and "zero width" in str(e),
              f"{mode}+oscillate refuses a 0 candidate in the ladder "
              "(message names the zero-width rule)")
try:
    _install_with_cfg({"candidate_steps": [3],
                       "tiers": {"1": {"candidate_steps": [0, 3]}}},
                      "armed", "oscillate", 4)
    check(False, "oscillate refuses a 0 candidate inside a tier ladder")
except ValueError:
    check(True, "oscillate refuses a 0 candidate inside a tier ladder")
# the clamp cannot smuggle a 0 through either: [0, 7] at launch 4 clamps to
# [0, 3], which is still an oscillating ladder containing 0 -> refused
try:
    _install_with_cfg({"candidate_steps": [0, 7]}, "armed", "oscillate", 4)
    check(False, "oscillate refuses a 0 candidate that survives the clamp")
except ValueError:
    check(True, "oscillate refuses a 0 candidate that survives the clamp")
# survivals: 0 stays legal whenever nothing would drive it
c, _ = _install_with_cfg({"candidate_steps": [0, 3],
                          "tiers": {"1": {"candidate_steps": [0, 3]}}},
                         "armed", None, 4)
check(c is not None and c.slot_for(1).candidate_steps == [0, 3],
      "armed WITHOUT oscillate keeps the donor 0 candidate (EMA paths)")
c.slot_for(1).current_steps = 0
check(c.k_for_step(1, 4) == 4,
      "armed maps a zero EMA slot to config_k - the stack never drives 0")
c, _ = _install_with_cfg({"candidate_steps": [0, 3]}, "shadow", "oscillate", None)
check(c is not None and c.k_for_step(1, 3) == 3,
      "shadow+oscillate+0 still installs (shadow never drives; FORCE ignored)")
# defense in depth: a post-install mutation planting a 0 cannot make the
# square wave drive it either (the k_for_step filter)
c, _ = _install_with_cfg({"candidate_steps": [3, 4]}, "force", "oscillate", 4)
c.tiers["1"]["candidate_steps"] = [0, 3]  # post-install mutation plants a 0
c.observe(1, [0] * 8)  # one schedule boundary -> the wave's low half
check(c.k_for_step(1, 4) == 3,
      "oscillate's low half maps a planted 0 to the nearest positive candidate")
c.tiers["1"]["candidate_steps"] = [0]  # mutated all-zero ladder
c.observe(1, [0] * 8)
check(c.k_for_step(1, 4) == 4,
      "a mutated all-zero ladder falls back to config_k (zero-slot mapping)")

# 5g-2. a tier VALUE that is not a dict is refused at install (round-2's
# validation loop skipped non-dict tiers; {'4': [7, 9]} then crashed
# _cfg_for_bs's dict update on the first k_for_step)
for bad_tier, label in (({"4": [7, 9]}, "list tier value"),
                        ({"4": "7"}, "string tier value"),
                        ({"4": 7}, "int tier value")):
    try:
        _install_with_cfg({"candidate_steps": [3, 7], "tiers": bad_tier},
                          "armed", None, 4)
        check(False, f"install refuses a {label}")
    except ValueError as e:
        check("must be a dict" in str(e) and "candidate_steps" in str(e),
              f"install refuses a {label} (clear ValueError names the fix)")

# 5g-3. tiers itself must be a dict: a list (["4"]) or a string ("4") used to
# slip past the per-key check (iterating them yields digit strings) and crash
# install() with an AttributeError instead of a config error
for bad_tiers, label in ((["4"], "a tiers list"), ("4", "a tiers string")):
    try:
        _install_with_cfg({"candidate_steps": [3, 7], "tiers": bad_tiers},
                          "armed", None, 4)
        check(False, f"install refuses {label} as the tiers container")
    except ValueError as e:
        check(type(bad_tiers).__name__ in str(e),
              f"install refuses {label} as the tiers container (names the type)")

# 5g-4. launch_k must be an int (bool refused): a float launch_k passed the
# old >= 1 check and derived a fractional ladder [3.5]; a bool derived [True];
# a str crashed install with a TypeError instead of a config error
for bad_lk, label in ((3.5, "a float launch_k"), (True, "a bool launch_k"),
                      (3.0, "a float-valued launch_k"), ("3", "a str launch_k")):
    try:
        install("armed", None, batch_size_fn=lambda: 1, launch_k=bad_lk)
        check(False, f"install refuses {label}")
    except ValueError as e:
        check("int" in str(e), f"install refuses {label} (clear ValueError)")
c = install("armed", None, batch_size_fn=lambda: 1, launch_k=4)
check(c.slot_for(1).candidate_steps == [3],
      "an int launch_k still installs and coordinates the ladder")

# 5g-5. zero-step nospec candidate parity with the donor (D-009 round 3 N7:
# real assertions of the donor's zero-step interval logic, replacing the
# harness selftest's tautological 'zero-width support is the empty set'
# case): the port keeps the donor's zero-step interval - a slot can HOLD the
# 0 candidate, climbs OUT of it at the next recompute (re-seeding the EMA to
# target-1), and drops INTO it from the candidate above at the donor's fixed
# 0.5 threshold (+ down_hysteresis) instead of prev-0.5 - while armed
# k_for_step maps a zero slot to config_k so the stack never DRIVES 0.
zcfg = {"candidate_steps": [0, 3, 7], "ema_alpha": 0.2, "update_interval": 5,
        "warmup_batches": 10, "down_hysteresis": -0.25, "up_hysteresis": 0.0,
        "ceiling_coeff": 0}
slot = AdaptiveStepSlot(0, zcfg)
check(slot.current_steps == 0 and slot.ema_accept_len == -1.0,
      "a slot can hold the donor 0 candidate (nospec interval, ema seeded at -1)")
slot._recompute_params()  # the donor's zero-step interval pass-through
check(slot.current_steps == 3,
      "donor zero-step interval: one recompute at 0 climbs to the next candidate")
check(abs(slot.ema_accept_len - 2.0) < 1e-9,
      "climb-out re-seeds the EMA to target-1 (donor parity)")
slot = AdaptiveStepSlot(3, zcfg)
slot.ema_accept_len = 0.25  # exactly the donor's fixed 0.5 + down_hysteresis
slot._recompute_params()
check(slot.current_steps == 0,
      "drop-INTO the zero interval at the donor's fixed 0.5 threshold (+hys)")
slot = AdaptiveStepSlot(3, zcfg)
slot.ema_accept_len = 0.3  # above the drop-in threshold
slot._recompute_params()
check(slot.current_steps == 3,
      "ema above the drop-in threshold stays at 3 (no spurious zero interval)")

# 6. tier routing
c = AdaptiveWidthController({"candidate_steps": [3, 7], "tiers": {"2": {"candidate_steps": [3]}}}, 7,
                            batch_size_fn=lambda: 1)
check(c._cfg_for_bs(1)["candidate_steps"] == [3, 7], "tier 1 keeps [3,7]")
check(c._cfg_for_bs(8)["candidate_steps"] == [3], "tier 2 pins [3]")

print(f"\n{len(fails)} failures")
sys.exit(1 if fails else 0)
