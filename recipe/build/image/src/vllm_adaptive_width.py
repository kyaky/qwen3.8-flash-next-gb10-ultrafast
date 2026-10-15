#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Adaptive MTP draft-width controller for the gb10-flashnext fork (P2 stage 1).

Faithful pure-Python port of Pennyroyal's AdaptiveStepSlot
(sglang-rtxpro6000 python/sglang/srt/speculative/adaptive_spec_params.py:140-259,
SHA-pinned donor; PLAN §8.2 allows non-kernel logic transfer) plus:
  - per-batch-size tier routing (Pennyroyal adaptive-next.json schema),
  - a JSON config loader with env overrides,
  - shadow/armed/forced modes for the stage-1 gate harness (G1/G2/G3/G4).

No vllm imports here: the controller is offline unit-testable; the wiring patch
(patch_adaptive_width.py) feeds it per-step acceptance and consumes its k.

Modes (VLLM_ADAPTIVE_WIDTH_MODE):
  off    controller not installed (byte-identical floor)
  shadow controller computes k/ema and logs, but k_for_step() returns config K
  armed  controller drives the per-step draft width from the EMA slot;
         VLLM_ADAPTIVE_WIDTH_FORCE=oscillate replaces the EMA decision with a
         forced candidate square wave (G2 correctness harness); any other
         FORCE value is refused at install, numeric pinning belongs to
         mode=force only, so armed stays EMA-driven pure (D-009 trap 2)
  force  pinned to VLLM_ADAPTIVE_WIDTH_FORCE for the G3/G4 locked arms
         ('oscillate' alternates candidates at every observe() boundary -
         one schedule step, for G2)

Launch coordination (D-009 trap 1): install(launch_k=...) clamps the
candidate ladder (default AND every tier) to the launch num_speculative_tokens
with a logged warning, derives the single-candidate ladder [launch_k] when a
scope has nothing left at or below it, and refuses a numeric FORCE above
launch_k (a pinned width the launch cannot hold is unsatisfiable, not
clampable). launch_k is REQUIRED for the driving modes (armed/force), the
armed wiring passes launch_k=self.num_spec_tokens, and optional only for
off/shadow, which never drive. The Pennyroyal config schema is untouched:
coercion happens at install time, never in the config files.

Config validation (D-009 round 2): install() is also the validation boundary
for the config shape, because the deployed armed branch calls k_for_step
UNGUARDED inside Scheduler.update_from_output, anything install() lets
through crashes the scheduler loop at the first driving step. Refused at
install with a clear ValueError: non-numeric tier keys (they crash
_cfg_for_bs's int(key) sort), non-integer candidate_steps elements (they
TypeError the clamp/sort into controller=None), empty ladders, unknown MODE
strings (a typo would fall through to the armed drive branch), and a numeric
FORCE below 1 (zero-width is a config error on this stack: armed maps a zero
slot to config_k and the launch pins K >= 1). FORCE parsing is strict: the
stripped value must be ASCII digits only, int()'s leniency (' 3 ', '+3') is
not accepted at the env boundary.

Config validation (D-009 round 3): the tier TABLE itself must be a dict
mapping positive-integer strings to dicts, a tiers list/string slipped past
the round-2 per-key check (iterating a list or a string yields the same digit
strings, so they 'validated') and then crashed install() itself with an
AttributeError, and a non-dict tier VALUE ({'4': [7, 9]}, the plausible typo
that forgets the candidate_steps nesting) crashed the first k_for_step.
launch_k must be a real int (bool refused): a float launch_k used to derive
a fractional candidate ladder. And the oscillate square wave may not drive
the zero width the force contract refuses: FORCE=oscillate with a 0 candidate
in the driving ladder (default scope or any tier, after clamping) is refused
at install, 0 stays a legal candidate for the EMA paths (armed maps a zero
slot to config_k) and for off/shadow, which never drive.
"""

import json
import logging
import math
import os

logger = logging.getLogger("vllm.adaptive_width")


class AdaptiveWidthController:
    """Tier-routed EMA controller over candidate draft widths (in STEPS).

    Width semantics on this stack: MTP=3 means num_speculative_tokens=3 draft
    steps + 1 bonus token; a candidate_steps value is the STEP count, so the
    draft window is candidate_steps + 1 tokens. Launch must pin
    num_speculative_tokens = max(candidate_steps) (capture-at-max-width rule,
    Pennyroyal adaptive_runtime_state.py:203-228).
    """

    def __init__(self, config: dict, initial_steps: int, batch_size_fn=None,
                 launch_k: int | None = None):
        cfg = dict(config)
        cfg.setdefault("candidate_steps", [initial_steps])
        self.tiers = cfg.get("tiers") or {}
        self.default_cfg = cfg
        self.batch_size_fn = batch_size_fn or (lambda: 1)
        self.launch_k = launch_k
        self._slots = {}
        self.mode = "shadow"
        self.force = None
        self._osc_flip = False

    # ------------------------------------------------------------------
    # tier routing
    # ------------------------------------------------------------------
    def _cfg_for_bs(self, bs: int) -> dict:
        best = None
        for floor in sorted(self.tiers, key=int):
            if bs >= int(floor):
                best = self.tiers[floor]
        if best is None:
            best = self.tiers.get("1", self.default_cfg)
        cfg = dict(self.default_cfg)
        cfg.update(best)
        return cfg

    def slot_for(self, bs: int) -> "AdaptiveStepSlot":
        cfg = self._cfg_for_bs(bs)
        if self.launch_k is not None and max(cfg["candidate_steps"]) > self.launch_k:
            # Defense in depth (D-009 trap 1): install() already coerces the
            # default ladder and every tier; this fires only if a config dict
            # is mutated after install. The controller must never drive a
            # width above the launch num_speculative_tokens, clamp instead.
            kept = sorted({c for c in cfg["candidate_steps"]
                           if c <= self.launch_k}) or [self.launch_k]
            logger.warning(
                "adaptive width: slot candidates %s exceed launch_k=%d; "
                "clamped to %s (post-install config mutation?)",
                sorted(set(cfg["candidate_steps"])), self.launch_k, kept)
            cfg["candidate_steps"] = kept
        key = (bs, tuple(cfg["candidate_steps"]))
        if key not in self._slots:
            initial = max(cfg["candidate_steps"])
            self._slots[key] = AdaptiveStepSlot(initial, cfg)
        return self._slots[key]

    # ------------------------------------------------------------------
    # per-step interface used by the wiring patch
    # ------------------------------------------------------------------
    def observe(self, bs: int, num_correct_drafts_per_req) -> None:
        """Feed one verify round's per-request accepted-draft counts."""
        slot = self.slot_for(bs)
        slot.update(num_correct_drafts_per_req)
        if self.force == "oscillate" and self.mode in ("force", "armed"):
            # One flip per observe() = one schedule boundary, so every
            # k_for_step() within the same step reads the same width.
            slot._osc_state = not getattr(slot, "_osc_state", False)

    def k_for_step(self, bs: int, config_k: int) -> int:
        mode = self.mode
        if mode == "off":
            return config_k
        slot = self.slot_for(bs)
        if mode == "shadow":
            return config_k
        if mode == "force":
            if self.force == "oscillate":
                # flip happens on observe() (one schedule boundary), so every
                # k_for_step within the same step returns the same width.
                # Zero candidates are filtered (D-009 round 3 defense in
                # depth): install() refuses a 0 in an oscillating ladder, so
                # this only fires on a post-install config mutation - the
                # square wave must never drive the zero width the force
                # contract refuses. The low half maps to the nearest positive
                # candidate; a mutated all-zero ladder falls back to config_k
                # (the same zero-slot mapping the armed EMA path uses).
                cands = sorted(c for c in slot.candidate_steps if c > 0)
                if not cands:
                    return config_k
                return cands[0] if getattr(slot, "_osc_state", False) else cands[-1]
            return int(self.force)
        # armed: EMA-driven width, or the forced square wave when the G2
        # harness sets VLLM_ADAPTIVE_WIDTH_FORCE=oscillate (same alternating
        # semantics as force/oscillate, but flowing through the armed path).
        if self.force == "oscillate":
            # Same zero-candidate filter as the force branch above (D-009
            # round 3): the wave never drives the zero width - the low half
            # maps to the nearest positive candidate, a mutated all-zero
            # ladder to config_k, matching the EMA path's zero-slot mapping
            # below.
            cands = sorted(c for c in slot.candidate_steps if c > 0)
            if not cands:
                return config_k
            return cands[0] if getattr(slot, "_osc_state", False) else cands[-1]
        k = slot.current_steps
        return k if k > 0 else config_k

    def state(self) -> dict:
        return {
            "mode": self.mode,
            "force": self.force,
            "slots": [
                {
                    "bs": bs,
                    "current_steps": slot.current_steps,
                    "ema_accept_len": round(slot.ema_accept_len, 4),
                    "batches": slot._batch_count,
                }
                for (bs, _), slot in sorted(self._slots.items())
            ],
        }


class AdaptiveStepSlot:
    """Tracks acceptance via EMA and adapts the draft step count.

    Ported from Pennyroyal AdaptiveStepSlot (Apache-2.0); log calls replaced
    with plain logging and the zero-step nospec interval kept for parity.
    """

    def __init__(self, initial_steps: int, cfg: dict):
        candidates = sorted(set(cfg["candidate_steps"]))
        assert len(candidates) >= 1, "candidate_steps must have at least 1 value"
        self.candidate_steps = candidates

        self.ema_alpha = cfg.get("ema_alpha", 0.2)
        self.update_interval = cfg.get("update_interval", 5)
        self.warmup_batches = cfg.get("warmup_batches", 10)
        self.down_hysteresis = cfg.get("down_hysteresis", -0.25)
        self.up_hysteresis = cfg.get("up_hysteresis", 0.0)
        self.ceiling_coeff = cfg.get("ceiling_coeff", 0)

        if initial_steps in self.candidate_steps:
            self.current_steps = initial_steps
        else:
            self.current_steps = self.candidate_steps[len(self.candidate_steps) // 2]

        self.ema_accept_len = float(self.current_steps - 1)
        self._batch_count = 0

    def update(self, num_correct_drafts_per_req) -> bool:
        if not num_correct_drafts_per_req:
            return False
        if self.current_steps > 0:
            batch_avg = sum(num_correct_drafts_per_req) / len(num_correct_drafts_per_req)
            self.ema_accept_len = (1 - self.ema_alpha) * self.ema_accept_len + self.ema_alpha * batch_avg
        self._batch_count += 1
        if self._batch_count <= self.warmup_batches:
            return False
        if (self._batch_count - self.warmup_batches) % self.update_interval != 0:
            return False
        return self._recompute_params()

    def _recompute_params(self) -> bool:
        old_steps = self.current_steps
        current_idx = self.candidate_steps.index(old_steps)
        old_idx = current_idx

        if old_steps == 0:
            current_idx = min(current_idx + 1, len(self.candidate_steps) - 1)
            target = self.candidate_steps[current_idx]
            if target > 0 and self.ema_accept_len < 0:
                self.ema_accept_len = float(target - 1)
            return self._apply_target_steps(old_steps, target)

        while current_idx > 0:
            prev_step = self.candidate_steps[current_idx - 1]
            drop_threshold = 0.5 if prev_step == 0 else prev_step - 0.5
            drop_threshold += self.down_hysteresis
            if self.ema_accept_len <= drop_threshold:
                current_idx -= 1
            else:
                break

        moved_down = current_idx < old_idx
        if not moved_down:
            while current_idx < len(self.candidate_steps) - 1:
                current_step = self.candidate_steps[current_idx]
                rise_threshold = current_step - 0.5 + self.up_hysteresis
                if self.ema_accept_len > rise_threshold:
                    current_idx += 1
                else:
                    break

        target = self.candidate_steps[current_idx]
        if self.ceiling_coeff > 0:
            ceiling = max(1, math.ceil(self.ema_accept_len * self.ceiling_coeff))
            if target > ceiling and target <= old_steps:
                while current_idx > 0 and self.candidate_steps[current_idx] > ceiling:
                    current_idx -= 1
                target = self.candidate_steps[current_idx]

        return self._apply_target_steps(old_steps, target)

    def _apply_target_steps(self, old_steps: int, target: int) -> bool:
        if target != old_steps:
            self.current_steps = target
            logger.info("Adaptive width: steps %s -> %s (ema=%.2f)", old_steps, target, self.ema_accept_len)
            return True
        return False


DEFAULT_CONFIG = {
    "candidate_steps": [3, 7],
    "tiers": {"1": {"candidate_steps": [3, 7]}},
    "ema_alpha": 0.2,
    "update_interval": 5,
    "warmup_batches": 10,
    "down_hysteresis": -0.25,
    "up_hysteresis": 0.0,
    "ceiling_coeff": 0,
}


def load_config() -> dict:
    """Config from VLLM_ADAPTIVE_WIDTH_CONFIG (Pennyroyal schema) + env overrides."""
    cfg = json.loads(json.dumps(DEFAULT_CONFIG))
    path = os.environ.get("VLLM_ADAPTIVE_WIDTH_CONFIG")
    if path:
        loaded = json.load(open(path))
        for k in ("candidate_steps", "ema_alpha", "update_interval", "warmup_batches",
                  "down_hysteresis", "up_hysteresis", "ceiling_coeff"):
            if k in loaded:
                cfg[k] = loaded[k]
        if "tiers" in loaded:
            cfg["tiers"] = loaded["tiers"]
    for envk, ck in (("VLLM_ADAPTIVE_WIDTH_DOWN_HYS", "down_hysteresis"),
                     ("VLLM_ADAPTIVE_WIDTH_UP_HYS", "up_hysteresis"),
                     ("VLLM_ADAPTIVE_WIDTH_ALPHA", "ema_alpha"),
                     ("VLLM_ADAPTIVE_WIDTH_INTERVAL", "update_interval"),
                     ("VLLM_ADAPTIVE_WIDTH_WARMUP", "warmup_batches")):
        if os.environ.get(envk):
            cfg[ck] = type(cfg[ck])(os.environ[envk])
    return cfg


def _clamp_candidates(cands, launch_k: int):
    """Split a candidate-step list into (kept, dropped) at the launch width.

    Never invents widths: it only filters the configured candidates. An empty
    `kept` means the scope has no usable width at this launch (the caller
    decides: derive [launch_k] or refuse).
    """
    kept = sorted({c for c in cands if c <= launch_k})
    dropped = sorted({c for c in cands if c > launch_k})
    return kept, dropped


def _parse_force_steps(force) -> int:
    """Strict numeric FORCE parse at the env boundary (D-009 round 2).

    int() is lenient in ways a launch env must not be: int(' 3 ') and int('+3')
    both succeed. Here the value is stripped, then the remainder must be ASCII
    digits ONLY, no sign, no decimal point, no internal whitespace. ' 3 ' is
    tolerated (surrounding shell whitespace is common); '+3' and '-3' are
    refused with a message naming the accepted forms.
    """
    text = str(force).strip()
    if not (text.isascii() and text.isdigit()):
        raise ValueError(
            f"VLLM_ADAPTIVE_WIDTH_FORCE must be 'oscillate' or a positive "
            f"integer written as plain digits, got {force!r}")
    return int(text)


def _validate_candidate_steps(cands, where: str) -> None:
    """Config-shape validation for one candidate ladder (D-009 round 2).

    install() is the single enforcement boundary: a malformed ladder that
    survives install crashes later with a TypeError inside the scheduler loop
    (sorted() on mixed int/str) or drives a fractional width. Elements must be
    ints (bool excluded), >= 0 (0 is the donor's legitimate zero-step nospec
    candidate), and the ladder must be non-empty.
    """
    if not isinstance(cands, (list, tuple)):
        raise ValueError(
            f"adaptive width config: {where} candidate_steps must be a list, "
            f"got {type(cands).__name__}")
    if not cands:
        raise ValueError(
            f"adaptive width config: {where} candidate_steps must not be empty")
    for c in cands:
        if isinstance(c, bool) or not isinstance(c, int):
            raise ValueError(
                f"adaptive width config: {where} candidate_steps must contain "
                f"integers only, got element {c!r} ({type(c).__name__}); "
                "non-integer elements TypeError the tier sort at the first "
                "driving step")
        if c < 0:
            raise ValueError(
                f"adaptive width config: {where} candidate_steps must be >= 0, "
                f"got {c!r}")


def _validate_tiers(tiers) -> None:
    """The tier table is a dict mapping positive-integer strings to dicts.

    Tier keys are batch-size floors: they must be positive-integer strings.
    A non-numeric key ('high') survives install() and then crashes
    _cfg_for_bs's int(key) sort on the FIRST k_for_step call, unguarded
    inside Scheduler.update_from_output on the deployed armed wiring. Refuse
    it here with a message naming the key.

    D-009 round 3, the container and the values are validated too:
      - tiers itself must be a DICT. A list (["4"]) or a string ("4") used to
        slip past the per-key check (iterating a list yields its key strings,
        iterating a string yields its digit characters, both 'validate') and
        then crash install() itself with an AttributeError (.items() on a
        non-dict), fail-safe through the wiring's except-arm, but not the
        promised clear config error;
      - every tier VALUE must be a dict. {'4': [7, 9]} (the plausible typo
        that forgets the candidate_steps nesting) used to pass install()
        (the validation loop skipped non-dict tiers) and then crash
        _cfg_for_bs's dict update on the first k_for_step.
    """
    if not isinstance(tiers, dict):
        raise ValueError(
            f"adaptive width config: tiers must be a dict mapping batch-size "
            f"floor strings to override dicts, got {type(tiers).__name__}")
    for key, tier in tiers.items():
        if not (isinstance(key, str) and key.isascii() and key.isdigit()
                and int(key) >= 1):
            raise ValueError(
                f"adaptive width config: tier key {key!r} must be a positive "
                "integer string (the batch-size floor the tier gates); "
                "non-numeric keys crash tier routing at the first driving step")
        if not isinstance(tier, dict):
            raise ValueError(
                f"adaptive width config: tier {key!r} must be a dict of "
                f"config overrides, got {type(tier).__name__} (a plausible "
                'typo omits the candidate_steps nesting, e.g. {"4": [7, 9]} '
                'instead of {"4": {"candidate_steps": [7, 9]}}); a non-dict '
                "tier value TypeErrors the tier merge at the first driving "
                "step")


def install(mode: str, force=None, batch_size_fn=None,
            launch_k: int | None = None) -> AdaptiveWidthController:
    """Build the controller from the env-parsed mode/force and the launch width.

    launch_k coordination (D-009 trap 1): launch pins num_speculative_tokens
    and everything downstream (draft buffers, verify graphs, QSA capacity) is
    sized at it (launch-at-max-width invariant, Pennyroyal
    adaptive_runtime_state.py:203-228), so the controller must never drive a
    width above launch_k. The configured candidate ladder, the default scope
    AND every tier override, is clamped to <= launch_k with a logged
    warning; a scope with no surviving candidate falls back to the
    single-candidate ladder [launch_k] (a tier with nothing left drops its
    override and inherits the coerced default ladder). Candidate values are
    never invented beyond that degenerate fallback. launch_k is REQUIRED for
    the driving modes (armed/force) and optional only for off/shadow: without
    it no ladder coordination is possible, so a driving install is refused
    here rather than allowed to drive an uncoordinated width (the wiring's
    except-arm falls back to the static launch K).

    Config validation (D-009 round 2): install() is the validation boundary
    for the config shape because k_for_step runs UNGUARDED inside
    Scheduler.update_from_output on the deployed armed wiring. Refused with a
    clear ValueError: unknown MODE strings, non-numeric tier keys, non-integer
    or negative candidate_steps elements (default and every tier), and empty
    ladders.

    Config validation (D-009 round 3): the tier TABLE must be a dict mapping
    positive-integer strings to dicts, a tiers list/string used to slip past
    the per-key check and crash install() with an AttributeError, and a
    non-dict tier value ({'4': [7, 9]}, the plausible typo that forgets the
    candidate_steps nesting) used to crash the first k_for_step. launch_k
    must be an int (bool refused): a float launch_k used to pass the >= 1
    check and derive a fractional candidate ladder.

    Zero-width contract (D-009 round 3): FORCE=oscillate with a 0 candidate
    in the driving ladder (the default scope or any tier, after clamping) is
    refused, the square wave drives the ladder's low extreme and would drive
    a REAL zero width, the same config error as FORCE=0. 0 stays a legal
    candidate for the EMA paths (armed maps a zero slot to config_k) and for
    off/shadow, which never drive; k_for_step keeps a defense-in-depth filter
    so a post-install mutation cannot make the wave drive 0 either.

    Force contract (D-009 trap 2): numeric pinning belongs to mode='force'
    only. mode='armed' accepts FORCE=oscillate (the G2 square-wave harness)
    or no FORCE at all, anything else is refused here (the env-parse
    boundary) instead of being silently ignored; the wiring's except-arm
    falls back to the static launch K. mode='force' requires FORCE to be
    'oscillate' or a positive integer parsed STRICTLY (stripped, then ASCII
    digits only, int()'s leniency for ' 3 '/'+3' is not accepted), and an
    integer below 1 is refused (zero-width is a config error on this stack:
    armed maps a zero slot to config_k and the launch pins K >= 1), as is an
    integer above launch_k: a pinned width the launch cannot hold is
    unsatisfiable, not clampable. A numeric FORCE is normalized to int at the
    boundary.
    """
    if force == "":
        force = None  # empty env value behaves as unset (load_config convention)
    if mode not in ("off", "shadow", "armed", "force"):
        raise ValueError(
            f"adaptive width mode {mode!r} is not one of "
            "['off', 'shadow', 'armed', 'force'] (check "
            "VLLM_ADAPTIVE_WIDTH_MODE for typos; an unknown mode would "
            "otherwise fall through to the armed drive branch)")
    if launch_k is not None:
        # D-009 round 3: launch_k must be an int (bool refused). A float
        # launch_k passed the old >= 1 check and derived a FRACTIONAL
        # single-candidate ladder [3.5], which the armed branch would ship
        # into num_spec_tokens_to_schedule; a bool derived [True]. A str
        # launch_k used to crash install() itself with a TypeError ('<' between
        # str and int) instead of a config error.
        if isinstance(launch_k, bool) or not isinstance(launch_k, int):
            raise ValueError(
                f"adaptive width launch_k must be an int (the launch "
                f"num_speculative_tokens), got {launch_k!r} "
                f"({type(launch_k).__name__}); a non-integer launch_k derives "
                "a non-integer candidate ladder and would drive a fractional "
                "width")
        if launch_k < 1:
            raise ValueError(f"adaptive width launch_k must be >= 1, got {launch_k}")
    if mode in ("armed", "force") and launch_k is None:
        raise ValueError(
            "adaptive width: launch_k is required for the driving modes "
            "(armed/force) - the candidate ladder must be coordinated with "
            "the launch num_speculative_tokens (the armed wiring passes "
            "launch_k=self.num_spec_tokens); off/shadow may omit it")
    cfg = load_config()

    # ---- config-shape validation (D-009 round 2/3): refuse malformed tiers
    # and ladders here instead of crashing k_for_step inside the scheduler
    # loop. _validate_tiers checks the container type, every key and every
    # value type, so the loop below can rely on tier being a dict. ----
    _validate_candidate_steps(cfg.get("candidate_steps"), "top-level config")
    tiers = cfg.get("tiers") or {}
    _validate_tiers(tiers)
    for tier_key, tier in tiers.items():
        if "candidate_steps" in tier:
            _validate_candidate_steps(tier["candidate_steps"],
                                      f"tier {tier_key!r}")

    # ---- D-009 trap 1: coordinate the candidate ladder with the launch K ----
    if launch_k is not None:
        original = list(cfg["candidate_steps"])
        kept, dropped = _clamp_candidates(original, launch_k)
        if dropped:
            if kept:
                logger.warning(
                    "adaptive width: candidate_steps %s exceed launch "
                    "num_speculative_tokens=%d; clamped to %s "
                    "(launch-at-max-width invariant: draft buffers, verify "
                    "graphs and QSA capacity are sized at the launch K)",
                    original, launch_k, kept)
                cfg["candidate_steps"] = kept
            else:
                logger.warning(
                    "adaptive width: every candidate in %s exceeds "
                    "launch_k=%d; derived single-candidate ladder [%d] "
                    "(adaptivity disabled at this launch)",
                    original, launch_k, launch_k)
                cfg["candidate_steps"] = [launch_k]
        for tier_key in list(cfg.get("tiers") or {}):
            tier = cfg["tiers"][tier_key]
            if not isinstance(tier, dict) or "candidate_steps" not in tier:
                continue
            tkept, tdropped = _clamp_candidates(tier["candidate_steps"], launch_k)
            if not tdropped:
                continue
            if tkept:
                logger.warning(
                    "adaptive width: tier %r candidate_steps %s exceed "
                    "launch_k=%d; clamped to %s",
                    tier_key, tier["candidate_steps"], launch_k, tkept)
                tier["candidate_steps"] = tkept
            else:
                logger.warning(
                    "adaptive width: tier %r candidate_steps %s has no width "
                    "<= launch_k=%d; tier inherits the default ladder %s",
                    tier_key, tier["candidate_steps"], launch_k,
                    cfg["candidate_steps"])
                del tier["candidate_steps"]

    # ---- D-009 round 3: the oscillate square wave must not drive the zero
    # width. The wave drives the ladder's extremes, so a 0 candidate at the
    # low extreme (legal for the EMA paths - the donor's zero-step nospec
    # interval) would drive a REAL zero width here: exactly the state the
    # force contract calls a config error ("zero-width is not a drivable
    # launch state"), and the armed EMA path never drives it either (a zero
    # slot maps to config_k). Chosen contract: REFUSE the combination at
    # install (the wiring's except-arm falls back to static K) rather than
    # silently skipping the 0 candidate, which would flatten the G2 square
    # wave it exists to exercise. 0 stays legal for the EMA paths and for
    # off/shadow, which never drive - only the oscillate combination is
    # refused. k_for_step keeps a matching defense-in-depth filter for
    # post-install mutations. ----
    if force == "oscillate" and mode in ("armed", "force"):
        osc_ladders = [("top-level config", cfg["candidate_steps"])]
        osc_ladders += [
            (f"tier {tier_key!r}", tier["candidate_steps"])
            for tier_key, tier in (cfg.get("tiers") or {}).items()
            if isinstance(tier, dict) and "candidate_steps" in tier]
        for where, lads in osc_ladders:
            if 0 in lads:
                raise ValueError(
                    f"adaptive width config: {where} candidate_steps {lads} "
                    "contains 0 (the donor's zero-step nospec candidate) "
                    "while FORCE=oscillate drives the ladder's low extreme: "
                    "the square wave would drive a REAL zero width, which is "
                    "a config error on this stack (the same rule that refuses "
                    "FORCE=0; the armed EMA path maps a zero slot to "
                    "config_k instead of driving it). Remove 0 from the "
                    "oscillating ladder or drop FORCE=oscillate")

    # ---- D-009 trap 2: the force contract, enforced at the env boundary ----
    if mode == "armed" and force is not None and force != "oscillate":
        raise ValueError(
            f"VLLM_ADAPTIVE_WIDTH_FORCE={force!r} is invalid in mode='armed': "
            "armed is EMA-driven and only accepts FORCE=oscillate (the G2 "
            "square-wave harness); numeric pinning belongs to mode='force'"
        )
    if mode == "force":
        if force is None:
            raise ValueError(
                "mode='force' requires VLLM_ADAPTIVE_WIDTH_FORCE "
                "('oscillate' or a positive integer step count)")
        if force != "oscillate":
            force_k = _parse_force_steps(force)
            if force_k < 1:
                raise ValueError(
                    f"VLLM_ADAPTIVE_WIDTH_FORCE={force!r}: a pinned width of "
                    f"{force_k} is a config error on this stack (zero-width "
                    "is not a drivable launch state: armed maps a zero slot "
                    "to config_k and the launch pins K >= 1); use a positive "
                    "integer or 'oscillate'")
            if launch_k is not None and force_k > launch_k:
                raise ValueError(
                    f"adaptive width force={force_k} exceeds launch "
                    f"num_speculative_tokens={launch_k} (launch-at-max-width "
                    "invariant); lower FORCE or raise the launch K")
            force = force_k  # normalize numeric FORCE to int at the boundary
    elif mode == "shadow" and force is not None:
        logger.warning(
            "adaptive width: FORCE=%r set but mode='shadow' never drives K; "
            "ignored (use MODE=force to pin a width)", force)

    initial = max(cfg["candidate_steps"])
    ctrl = AdaptiveWidthController(cfg, initial, batch_size_fn, launch_k=launch_k)
    ctrl.mode = mode
    ctrl.force = force
    if force == "oscillate" and launch_k is not None \
            and len(cfg["candidate_steps"]) < 2:
        logger.warning(
            "adaptive width: FORCE=oscillate needs >= 2 candidates at or "
            "below launch_k=%d; the square wave will hold a constant width",
            launch_k)
    logger.info("adaptive width installed: mode=%s force=%s candidates=%s "
                "launch_k=%s",
                mode, force, sorted(cfg["candidate_steps"]), launch_k)
    return ctrl


def gb10_aw_bind(scheduler_cls):
    """Bind the per-step acceptance flush onto the scheduler class.

    Called from the appended block in scheduler.py (patch_adaptive_width.py).
    The wrapper runs after update_from_output and feeds the step's collected
    per-request accepted counts into the controller; when no controller is
    installed (MODE=off) it is a no-op getattr.
    """
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
