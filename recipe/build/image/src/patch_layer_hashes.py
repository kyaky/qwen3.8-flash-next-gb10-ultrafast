#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Install the DETHASH per-layer hash hook into GPUModelRunner.load_model
(D-011 determinism localization, 2026-10-14).

The hook module awdbg_hash.py is copied to site-packages next to vllm; this
patch wires its maybe_install() into the end of load_model so the hooks are
attached to the freshly loaded target model before any forward runs.  The
wiring is env-gated inside awdbg_hash (DET_HASH_LAYERS=1) and doubly guarded
by try/except so it can never break model loading.

Assert-guarded: the anchor must occur exactly once and the patch refuses to
double-install.  A re-run without "off" fails loudly (anchor no longer
present); run with "off" to restore the stock load_model tail.

Usage: python3 patch_layer_hashes.py <model_runner.py> [off]
"""
import sys

TARGET = sys.argv[1]

ANCHOR = """        get_offloader().post_init()

    def get_model(self) -> nn.Module:
"""

NEW = """        get_offloader().post_init()

        # ---- DETHASH (D-011 layer-hash localization; gb10-flashnext 2026-10-14) ----
        # Env-gated per-layer output hashing (DET_HASH_LAYERS=1); inert and
        # zero-overhead otherwise.  Never allowed to break model loading.
        try:
            import awdbg_hash as _awdbg_hash
            _awdbg_hash.maybe_install(self.model, runner=self)
        except Exception as _dethash_err:
            logger.error("DETHASH install failed: %s", _dethash_err)
        # ---- end DETHASH ----

    def get_model(self) -> nn.Module:
"""

s = open(TARGET).read()
off = sys.argv[2:] and sys.argv[2] == "off"
if off:
    if NEW in s:
        open(TARGET, "w").write(s.replace(NEW, ANCHOR))
        print("  dethash REMOVED")
    else:
        print("  dethash not present; nothing to remove")
elif "DETHASH" in s:
    if NEW in s:
        print("  dethash already installed")
    else:
        raise SystemExit("  dethash marker present but anchor block missing; "
                         "refusing to touch the file")
elif s.count(ANCHOR) == 1:
    open(TARGET, "w").write(s.replace(ANCHOR, NEW))
    print(f"  dethash INSTALLED in {TARGET} (inert unless DET_HASH_LAYERS=1)")
else:
    raise SystemExit(f"  anchor count {s.count(ANCHOR)} != 1 in {TARGET}; "
                     "refusing to patch")
