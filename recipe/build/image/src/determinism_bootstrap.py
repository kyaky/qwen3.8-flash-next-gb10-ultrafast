# SPDX-License-Identifier: Apache-2.0
"""Determinism bootstrap for the torch-determinism experiment lane
(2026-10-14, D-011/D-012 decode tie-flip: does the PyTorch global
deterministic-algorithms switch cover the noise source?).

Arms PyTorch's global switch at interpreter startup in EVERY process of the
container (engine core, workers, API server, PLE prewarm pool) via the
``zz_determinism_bootstrap.pth`` import hook in site-packages.  This is
needed because neither torch 2.13 nor this vLLM snapshot reads any
environment variable for the switch (verified in-image 2026-10-14:
``torch/utils/deterministic.py`` has no ``getenv``, and vLLM only touches
``CUBLAS_WORKSPACE_CONFIG`` inside the unused batch-invariant module).

What it arms (guarded by env ``DETERM_BOOTSTRAP``; default on, ``=0`` gives
stock iter7d behavior from the same image - the same-image control arm):

  torch.use_deterministic_algorithms(True, warn_only=True)

``warn_only=True``: ops that have a deterministic CUDA implementation use
it; ops flagged nondeterministic without a deterministic alternative emit a
warning naming the op instead of raising, so the boot survives and the
warning list is the experiment's "which ops complained" result.  A strict
boot was not attempted first: it would die on the first nondeterministic op
reached during startup (the batch-invariant arm already failed that way on
GDN_ATTN, see D-011).

Also pins ``cudnn.deterministic=True`` and ``cudnn.benchmark=False``
(cudnn algo selection), and relies on the image ENV
``CUBLAS_WORKSPACE_CONFIG=:16:8`` - the workspace config cuBLAS requires
once the switch is on, which additionally disables cuBLAS's split-k atomic
GEMM algorithms (run-to-run nondeterministic on identical inputs).

Zero model-code impact: the switch only changes ATen kernel selection and
warning behavior.  Custom CUDA kernels (Marlin GEMM, FLA/GDN Triton, QSA
Triton, FlashInfer, PLE lookup) are outside torch's reach; whether the
tie-flip survives this boot is exactly what the experiment discriminates.

The bootstrap must never break interpreter startup: everything is wrapped,
failures print ``DETERM-FATAL`` and continue.
"""
import os


def _arm():
    try:
        import torch

        torch.use_deterministic_algorithms(True, warn_only=True)
        try:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except Exception:  # cudnn-less builds
            pass
        warn_only = getattr(
            torch._C, "_get_deterministic_algorithms_warn_only", lambda: None
        )()
        print(
            "DETERM-INSTALLED enabled=%s warn_only=%s cudnn_det=%s "
            "cublas_ws=%s torch=%s"
            % (
                torch.are_deterministic_algorithms_enabled(),
                warn_only,
                torch.backends.cudnn.deterministic,
                os.environ.get("CUBLAS_WORKSPACE_CONFIG") or "UNSET",
                torch.__version__,
            ),
            flush=True,
        )
    except Exception as exc:  # never break interpreter startup
        print("DETERM-FATAL %r" % (exc,), flush=True)


# Env-gated so the same image can serve as its own control arm
# (DETERM_BOOTSTRAP=0) and so non-experiment reuses stay stock.
if os.environ.get("DETERM_BOOTSTRAP", "1") != "0":
    _arm()
