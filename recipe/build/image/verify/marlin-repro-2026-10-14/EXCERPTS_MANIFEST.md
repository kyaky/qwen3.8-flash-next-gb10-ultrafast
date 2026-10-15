# Excerpts manifest - D-011 marlin repro (2026-10-14)

Provenance:
- python: docker image `qwen38-flash-dgx:iter7c-20261014` (id e21ef9c8f950), vllm 0.1.dev20073+g8e685d198, site-packages /usr/local/lib/python3.12/dist-packages/vllm (docker cp, read-only).
- csrc: upstream `vllm-project/vllm` main fetched 2026-10-14 (https://github.com/vllm-project/vllm csrc/moe/marlin_moe_wna16/). The image ships the native marlin compiled-only (`vllm/_moe_C_stable_libtorch.abi3.so`, PyInit__moe_C_stable_libtorch); no .cu/.h in image or build context, and commit `g8e685d198` does not resolve upstream, so these upstream files are the closest matching reference for the compiled kernel: the image's python wrapper signature (moe_block_size/top_k/mul_topk_weights/num_tokens_past_padded/blocks_per_sm/use_atomic_add/use_fp32_reduce) matches upstream main's ops.cu exactly.

| file | source | lines | sha256[:16] | bytes | note |
|---|---|---|---|---|---|
| marlin_moe_120-235_gemm_calls.py | marlin_moe.py | 120-235 | 992a4576a6327a76 | 3367 | |
| marlin_moe_59-119_workspace_caches.py | marlin_moe.py | 59-120 | 17f838e8da36b435 | 2374 | |
| marlin_moe_311-346_blocksize_align.py | marlin_moe.py | 311-346 | 88e324d16489d3ec | 1383 | |
| custom_ops_2509-2606_moe_wna16_marlin_gemm.py | _custom_ops.py | 2509-2606 | de852c7c64319f26 | 2732 | |
| marlin_utils_334-352_padded_intermediate.py | marlin_utils.py | 334-352 | 50bd04f530fbfa61 | 1152 | |
| marlin_utils_408-433_make_workspace.py | marlin_utils.py | 408-433 | f0e25dd6e1017176 | 1279 | |
| marlin_utils_620-653_atomic_add_gate.py | marlin_utils.py | 620-653 | bfd776d6d6fbafe9 | 1318 | |
| marlin_utils_666-685_input_dtype.py | marlin_utils.py | 666-685 | 968a94b0b966e46c | 904 | |
| auto_gptq_467-500_moe_method_head.py | auto_gptq.py | 467-500 | a094d19b9293516a | 1252 | |
| auto_gptq_673-782_workspace_repack.py | auto_gptq.py | 673-782 | 71a25dedc457cc84 | 4467 | |
| int_wna16_533-700_process_weights_marlin.py | oracle_int_wna16.py | 533-700 | a138ea4e1a29d6a9 | 6823 | |
| awdbg_hash_306-345_detgemm_probe.py | awdbg_hash.py | 306-345 | aa641dd2c08e09c9 | 1793 | |
| ops_cu_46-243_exec_config.cu | upstream_csrc/ops.cu | 46-243 | bd7ea6cbace551b8 | 7016 | |
| ops_cu_245-388_marlin_mm.cu | upstream_csrc/ops.cu | 245-388 | 587aec9cb7d4a555 | 6740 | |
| ops_cu_392-566_host_wrapper.cu | upstream_csrc/ops.cu | 392-566 | 16a9c13d91f4d350 | 7524 | |
| ops_cu_662-702_workspace_check_call.cu | upstream_csrc/ops.cu | 662-702 | 5217e6c35fc90f7d | 1987 | |
| marlin_h_155-205_barriers.cu | upstream_csrc/marlin_template.h | 155-205 | 83f0839b8fbcabaf | 1664 | |
| marlin_h_355-631_streamk_init.cu | upstream_csrc/marlin_template.h | 355-631 | e63b7e6cc27ecb59 | 9894 | |
| marlin_h_1500-1620_reduce_store.cu | upstream_csrc/marlin_template.h | 1500-1620 | 7996ef701da673c2 | 4387 | |
| marlin_h_1810-1958_main_loop.cu | upstream_csrc/marlin_template.h | 1810-1958 | a856f128f7dc5374 | 5481 | |
| torch_bindings_1-120_head.cpp | upstream_csrc/torch_bindings.cpp | 1-120 | ab7518ac397c9dd4 | 5622 | |

| align_scatter_atomicAdd_341-368.cu | upstream_csrc/moe_align_sum_kernels.cu | 341-368/371-386 | 378726d756317dd9 | 1431 | SMOKING GUN: align kernel assigns the within-group slot by atomicAdd arrival order |
| align_kernel_entry_371-386.cu | upstream_csrc/moe_align_sum_kernels.cu | 341-368/371-386 | 2a593bdb297ee22d | 958 | moe_align_block_size_kernel entry |

- raw/_custom_ops.py: 123207 bytes sha256[:16]=9f11980f43c44857
- raw/auto_gptq.py: 31664 bytes sha256[:16]=35bdb229a83a3900
- raw/awdbg_hash.py: 19482 bytes sha256[:16]=abc5890358de1da1
- raw/int_wna16.py: 59508 bytes sha256[:16]=53964e8177ee062c
- raw/marlin_moe.py: 37862 bytes sha256[:16]=8aea91f825588e8f
- raw/marlin_utils.py: 26649 bytes sha256[:16]=6f91cbf2b0ae276a
- raw/oracle_int_wna16.py: 59508 bytes sha256[:16]=53964e8177ee062c
- raw/upstream_csrc/.gitignore: 34 bytes sha256[:16]=700b5908034e14a4
- raw/upstream_csrc/kernel.h: 2570 bytes sha256[:16]=f62a9719689b74d3
- raw/upstream_csrc/marlin_cuh_ref.cuh: 4353 bytes sha256[:16]=edaa19c541dfbe5e
- raw/upstream_csrc/marlin_template.h: 77766 bytes sha256[:16]=516749049e46b27b
- raw/upstream_csrc/moe_align_sum_kernels.cu: 50194 bytes sha256[:16]=d21e3309a45fd243
- raw/upstream_csrc/moe_wna16.cu: 14334 bytes sha256[:16]=ce2f79438f7fafb5
- raw/upstream_csrc/moe_wna16_utils.h: 6401 bytes sha256[:16]=b14d161c75db659a
- raw/upstream_csrc/ops.cu: 28066 bytes sha256[:16]=4a42e3a1aff950e6
- raw/upstream_csrc/torch_bindings.cpp: 7822 bytes sha256[:16]=eda00c72aa069b93
