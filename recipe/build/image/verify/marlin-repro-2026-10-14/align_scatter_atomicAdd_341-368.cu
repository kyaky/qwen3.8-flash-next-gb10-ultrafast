# excerpt: upstream_csrc/moe_align_sum_kernels.cu lines 341-368 -- the SMOKING GUN: within-group slot assigned by atomicAdd arrival order -> run-to-run nondeterministic sorted_token_ids ordering
__device__ void _count_and_sort_expert_tokens(
    const scalar_t* __restrict__ topk_ids,
    int32_t* __restrict__ sorted_token_ids, int32_t* __restrict__ cumsum_buffer,
    int32_t* __restrict__ expert_map, size_t numel, int32_t num_experts,
    int32_t max_num_tokens_padded, int32_t* __restrict__ token_mask,
    int32_t model_offset, int32_t topk_num, bool has_expert_map,
    int32_t* scatter_idx = nullptr) {
  const size_t tid = blockIdx.y * blockDim.x + threadIdx.x;
  const size_t stride = blockDim.x * gridDim.y;

  for (size_t i = tid; i < numel; i += stride) {
    int expert_id = get_local_expert_id(i, topk_ids, expert_map, num_experts,
                                        has_expert_map);
    bool valid =
        expert_id != -1 && (token_mask == nullptr || token_mask[i / topk_num]);
    if (scatter_idx != nullptr) {
      scatter_idx[i] = valid ? static_cast<int32_t>(i) : -1;
    }
    if (expert_id != -1) {
      if (token_mask == nullptr || token_mask[i / topk_num]) {
        int32_t rank_post_pad = atomicAdd(
            &cumsum_buffer[(model_offset * (num_experts + 1)) + expert_id], 1);
        sorted_token_ids[max_num_tokens_padded * model_offset + rank_post_pad] =
            i;
      }
    }
  }
}
