import torch
import triton
import triton.language as tl


@triton.jit
def _prepare_dflash_draft_block_kernel(
    verified_id_ptr,
    prefix_lens_ptr,
    req_pool_indices_ptr,
    req_to_token_ptr,
    block_ids_out_ptr,
    positions_out_ptr,
    cache_loc_out_ptr,
    verified_id_stride,
    prefix_lens_stride,
    req_pool_indices_stride,
    req_to_token_row_stride,
    req_to_token_col_stride,
    block_ids_row_stride,
    block_ids_col_stride,
    positions_row_stride,
    positions_col_stride,
    cache_loc_row_stride,
    cache_loc_col_stride,
    req_to_token_width,
    block_size,
    mask_token_id,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    row_mask = cols < block_size

    prefix_len = tl.load(prefix_lens_ptr + row * prefix_lens_stride)
    req_idx = tl.load(req_pool_indices_ptr + row * req_pool_indices_stride)
    verified_id = tl.load(verified_id_ptr + row * verified_id_stride)

    logical_pos = prefix_len + cols
    valid = row_mask & (logical_pos < req_to_token_width)
    token_offsets = (
        req_idx * req_to_token_row_stride + logical_pos * req_to_token_col_stride
    )
    slot_ids = tl.load(req_to_token_ptr + token_offsets, mask=valid, other=0)

    block_ids = tl.where(cols == 0, verified_id.to(tl.int64), mask_token_id)
    tl.store(
        block_ids_out_ptr + row * block_ids_row_stride + cols * block_ids_col_stride,
        block_ids,
        mask=row_mask,
    )
    tl.store(
        positions_out_ptr + row * positions_row_stride + cols * positions_col_stride,
        logical_pos.to(tl.int64),
        mask=row_mask,
    )
    tl.store(
        cache_loc_out_ptr + row * cache_loc_row_stride + cols * cache_loc_col_stride,
        slot_ids.to(tl.int64),
        mask=row_mask,
    )


def _prepare_dflash_draft_block_unchecked(
    verified_id: torch.Tensor,
    prefix_lens: torch.Tensor,
    req_pool_indices: torch.Tensor,
    req_to_token: torch.Tensor,
    block_ids_out: torch.Tensor,
    positions_out: torch.Tensor,
    cache_loc_out: torch.Tensor,
    mask_token_id: int,
) -> None:
    batch_size = int(verified_id.numel())
    if batch_size == 0:
        return

    block_size = int(block_ids_out.shape[1])
    block = min(64, triton.next_power_of_2(block_size))
    _prepare_dflash_draft_block_kernel[(batch_size,)](
        verified_id,
        prefix_lens,
        req_pool_indices,
        req_to_token,
        block_ids_out,
        positions_out,
        cache_loc_out,
        verified_id.stride(0),
        prefix_lens.stride(0),
        req_pool_indices.stride(0),
        req_to_token.stride(0),
        req_to_token.stride(1),
        block_ids_out.stride(0),
        block_ids_out.stride(1),
        positions_out.stride(0),
        positions_out.stride(1),
        cache_loc_out.stride(0),
        cache_loc_out.stride(1),
        int(req_to_token.shape[1]),
        block_size,
        int(mask_token_id),
        BLOCK_SIZE=block,
        num_warps=1,
    )
