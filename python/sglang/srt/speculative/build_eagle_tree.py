# NOTE: Please run this file to make sure the test cases are correct.

import math
from enum import IntEnum
from typing import List, Optional

import torch

# Import EAGLE3 tracing utilities
from sglang.srt.speculative.eagle_trace_utils import (
    eagle_trace, trace_call, trace_return, trace_intermediate, 
    trace_gpu_kernel, trace_memory_op, trace_enabled
)

from sglang.srt.utils import is_cuda, is_hip

if is_cuda() or is_hip():
    from sgl_kernel import (
        build_tree_kernel_efficient as sgl_build_tree_kernel_efficient,
    )


@eagle_trace
def build_tree_kernel_efficient_preprocess(
    verified_id: torch.Tensor,
    score_list: List[torch.Tensor],
    token_list: List[torch.Tensor],
    parents_list: List[torch.Tensor],
    num_verify_tokens: int,
):
    trace_intermediate("PREPROCESS_INPUTS",
                      verified_id=verified_id,
                      score_list_lengths=[s.shape for s in score_list],
                      token_list_lengths=[t.shape for t in token_list],
                      parents_list_lengths=[p.shape for p in parents_list],
                      num_verify_tokens=num_verify_tokens)
    
    score_list = torch.cat(score_list, dim=1).flatten(
        1
    )  # b, n, topk; n= 1 + (num_steps-1) * self.topk
    
    trace_intermediate("SCORE_LIST_CONCATENATED",
                      concatenated_score_list=score_list)
    
    ss_token_list = torch.cat(
        token_list, dim=1
    )  # b, (self.topk + (num_steps-1) * self.topk)
    
    trace_intermediate("TOKEN_LIST_CONCATENATED",
                      concatenated_token_list=ss_token_list)
    
    top_scores = torch.topk(score_list, num_verify_tokens - 1, dim=-1)
    top_scores_index = top_scores.indices
    
    trace_intermediate("TOP_SCORES_COMPUTED",
                      top_scores_values=top_scores.values,
                      top_scores_indices=top_scores_index)
    
    top_scores_index = torch.sort(top_scores_index).values
    
    trace_intermediate("TOP_SCORES_SORTED",
                      sorted_top_scores_index=top_scores_index)
    
    draft_tokens = torch.gather(ss_token_list, index=top_scores_index, dim=1)
    
    trace_intermediate("DRAFT_TOKENS_GATHERED",
                      draft_tokens_without_verified=draft_tokens)
    
    draft_tokens = torch.cat((verified_id.unsqueeze(1), draft_tokens), dim=1).flatten()
    
    trace_intermediate("DRAFT_TOKENS_WITH_VERIFIED",
                      final_draft_tokens=draft_tokens)

    if len(parents_list) > 1:
        parent_list = torch.cat(parents_list[:-1], dim=1)
        trace_intermediate("PARENT_LIST_CONCATENATED",
                          parent_list=parent_list,
                          num_parents_lists=len(parents_list))
    else:
        batch_size = parents_list[0].shape[0]
        parent_list = torch.empty(batch_size, 0, device=parents_list[0].device)
        trace_intermediate("EMPTY_PARENT_LIST_CREATED",
                          parent_list=parent_list,
                          batch_size=batch_size)

    trace_intermediate("PREPROCESS_OUTPUTS",
                      parent_list=parent_list,
                      top_scores_index=top_scores_index,
                      draft_tokens=draft_tokens)

    return parent_list, top_scores_index, draft_tokens


class TreeMaskMode(IntEnum):
    FULL_MASK = 0
    QLEN_ONLY = 1
    QLEN_ONLY_BITPACKING = 2


@eagle_trace
def build_tree_kernel_efficient(
    verified_id: torch.Tensor,
    score_list: List[torch.Tensor],
    token_list: List[torch.Tensor],
    parents_list: List[torch.Tensor],
    seq_lens: torch.Tensor,
    seq_lens_sum: int,
    topk: int,
    spec_steps: int,
    num_verify_tokens: int,
    tree_mask_mode: TreeMaskMode = TreeMaskMode.FULL_MASK,
    tree_mask_buf: Optional[torch.Tensor] = None,
    position_buf: Optional[torch.Tensor] = None,
):
    trace_intermediate("TREE_BUILD_INIT",
                      verified_id=verified_id,
                      seq_lens=seq_lens,
                      seq_lens_sum=seq_lens_sum,
                      topk=topk,
                      spec_steps=spec_steps,
                      num_verify_tokens=num_verify_tokens,
                      tree_mask_mode=tree_mask_mode,
                      has_tree_mask_buf=tree_mask_buf is not None,
                      has_position_buf=position_buf is not None)
    
    parent_list, top_scores_index, draft_tokens = (
        build_tree_kernel_efficient_preprocess(
            verified_id,
            score_list,
            token_list,
            parents_list,
            num_verify_tokens,
        )
    )
    
    trace_intermediate("PREPROCESSING_COMPLETED",
                      parent_list=parent_list,
                      top_scores_index=top_scores_index,
                      draft_tokens=draft_tokens)

    # seq_lens_sum == sum(seq_lens); seq_lens: sequence length without draft tokens
    bs = seq_lens.numel()
    device = seq_lens.device
    
    trace_intermediate("BATCH_INFO",
                      batch_size=bs,
                      device=device)
    
    # e.g. for bs=1, tree_mask: num_draft_token, seq_lens_sum + num_draft_token (flattened)
    # where each row indicates the attending pattern of each draft token
    # if use_partial_packed_tree_mask is True, tree_mask: num_draft_token (flattened, packed)
    if tree_mask_buf is not None:
        tree_mask = tree_mask_buf
        trace_intermediate("USING_PROVIDED_TREE_MASK_BUF",
                          tree_mask_shape=tree_mask.shape)
    elif tree_mask_mode == TreeMaskMode.QLEN_ONLY:
        tree_mask_size = num_verify_tokens * bs * num_verify_tokens
        tree_mask = torch.full(
            (tree_mask_size,),
            True,
            dtype=torch.bool,
            device=device,
        )
        trace_intermediate("CREATED_QLEN_ONLY_TREE_MASK",
                          tree_mask_size=tree_mask_size,
                          tree_mask=tree_mask)
    elif tree_mask_mode == TreeMaskMode.QLEN_ONLY_BITPACKING:
        packed_dtypes = [torch.uint8, torch.uint16, torch.uint32]
        packed_dtype_idx = int(math.ceil(math.log2((num_verify_tokens + 7) // 8)))
        tree_mask_size = num_verify_tokens * bs
        tree_mask = torch.zeros(
            (tree_mask_size,),
            dtype=packed_dtypes[packed_dtype_idx],
            device=device,
        )
        trace_intermediate("CREATED_BITPACKED_TREE_MASK",
                          tree_mask_size=tree_mask_size,
                          packed_dtype_idx=packed_dtype_idx,
                          packed_dtype=packed_dtypes[packed_dtype_idx],
                          tree_mask=tree_mask)
    elif tree_mask_mode == TreeMaskMode.FULL_MASK:
        tree_mask_size = (
            seq_lens_sum * num_verify_tokens
            + num_verify_tokens * num_verify_tokens * bs
        )
        tree_mask = torch.full(
            (tree_mask_size,),
            True,
            device=device,
        )
        trace_intermediate("CREATED_FULL_TREE_MASK",
                          tree_mask_size=tree_mask_size,
                          seq_lens_component=seq_lens_sum * num_verify_tokens,
                          draft_component=num_verify_tokens * num_verify_tokens * bs,
                          tree_mask=tree_mask)
    else:
        trace_intermediate("INVALID_TREE_MASK_MODE", tree_mask_mode=tree_mask_mode)
        raise NotImplementedError(f"Invalid tree mask: {tree_mask_mode=}")

    # TODO: make them torch.empty and fuse them into `sgl_build_tree_kernel`
    retrive_index = torch.full(
        (bs, num_verify_tokens), -1, device=device, dtype=torch.long
    )
    retrive_next_token = torch.full(
        (bs, num_verify_tokens), -1, device=device, dtype=torch.long
    )
    retrive_next_sibling = torch.full(
        (bs, num_verify_tokens), -1, device=device, dtype=torch.long
    )
    
    trace_intermediate("RETRIEVAL_TENSORS_ALLOCATED",
                      retrive_index=retrive_index,
                      retrive_next_token=retrive_next_token,
                      retrive_next_sibling=retrive_next_sibling)
    
    # position: where each token belongs to
    # e.g. if depth of each draft token is [0, 1, 1, 2] and the prompt length is 7
    # then, positions = [7, 8, 8, 9]
    if position_buf is not None:
        positions = position_buf
        trace_intermediate("USING_PROVIDED_POSITION_BUF",
                          positions_shape=positions.shape)
    else:
        positions = torch.empty(
            (bs * num_verify_tokens,), device=device, dtype=torch.long
        )
        trace_intermediate("ALLOCATED_POSITION_TENSOR",
                          positions_shape=positions.shape)

    trace_intermediate("CALLING_SGL_BUILD_TREE_KERNEL",
                      parent_list=parent_list,
                      top_scores_index=top_scores_index,
                      seq_lens=seq_lens,
                      tree_mask_shape=tree_mask.shape,
                      positions_shape=positions.shape,
                      retrieval_tensors_shapes={
                          "retrive_index": retrive_index.shape,
                          "retrive_next_token": retrive_next_token.shape,
                          "retrive_next_sibling": retrive_next_sibling.shape
                      })

    trace_gpu_kernel("sgl_build_tree_kernel_efficient",
                   inputs={
                       "parent_list": parent_list,
                       "top_scores_index": top_scores_index,
                       "seq_lens": seq_lens,
                       "topk": topk,
                       "spec_steps": spec_steps,
                       "num_verify_tokens": num_verify_tokens,
                       "tree_mask_mode": tree_mask_mode
                   },
                   outputs={
                       "tree_mask": "mutable - attention mask patterns",
                       "positions": "mutable - token depth positions",
                       "retrive_index": "mutable - token retrieval indices",
                       "retrive_next_token": "mutable - next token pointers",
                       "retrive_next_sibling": "mutable - sibling pointers"
                   })

    sgl_build_tree_kernel_efficient(
        parent_list,
        top_scores_index,
        seq_lens,
        tree_mask,
        positions,
        retrive_index,
        retrive_next_token,
        retrive_next_sibling,
        topk,
        spec_steps,
        num_verify_tokens,
        tree_mask_mode,
    )
    
    trace_intermediate("TREE_KERNEL_COMPLETED",
                      tree_mask=tree_mask,
                      positions=positions,
                      retrive_index=retrive_index,
                      retrive_next_token=retrive_next_token,
                      retrive_next_sibling=retrive_next_sibling,
                      draft_tokens=draft_tokens)
    
    output = (
        tree_mask,
        positions,
        retrive_index,
        retrive_next_token,
        retrive_next_sibling,
        draft_tokens,
    )
    
    trace_intermediate("TREE_BUILD_OUTPUT",
                      output_tree_mask=output[0],
                      output_positions=output[1],
                      output_retrive_index=output[2],
                      output_retrive_next_token=output[3],
                      output_retrive_next_sibling=output[4],
                      output_draft_tokens=output[5])
    
    return output


def test_build_tree_kernel_efficient():
    verified_id = torch.tensor([29974, 13], device="cuda", dtype=torch.int32)
    score_list = [
        torch.tensor(
            [
                [[7.1127e-01, 2.8292e-01, 2.2995e-03, 1.7357e-03]],
                [[9.7476e-01, 2.2219e-02, 6.5031e-04, 1.3212e-04]],
            ],
            dtype=torch.float32,
            device="cuda",
        ),
        torch.tensor(
            [
                [
                    [6.9142e-01, 1.2863e-02, 1.6873e-03, 1.1871e-03],
                    [2.4787e-01, 1.8818e-02, 1.4204e-02, 9.2235e-04],
                    [2.2971e-03, 1.6700e-06, 1.8737e-07, 8.3146e-08],
                    [1.2771e-03, 2.4374e-04, 1.7832e-04, 1.1947e-05],
                ],
                [
                    [8.4832e-02, 6.6068e-02, 5.8304e-02, 5.7851e-02],
                    [2.3616e-03, 1.1243e-03, 5.4368e-04, 2.7768e-04],
                    [2.5286e-04, 1.5578e-04, 2.8817e-05, 1.2888e-05],
                    [1.2834e-04, 2.5417e-06, 1.1279e-06, 1.6088e-08],
                ],
            ],
            dtype=torch.float32,
            device="cuda",
        ),
        torch.tensor(
            [
                [
                    [6.6438e-01, 2.6997e-02, 2.4236e-05, 4.0821e-06],
                    [2.4402e-01, 2.8409e-03, 5.0935e-04, 2.9022e-04],
                    [1.6178e-02, 2.0567e-03, 4.5892e-04, 3.0034e-05],
                    [1.3023e-02, 5.0497e-04, 3.6371e-04, 8.7750e-05],
                ],
                [
                    [2.3263e-02, 2.0054e-02, 9.3990e-03, 2.7783e-03],
                    [6.4156e-02, 5.5506e-04, 1.0429e-04, 9.7211e-05],
                    [4.9950e-02, 5.0630e-03, 9.0068e-04, 3.3656e-04],
                    [7.5817e-03, 8.5731e-04, 6.9972e-04, 6.0793e-04],
                ],
            ],
            dtype=torch.float32,
            device="cuda",
        ),
        torch.tensor(
            [
                [
                    [6.6420e-01, 1.0525e-04, 6.5864e-05, 1.2253e-06],
                    [1.3019e-01, 1.0461e-01, 5.2083e-03, 1.6777e-03],
                    [2.0103e-02, 6.7335e-03, 1.2625e-04, 1.0364e-05],
                    [1.5142e-02, 7.0819e-04, 9.6595e-05, 8.7951e-05],
                ],
                [
                    [5.8608e-02, 1.8840e-03, 7.8535e-04, 4.4400e-04],
                    [1.2185e-02, 2.0684e-03, 1.7418e-03, 1.4327e-03],
                    [6.2455e-03, 6.1487e-03, 2.6862e-03, 1.8034e-03],
                    [1.8590e-03, 1.6151e-03, 1.2481e-03, 3.6038e-04],
                ],
            ],
            dtype=torch.float32,
            device="cuda",
        ),
    ]
    token_list = [
        torch.tensor(
            [[29896, 29906, 29900, 29945], [13, 2, 29871, 28956]],
            dtype=torch.int64,
            device="cuda",
        ),
        torch.tensor(
            [
                [
                    29889,
                    29974,
                    29945,
                    29900,
                    29974,
                    29922,
                    29930,
                    29958,
                    29889,
                    29974,
                    29930,
                    29945,
                    29974,
                    29922,
                    29930,
                    29958,
                ],
                [
                    22550,
                    4136,
                    16492,
                    8439,
                    29871,
                    2,
                    3001,
                    13,
                    2,
                    13,
                    29906,
                    29946,
                    2,
                    13,
                    29871,
                    259,
                ],
            ],
            device="cuda",
        ),
        torch.tensor(
            [
                [
                    29946,
                    29945,
                    29953,
                    29906,
                    29896,
                    29945,
                    29900,
                    29906,
                    29896,
                    29945,
                    29906,
                    29953,
                    29896,
                    29945,
                    29906,
                    29946,
                ],
                [
                    29871,
                    2,
                    29901,
                    29889,
                    29871,
                    2,
                    395,
                    259,
                    29901,
                    29871,
                    2,
                    29889,
                    3001,
                    1234,
                    7146,
                    2186,
                ],
            ],
            device="cuda",
        ),
        torch.tensor(
            [
                [
                    29946,
                    29974,
                    29945,
                    29930,
                    29889,
                    29922,
                    29974,
                    29930,
                    29974,
                    29946,
                    29930,
                    29922,
                    29889,
                    29974,
                    29945,
                    29922,
                ],
                [
                    29941,
                    29906,
                    2,
                    29946,
                    29871,
                    450,
                    319,
                    14990,
                    29946,
                    29941,
                    2,
                    29906,
                    29871,
                    2,
                    3001,
                    13,
                ],
            ],
            device="cuda",
        ),
    ]
    parents_list = [
        torch.tensor(
            [[-1, 0, 1, 2, 3], [-1, 0, 1, 2, 3]], dtype=torch.int64, device="cuda"
        ),
        torch.tensor([[4, 8, 9, 10], [4, 5, 6, 7]], dtype=torch.int64, device="cuda"),
        torch.tensor(
            [[20, 24, 21, 28], [24, 28, 20, 21]], dtype=torch.int64, device="cuda"
        ),
        torch.tensor(
            [[36, 40, 41, 44], [36, 40, 44, 45]], dtype=torch.int64, device="cuda"
        ),
    ]
    seq_lens = torch.tensor([5, 10], dtype=torch.int64, device="cuda")
    topk = 4
    depth = 4
    num_draft_token = 8

    (
        tree_mask,
        position,
        retrive_index,
        retrive_next_token,
        retrive_next_sibling,
        draft_tokens,
    ) = build_tree_kernel_efficient(
        verified_id=verified_id,
        score_list=score_list,
        token_list=token_list,
        parents_list=parents_list,
        seq_lens=seq_lens,
        seq_lens_sum=torch.sum(seq_lens).item(),
        topk=topk,
        spec_steps=depth,
        num_verify_tokens=num_draft_token,
    )

    print("=========== build tree kernel efficient ==========")
    print(f"{tree_mask=}")
    print(f"{position=}")
    print(f"{retrive_index=}")
    print(f"{retrive_next_token=}")
    print(f"{retrive_next_sibling=}")
    print(f"{draft_tokens=}")
    assert position.tolist() == [5, 6, 6, 7, 7, 8, 8, 9, 10, 11, 12, 12, 12, 12, 13, 14]
    assert retrive_index.tolist() == [
        [0, 1, 2, 3, 4, 5, 6, 7],
        [8, 9, 10, 11, 12, 13, 14, 15],
    ]
    assert retrive_next_token.tolist() == [
        [1, 3, 4, 5, 6, 7, -1, -1],
        [1, 2, -1, 6, -1, -1, 7, -1],
    ]
    assert retrive_next_sibling.tolist() == [
        [-1, 2, -1, -1, -1, -1, -1, -1],
        [-1, -1, 3, 4, 5, -1, -1, -1],
    ]
    assert draft_tokens.tolist() == [
        29974,
        29896,
        29906,
        29889,
        29974,
        29946,
        29896,
        29946,
        13,
        13,
        22550,
        4136,
        16492,
        8439,
        29871,
        29941,
    ]


if __name__ == "__main__":
    test_build_tree_kernel_efficient()
