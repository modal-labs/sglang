from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from sglang.srt.environ import envs
from sglang.srt.managers.overlap_utils import FutureIndices
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.utils import get_alloc_len_per_decode
from sglang.srt.mem_cache.common import (
    alloc_paged_token_slots_extend,
    alloc_token_slots,
    get_last_loc,
)
from sglang.srt.speculative.spec_info import SpecInput, SpecInputType
from sglang.srt.speculative.spec_utils import assign_req_to_token_pool_func

_OVERLAP_PLAN_STREAMS: dict[str, torch.cuda.Stream] = {}


def _get_overlap_plan_stream(
    device: torch.device | str,
) -> tuple[Optional[torch.cuda.Stream], contextlib.AbstractContextManager]:
    if not envs.SGLANG_ENABLE_OVERLAP_PLAN_STREAM.get():
        return None, contextlib.nullcontext()

    device_str = str(device)
    stream = _OVERLAP_PLAN_STREAMS.get(device_str)
    if stream is None:
        stream = torch.get_device_module(device_str).Stream()
        _OVERLAP_PLAN_STREAMS[device_str] = stream
    return stream, torch.get_device_module(device_str).stream(stream)


@dataclass
class DFlashDraftInputV2(SpecInput):
    """Draft-side state carried across overlap iterations.

    The overlap `FutureMap` currently expects the same field layout used by Eagle
    v2 draft inputs, so DFLASH mirrors that envelope here. `draft_seq_lens`
    tracks the logical resident prefix lengths in the draft-local cache, with
    `draft_seq_lens_cpu` as the CPU mirror used by attention backend planning.
    """

    topk_p: torch.Tensor
    topk_index: torch.Tensor
    verified_id: torch.Tensor
    new_seq_lens: torch.Tensor
    draft_seq_lens: torch.Tensor
    draft_seq_lens_cpu: torch.Tensor
    hidden_states: torch.Tensor
    verify_done: Optional[torch.cuda.Event] = None

    # Filled by the scheduler after dispatch into the overlap future map.
    future_indices: Optional[FutureIndices] = None

    def __post_init__(self):
        super().__init__(spec_input_type=SpecInputType.DFLASH_DRAFT)

    def get_spec_adjust_token_coefficient(self) -> Tuple[int, int]:
        return (1, 1)

    @classmethod
    def create_idle_input(cls, device: torch.device) -> "DFlashDraftInputV2":
        return cls(
            topk_p=torch.empty((0, 1), device=device, dtype=torch.float32),
            topk_index=torch.empty((0, 1), device=device, dtype=torch.int64),
            verified_id=torch.empty((0,), device=device, dtype=torch.int32),
            new_seq_lens=torch.empty((0,), device=device, dtype=torch.int32),
            draft_seq_lens=torch.empty((0,), device=device, dtype=torch.int32),
            draft_seq_lens_cpu=torch.empty((0,), device="cpu", dtype=torch.int32),
            hidden_states=torch.empty((0, 1), device=device, dtype=torch.float16),
            verify_done=None,
        )

    def prepare_for_decode(self, batch: ScheduleBatch):
        """Reserve future target-KV headroom for overlap scheduling."""
        batch.maybe_evict_swa()

        plan_stream, plan_stream_ctx = _get_overlap_plan_stream(batch.device)
        if plan_stream is None:
            batch.maybe_wait_verify_done()

        bs = batch.batch_size()
        if bs == 0:
            return

        page_size = batch.token_to_kv_pool_allocator.page_size
        alloc_len_per_decode = int(get_alloc_len_per_decode())

        cur_kv_lens_cpu: list[int] = []
        nxt_kv_lens_cpu: list[int] = []
        num_needed_tokens = 0

        for req in batch.reqs:
            # Keep one additional decode step of headroom beyond the next verify block.
            needed = (
                req.kv_committed_len + 2 * alloc_len_per_decode - req.kv_allocated_len
            )
            if needed < 0:
                needed = 0
            cur_kv_lens_cpu.append(req.kv_allocated_len)
            nxt_kv_lens_cpu.append(req.kv_allocated_len + needed)
            num_needed_tokens += needed
            req.kv_allocated_len += needed
            req.decode_batch_idx += 1

        if num_needed_tokens > 0:
            cur_kv_lens_cpu_t = torch.tensor(
                cur_kv_lens_cpu, dtype=torch.int32, device="cpu"
            )
            nxt_kv_lens_cpu_t = torch.tensor(
                nxt_kv_lens_cpu, dtype=torch.int32, device="cpu"
            )

            with plan_stream_ctx:
                if page_size == 1:
                    out_cache_loc = alloc_token_slots(
                        batch.tree_cache, num_needed_tokens
                    )
                else:
                    cur_kv_lens = cur_kv_lens_cpu_t.to(device=batch.device)
                    nxt_kv_lens = nxt_kv_lens_cpu_t.to(device=batch.device)
                    last_loc = get_last_loc(
                        batch.req_to_token_pool.req_to_token,
                        batch.req_pool_indices,
                        cur_kv_lens,
                    )
                    out_cache_loc = alloc_paged_token_slots_extend(
                        batch.tree_cache,
                        cur_kv_lens,
                        cur_kv_lens_cpu_t,
                        nxt_kv_lens,
                        nxt_kv_lens_cpu_t,
                        last_loc,
                        num_needed_tokens,
                    )

                if plan_stream is not None and self.verify_done is not None:
                    plan_stream.wait_event(self.verify_done)

                assign_req_to_token_pool_func(
                    batch.req_pool_indices,
                    batch.req_to_token_pool.req_to_token,
                    cur_kv_lens_cpu_t.to(device=batch.device),
                    nxt_kv_lens_cpu_t.to(device=batch.device),
                    out_cache_loc,
                    bs,
                )

            if plan_stream is not None:
                torch.get_device_module(batch.device).current_stream().wait_stream(
                    plan_stream
                )

        # In overlap mode, CPU request state can lag by one iteration. Use the
        # allocated KV lengths as a safe upper bound for backend planning buffers.
        batch.seq_lens_cpu = torch.tensor(
            nxt_kv_lens_cpu, dtype=torch.int64, device="cpu"
        )
        batch.seq_lens_sum = int(sum(nxt_kv_lens_cpu))

    def filter_batch(self, new_indices: torch.Tensor, has_been_filtered: bool = True):
        if self.future_indices is not None:
            self.future_indices.indices = self.future_indices.indices[new_indices]
            return

        self.topk_p = self.topk_p[new_indices]
        self.topk_index = self.topk_index[new_indices]
        self.verified_id = self.verified_id[new_indices]
        self.new_seq_lens = self.new_seq_lens[new_indices]
        self.draft_seq_lens = self.draft_seq_lens[new_indices]
        self.draft_seq_lens_cpu = self.draft_seq_lens_cpu[new_indices.cpu()]
        self.hidden_states = self.hidden_states[new_indices]

    def merge_batch(self, spec_info: "DFlashDraftInputV2"):
        if self.future_indices is not None:
            assert spec_info.future_indices is not None
            self.future_indices = FutureIndices(
                indices=torch.cat(
                    [self.future_indices.indices, spec_info.future_indices.indices]
                )
            )
            return

        self.topk_p = torch.cat([self.topk_p, spec_info.topk_p], dim=0)
        self.topk_index = torch.cat([self.topk_index, spec_info.topk_index], dim=0)
        self.verified_id = torch.cat([self.verified_id, spec_info.verified_id], dim=0)
        self.new_seq_lens = torch.cat(
            [self.new_seq_lens, spec_info.new_seq_lens], dim=0
        )
        self.draft_seq_lens = torch.cat(
            [self.draft_seq_lens, spec_info.draft_seq_lens], dim=0
        )
        self.draft_seq_lens_cpu = torch.cat(
            [self.draft_seq_lens_cpu, spec_info.draft_seq_lens_cpu], dim=0
        )
        self.hidden_states = torch.cat(
            [self.hidden_states, spec_info.hidden_states], dim=0
        )
