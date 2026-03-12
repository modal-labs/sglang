from __future__ import annotations

from typing import Optional, Union

import torch

from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.managers.schedule_batch import ModelWorkerBatch, ScheduleBatch
from sglang.srt.managers.scheduler import GenerationBatchResult
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
    compute_position,
)
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.dflash_info import (
    DFlashVerifyInput,
    build_dflash_verify_custom_mask,
    compute_dflash_verify_accept_len_and_bonus_from_logits,
)
from sglang.srt.speculative.dflash_info_v2 import DFlashDraftInputV2
from sglang.srt.speculative.dflash_utils import resolve_dflash_verify_mask_policy
from sglang.srt.speculative.dflash_worker import DFlashWorker
from sglang.srt.speculative.eagle_info_v2 import assign_extend_cache_locs_func
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.speculative.spec_utils import assign_req_to_token_pool_func


class DFlashWorkerV2(DFlashWorker):
    """DFLASH speculative decoding worker (spec-v2 overlap scheduling)."""

    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        tp_rank: int,
        dp_rank: Optional[int],
        moe_ep_rank: int,
        attn_cp_rank: int,
        moe_dp_rank: int,
        nccl_port: int,
        target_worker: TpModelWorker,
    ):
        super().__init__(
            server_args=server_args,
            gpu_id=gpu_id,
            tp_rank=tp_rank,
            dp_rank=dp_rank,
            moe_ep_rank=moe_ep_rank,
            attn_cp_rank=attn_cp_rank,
            moe_dp_rank=moe_dp_rank,
            nccl_port=nccl_port,
            target_worker=target_worker,
        )

    def _make_next_draft_input(
        self,
        *,
        verified_id: torch.Tensor,
        new_seq_lens: torch.Tensor,
        verify_done: Optional[torch.cuda.Event],
    ) -> DFlashDraftInputV2:
        bs = int(new_seq_lens.numel())
        device = verified_id.device
        return DFlashDraftInputV2(
            topk_p=torch.ones((bs, 1), device=device, dtype=torch.float32),
            topk_index=torch.zeros((bs, 1), device=device, dtype=torch.int64),
            verified_id=verified_id.to(dtype=torch.int32),
            new_seq_lens=new_seq_lens.to(dtype=torch.int32),
            hidden_states=torch.empty((bs, 1), device=device, dtype=torch.float16),
            verify_done=verify_done,
        )

    def _should_keep_hidden_states_for_output(self, batch: ModelWorkerBatch) -> bool:
        reqs = getattr(batch, "reqs", None)
        return bool(reqs and any(req.return_hidden_states for req in reqs))

    def _forward_overlap_prefill(
        self,
        model_worker_batch: ModelWorkerBatch,
        **kwargs,
    ) -> GenerationBatchResult:
        model_worker_batch.capture_hidden_mode = CaptureHiddenMode.FULL
        batch_output = self.target_worker.forward_batch_generation(
            model_worker_batch, **kwargs
        )
        logits_output = batch_output.logits_output
        next_token_ids = batch_output.next_token_ids

        if logits_output.hidden_states is None:
            raise RuntimeError(
                "DFLASH requires target aux hidden capture for prefill, but got None. "
                "Make sure the target model has DFlash layers-to-capture configured."
            )
        if (
            model_worker_batch.extend_seq_lens is None
            or model_worker_batch.extend_prefix_lens is None
        ):
            raise RuntimeError(
                "DFLASH expected extend_seq_lens / extend_prefix_lens to be populated "
                "in overlap prefill mode, but got None."
            )
        if model_worker_batch.out_cache_loc is None:
            raise RuntimeError(
                "DFLASH overlap prefill expected out_cache_loc, but got None."
            )

        extend_seq_lens = torch.as_tensor(
            model_worker_batch.extend_seq_lens, dtype=torch.int32, device=self.device
        )
        extend_prefix_lens = torch.as_tensor(
            model_worker_batch.extend_prefix_lens, dtype=torch.int32, device=self.device
        )
        positions, _ = compute_position(
            self.model_runner.server_args.attention_backend,
            extend_prefix_lens,
            extend_seq_lens,
            int(extend_seq_lens.sum().item()),
        )
        self._append_target_hidden_to_draft_kv_by_loc(
            target_hidden=logits_output.hidden_states,
            cache_loc=model_worker_batch.out_cache_loc,
            positions=positions,
        )
        if self.use_compact_draft_cache:
            self._rebuild_compact_draft_cache_view(
                batch=model_worker_batch,
                target_seq_lens=model_worker_batch.seq_lens,
            )

        verify_done = torch.get_device_module(self.device).Event()
        verify_done.record()

        if not self._should_keep_hidden_states_for_output(model_worker_batch):
            logits_output.hidden_states = None

        batch_output.next_draft_input = self._make_next_draft_input(
            verified_id=next_token_ids,
            new_seq_lens=model_worker_batch.seq_lens,
            verify_done=verify_done,
        )
        return batch_output

    def _forward_overlap_decode(
        self,
        model_worker_batch: ModelWorkerBatch,
        **kwargs,
    ) -> GenerationBatchResult:
        if model_worker_batch.spec_info is None:
            model_worker_batch.spec_info = DFlashDraftInputV2.create_idle_input(
                device=self.device
            )

        draft_input = model_worker_batch.spec_info
        if not isinstance(draft_input, DFlashDraftInputV2):
            raise RuntimeError(
                "DFLASH overlap scheduling expected DFlashDraftInputV2 state on the "
                "running batch."
            )

        if model_worker_batch.forward_mode.is_idle():
            verify_done = torch.get_device_module(self.device).Event()
            verify_done.record()
            empty_ids = torch.empty((0,), dtype=torch.int64, device=self.device)
            empty_lens = torch.empty((0,), dtype=torch.int32, device=self.device)
            return GenerationBatchResult(
                logits_output=LogitsProcessorOutput(next_token_logits=None),
                next_token_ids=empty_ids,
                accept_lens=empty_lens,
                next_draft_input=self._make_next_draft_input(
                    verified_id=empty_ids,
                    new_seq_lens=empty_lens,
                    verify_done=verify_done,
                ),
                can_run_cuda_graph=False,
            )

        model_worker_batch.seq_lens.record_stream(
            torch.get_device_module(self.device).current_stream()
        )

        bs = int(model_worker_batch.seq_lens.shape[0])
        target_prefix_lens = model_worker_batch.seq_lens

        if self.use_compact_draft_cache:
            block_cache_loc = self._gather_target_future_block_cache_locs(
                req_to_token=self.req_to_token_pool.req_to_token,
                req_pool_indices=model_worker_batch.req_pool_indices,
                start_offset=target_prefix_lens,
            )
            draft_prefix_lens = self._rebuild_compact_draft_cache_view(
                batch=model_worker_batch,
                target_seq_lens=target_prefix_lens,
            )
            block_end = draft_prefix_lens + int(self.block_size)
            assign_req_to_token_pool_func(
                model_worker_batch.req_pool_indices,
                self.draft_model_runner.req_to_token_pool.req_to_token,
                draft_prefix_lens,
                block_end,
                block_cache_loc,
                bs,
            )
            draft_seq_lens_sum = int(draft_prefix_lens.sum().item())
        else:
            draft_prefix_lens = target_prefix_lens.to(dtype=torch.int32)
            block_cache_loc = assign_extend_cache_locs_func(
                req_pool_indices=model_worker_batch.req_pool_indices,
                req_to_token=self.draft_model_runner.req_to_token_pool.req_to_token,
                start_offset=target_prefix_lens,
                end_offset=target_prefix_lens + int(self.block_size),
                batch_size=bs,
                draft_token_num=int(self.block_size),
                device=self.device,
            )
            draft_seq_lens_sum = int(model_worker_batch.seq_lens_sum)

        target_model = self.target_worker.model_runner.model
        embed_module = target_model.get_input_embeddings()
        lm_head = getattr(target_model, "lm_head", None)
        if (
            lm_head is None
            or not hasattr(lm_head, "weight")
            or not hasattr(lm_head, "shard_indices")
        ):
            raise RuntimeError(
                "DFLASH requires the target model to expose a vocab-parallel `lm_head` "
                "with `weight` and `shard_indices` attributes."
            )

        self._ensure_draft_block_buffers(bs)
        assert self._draft_block_ids_buf is not None
        assert self._draft_block_positions_buf is not None
        assert self._draft_block_tokens_buf is not None
        assert self._draft_seq_lens_cpu_buf is not None

        block_ids = self._draft_block_ids_buf[:bs]
        block_ids.fill_(int(self._mask_token_id))
        block_ids[:, 0].copy_(draft_input.verified_id.to(torch.long))

        noise_embedding = embed_module(block_ids)
        input_embeds = noise_embedding.view(-1, noise_embedding.shape[-1])

        positions_2d = self._draft_block_positions_buf[:bs]
        torch.add(
            target_prefix_lens.to(torch.int64).unsqueeze(1),
            self._block_pos_offsets,
            out=positions_2d,
        )
        positions = positions_2d.reshape(-1)

        draft_seq_lens_cpu = self._draft_seq_lens_cpu_buf[:bs]
        if (
            not self.use_compact_draft_cache
            and model_worker_batch.seq_lens_cpu is not None
        ):
            if model_worker_batch.seq_lens_cpu.dtype == torch.int32:
                draft_seq_lens_cpu.copy_(model_worker_batch.seq_lens_cpu)
            else:
                draft_seq_lens_cpu.copy_(
                    model_worker_batch.seq_lens_cpu.to(dtype=torch.int32)
                )
        else:
            draft_seq_lens_cpu.copy_(
                draft_prefix_lens.to(device="cpu", dtype=torch.int32)
            )

        draft_spec_info = self._draft_block_spec_info
        draft_spec_info.custom_mask = None
        draft_forward_batch = ForwardBatch(
            forward_mode=ForwardMode.TARGET_VERIFY,
            batch_size=bs,
            input_ids=block_ids.reshape(-1),
            req_pool_indices=model_worker_batch.req_pool_indices,
            seq_lens=draft_prefix_lens,
            out_cache_loc=block_cache_loc,
            seq_lens_sum=draft_seq_lens_sum,
            seq_lens_cpu=draft_seq_lens_cpu,
            positions=positions,
            req_to_token_pool=self.draft_model_runner.req_to_token_pool,
            token_to_kv_pool=self.draft_model_runner.token_to_kv_pool,
            attn_backend=self.draft_model_runner.attn_backend,
            input_embeds=input_embeds,
            spec_algorithm=SpeculativeAlgorithm.DFLASH,
            spec_info=draft_spec_info,
            capture_hidden_mode=CaptureHiddenMode.NULL,
        )

        with torch.inference_mode():
            draft_hidden = self.draft_model_runner.forward(
                draft_forward_batch
            ).logits_output

        draft_hidden = draft_hidden.view(bs, self.block_size, -1)
        draft_next = self._greedy_sample_from_vocab_parallel_head(
            hidden_states=draft_hidden[:, 1:, :].reshape(-1, draft_hidden.shape[-1]),
            lm_head=lm_head,
        ).view(bs, self.block_size - 1)
        draft_tokens = self._draft_block_tokens_buf[:bs]
        draft_tokens[:, 0].copy_(block_ids[:, 0])
        draft_tokens[:, 1:].copy_(draft_next)

        verify_input = DFlashVerifyInput(
            draft_token=draft_tokens.reshape(-1),
            positions=positions,
            draft_token_num=self.block_size,
            capture_hidden_mode=CaptureHiddenMode.FULL,
        )
        _, build_custom_mask = resolve_dflash_verify_mask_policy(
            self.model_runner.attn_backend
        )
        if build_custom_mask:
            target_prefix_lens_cpu = target_prefix_lens.to("cpu", dtype=torch.int32)
            model_worker_batch.seq_lens_cpu = target_prefix_lens_cpu
            model_worker_batch.seq_lens_sum = int(target_prefix_lens.sum().item())
            verify_input.custom_mask = build_dflash_verify_custom_mask(
                seq_lens_cpu=target_prefix_lens_cpu,
                draft_token_num=int(self.block_size),
                device=self.device,
            )

        model_worker_batch.forward_mode = ForwardMode.TARGET_VERIFY
        model_worker_batch.input_ids = verify_input.draft_token
        model_worker_batch.out_cache_loc = block_cache_loc
        model_worker_batch.spec_info = verify_input
        model_worker_batch.capture_hidden_mode = CaptureHiddenMode.FULL

        need_mamba_verify_commit = hasattr(
            self.target_worker.model_runner.attn_backend,
            "update_mamba_state_after_mtp_verify",
        )
        seq_lens_pre_verify = (
            target_prefix_lens.clone() if need_mamba_verify_commit else None
        )

        target_out = self.target_worker.forward_batch_generation(
            model_worker_batch, is_verify=True, **kwargs
        )
        logits_output = target_out.logits_output
        can_run_cuda_graph = target_out.can_run_cuda_graph

        accept_len, bonus = compute_dflash_verify_accept_len_and_bonus_from_logits(
            candidates=draft_tokens,
            logits_output=logits_output,
            sampling_info=model_worker_batch.sampling_info,
        )
        commit_lens = accept_len.to(torch.int32) + 1

        out_tokens = torch.empty(
            (bs, int(self.block_size)), dtype=torch.int64, device=self.device
        )
        if int(self.block_size) > 1:
            out_tokens[:, : int(self.block_size) - 1].copy_(draft_tokens[:, 1:])
        out_tokens[:, int(self.block_size) - 1].fill_(0)
        out_tokens.scatter_(1, accept_len.to(torch.int64)[:, None], bonus[:, None])

        hidden = logits_output.hidden_states
        if hidden is None:
            raise RuntimeError(
                "DFLASH verify requires target hidden states, but got None."
            )
        offsets = self._block_pos_offsets
        mask2d = offsets[None, :] < commit_lens.to(torch.int64)[:, None]
        hidden = hidden.view(bs, self.block_size, -1)
        loc2d = block_cache_loc.view(bs, self.block_size)
        loc2d = torch.where(mask2d, loc2d, loc2d.new_zeros(()))
        self._append_target_hidden_to_draft_kv_by_loc(
            target_hidden=hidden.reshape(-1, hidden.shape[-1]),
            cache_loc=loc2d.reshape(-1),
            positions=positions,
            mask_valid=mask2d.reshape(-1),
        )

        new_seq_lens = target_prefix_lens + commit_lens.to(target_prefix_lens.dtype)
        if need_mamba_verify_commit:
            assert seq_lens_pre_verify is not None
            self._update_target_mamba_state_after_verify(
                batch=model_worker_batch,
                seq_lens_pre_verify=seq_lens_pre_verify,
                commit_lens=commit_lens,
            )

        if self.use_compact_draft_cache:
            self._rebuild_compact_draft_cache_view(
                batch=model_worker_batch,
                target_seq_lens=new_seq_lens,
            )

        verify_done = torch.get_device_module(self.device).Event()
        verify_done.record()

        # Match spec-v1 DFLASH decode semantics: decode hidden states are not surfaced
        # to the scheduler output path.
        logits_output.hidden_states = None

        next_draft_input = self._make_next_draft_input(
            verified_id=bonus,
            new_seq_lens=new_seq_lens,
            verify_done=verify_done,
        )
        return GenerationBatchResult(
            logits_output=logits_output,
            next_token_ids=out_tokens.reshape(-1),
            accept_lens=commit_lens,
            can_run_cuda_graph=can_run_cuda_graph,
            next_draft_input=next_draft_input,
        )

    def forward_batch_generation(
        self,
        batch: Union[ScheduleBatch, ModelWorkerBatch],
        **kwargs,
    ) -> GenerationBatchResult:
        if isinstance(batch, ScheduleBatch):
            return super().forward_batch_generation(batch, **kwargs)

        if getattr(batch, "return_logprob", False):
            raise RuntimeError(
                "Invariant broken: DFLASH batch requested return_logprob, but scheduler "
                "should have rejected this request."
            )

        if batch.forward_mode.is_extend() or batch.is_extend_in_batch:
            return self._forward_overlap_prefill(batch, **kwargs)
        return self._forward_overlap_decode(batch, **kwargs)
