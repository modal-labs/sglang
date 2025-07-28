# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""An EAGLE speculative decoding worker with overlap thread."""

import dataclasses
import logging
import signal
import threading
from queue import Queue
from typing import List, Optional, Tuple

import psutil
import torch

from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.speculative.eagle_utils import EagleDraftInput
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.managers.schedule_batch import ModelWorkerBatch, Req, ScheduleBatch
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.eagle_worker import EAGLEWorker
from sglang.srt.utils import DynamicGradMode, get_compiler_backend
from sglang.utils import get_exception_traceback

logger = logging.getLogger(__name__)


class FutureSpecInfo:
    """
    Stores spec info for future batches.

    For overlapped scheduling, EagleWorkerClient returns to the scheduler a list of _pointers_ to
    the spec info that will eventually be populated when the current batch finishes running.

    The scheduler will filter this list of pointers to remove requests that are completed,
    then give the filtered list of pointers to EagleWorkerClient as part of the next batch.

    Before executing a batch, EagleWorkerClient will take this list of pointers and "dereference"
    them into actual spec info objects.

    The process is something like this:

    - [scheduler] get batch 1
    - [eagle]     enqueue batch 1  -> returns pointers to spec info 1
    - [eagle]     execute batch 0
    - [scheduler] process batch 0  -> receives list of next token IDs, updates running requests
    - [scheduler] get batch 2      -> uses result of process batch 0 to filter spec info 1 pointers, removing completed requests
    - [eagle]     enqueue batch 2  -> receives filtered spec info 1 pointers, returns pointers to spec info 2
    - [eagle]     execute batch 1  -> populates spec info 1
    - [scheduler] process batch 1
    - [scheduler] get batch 3
    - [eagle]     enqueue batch 3
    - [eagle]     execute batch 2  -> takes filtered spec info 1 pointers, reconstructs filtered spec info 1 tensors, populates spec info 2
    - [scheduler] process batch 2

    The scheduler is freely able to mutate and transform pointers that EagleWorkerClient returns.
    The actual spec info objects are owned by EagleWorkerClient and should not be accessed by the scheduler.
    """

    def __init__(self, max_items: int, topk: int, hidden_size: int, device: torch.device):
        self.topk_p: torch.Tensor = torch.empty((max_items, topk), device=device, dtype=torch.float32)
        self.topk_index: torch.Tensor = torch.empty((max_items, topk), device=device, dtype=torch.int64)
        self.hidden_states: torch.Tensor = torch.empty((max_items, hidden_size), device=device, dtype=torch.bfloat16)
        self.verified_id: torch.Tensor = torch.empty((max_items, ), device=device, dtype=torch.int32)

        self.topk: int = topk
        self.hidden_size: int = hidden_size
        self.device: torch.device = device
        self.max_items: int = max_items
        self.current_index: int = 0

    def get_pointers(self, num_items: int) -> torch.Tensor:
        """Returns a device tensor of shape (num_items, ) each containing a pointer to one spec info entry."""
        # TODO(nathan): This is a really sketchy to implement a circular buffer.
        if self.current_index + num_items > self.max_items:
            self.current_index = 0

        pointers = torch.arange(self.current_index, self.current_index + num_items, device=self.device, dtype=torch.int32)
        self.current_index += num_items
        return pointers

    def put_data(self, pointers: torch.Tensor, output_spec_info: EagleDraftInput):
        """Puts the data from output_spec_info into the spec info objects at the indices specified by pointers."""

        if pointers.shape[0] == 0:
            # Special case where all requests in the batch have finished. In this case, output_spec_info can contain None values.
            return

        assert output_spec_info.topk_p.shape == (pointers.shape[0], self.topk)
        assert output_spec_info.topk_index.shape == (pointers.shape[0], self.topk)
        assert output_spec_info.hidden_states.shape == (pointers.shape[0], self.hidden_size)
        assert output_spec_info.verified_id.shape == (pointers.shape[0], )

        assert output_spec_info.topk_p.dtype == self.topk_p.dtype
        assert output_spec_info.topk_index.dtype == self.topk_index.dtype
        assert output_spec_info.hidden_states.dtype == self.hidden_states.dtype
        assert output_spec_info.verified_id.dtype == self.verified_id.dtype

        self.topk_p[pointers] = output_spec_info.topk_p
        self.topk_index[pointers] = output_spec_info.topk_index
        self.hidden_states[pointers] = output_spec_info.hidden_states
        self.verified_id[pointers] = output_spec_info.verified_id
    
    def get_data(self, pointers: torch.Tensor) -> EagleDraftInput:
        """Returns the spec info objects at the indices specified by pointers."""
        assert pointers.shape == (pointers.shape[0], )
        assert pointers.dtype == torch.int32

        return EagleDraftInput(
            topk_p=self.topk_p[pointers],
            topk_index=self.topk_index[pointers],
            hidden_states=self.hidden_states[pointers],
            verified_id=self.verified_id[pointers],
        )

class EAGLEWorkerClient:
    """An EAGLE speculative decoding worker with overlap thread."""

    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        tp_rank: int,
        dp_rank: Optional[int],
        nccl_port: int,
        target_worker: TpModelWorker,
    ):
        # Load the model
        self.worker = EAGLEWorker(
            server_args=server_args,
            gpu_id=gpu_id,
            tp_rank=tp_rank,
            dp_rank=dp_rank,
            nccl_port=nccl_port,
            target_worker=target_worker,
        )
        self.device = self.worker.device
        self.gpu_id = gpu_id
        self.max_running_requests = self.worker.max_running_requests

        assert self.worker.topk is not None
        assert self.device is not None
        self.future_spec_infos: FutureSpecInfo = FutureSpecInfo(
            max_items=self.max_running_requests * 1000,  # TODO(nathan): Make this something more reasonable. Have to fix the circular buffer implementation of FutureSpecInfo first though.
            topk=self.worker.topk,
            hidden_size=self.worker.model_runner.model_config.hidden_size,
            device=torch.device(self.device),
        )
        
        # Launch threads
        self.input_queue = Queue()
        self.output_queue = Queue()
        self.forward_stream = torch.get_device_module(self.device).Stream()
        self.forward_thread = threading.Thread(
            target=self.forward_thread_func,
        )
        self.forward_thread.start()
        self.parent_process = psutil.Process().parent()
        self.scheduler_stream = torch.get_device_module(self.device).current_stream()
        if self.device == "cpu":
            self.scheduler_stream.synchronize = lambda: None  # No-op for CPU

    def forward_thread_func(self):
        try:
            with torch.get_device_module(self.device).stream(self.forward_stream):
                self.forward_thread_func_()
        except Exception:
            traceback = get_exception_traceback()
            logger.error(f"EAGLEWorkerClient hit an exception: {traceback}")
            self.parent_process.send_signal(signal.SIGQUIT)

    @DynamicGradMode()
    def forward_thread_func_(self):
        batch_pt = 0
        batch_lists = [None] * 2

        while True:
            batch, sync_event, output_spec_info_pointers = self.input_queue.get()
            if batch is None:
                break

            assert isinstance(batch, ModelWorkerBatch)
            sync_event.wait()

            # Keep a reference of batch by storing it into a list.
            # Otherwise, the tensor members of batch will be released
            # by pytorch and cause CUDA illegal memory access errors.
            batch_lists[batch_pt % 2] = batch
            batch_pt += 1

            # Create event
            copy_done = torch.get_device_module(self.device).Event()

            if batch.forward_mode == ForwardMode.DECODE:
                batch.spec_info = self.future_spec_infos.get_data(batch.spec_info.verified_id)
                assert batch.spec_info.verified_id.shape == batch.seq_lens.shape
            else:
                assert batch.forward_mode == ForwardMode.EXTEND
                assert batch.spec_info is None

            # Run forward
            (
                logits_output,
                next_token_ids,
                free_cache_loc_cpu,
                bid,
                can_run_cuda_graph,
                output_spec_info,
            ) = self.worker.forward_batch_speculative_generation(batch)

            self.future_spec_infos.put_data(output_spec_info_pointers, output_spec_info)

            # NOTE(Nathan): not super sure about the placement of this
            if batch.launch_done is not None:
                batch.launch_done.set()

            # Copy results to CPU if needed
            if logits_output is not None:
                if logits_output.next_token_logprobs is not None:
                    logits_output.next_token_logprobs = (
                        logits_output.next_token_logprobs.to("cpu", non_blocking=True)
                    )
                if logits_output.input_token_logprobs is not None:
                    logits_output.input_token_logprobs = (
                        logits_output.input_token_logprobs.to("cpu", non_blocking=True)
                    )
            if logits_output.hidden_states is not None:
                logits_output.hidden_states = logits_output.hidden_states.to(
                    "cpu", non_blocking=True
                )
            next_token_ids = next_token_ids.to("cpu", non_blocking=True)
            copy_done.record()

            self.output_queue.put(
                (copy_done, logits_output, next_token_ids, free_cache_loc_cpu, bid, can_run_cuda_graph)
            )

    def resolve_last_batch_result(self, launch_done: Optional[threading.Event] = None):
        """
        Resolve the last batch result and wait for the current batch to be launched.
        Used in overlap mode.
        """
        copy_done, logits_output, next_token_ids, free_cache_loc_cpu, bid, can_run_cuda_graph = (
            self.output_queue.get()
        )

        if launch_done is not None:
            launch_done.wait()
        copy_done.synchronize()

        if logits_output.next_token_logprobs is not None:
            logits_output.next_token_logprobs = (
                logits_output.next_token_logprobs.tolist()
            )
            if logits_output.input_token_logprobs is not None:
                logits_output.input_token_logprobs = tuple(
                    logits_output.input_token_logprobs.tolist()
                )
        next_token_ids = next_token_ids.tolist()
        return logits_output, next_token_ids, free_cache_loc_cpu, bid, can_run_cuda_graph

    def forward_batch_speculative_generation(
        self, model_worker_batch: ModelWorkerBatch
    ) -> Tuple[LogitsProcessorOutput, torch.Tensor, int, int, bool, EagleDraftInput]:
        # Create a new copy of sampling_info because it will be updated in-place by the scheduler for the next batch.
        sampling_info = model_worker_batch.sampling_info
        model_worker_batch.sampling_info = self.cur_sampling_info = dataclasses.replace(
            sampling_info,
            sampling_info_done=threading.Event(),
        )

        # TODO(nathan): Remove this restriction
        assert len(model_worker_batch.seq_lens) == 1, "only batch size 1 is supported for overlap scheduling"

        # Create sync event to coordinate streams
        sync_event = torch.get_device_module(self.device).Event()
        sync_event.record(self.scheduler_stream)

        if model_worker_batch.forward_mode == ForwardMode.DECODE:
            assert model_worker_batch.spec_info is not None
            assert isinstance(model_worker_batch.spec_info, EagleDraftInput)
            assert model_worker_batch.spec_info.verified_id is not None
        
        spec_info_pointers = self.future_spec_infos.get_pointers(len(model_worker_batch.seq_lens))

        # Push batch to queue
        self.input_queue.put((model_worker_batch, sync_event, spec_info_pointers))

        # For overlap scheduling, we return a list of pointers to the spec info we will eventually populate (once the current batch finishes running).
        # The scheduler will mutate this list of pointers as needed and give it back to us as part of the next batch's model_worker_batch.spec_info.
        output_spec_info_pointers = EagleDraftInput(
            verified_id=spec_info_pointers,
        )
        return None, None, -1, 0, False, output_spec_info_pointers

    def get_worker_info(self):
        return self.worker.get_worker_info()

    def get_tokens_per_layer_info(self):
        return self.worker.get_tokens_per_layer_info()

    @property
    def sliding_window_size(self) -> Optional[int]:
        return self.worker.sliding_window_size

    @property
    def is_hybrid(self) -> bool:
        return self.worker.is_hybrid

    def get_pad_input_ids_func(self):
        return self.worker.get_pad_input_ids_func()

    def get_tp_group(self):
        return self.worker.get_tp_group()

    def get_attention_tp_group(self):
        return self.worker.get_attention_tp_group()

    def get_attention_tp_cpu_group(self):
        return self.worker.get_attention_tp_cpu_group()

    def get_memory_pool(self):
        return self.worker.get_memory_pool()

    def get_kv_cache(self):
        return self.worker.model_runner.token_to_kv_pool

    def register_hicache_layer_transfer_counter(self, counter):
        self.worker.register_hicache_layer_transfer_counter(counter)

    def set_hicache_consumer(self, consumer_index):
        self.worker.set_hicache_consumer(consumer_index)

    def update_weights_from_disk(self, recv_req):
        return self.worker.update_weights_from_disk(recv_req)

    def init_weights_update_group(self, recv_req):
        return self.worker.init_weights_update_group(recv_req)

    def update_weights_from_distributed(self, recv_req):
        return self.worker.update_weights_from_distributed(recv_req)

    def update_weights_from_tensor(self, recv_req):
        return self.worker.update_weights_from_tensor(recv_req)

    def get_weights_by_name(self, recv_req):
        return self.worker.get_weights_by_name(recv_req)

    def load_lora_adapter(self, recv_req):
        return self.worker.load_lora_adapter(recv_req)

    def unload_lora_adapter(self, recv_req):
        return self.worker.unload_lora_adapter(recv_req)

    def __delete__(self):
        self.input_queue.put((None, None, None))
