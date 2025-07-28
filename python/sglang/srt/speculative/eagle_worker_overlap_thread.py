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

from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.eagle_worker import EAGLEWorker
from sglang.srt.utils import DynamicGradMode, get_compiler_backend
from sglang.utils import get_exception_traceback

logger = logging.getLogger(__name__)


@torch.compile(dynamic=True, backend=get_compiler_backend())
def resolve_future_token_ids_eagle(batch, future_token_ids_map):
    pass


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

        # Init future mappings
        self.future_token_ids_ct = 0
        self.future_token_ids_limit = self.max_running_requests * (self.worker.speculative_num_steps + 1) * 3
        self.future_token_ids_map = torch.empty(
            (self.max_running_requests * (self.worker.speculative_num_steps + 1) * 5,), dtype=torch.int64, device=self.device
        )
        # Track accepted token counts per request
        self.future_acceptance_counts = torch.zeros(
            (self.max_running_requests * 5,), dtype=torch.int64, device=self.device
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
            batch, future_token_ids_ct, sync_event = self.input_queue.get()
            if batch is None:
                break

            sync_event.wait()

            # Keep a reference of batch by storing it into a list.
            # Otherwise, the tensor members of batch will be released
            # by pytorch and cause CUDA illegal memory access errors.
            batch_lists[batch_pt % 2] = batch
            batch_pt += 1

            # Create event
            copy_done = torch.get_device_module(self.device).Event()

            # Resolve future tokens in the input
            resolve_future_token_ids_eagle(batch, self.future_token_ids_map)

            # Run forward
            (
                logits_output,
                next_token_ids,
                free_cache_loc_cpu,
                bid,
                can_run_cuda_graph,
            ) = self.worker.forward_batch_speculative_generation(batch)

            # Update the future token ids map
            self.future_token_ids_map[
                future_token_ids_ct + 1 : future_token_ids_ct + len(next_token_ids) + 1
            ] = next_token_ids

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
        self, batch: ScheduleBatch
    ) -> Tuple[None, torch.Tensor, Optional[torch.Tensor], int, bool]:
        # Create a new copy of sampling_info because it will be updated in-place by the scheduler for the next batch.
        sampling_info = batch.sampling_info
        batch.sampling_info = self.cur_sampling_info = dataclasses.replace(
            sampling_info,
            sampling_info_done=threading.Event(),
        )

        # Create sync event to coordinate streams
        sync_event = torch.get_device_module(self.device).Event()
        sync_event.record(self.scheduler_stream)

        # Push batch to queue
        self.input_queue.put((batch, self.future_token_ids_ct, sync_event))
        print(f'added {batch.forward_mode} to queue')

        # Allocate future token placeholders
        bs = batch.batch_size()
        max_spec_tokens = bs * (self.worker.speculative_num_steps + 1)
        future_next_token_ids = torch.arange(
            -(self.future_token_ids_ct + 1),
            -(self.future_token_ids_ct + 1 + max_spec_tokens),
            -1,
            dtype=torch.int64,
            device=self.device,
        )
        self.future_token_ids_ct = (
            self.future_token_ids_ct + max_spec_tokens
        ) % self.future_token_ids_limit

        return None, future_next_token_ids, None, -1, False

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
