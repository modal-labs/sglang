"""
Usage:
# Basic disaggregation benchmark
python3 bench_disagg.py --model-path Qwen/Qwen3-8B --specdec --bench-max-concurrency 8 --page-size 16

# With different concurrency levels
python3 bench_disagg.py --model-path Qwen/Qwen3-8B --bench-max-concurrency 1 4 8 16 --page-size 16
"""

import argparse
import asyncio
import json
import os
import subprocess
import time
from types import SimpleNamespace

import numpy as np
import requests

from sglang.bench_serving import DatasetRow, benchmark, set_global_args
from sglang.srt.server_args import ServerArgs
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    kill_process_tree,
    popen_launch_server,
)


OUTPUT_TOKEN_LEN = 32


def wait_for_port(process: subprocess.Popen, port: int):
    import socket

    while True:
        try:
            with socket.create_connection(("localhost", port), timeout=1):
                break
        except (ConnectionRefusedError, OSError):
            if process.poll() is not None:
                raise Exception(f"Process {process.pid} exited with code {process.returncode}")


class FakeTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False):
        return []


def send_one_batch(base_url, num_prompts, batch_size, profile=False):
    prompts = [
        "Human: Give me a fully functional FastAPI server. Show the full, long python code without stop.\n\nAssistant:",
        "Human: Imagine you are an experienced Ethereum developer tasked with creating a smart contract for a blockchain messenger. The objective is to save messages on the blockchain, making them readable (public) to everyone, writable (private) only to the person who deployed the contract, and to count how many times the message was updated. Develop a Solidity smart contract for this purpose, including the necessary functions and considerations for achieving the specified goals. Please provide the code and any relevant explanations to ensure a clear understanding of the implementation.\n\nAssistant:",
        "Human: Write a travel blog post to Hawaii.\n\nAssistant:",
        "Human: I want you to act as an English translator, spelling corrector and improver. I will speak to you in any language and you will detect the language, translate it and answer in the corrected and improved version of my text, in English. I want you to replace my simplified A0-level words and sentences with more beautiful and elegant, upper level English words and sentences. Keep the meaning same, but make them more literary. My first sentence is 'istanbulu cok seviyom burada olmak cok guzel'. Answer in more than 5000 words.\n\nAssistant:",
        "Human: I want you to act as a storyteller. You will come up with entertaining stories that are engaging, imaginative and captivating for the audience. It can be fairy tales, educational stories or any other type of stories which has the potential to capture people's attention and imagination. Depending on the target audience, you may choose specific themes or topics for your storytelling session e.g., if it's children then you can talk about animals; If it's adults then history-based tales might engage them better etc. Answer in more than 5000 words. My first request is 'I need an interesting story on perseverance.'\n\nAssistant:",
        "Human: Solve x^2 = -1. Think step-by-step. Give me a long detailed explanation. \n\nAssistant:",
        "Human: Tell me about the president of the USA in wikipedia style.\n\nAssistant:",
        "Human: Hello? Who are you? Write code, math, and poem to explanin yourself.\n\nAssistant:",
    ]

    padded_prompts = (prompts * ((num_prompts + len(prompts) - 1) // len(prompts)))[
        :num_prompts
    ]

    input_requests = [DatasetRow(p, 0, OUTPUT_TOKEN_LEN) for p in padded_prompts]

    args = SimpleNamespace(
        disable_ignore_eos=False,
        disable_stream=False,
        return_logprob=False,
        backend="sglang",
        dataset_name="custom",
        num_prompts=None,
        sharegpt_output_len=None,
        random_input_len=None,
        random_output_len=None,
        random_range_ratio=None,
        output_file=None,
        warmup_requests=1,
        output_details=False,
    )
    set_global_args(args)
    tokenizer = FakeTokenizer()

    # Run benchmark
    asyncio.run(
        benchmark(
            backend="sglang",
            api_url=f"{base_url}/generate",
            base_url=base_url,
            model_id="default",
            tokenizer=tokenizer,
            input_requests=input_requests,
            request_rate=float("inf"),
            max_concurrency=batch_size,
            disable_tqdm=False,
            lora_names=None,
            extra_request_body={},
            profile=profile,
        )
    )


def main(args, server_args):
    base_url = "http://127.0.0.1:8000"
    prefill_url = "http://127.0.0.1:30000"
    decode_url = "http://127.0.0.1:30001"

    bench_max_concurrency = args.bench_max_concurrency
    print(f"Start {bench_max_concurrency=}, {args.specdec=}, {args.page_size=}")

    # Create base command for disaggregation
    base_cmd = [
        "--model-path", args.model_path,
        "--disaggregation-transfer-backend", args.disaggregation_transfer_backend,
        "--dtype", "float16",
        "--cuda-graph-max-bs", str(bench_max_concurrency),
        "--mem-fraction", "0.7",
        "--page-size", str(args.page_size),
    ]

    # Add speculative decoding config if enabled
    specdec_config = []
    if args.specdec:
        specdec_config = [
            "--speculative-algorithm", "EAGLE3",
            "--speculative-draft-model-path", "Tengyunw/qwen3_8b_eagle3",
            "--speculative-num-steps", "8",
            "--speculative-eagle-topk", "4",
            "--speculative-num-draft-tokens", "128",
        ]

# Launch prefill server
    prefill_cmd = base_cmd + ["--disaggregation-mode", "prefill"]
    prefill_process = popen_launch_server(
        args.model_path,
        prefill_url,
        timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
        other_args=prefill_cmd + ["--base-gpu-id", "0", "--port", "30000", "--disaggregation-bootstrap-port", "8200"],
    )

    # Launch decode server
    decode_cmd = base_cmd + specdec_config + ["--disaggregation-mode", "decode"]
    decode_process = popen_launch_server(
        args.model_path,
        decode_url,
        timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
        other_args=decode_cmd + ["--base-gpu-id", "1", "--port", "30001"],
    )

    # Wait for both servers to be ready
    wait_for_port(prefill_process, 30000)
    wait_for_port(decode_process, 30001)

    # Launch load balancer
    lb_cmd = [
        "python", "-m", "sglang.srt.disaggregation.mini_lb",
        "--host", "0.0.0.0",
        "--port", "8000",
        "--prefill", prefill_url,
        "--prefill-bootstrap-ports", "8200",
        "--decode", decode_url,
    ]
    print("Starting load balancer:", " ".join(lb_cmd))
    lb_process = subprocess.Popen(lb_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    wait_for_port(lb_process, 8000)

    try:
        # Warmup
        send_one_batch(base_url, bench_max_concurrency, bench_max_concurrency)

        # Benchmark
        send_one_batch(
            base_url, max(args.num_prompts, bench_max_concurrency), bench_max_concurrency, profile=args.profile
        )
    finally:
        # Clean up processes
        kill_process_tree(lb_process.pid)
        kill_process_tree(prefill_process.pid)
        kill_process_tree(decode_process.pid)

    # Wait for the servers to shutdown
    time.sleep(5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    parser.add_argument(
        "--bench-max-concurrency",
        type=int,
        default=1,
        help="Maximum concurrency levels to benchmark",
    )
    parser.add_argument(
        "--specdec",
        action="store_true",
        help="Enable speculative decoding",
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=16,
        help="Number of prompts for benchmarking",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable profiling",
    )
    args = parser.parse_args()
    server_args: ServerArgs = ServerArgs.from_cli_args(args)

    if args.profile:
        args.num_prompts = args.bench_max_concurrency

    main(args, server_args)
