"""DFLASH vs baseline request sweep.

This is a *benchmark script* (not a CI test): it can take a long time because it
launches servers for multiple (attention_backend, tp_size) configs and runs a
request workload for each (concurrency, num_questions) setting.

By default it benchmarks GSM8K. It can also build synthetic query-generator /
query-expander / trace-replay request corpora using the request construction
from `~/latest_parallel_bench_modal_openloop.py`.

Example usage:
  ./venv/bin/python benchmark/dflash/bench_dflash_gsm8k_sweep.py
  ./venv/bin/python benchmark/dflash/bench_dflash_gsm8k_sweep.py --skip-baseline --concurrencies 32 --tp-sizes 8
"""

from __future__ import annotations

import argparse
import ast
import copy
import json
import os
import random
import re
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import requests
import torch
from transformers import AutoTokenizer

from sglang.srt.utils import get_device_sm, kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    find_available_port,
    popen_launch_server,
)
from sglang.utils import convert_json_schema_to_str, download_and_cache_file, read_jsonl

INVALID = -9999999
DEFAULT_SEED = 42
DEFAULT_EXAMPLES_PATH = Path.home() / "latest_parallel_examples.json"

MAX_TOKENS_PROFILES = {
    "legacy_generator": [
        {"weight": 0.50, "min": 50, "max": 50},
        {"weight": 0.40, "min": 51, "max": 99},
        {"weight": 0.05, "min": 91, "max": 115},
        {"weight": 0.05, "min": 116, "max": 150},
    ],
    "customer_generator_v1": [
        {"weight": 0.50, "min": 50, "max": 50},
        {"weight": 0.40, "min": 51, "max": 100},
        {"weight": 0.10, "min": 101, "max": 150},
    ],
}


def _parse_int_csv(value: str) -> list[int]:
    return [int(x) for x in value.split(",") if x.strip()]


def _filter_attention_backends(backends: list[str], *, device_sm: int) -> list[str]:
    if not (80 <= device_sm <= 90):
        backends = [b for b in backends if b != "fa3"]
    if device_sm < 100:
        backends = [b for b in backends if b not in ("fa4", "trtllm_mha")]
    return backends or ["flashinfer"]


def _get_answer_value(answer_str: str) -> int:
    answer_str = answer_str.replace(",", "")
    numbers = re.findall(r"\d+", answer_str)
    if len(numbers) < 1:
        return INVALID
    try:
        return ast.literal_eval(numbers[-1])
    except SyntaxError:
        return INVALID


def _maybe_download_gsm8k(data_path: str) -> str:
    url = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"
    if os.path.isfile(data_path):
        return data_path
    return download_and_cache_file(url)


def _flush_cache(base_url: str) -> None:
    resp = requests.get(base_url + "/flush_cache", timeout=60)
    resp.raise_for_status()


@dataclass(frozen=True)
class CorpusItem:
    corpus_index: int
    body: dict[str, Any]
    label: str


@dataclass(frozen=True)
class BenchRequest:
    prompt: str
    sampling_params: dict[str, Any]
    label: Optional[int] = None


class CorpusSampler:
    def __init__(
        self,
        items: list[CorpusItem],
        *,
        mode: str,
        rng: random.Random,
    ) -> None:
        if not items:
            raise ValueError("Corpus is empty.")
        self._items = items
        self._mode = mode
        self._rng = rng
        self._cursor = 0
        self._order: list[int] = []

    def next_item(self) -> CorpusItem:
        if self._mode == "random_with_replacement":
            return self._items[self._rng.randrange(len(self._items))]

        if self._mode == "sequential_cycle":
            item = self._items[self._cursor % len(self._items)]
            self._cursor += 1
            return item

        if self._mode == "unique_once":
            if self._cursor >= len(self._items):
                raise RuntimeError(
                    "Corpus exhausted in unique_once mode. "
                    "Use a larger request corpus or another sample mode."
                )
            item = self._items[self._cursor]
            self._cursor += 1
            return item

        if self._cursor >= len(self._order):
            self._order = list(range(len(self._items)))
            self._rng.shuffle(self._order)
            self._cursor = 0
        item = self._items[self._order[self._cursor]]
        self._cursor += 1
        return item


class MaxTokensSampler:
    def __init__(self, buckets: list[dict[str, Any]]) -> None:
        if not buckets:
            raise ValueError("At least one max_tokens bucket is required.")
        total_weight = 0.0
        self._buckets: list[tuple[float, int, int]] = []
        for bucket in buckets:
            weight = float(bucket["weight"])
            lo = int(bucket["min"])
            hi = int(bucket["max"])
            if weight <= 0:
                raise ValueError(f"Invalid bucket weight: {weight}")
            if lo <= 0 or hi <= 0 or hi < lo:
                raise ValueError(f"Invalid bucket bounds: {lo}..{hi}")
            total_weight += weight
            self._buckets.append((total_weight, lo, hi))
        if abs(total_weight - 1.0) > 1e-6:
            raise ValueError(f"Bucket weights must sum to 1.0, got {total_weight:.6f}")

    @classmethod
    def from_profile(cls, profile_name: str) -> "MaxTokensSampler":
        if profile_name not in MAX_TOKENS_PROFILES:
            raise ValueError(
                f"Unknown max_tokens profile '{profile_name}'. "
                f"Available: {list(MAX_TOKENS_PROFILES)}"
            )
        return cls(MAX_TOKENS_PROFILES[profile_name])

    @classmethod
    def from_file(cls, spec_path: Path) -> "MaxTokensSampler":
        payload = json.loads(spec_path.read_text())
        if not isinstance(payload, dict) or "buckets" not in payload:
            raise ValueError(
                "Max tokens spec must be a JSON object with a 'buckets' field."
            )
        buckets = payload["buckets"]
        if not isinstance(buckets, list):
            raise ValueError("'buckets' must be a list.")
        return cls(buckets)

    def sample(self, rng: random.Random) -> int:
        roll = rng.random()
        for cumulative_weight, lo, hi in self._buckets:
            if roll <= cumulative_weight:
                if lo == hi:
                    return lo
                return rng.randint(lo, hi)
        _, lo, hi = self._buckets[-1]
        return hi if lo == hi else rng.randint(lo, hi)


def _send_generate(
    base_url: str,
    request: BenchRequest | list[BenchRequest],
    *,
    timeout_s: int,
) -> list[dict]:
    if isinstance(request, list) and not request:
        return []
    if isinstance(request, list):
        text: str | list[str] = [item.prompt for item in request]
        sampling_params: dict[str, Any] | list[dict[str, Any]] = [
            item.sampling_params for item in request
        ]
    else:
        text = request.prompt
        sampling_params = request.sampling_params
    resp = requests.post(
        base_url + "/generate",
        json={
            "text": text,
            "sampling_params": sampling_params,
        },
        timeout=int(timeout_s),
    )
    resp.raise_for_status()
    out = resp.json()
    if isinstance(text, list):
        if not isinstance(out, list):
            raise RuntimeError(
                "Expected a list response for batched /generate, but got "
                f"type={type(out).__name__}."
            )
        if len(out) != len(text):
            raise RuntimeError(
                "Batched /generate output length mismatch: "
                f"got {len(out)} outputs for {len(text)} prompts."
            )
        return out

    if isinstance(out, list):
        raise RuntimeError(
            "Expected an object response for single /generate, but got "
            f"type={type(out).__name__}."
        )
    return [out]


@dataclass(frozen=True)
class BenchMetrics:
    latency_s: float
    output_tokens: int
    output_toks_per_s: float
    e2e_latency_p50_s: float
    e2e_latency_p90_s: float
    e2e_latency_p99_s: float
    accuracy: Optional[float]
    invalid_rate: Optional[float]
    spec_accept_length: Optional[float]
    spec_verify_ct_sum: int


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])

    sorted_values = sorted(values)
    idx = (len(sorted_values) - 1) * q
    lower = int(idx)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = idx - lower
    return float(
        sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight
    )


def _load_examples(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _fallback_chat_prompt(messages: list[dict[str, Any]]) -> str:
    role_prefix = {
        "system": "System",
        "user": "Human",
        "assistant": "Assistant",
    }
    parts: list[str] = []
    for message in messages:
        role = role_prefix.get(str(message.get("role", "user")), "Human")
        content = message.get("content", "")
        if isinstance(content, list):
            text_parts = [
                part.get("text", "")
                for part in content
                if isinstance(part, dict) and part.get("type") == "text"
            ]
            content = " ".join(x for x in text_parts if x)
        parts.append(f"{role}: {str(content).strip()}".rstrip())
    return "\n\n".join(parts) + "\n\nAssistant:"


def _render_messages_to_prompt(
    tokenizer: AutoTokenizer,
    messages: list[dict[str, Any]],
) -> str:
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except ValueError as exc:
            if "chat_template is not set" not in str(exc):
                raise
    except ValueError as exc:
        if "chat_template is not set" not in str(exc):
            raise
    return _fallback_chat_prompt(messages)


def _load_queries(
    examples: dict[str, Any],
    *,
    n: int,
    rng: random.Random,
) -> list[str]:
    queries: list[str] = []
    seen: set[str] = set()
    attempts = 0
    while len(queries) < n and attempts < n * 20:
        attempts += 1
        template = rng.choice(examples["query_templates"])
        replacements = {
            "entity": rng.choice(examples["entities"]),
            "product": rng.choice(examples["products"]),
            "product2": rng.choice(examples["products"]),
            "brand": rng.choice(examples["brands"]),
            "action": rng.choice(examples["actions"]),
            "topic": rng.choice(examples["topics"]),
            "topicA": rng.choice(examples["topics"]),
            "topicB": rng.choice(examples["topics"]),
            "tech": rng.choice(examples["techs"]),
            "tech2": rng.choice(examples["techs"]),
            "error_code": rng.choice(examples["error_codes"]),
            "city": rng.choice(examples["cities"]),
            "service": rng.choice(examples["services"]),
        }
        try:
            query = template.format(**replacements)
        except KeyError:
            continue
        if query not in seen:
            seen.add(query)
            queries.append(query)
    if len(queries) < n:
        raise RuntimeError(f"Only generated {len(queries)} unique queries out of {n}.")
    return queries


def _build_output_schema(n: int) -> dict[str, Any]:
    properties = {f"query_{i}": {"type": "string"} for i in range(n)}
    required = [f"query_{i}" for i in range(n)]
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "QueryExpanderOutput",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
        },
    }


def _build_schema_hint(n: int) -> str:
    fields = ", ".join(f'"query_{i}": "<string>"' for i in range(n))
    return "{" + fields + "}"


def _build_query_generator_corpus(
    *,
    examples: dict[str, Any],
    max_expansions: int,
) -> list[CorpusItem]:
    num_additional = max(max_expansions - 1, 1)
    system_prompt = examples["generator_system_prompt_template"].format(
        num_additional_queries=num_additional,
    )
    items: list[CorpusItem] = []
    for idx, intent in enumerate(examples["generator_intent_templates"]):
        body = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "intent": intent,
                            "num_additional_queries": num_additional,
                        }
                    ),
                },
            ],
            "temperature": 0,
        }
        items.append(CorpusItem(corpus_index=idx, body=body, label=intent[:120]))
    return items


def _build_query_expander_corpus(
    *,
    examples: dict[str, Any],
    max_expansions: int,
    corpus_size: int,
    rng: random.Random,
) -> list[CorpusItem]:
    num_llm_queries = max(max_expansions - 1, 1)
    system_prompt = examples["expander_system_prompt_template"].format(
        num_queries=num_llm_queries,
        schema_hint=_build_schema_hint(num_llm_queries),
    )
    output_schema = _build_output_schema(num_llm_queries)
    queries = _load_queries(examples, n=corpus_size, rng=rng)
    items: list[CorpusItem] = []
    for idx, query in enumerate(queries):
        body = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "user_query": query,
                            "num_queries": num_llm_queries,
                        }
                    ),
                },
            ],
            "max_tokens": 256,
            "temperature": 0,
            "response_format": output_schema,
        }
        items.append(CorpusItem(corpus_index=idx, body=body, label=query))
    return items


def _load_trace_replay_corpus(
    *,
    requests_jsonl: Path,
    default_max_tokens: Optional[int],
) -> list[CorpusItem]:
    items: list[CorpusItem] = []
    with requests_jsonl.open() as handle:
        for idx, raw_line in enumerate(handle):
            line = raw_line.strip()
            if not line:
                continue
            payload = json.loads(line)
            body = (
                payload["body"]
                if isinstance(payload, dict) and "body" in payload
                else payload
            )
            if not isinstance(body, dict):
                raise ValueError(f"Line {idx + 1} is not a JSON object.")
            if "messages" not in body:
                raise ValueError(f"Line {idx + 1} is missing 'messages'.")
            item_body = copy.deepcopy(body)
            item_body.pop("stream", None)
            item_body.pop("stream_options", None)
            if default_max_tokens is not None:
                item_body.setdefault("max_tokens", default_max_tokens)
            label = (
                payload.get("label", f"trace:{idx}")
                if isinstance(payload, dict)
                else f"trace:{idx}"
            )
            items.append(CorpusItem(corpus_index=idx, body=item_body, label=str(label)))
    if not items:
        raise ValueError(f"No usable request bodies found in {requests_jsonl}")
    return items


def _build_parallel_bench_corpus(
    *,
    mode: str,
    examples_path: Path,
    requests_jsonl: Optional[Path],
    max_expansions: int,
    expander_corpus_size: int,
    default_max_tokens: Optional[int],
    rng: random.Random,
) -> list[CorpusItem]:
    if mode == "trace_replay":
        if requests_jsonl is None:
            raise ValueError("--requests-jsonl is required for trace_replay mode.")
        return _load_trace_replay_corpus(
            requests_jsonl=requests_jsonl,
            default_max_tokens=default_max_tokens,
        )

    examples = _load_examples(examples_path)
    if mode == "query_generator":
        return _build_query_generator_corpus(
            examples=examples,
            max_expansions=max_expansions,
        )
    if mode == "query_expander":
        return _build_query_expander_corpus(
            examples=examples,
            max_expansions=max_expansions,
            corpus_size=expander_corpus_size,
            rng=rng,
        )
    raise ValueError(f"Unsupported workload mode: {mode}")


def _sampling_params_from_body(
    *,
    body: dict[str, Any],
    args: argparse.Namespace,
    max_tokens_sampler: Optional[MaxTokensSampler],
    rng: random.Random,
) -> dict[str, Any]:
    if int(body.get("n", 1)) != 1:
        raise ValueError("Only n=1 request bodies are supported in this sweep.")
    if body.get("tools") is not None or body.get("tool_choice") is not None:
        raise ValueError(
            "Tool-calling request bodies are not supported in this /generate-based sweep."
        )

    max_new_tokens = body.get("max_tokens", body.get("max_new_tokens"))
    if max_new_tokens is None:
        max_new_tokens = (
            max_tokens_sampler.sample(rng)
            if max_tokens_sampler is not None
            else int(args.max_new_tokens)
        )

    sampling_params: dict[str, Any] = {
        "temperature": float(body.get("temperature", args.temperature)),
        "top_p": float(body.get("top_p", args.top_p)),
        "top_k": int(body.get("top_k", args.top_k)),
        "max_new_tokens": int(max_new_tokens),
    }

    stop = body.get("stop")
    if stop:
        sampling_params["stop"] = stop

    response_format = body.get("response_format")
    if isinstance(response_format, dict):
        fmt_type = response_format.get("type")
        if fmt_type == "json_schema":
            json_schema = response_format.get("json_schema")
            if not isinstance(json_schema, dict):
                raise ValueError("response_format.json_schema must be an object.")
            schema = json_schema.get("schema", json_schema.get("schema_"))
            if schema is None:
                raise ValueError(
                    "response_format.json_schema.schema is required for json_schema mode."
                )
            sampling_params["json_schema"] = convert_json_schema_to_str(schema)
        elif fmt_type == "json_object":
            sampling_params["json_schema"] = '{"type": "object"}'
        elif fmt_type != "text":
            raise ValueError(f"Unsupported response_format type: {fmt_type}")

    return sampling_params


def _build_parallel_bench_requests(
    *,
    args: argparse.Namespace,
    tokenizer: AutoTokenizer,
    max_requests: int,
) -> list[BenchRequest]:
    rng = random.Random(int(args.seed))
    default_max_tokens = int(args.max_new_tokens) if args.max_new_tokens > 0 else None
    requests_jsonl = Path(args.requests_jsonl).expanduser() if args.requests_jsonl else None
    examples_path = Path(args.examples_path).expanduser()

    max_tokens_sampler: Optional[MaxTokensSampler] = None
    if args.workload == "query_generator":
        if args.request_max_tokens_spec:
            max_tokens_sampler = MaxTokensSampler.from_file(
                Path(args.request_max_tokens_spec).expanduser()
            )
        else:
            max_tokens_sampler = MaxTokensSampler.from_profile(
                args.request_max_tokens_profile
            )

    corpus = _build_parallel_bench_corpus(
        mode=args.workload,
        examples_path=examples_path,
        requests_jsonl=requests_jsonl,
        max_expansions=int(args.max_expansions),
        expander_corpus_size=max(int(args.expander_corpus_size), int(max_requests)),
        default_max_tokens=default_max_tokens,
        rng=rng,
    )
    sampler = CorpusSampler(corpus, mode=args.sample_mode, rng=rng)

    requests_list: list[BenchRequest] = []
    for _ in range(max_requests):
        item = sampler.next_item()
        body = copy.deepcopy(item.body)
        prompt = _render_messages_to_prompt(tokenizer, body["messages"])
        requests_list.append(
            BenchRequest(
                prompt=prompt,
                sampling_params=_sampling_params_from_body(
                    body=body,
                    args=args,
                    max_tokens_sampler=max_tokens_sampler,
                    rng=rng,
                ),
                label=None,
            )
        )
    return requests_list


def _build_gsm8k_requests(
    *,
    data_path: str,
    tokenizer: AutoTokenizer,
    max_requests: int,
    args: argparse.Namespace,
) -> list[BenchRequest]:
    data_path = _maybe_download_gsm8k(data_path)
    lines = list(read_jsonl(data_path))
    if len(lines) < max_requests:
        raise RuntimeError(
            f"GSM8K file only has {len(lines)} lines, but need {max_requests}."
        )

    requests_list: list[BenchRequest] = []
    for i in range(max_requests):
        user_content = (
            lines[i]["question"]
            + "\nPlease reason step by step, and put your final answer within \\boxed{}."
        )
        requests_list.append(
            BenchRequest(
                prompt=_render_messages_to_prompt(
                    tokenizer,
                    [{"role": "user", "content": user_content}],
                ),
                sampling_params={
                    "temperature": float(args.temperature),
                    "top_p": float(args.top_p),
                    "top_k": int(args.top_k),
                    "max_new_tokens": int(args.max_new_tokens),
                },
                label=_get_answer_value(lines[i]["answer"]),
            )
        )
    if not all(req.label != INVALID for req in requests_list):
        raise RuntimeError("Invalid labels in GSM8K data.")
    return requests_list


def _run_benchmark_requests(
    base_url: str,
    *,
    requests_list: list[BenchRequest],
    concurrency: int,
    batch_requests: bool,
    timeout_s: int,
    expect_dflash: bool,
) -> BenchMetrics:
    # Drop the first batch from metrics to exclude one-time JIT/cuda-graph overhead
    # that often happens immediately after /flush_cache for large batch sizes.
    bs = max(int(concurrency), 1)
    if len(requests_list) > bs:
        warmup_requests = requests_list[:bs]
        if batch_requests:
            _send_generate(
                base_url,
                warmup_requests,
                timeout_s=timeout_s,
            )
        else:
            with ThreadPoolExecutor(max_workers=int(concurrency)) as pool:
                futures = [
                    pool.submit(
                        _send_generate,
                        base_url=base_url,
                        request=request_item,
                        timeout_s=timeout_s,
                    )
                    for request_item in warmup_requests
                ]
                for fut in as_completed(futures):
                    outs = fut.result()
                    if len(outs) != 1:
                        raise RuntimeError(
                            "Expected exactly one output for single /generate warmup request."
                        )

        requests_list = requests_list[bs:]

    start = time.perf_counter()
    total_tokens = 0
    spec_verify_ct_sum = 0
    spec_accept_lengths: list[float] = []
    request_latencies_s: list[float] = []
    correct = 0
    invalid = 0

    def _handle_output(out: dict, label: Optional[int]) -> None:
        nonlocal total_tokens, spec_verify_ct_sum, correct, invalid
        meta = out.get("meta_info", {}) or {}
        total_tokens += int(meta.get("completion_tokens", 0))
        spec_verify_ct_sum += int(meta.get("spec_verify_ct", 0))
        if "spec_accept_length" in meta:
            try:
                spec_accept_lengths.append(float(meta["spec_accept_length"]))
            except (TypeError, ValueError):
                pass

        if label is not None:
            pred = _get_answer_value(out.get("text", ""))
            if pred == INVALID:
                invalid += 1
            if pred == label:
                correct += 1

    if batch_requests:
        bs = max(int(concurrency), 1)
        for start_idx in range(0, len(requests_list), bs):
            chunk_requests = requests_list[start_idx : start_idx + bs]
            chunk_start = time.perf_counter()
            outs = _send_generate(
                base_url,
                chunk_requests,
                timeout_s=timeout_s,
            )
            chunk_latency_s = time.perf_counter() - chunk_start
            # Batched /generate returns once the whole batch is complete, so each
            # prompt in the chunk observes the same client-side e2e latency.
            request_latencies_s.extend([float(chunk_latency_s)] * len(chunk_requests))
            for out, request_item in zip(outs, chunk_requests):
                _handle_output(out, request_item.label)
    else:
        def _timed_single_generate(
            request_item: BenchRequest,
        ) -> tuple[list[dict], float]:
            req_start = time.perf_counter()
            outs = _send_generate(
                base_url=base_url,
                request=request_item,
                timeout_s=timeout_s,
            )
            return outs, float(time.perf_counter() - req_start)

        with ThreadPoolExecutor(max_workers=int(concurrency)) as pool:
            futures = {
                pool.submit(_timed_single_generate, request_item): i
                for i, request_item in enumerate(requests_list)
            }
            for fut in as_completed(futures):
                i = futures[fut]
                outs, request_latency_s = fut.result()
                if len(outs) != 1:
                    raise RuntimeError(
                        "Expected exactly one output for single /generate request."
                    )
                request_latencies_s.append(float(request_latency_s))
                _handle_output(outs[0], requests_list[i].label)

    latency = time.perf_counter() - start
    toks_per_s = total_tokens / max(latency, 1e-6)

    if expect_dflash and spec_verify_ct_sum <= 0:
        raise RuntimeError(
            "DFLASH sanity check failed: did not observe any `spec_verify_ct` in responses "
            "(DFLASH may not have been enabled)."
        )

    spec_accept_length = (
        float(statistics.mean(spec_accept_lengths)) if spec_accept_lengths else None
    )

    has_labels = any(req.label is not None for req in requests_list)
    if not has_labels:
        acc = None
        invalid_rate = None
    else:
        acc = correct / max(len(requests_list), 1)
        invalid_rate = invalid / max(len(requests_list), 1)

    return BenchMetrics(
        latency_s=float(latency),
        output_tokens=int(total_tokens),
        output_toks_per_s=float(toks_per_s),
        e2e_latency_p50_s=_percentile(request_latencies_s, 0.50),
        e2e_latency_p90_s=_percentile(request_latencies_s, 0.90),
        e2e_latency_p99_s=_percentile(request_latencies_s, 0.99),
        accuracy=acc,
        invalid_rate=invalid_rate,
        spec_accept_length=spec_accept_length,
        spec_verify_ct_sum=int(spec_verify_ct_sum),
    )


def _format_table(
    *,
    tp_sizes: list[int],
    concurrencies: list[int],
    values: dict[tuple[int, int], Optional[float]],
    float_fmt: str,
) -> str:
    header = ["tp\\conc"] + [str(c) for c in concurrencies]
    rows: list[list[str]] = [header]
    for tp in tp_sizes:
        row = [str(tp)]
        for c in concurrencies:
            v = values.get((tp, c), None)
            row.append("N/A" if v is None else format(v, float_fmt))
        rows.append(row)

    col_widths = [
        max(len(row[col_idx]) for row in rows) for col_idx in range(len(rows[0]))
    ]

    lines: list[str] = []
    lines.append("  ".join(cell.rjust(col_widths[i]) for i, cell in enumerate(rows[0])))
    lines.append("  ".join("-" * w for w in col_widths))
    for row in rows[1:]:
        lines.append("  ".join(cell.rjust(col_widths[i]) for i, cell in enumerate(row)))
    return "\n".join(lines)


def _build_common_server_args(
    args: argparse.Namespace, *, backend: str, tp: int
) -> list[str]:
    common_server_args: list[str] = [
        "--trust-remote-code",
        "--attention-backend",
        backend,
        "--tp-size",
        str(tp),
        "--dtype",
        str(args.dtype),
        "--max-running-requests",
        str(args.max_running_requests),
        "--cuda-graph-max-bs",
        "32",
        "--mamba-scheduler-strategy",
        str(args.mamba_scheduler_strategy),
        "--enforce-piecewise-cuda-graph",
    ]
    if args.mem_fraction_static is not None:
        common_server_args.extend(
            ["--mem-fraction-static", str(args.mem_fraction_static)]
        )
    if args.disable_radix_cache:
        common_server_args.append("--disable-radix-cache")
    if args.page_size is not None:
        common_server_args.extend(["--page-size", str(int(args.page_size))])
    return common_server_args


def _build_mode_runs(
    args: argparse.Namespace, common_server_args: list[str]
) -> list[tuple[str, str, list[str], bool]]:
    mode_runs: list[tuple[str, str, list[str], bool]] = []
    if not args.skip_baseline:
        mode_runs.append(("baseline", "baseline", common_server_args, False))
    mode_runs.append(
        (
            "dflash",
            "DFLASH",
            [
                *common_server_args,
                "--speculative-algorithm",
                "DFLASH",
                "--speculative-draft-model-path",
                args.draft_model,
                "--speculative-dflash-block-size",
                str(int(args.speculative_dflash_block_size)),
                *(
                    [
                        "--speculative-dflash-draft-window-size",
                        str(int(args.speculative_dflash_draft_window_size)),
                    ]
                    if args.speculative_dflash_draft_window_size is not None
                    else []
                ),
                *(
                    [
                        "--speculative-draft-attention-backend",
                        args.speculative_draft_attention_backend,
                    ]
                    if args.speculative_draft_attention_backend
                    else []
                ),
            ],
            True,
        )
    )
    return mode_runs


def _collect_metric(
    *,
    results: dict[tuple[str, int, int, str], BenchMetrics],
    backend: str,
    tp_sizes: list[int],
    concurrencies: list[int],
    mode: str,
    field: str,
) -> dict[tuple[int, int], Optional[float]]:
    out: dict[tuple[int, int], Optional[float]] = {}
    for tp in tp_sizes:
        for conc in concurrencies:
            metrics = results.get((backend, tp, conc, mode), None)
            out[(tp, conc)] = None if metrics is None else getattr(metrics, field)
    return out


def _compute_speedup(
    baseline: dict[tuple[int, int], Optional[float]],
    dflash: dict[tuple[int, int], Optional[float]],
) -> dict[tuple[int, int], Optional[float]]:
    return {
        key: None if (b is None or d is None or b <= 0) else (d / b)
        for key, b in baseline.items()
        for d in [dflash.get(key, None)]
    }


def _print_kv_lines(items: list[tuple[str, object]]) -> None:
    for key, value in items:
        print(f"{key}={value}")


def _run_mode_for_backend_tp(
    *,
    mode_label: str,
    model_path: str,
    base_url: str,
    server_args: list[str],
    expect_dflash: bool,
    requests_list: list[BenchRequest],
    concurrencies: list[int],
    num_questions_by_conc: dict[int, int],
    args: argparse.Namespace,
) -> dict[int, BenchMetrics]:
    print(f"\n=== {mode_label} ===")
    server_start_timeout_s = int(max(DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH, args.timeout_s))
    proc = popen_launch_server(
        model_path,
        base_url,
        timeout=server_start_timeout_s,
        other_args=server_args,
    )
    try:
        _send_generate(
            base_url,
            BenchRequest(
                prompt="Hello",
                sampling_params={
                    "temperature": float(args.temperature),
                    "top_p": float(args.top_p),
                    "top_k": int(args.top_k),
                    "max_new_tokens": 8,
                },
            ),
            timeout_s=min(int(args.timeout_s), 300),
        )

        metrics_by_conc: dict[int, BenchMetrics] = {}
        for conc in concurrencies:
            n = num_questions_by_conc[conc]
            _flush_cache(base_url)
            print(
                f"[warmup] run 1 warmup batch (size={conc}) after /flush_cache; excluded from metrics."
            )
            metrics = _run_benchmark_requests(
                base_url,
                requests_list=requests_list[: n + conc],
                concurrency=int(conc),
                batch_requests=bool(args.batch_requests),
                timeout_s=int(args.timeout_s),
                expect_dflash=expect_dflash,
            )
            metrics_by_conc[conc] = metrics
            acc = "N/A" if metrics.accuracy is None else f"{metrics.accuracy:.3f}"
            invalid = (
                "N/A" if metrics.invalid_rate is None else f"{metrics.invalid_rate:.3f}"
            )
            line = (
                f"[{mode_label}] conc={conc:>2} n={n:<4} "
                f"toks/s={metrics.output_toks_per_s:,.2f} "
                f"latency={metrics.latency_s:.1f}s "
                f"e2e_p50={metrics.e2e_latency_p50_s:.3f}s "
                f"e2e_p90={metrics.e2e_latency_p90_s:.3f}s "
                f"e2e_p99={metrics.e2e_latency_p99_s:.3f}s "
                f"acc={acc} invalid={invalid}"
            )
            if expect_dflash:
                accept_len = (
                    "N/A"
                    if metrics.spec_accept_length is None
                    else f"{metrics.spec_accept_length:.3f}"
                )
                line += (
                    f" accept_len={accept_len} "
                    f"spec_verify_ct_sum={metrics.spec_verify_ct_sum}"
                )
            print(line)
        return metrics_by_conc
    finally:
        kill_process_tree(proc.pid)
        try:
            proc.wait(timeout=30)
        except Exception:
            pass


def _print_summary(
    *,
    args: argparse.Namespace,
    attention_backends: list[str],
    tp_sizes: list[int],
    concurrencies: list[int],
    device_sm: int,
    results: dict[tuple[str, int, int, str], BenchMetrics],
) -> None:
    print("\n=== DFLASH Sweep Summary ===")
    _print_kv_lines(
        [
            ("workload", args.workload),
            ("target_model", args.target_model),
            ("draft_model", args.draft_model),
            ("max_new_tokens", args.max_new_tokens),
            (
                "sampling",
                f"temperature:{args.temperature}, top_p:{args.top_p}, top_k:{args.top_k}",
            ),
            ("attention_backends", ",".join(attention_backends)),
            (
                "speculative_draft_attention_backend",
                args.speculative_draft_attention_backend,
            ),
            (
                "speculative_dflash_block_size",
                args.speculative_dflash_block_size,
            ),
            (
                "speculative_dflash_draft_window_size",
                args.speculative_dflash_draft_window_size,
            ),
            ("tp_sizes", ",".join(str(x) for x in tp_sizes)),
            ("concurrencies", ",".join(str(x) for x in concurrencies)),
            (
                "questions_per_concurrency_base",
                args.questions_per_concurrency_base,
            ),
            ("device_sm", device_sm),
            ("skip_baseline", bool(args.skip_baseline)),
        ]
    )

    section_fields = [
        ("Baseline output tok/s", "baseline", "output_toks_per_s", ",.2f"),
        (
            "Baseline request e2e latency p50 (s)",
            "baseline",
            "e2e_latency_p50_s",
            ".3f",
        ),
        (
            "Baseline request e2e latency p90 (s)",
            "baseline",
            "e2e_latency_p90_s",
            ".3f",
        ),
        (
            "Baseline request e2e latency p99 (s)",
            "baseline",
            "e2e_latency_p99_s",
            ".3f",
        ),
        ("Baseline accuracy", "baseline", "accuracy", ".3f"),
        ("DFLASH output tok/s", "dflash", "output_toks_per_s", ",.2f"),
        (
            "DFLASH request e2e latency p50 (s)",
            "dflash",
            "e2e_latency_p50_s",
            ".3f",
        ),
        (
            "DFLASH request e2e latency p90 (s)",
            "dflash",
            "e2e_latency_p90_s",
            ".3f",
        ),
        (
            "DFLASH request e2e latency p99 (s)",
            "dflash",
            "e2e_latency_p99_s",
            ".3f",
        ),
        ("DFLASH accuracy", "dflash", "accuracy", ".3f"),
        (
            "DFLASH acceptance length (mean spec_accept_length)",
            "dflash",
            "spec_accept_length",
            ".3f",
        ),
    ]

    for backend in attention_backends:
        print(f"\n=== Backend: {backend} ===")
        metrics_map = {
            (mode, field): _collect_metric(
                results=results,
                backend=backend,
                tp_sizes=tp_sizes,
                concurrencies=concurrencies,
                mode=mode,
                field=field,
            )
            for _, mode, field, _ in section_fields
        }
        sections: list[tuple[str, dict[tuple[int, int], Optional[float]], str]] = [
            (title, metrics_map[(mode, field)], fmt)
            for title, mode, field, fmt in section_fields
        ]
        sections.insert(
            max(len(sections) - 1, 0),
            (
                "Speedup (DFLASH / baseline)",
                _compute_speedup(
                    metrics_map[("baseline", "output_toks_per_s")],
                    metrics_map[("dflash", "output_toks_per_s")],
                ),
                ".3f",
            ),
        )

        for title, values, fmt in sections:
            print(f"\n{title}")
            print(
                _format_table(
                    tp_sizes=tp_sizes,
                    concurrencies=concurrencies,
                    values=values,
                    float_fmt=fmt,
                )
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--workload",
        choices=["gsm8k", "query_generator", "query_expander", "trace_replay"],
        default="gsm8k",
        help=(
            "Request corpus to benchmark. `gsm8k` preserves the original sweep; "
            "the other modes port over request construction from "
            "`~/latest_parallel_bench_modal_openloop.py`."
        ),
    )
    parser.add_argument(
        "--data-path",
        default="test.jsonl",
        help="GSM8K jsonl path used when --workload=gsm8k.",
    )
    parser.add_argument(
        "--examples-path",
        default=str(DEFAULT_EXAMPLES_PATH),
        help="Examples JSON used for query_generator/query_expander workloads.",
    )
    parser.add_argument(
        "--requests-jsonl",
        default="",
        help="Trace replay request corpus used when --workload=trace_replay.",
    )
    parser.add_argument(
        "--sample-mode",
        choices=[
            "shuffle_cycle",
            "random_with_replacement",
            "sequential_cycle",
            "unique_once",
        ],
        default="shuffle_cycle",
        help="How to reuse the synthetic corpus when the sweep needs more requests.",
    )
    parser.add_argument(
        "--max-expansions",
        type=int,
        default=2,
        help="Synthetic workload parameter copied from the parallel benchmark.",
    )
    parser.add_argument(
        "--expander-corpus-size",
        type=int,
        default=2048,
        help="Base synthetic corpus size for --workload=query_expander.",
    )
    parser.add_argument(
        "--request-max-tokens-profile",
        default="customer_generator_v1",
        choices=list(MAX_TOKENS_PROFILES),
        help="Per-request max_tokens profile for --workload=query_generator.",
    )
    parser.add_argument(
        "--request-max-tokens-spec",
        default="",
        help=(
            "Optional JSON file overriding --request-max-tokens-profile. "
            'Format: {"buckets": [{"weight": 0.5, "min": 50, "max": 50}, ...]}'
        ),
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--target-model", default="Qwen/Qwen3-8B")
    parser.add_argument("--draft-model", default="z-lab/Qwen3-8B-DFlash-b16")
    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        help="Skip running the baseline (target-only) sweep; only run DFLASH and report N/A for baseline/speedup.",
    )
    parser.add_argument(
        "--batch-requests",
        action="store_true",
        help="Send prompts as server-side batched /generate requests (batch size = concurrency) instead of client-side concurrent requests.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument(
        "--timeout-s",
        type=int,
        default=3600,
        help=(
            "Timeout in seconds for benchmarked /generate calls and server startup "
            "health checks."
        ),
    )
    parser.add_argument(
        "--mem-fraction-static",
        type=float,
        default=None,
        help="Optional server --mem-fraction-static override. If unset, use the server auto heuristic.",
    )
    parser.add_argument("--disable-radix-cache", action="store_true")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument(
        "--page-size",
        type=int,
        default=None,
        help="Optional server --page-size override for both baseline and DFLASH runs.",
    )
    parser.add_argument("--max-running-requests", type=int, default=32)
    parser.add_argument(
        "--mamba-scheduler-strategy",
        default="no_buffer",
        help=(
            "Server --mamba-scheduler-strategy value to pass through to benchmark "
            "runs, e.g. `no_buffer` or `extra_buffer`."
        ),
    )
    parser.add_argument("--tp-sizes", default="1,2,4,8")
    parser.add_argument("--concurrencies", default="1,2,4,8,16,32")
    parser.add_argument(
        "--questions-per-concurrency-base",
        type=int,
        default=128,
        help="num_questions = base * concurrency (default matches the sweep plan).",
    )
    parser.add_argument(
        "--max-questions-per-config",
        type=int,
        default=1024,
        help="Cap num_questions per (tp, concurrency) run (default: 1024).",
    )
    parser.add_argument("--attention-backends", default="flashinfer,fa3,trtllm_mha,fa4")
    parser.add_argument(
        "--speculative-draft-attention-backend",
        default=None,
        help="Optional server --speculative-draft-attention-backend override for DFLASH runs.",
    )
    parser.add_argument(
        "--speculative-dflash-block-size",
        type=int,
        default=4,
        help="Server --speculative-dflash-block-size override for DFLASH runs.",
    )
    parser.add_argument(
        "--speculative-dflash-draft-window-size",
        type=int,
        default=None,
        help="Optional server --speculative-dflash-draft-window-size override for DFLASH runs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this sweep.")
    if args.temperature < 0.0:
        raise RuntimeError(f"--temperature must be >= 0, got {args.temperature}.")
    if not (0.0 < args.top_p <= 1.0):
        raise RuntimeError(f"--top-p must be in (0, 1], got {args.top_p}.")
    if args.top_k == 0 or args.top_k < -1:
        raise RuntimeError(f"--top-k must be -1 (all vocab) or >= 1, got {args.top_k}.")
    if args.timeout_s <= 0:
        raise RuntimeError(f"--timeout-s must be > 0, got {args.timeout_s}.")

    visible_gpus = int(torch.cuda.device_count())
    tp_sizes = _parse_int_csv(args.tp_sizes)
    tp_sizes = [tp for tp in tp_sizes if tp >= 1 and tp <= visible_gpus]
    if not tp_sizes:
        raise RuntimeError(
            f"No tp sizes are runnable with visible_gpus={visible_gpus}. "
            "Set CUDA_VISIBLE_DEVICES accordingly."
        )

    concurrencies = _parse_int_csv(args.concurrencies)
    concurrencies = [c for c in concurrencies if c >= 1]
    if not concurrencies:
        raise RuntimeError("No concurrencies specified.")

    num_questions_by_conc = {
        c: min(
            int(args.questions_per_concurrency_base) * int(c),
            int(args.max_questions_per_config),
        )
        for c in concurrencies
    }
    max_questions = max(num_questions_by_conc.values())

    attention_backends = [
        s.strip() for s in args.attention_backends.split(",") if s.strip()
    ]
    device_sm = get_device_sm()
    attention_backends = _filter_attention_backends(
        attention_backends, device_sm=device_sm
    )

    tokenizer = AutoTokenizer.from_pretrained(args.target_model)
    if args.workload == "gsm8k":
        requests_list = _build_gsm8k_requests(
            data_path=args.data_path,
            tokenizer=tokenizer,
            max_requests=max_questions,
            args=args,
        )
    else:
        requests_list = _build_parallel_bench_requests(
            args=args,
            tokenizer=tokenizer,
            max_requests=max_questions,
        )

    # Results indexed by (backend, tp, concurrency, mode).
    results: dict[tuple[str, int, int, str], BenchMetrics] = {}
    # Baseline metrics are backend-agnostic in this sweep; run once per TP and reuse.
    baseline_cache_by_tp: dict[int, dict[int, BenchMetrics]] = {}

    for backend_idx, backend in enumerate(attention_backends):
        for tp in tp_sizes:
            port_base = find_available_port(20000)
            common_server_args = _build_common_server_args(args, backend=backend, tp=tp)
            mode_runs = _build_mode_runs(args, common_server_args)

            for idx, (
                mode_key,
                mode_name,
                mode_server_args,
                expect_dflash,
            ) in enumerate(mode_runs):
                if (
                    mode_key == "baseline"
                    and not args.skip_baseline
                    and backend_idx > 0
                    and tp in baseline_cache_by_tp
                ):
                    mode_metrics = baseline_cache_by_tp[tp]
                else:
                    mode_metrics = _run_mode_for_backend_tp(
                        mode_label=f"backend={backend} tp={tp} ({mode_name})",
                        model_path=args.target_model,
                        base_url=f"http://127.0.0.1:{find_available_port(port_base + idx)}",
                        server_args=mode_server_args,
                        expect_dflash=expect_dflash,
                        requests_list=requests_list,
                        concurrencies=concurrencies,
                        num_questions_by_conc=num_questions_by_conc,
                        args=args,
                    )
                    if mode_key == "baseline" and not args.skip_baseline:
                        baseline_cache_by_tp[tp] = mode_metrics

                for conc, metrics in mode_metrics.items():
                    results[(backend, tp, conc, mode_key)] = metrics

    _print_summary(
        args=args,
        attention_backends=attention_backends,
        tp_sizes=tp_sizes,
        concurrencies=concurrencies,
        device_sm=device_sm,
        results=results,
    )


if __name__ == "__main__":
    main()
