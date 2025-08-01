import argparse
import asyncio
import json
import os
import time
import urllib

import httpx
import numpy as np
from datasets import load_dataset


def download_dataset():
    def formatting_prompts_func(examples):
        prompts = []
        oracle_outputs = []
        for inputx, outputx in zip(examples["input"], examples["output"]):
            prompts.append(inputx)
            oracle_outputs.append(outputx)

        return {"prompt": prompts, "oracle_output": oracle_outputs}

    # Load the dataset
    min_tokens = 1000
    min_chars = 2 * min_tokens
    ds = load_dataset("akoksal/LongForm")
    ds = ds.filter(lambda example: len(example["input"]) > min_chars)
    ds = ds.map(
        formatting_prompts_func,
        batched=True,
    )

    # Check what splits are available in the dataset
    print(f"Available splits: {list(ds.keys())}")

    # Save each split to a separate JSONL file
    for split_name, split_data in ds.items():
        output_file = f"datasets/longform/{split_name}.jsonl"
        split_data.to_json(output_file)
        print(f"Saved {split_name} split to {output_file}")
        print(f"Number of examples: {len(split_data)}")


async def fetch(semaphore, client, url, data, headers, i):
    await asyncio.sleep(0.08 * i)
    async with semaphore:
        try:
            t = time.perf_counter()
            print(f"sending request #{i} to {url}")
            print(url)
            r = await client.post(url, json=data, headers=headers)
            print(f"received response for #{i}")
            client_ttlt_s = time.perf_counter() - t
            response_json = r.json()
            usage = response_json["usage"]
            return client_ttlt_s, usage
        except Exception as e:
            print(f"{i}: Error - {e}")
            # breakpoint()
            print(e)


def get_extra_query_args(base_url, model_id, engine_kwargs):
    extra_query = {"model_id": model_id, "engine_kwargs": json.dumps(engine_kwargs)}
    # FIXME add engine
    return urllib.parse.urlencode(extra_query)


async def run_benchmark(url, args):
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        # "Authorization": f"Bearer {api_key}",
    }

    prompts = []
    with open("datasets/longform/train.jsonl", "r") as f:
        for line in f:
            if len(json.loads(line)["prompt"]) > 10_000 and len(json.loads(line)["prompt"]) < 40_000:
                prompts.append(json.loads(line)["prompt"])
    print(len(prompts), "prompts available over 10,000 chars and less than 40,000 chars.")

    print(f"loaded {len(prompts)} prompts, {args.num_prompts} requested")
    while len(prompts) < args.num_prompts:
        print(f"duplicating prompts from {len(prompts)} to {len(prompts) * 2}")
        prompts *= 2

    def get_data(i):
        return {
            "model": args.model,
            "messages": [
                {"role": "user", "content": prompts[i % len(prompts)]},
            ],
            "temperature": 0,  # v20
            "max_tokens": args.output_len,
        }

    semaphore = asyncio.Semaphore(args.max_concurrency)

    # flush cache
    # cold boot check, i.e. make sure it's responding
    async with httpx.AsyncClient(timeout=120) as client:
        await client.post(url.split("/v1")[0] + "/flush_cache")
        await fetch(semaphore, client, url, get_data(0), headers, 0)

    t = time.perf_counter()
    async with httpx.AsyncClient(timeout=120) as client:
        tasks = [fetch(semaphore, client, url, get_data(i), headers, i) for i in range(args.num_prompts)]
        results = await asyncio.gather(*tasks)
    total_time_s = time.perf_counter() - t

    client_ttlts_s, isls, osls = [], [], []
    for client_ttlt_s, usage in results:
        client_ttlts_s.append(client_ttlt_s)
        isls.append(usage["prompt_tokens"])
        osls.append(usage["completion_tokens"])

    results = {}
    for p in (50, 90, 95):
        results[f"client_ttlt_p{p}_s"] = np.percentile(client_ttlts_s, p)
        results[f"isl_p{p}"] = np.percentile(isls, p)
        results[f"osl_p{p}"] = np.percentile(osls, p)

    results["max_concurrency"] = args.max_concurrency

    results["backend"] = args.backend

    results["total_time_s"] = total_time_s
    results["completed_request_rate"] = args.num_prompts / total_time_s
    results["client_region"] = os.getenv("MODAL_REGION", "unknown")

    results["total_prompt_tokens"] = int(np.sum(isls))
    results["total_output_tokens"] = int(np.sum(osls))
    results["total_tokens"] = results["total_prompt_tokens"] + results["total_output_tokens"]

    results["output_throughput"] = results["total_output_tokens"] / total_time_s
    results["p5_e2e_latency_ms"] = np.percentile(client_ttlts_s, 5)
    results["median_e2e_latency_ms"] = np.percentile(client_ttlts_s, 50)
    results["p95_e2e_latency_ms"] = np.percentile(client_ttlts_s, 95)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", type=str)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument("--max-concurrency", type=int, default=1)
    parser.add_argument("--num-prompts", type=int, default=200)
    parser.add_argument("--output-len", type=int, default=100)
    args = parser.parse_args()

    url = f"http://{args.host}:{args.port}/v1/chat/completions"

    print("downloading dataset")
    download_dataset()

    print(f"running benchmark with max_concurrency={args.max_concurrency} and num_prompts={args.num_prompts}")
    results = asyncio.run(run_benchmark(url, args))

    print("printing results")
    for p in (50, 90, 95):
        k = f"isl_p{p}"  # input seq length
        print(f"{k} = {int(results[k])}")
    for p in (50, 90, 95):
        k = f"osl_p{p}"  # output seq length
        print(f"{k} = {int(results[k])}")
    for p in (50, 90, 95):
        k = f"client_ttlt_p{p}_s"
        print(f"{k} = {results[k]:.2f}")

    print(json.dumps(results))

    # print("saving results")
    # with open("benchmark_results.jsonl", "a") as f:
    #     f.write(json.dumps(results) + "\n")
