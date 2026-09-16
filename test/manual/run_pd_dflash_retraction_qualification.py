#!/usr/bin/env python3
"""Bounded synthetic GPU qualification for PD DFlash save/restore.

Run the same script against an isolated normal app and an otherwise identical
app whose DECODE process has SGLANG_TEST_RETRACT=true and
SGLANG_TEST_RETRACT_INTERVAL=16. Never point the forced app at mirrored traffic.

Example:
  python run_pd_dflash_retraction_qualification.py --url http://127.0.0.1:9000 \
    --decode-url http://DECODE:8000 --label baseline --output /tmp/baseline.json
  python run_pd_dflash_retraction_qualification.py --url http://127.0.0.1:9000 \
    --decode-url http://DECODE:8000 --label forced --output /tmp/forced.json \
    --compare /tmp/baseline.json --require-retractions

Saves synthetic outputs only. No flush, pause, global abort, or runtime mutation.
Exact greedy token equality is a qualification gate, not an assumption:
a mismatch needs investigation even if batching can cause numerical drift.
"""

import argparse
import asyncio
import base64
import hashlib
import json
import struct
import time
import zlib
from pathlib import Path

import aiohttp


def image_url():
    """A deterministic two-color image, encoded without image dependencies."""
    width, height = 192, 128
    raw = b"".join(
        b"\0"
        + b"".join(
            bytes((255, 0, 0) if x < width // 2 else (0, 0, 255)) for x in range(width)
        )
        for _ in range(height)
    )

    def chunk(kind, body):
        return (
            struct.pack(">I", len(body))
            + kind
            + body
            + struct.pack(">I", zlib.crc32(kind + body))
        )

    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )
    return "data:image/png;base64," + base64.b64encode(png).decode()


async def fetch(session, url, payload=None, timeout=180):
    start = time.monotonic()
    method = "GET" if payload is None else "POST"
    try:
        async with session.request(
            method,
            url,
            json=payload,
            timeout=aiohttp.ClientTimeout(total=timeout),
        ) as response:
            raw = await response.text()
            try:
                body = json.loads(raw)
            except json.JSONDecodeError:
                body = raw
            return {
                "status": response.status,
                "body": body,
                "seconds": time.monotonic() - start,
            }
    except Exception as exc:
        return {"status": 0, "error": repr(exc), "seconds": time.monotonic() - start}


def metric_subset(text):
    if not isinstance(text, str):
        return text
    names = (
        "num_retracted_requests_total",
        "num_running_reqs",
        "num_queue_reqs",
        "num_decode_prealloc_queue_reqs",
        "num_decode_transfer_queue_reqs",
        "full_token_usage",
        "mamba_usage",
        "spec_accept",
        "spec_verify_calls",
    )
    return "\n".join(
        line
        for line in text.splitlines()
        if not line.startswith("#") and any(n in line for n in names)
    )


def summary(body, multimodal):
    if not isinstance(body, dict):
        return {}
    if multimodal:
        choice = (body.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        text = (msg.get("reasoning_content") or "") + (msg.get("content") or "")
        return {
            "text": text,
            "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "finish_reason": choice.get("finish_reason"),
            "usage": body.get("usage"),
            "mentions_red_and_blue": "red" in text.lower() and "blue" in text.lower(),
        }
    meta = body.get("meta_info") or {}
    text = body.get("text", "")
    logprobs = meta.pop("output_token_logprobs", [])
    # Keep token IDs for exact comparison; drop large input/output probabilities.
    meta.pop("input_token_logprobs", None)
    return {
        "text": text,
        "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "token_ids": body.get("output_ids") or [item[1] for item in logprobs],
        "meta_info": meta,
    }


async def run(args):
    report = {
        "label": args.label,
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "waves": args.waves,
        "max_new_tokens": args.max_new_tokens,
        "results": [],
        "decode_snapshots": [],
        "limitations": [
            "Final responses establish progress; request counts establish forced retraction.",
            "Repeated waves exercise allocation/free/reuse without claiming exact reused addresses.",
            "GPU sanitizer/log inspection and allocator leak checks remain separate gates.",
            "Only synthetic two-color image coverage; not comprehensive audio/video coverage.",
        ],
    }
    async with aiohttp.ClientSession() as session:
        models = await fetch(session, args.url + "/v1/models", timeout=10)
        if models["status"] != 200:
            raise RuntimeError(f"Model discovery failed: {models}")
        model = models["body"]["data"][0]["id"]

        async def snapshot(tag):
            if not args.decode_url:
                return
            for path in ("/v1/loads?include=all", "/metrics"):
                result = await fetch(session, args.decode_url + path, timeout=10)
                if path == "/metrics":
                    result["body"] = metric_subset(result.get("body"))
                report["decode_snapshots"].append({"tag": tag, "path": path, **result})

        async def request(wave, case):
            mm = case >= 4
            key = f"wave{wave}-case{case}"
            rid = f"dflash-retract-{args.label}-{key}"
            if mm:
                payload = {
                    "model": model,
                    "rid": rid,
                    "temperature": 0,
                    "seed": case,
                    "max_tokens": args.max_new_tokens,
                    "stream": False,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {"url": image_url()},
                                },
                                {
                                    "type": "text",
                                    "text": "State which color is on the left and which is on the right. "
                                    "Then write a detailed 40-item numbered explanation of the two "
                                    "colors in this image. Be explicit about red and blue.",
                                },
                            ],
                        }
                    ],
                }
                path = "/v1/chat/completions"
            else:
                # Different prefixes and ragged input lengths exercise partial DCP pages.
                facts = (
                    f"Record {case}: a copper key opens the blue door. "
                    "The red door stays closed. "
                ) * (32 + case * 11)
                payload = {
                    "rid": rid,
                    "text": facts
                    + "\nWrite 80 numbered sentences explaining these records.",
                    "return_logprob": True,
                    "logprob_start_len": -1,
                    "sampling_params": {
                        "temperature": 0,
                        "sampling_seed": case,
                        "ignore_eos": True,
                        "max_new_tokens": args.max_new_tokens,
                    },
                }
                path = "/generate"
            result = await fetch(session, args.url + path, payload, args.timeout)
            body = result.pop("body", None)
            result.update(
                {
                    "key": key,
                    "rid": rid,
                    "multimodal": mm,
                    **summary(body, mm),
                }
            )
            if result["status"] != 200:
                result["failure_body"] = body
            report["results"].append(result)
            print(
                json.dumps(
                    {
                        "event": "complete",
                        "key": key,
                        "status": result["status"],
                        "seconds": round(result["seconds"], 2),
                        "retractions": result.get("meta_info", {}).get(
                            "num_retractions"
                        ),
                        "accept_length": result.get("meta_info", {}).get(
                            "spec_accept_length"
                        ),
                    }
                ),
                flush=True,
            )

        await snapshot("before")
        for wave in range(args.waves):
            pending = {asyncio.create_task(request(wave, case)) for case in range(6)}
            while pending:
                completed, pending = await asyncio.wait(pending, timeout=20)
                for task in completed:
                    task.result()
                if pending:
                    print(
                        json.dumps(
                            {"event": "progress", "wave": wave, "pending": len(pending)}
                        ),
                        flush=True,
                    )
            await snapshot(f"after-wave-{wave}")
        health = await fetch(session, args.url + "/health", timeout=10)
        report["final_health"] = health
    errors = []
    for result in report["results"]:
        if result["status"] != 200:
            errors.append(f"{result['key']}: HTTP {result['status']}")
        elif not result.get("text"):
            errors.append(f"{result['key']}: empty output")
        elif result["multimodal"] and not result.get("mentions_red_and_blue"):
            errors.append(f"{result['key']}: image color check failed")
        elif not result["multimodal"] and not result.get("token_ids"):
            errors.append(f"{result['key']}: no output token IDs")
        elif not result["multimodal"]:
            meta = result.get("meta_info", {})
            if meta.get("completion_tokens") != args.max_new_tokens:
                errors.append(f"{result['key']}: unexpected generated-token count")
            if (meta.get("finish_reason") or {}).get("type") == "abort":
                errors.append(f"{result['key']}: engine aborted generation")
    metas = [r.get("meta_info", {}) for r in report["results"] if not r["multimodal"]]
    retractions = sum(m.get("num_retractions", 0) for m in metas)
    verifies = sum(m.get("spec_verify_ct", 0) for m in metas)
    proposed = sum(m.get("spec_num_proposed_drafts", 0) for m in metas)
    accepted = sum(m.get("spec_num_correct_drafts", 0) for m in metas)
    tokens = sum(m.get("completion_tokens", 0) for m in metas)
    report["aggregate"] = {
        "native_retractions": retractions,
        "native_verify_calls": verifies,
        "native_completion_tokens": tokens,
        "accept_length": tokens / verifies if verifies else None,
        "accept_rate": accepted / proposed if proposed else None,
    }
    if args.require_retractions and not retractions:
        errors.append(
            "No per-request forced retractions observed; qualification inconclusive."
        )
    if not verifies:
        errors.append("No per-request draft verification counts observed.")
    if report["final_health"]["status"] != 200:
        errors.append("Frontend unhealthy after final wave.")
    if args.compare:
        baseline = json.loads(Path(args.compare).read_text())
        reference = {r["key"]: r for r in baseline["results"]}
        comparisons = []
        for result in report["results"]:
            ref = reference.get(result["key"], {})
            same = (
                result.get("token_ids") == ref.get("token_ids")
                if not result["multimodal"]
                else result.get("text_sha256") == ref.get("text_sha256")
            )
            comparisons.append({"key": result["key"], "exact_match": same})
            if not same:
                errors.append(f"{result['key']}: greedy output differs from baseline")
        report["comparisons"] = comparisons
        report["baseline_aggregate"] = baseline.get("aggregate")
    report["errors"] = errors
    report["passed"] = not errors
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "event": "summary",
                "passed": report["passed"],
                "aggregate": report["aggregate"],
                "errors": errors,
                "artifact": args.output,
            }
        ),
        flush=True,
    )
    return 0 if report["passed"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--decode-url")
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--compare")
    parser.add_argument("--require-retractions", action="store_true")
    parser.add_argument("--waves", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()
    if not 1 <= args.waves <= 5 or not 128 <= args.max_new_tokens <= 1024:
        parser.error("Keep qualification bounded: waves 1..5, max-new-tokens 128..1024")
    args.url = args.url.rstrip("/")
    if args.decode_url:
        args.decode_url = args.decode_url.rstrip("/")
    raise SystemExit(asyncio.run(run(args)))


if __name__ == "__main__":
    main()
