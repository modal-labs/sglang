#!/usr/bin/env python3
"""Known-answer text/image QA under repeated PD retraction.

Run on an isolated normal and forced-retraction pair. Schema constrains structure,
never the answer values. Outputs remain long enough to cross forced boundaries.
Uses only synthetic bodies; no mirrored request storage or runtime mutation.
"""

import argparse
import asyncio
import json
import re
from pathlib import Path

import aiohttp

from run_pd_dflash_retraction_qualification import fetch, image_url

_SEQUENCE = list(range(1, 97))


def make_case(case, model, rid):
    if case < 2:
        operands = ((173, 269), (187, 356))[case]
        prompt = f"Compute {operands[0]} + {operands[1]}. Put the decimal result as a string in answer."
        expected = {"answer": str(sum(operands))}
        properties = {"answer": {"type": "string"}}
        content = prompt
    elif case < 4:
        code = ("LARCH742", "COBALT319")[case - 2]
        prompt = (
            "This is an unrelated audit record; ignore the filler. " * (47 + case)
        ) + f"\nThe unique audit seal is {code}. Copy that seal verbatim into answer."
        expected = {"answer": code}
        properties = {"answer": {"type": "string"}}
        content = prompt
    else:
        prompt = "Inspect the image. Put the left half color in left_color and right half color in right_color, using lowercase English color names."
        expected = {"left_color": "red", "right_color": "blue"}
        properties = {key: {"type": "string"} for key in expected}
        content = [
            {"type": "image_url", "image_url": {"url": image_url()}},
            {"type": "text", "text": prompt},
        ]
    suffix = " Also put every integer from 1 through 96 inclusive, in order, in sequence. Return only the JSON object."
    if isinstance(content, str):
        content += suffix
    else:
        content[-1]["text"] += suffix
    properties["sequence"] = {
        "type": "array",
        "items": {"type": "integer"},
        "minItems": len(_SEQUENCE),
        "maxItems": len(_SEQUENCE),
    }
    payload = {
        "model": model,
        "rid": rid,
        "temperature": 0,
        "seed": case,
        "reasoning_effort": "low",
        "max_tokens": 2048,
        "messages": [{"role": "user", "content": content}],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "retraction_semantics",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": properties,
                    "required": list(properties),
                    "additionalProperties": False,
                },
            },
        },
    }
    return payload, expected


def check_answer(text, expected):
    try:
        value = json.loads(text)
    except (TypeError, ValueError):
        return False, "invalid_json"
    if not isinstance(value, dict):
        return False, "not_an_object"
    if any(value.get(key) != answer for key, answer in expected.items()):
        return False, "incorrect_answer"
    if value.get("sequence") != _SEQUENCE:
        return False, "incorrect_sequence"
    return True, None


def retraction_counts(text):
    counts = {}
    for line in text.splitlines():
        if not line.startswith("sglang:num_retracted_requests_total{"):
            continue
        rank = re.search(r'tp_rank="([^"]+)"', line)
        counts[rank.group(1) if rank else "unknown"] = float(line.rsplit(" ", 1)[1])
    return counts


async def run(args):
    report = {"label": args.label, "results": [], "errors": []}
    async with aiohttp.ClientSession() as session:
        models = await fetch(session, args.url + "/v1/models", timeout=10)
        model = models["body"]["data"][0]["id"]
        before = await fetch(session, args.decode_url + "/metrics", timeout=10)
        report["retractions_before"] = retraction_counts(before.get("body", ""))
        for wave in range(args.waves):

            async def send(case):
                key = f"wave{wave}-case{case}"
                rid = f"retract-semantic-{args.label}-{key}"
                payload, expected = make_case(case, model, rid)
                result = await fetch(
                    session, args.url + "/v1/chat/completions", payload, 180
                )
                body = result.pop("body", {})
                choice = (
                    (body.get("choices") or [{}])[0] if isinstance(body, dict) else {}
                )
                text = (choice.get("message") or {}).get("content")
                good, reason = (
                    check_answer(text, expected)
                    if result["status"] == 200
                    else (False, "http_error")
                )
                if result["status"] != 200:
                    result["error_body"] = body
                result.update(
                    key=key,
                    rid=rid,
                    passed=good and result["status"] == 200,
                    reason=reason,
                    answer=text,
                    reasoning=(choice.get("message") or {}).get("reasoning_content"),
                    finish_reason=choice.get("finish_reason"),
                    usage=body.get("usage") if isinstance(body, dict) else None,
                )
                report["results"].append(result)
                print(
                    json.dumps(
                        {
                            key: result[key]
                            for key in ("key", "status", "passed", "reason", "seconds")
                        }
                    ),
                    flush=True,
                )

            await asyncio.gather(*(send(case) for case in range(6)))
        after = await fetch(session, args.decode_url + "/metrics", timeout=10)
        report["retractions_after"] = retraction_counts(after.get("body", ""))
        report["health"] = await fetch(session, args.url + "/health", timeout=10)
        report["loads_after"] = await fetch(
            session, args.decode_url + "/v1/loads?include=all", timeout=10
        )
    report["errors"] = [
        row["key"] + ":" + str(row["reason"])
        for row in report["results"]
        if not row["passed"]
    ]
    report["retraction_delta_by_rank"] = {
        rank: value - report["retractions_before"].get(rank, 0)
        for rank, value in report["retractions_after"].items()
    }
    if args.require_retractions and not any(
        report["retraction_delta_by_rank"].values()
    ):
        report["errors"].append("No forced retractions observed; inconclusive.")
    if report["health"]["status"] != 200:
        report["errors"].append("Frontend unhealthy after test.")
    report["passed"] = not report["errors"]
    Path(args.output).write_text(json.dumps(report, indent=2))
    print(
        json.dumps(
            {
                "event": "summary",
                "passed": report["passed"],
                "errors": report["errors"],
                "retractions": report["retraction_delta_by_rank"],
            }
        ),
        flush=True,
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--decode-url", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--waves", type=int, default=3)
    parser.add_argument("--require-retractions", action="store_true")
    raise SystemExit(asyncio.run(run(parser.parse_args())))
