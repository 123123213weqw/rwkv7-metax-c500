#!/usr/bin/env python3
"""Validate continuously arriving public AsyncLLM requests on MetaX C500."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import platform
import time
from pathlib import Path
from typing import Any


PROMPT_SPECS = (("request_a", 33, 19), ("request_b", 47, 211))


def _prompt_map() -> dict[str, dict[str, list[int]]]:
    return {
        name: {
            "prompt_token_ids": [
                ((position * 43 + seed) % 65000) + 1 for position in range(length)
            ]
        }
        for name, length, seed in PROMPT_SPECS
    }


def _write_json(path: Path, result: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


async def _collect_request(
    engine: Any,
    prompt: dict[str, list[int]],
    sampling_params: Any,
    request_id: str,
    first_token_ready: asyncio.Event | None = None,
) -> dict[str, object]:
    started_at = time.perf_counter()
    first_event_at: float | None = None
    first_event_finished: bool | None = None
    finished_at: float | None = None
    event_count = 0
    token_ids: list[int] = []

    async for output in engine.generate(
        prompt=prompt,
        sampling_params=sampling_params,
        request_id=request_id,
    ):
        event_count += 1
        now = time.perf_counter()
        if output.outputs:
            token_ids = list(output.outputs[0].token_ids)
        if token_ids and first_event_at is None:
            first_event_at = now
            first_event_finished = bool(output.finished)
            if first_token_ready is not None:
                first_token_ready.set()
        if output.finished:
            finished_at = now

    if finished_at is None:
        finished_at = time.perf_counter()
    return {
        "request_id": request_id,
        "token_ids": token_ids,
        "event_count": event_count,
        "started_at": started_at,
        "first_event_at": first_event_at,
        "first_event_finished": first_event_finished,
        "finished_at": finished_at,
    }


def _evaluate_dynamic_batch(
    references: dict[str, list[int]],
    dynamic: dict[str, dict[str, object]],
    request_b_submitted_at: float,
) -> dict[str, bool]:
    request_a = dynamic["request_a"]
    request_b = dynamic["request_b"]
    a_first = request_a["first_event_at"]
    a_finished = request_a["finished_at"]
    b_first = request_b["first_event_at"]

    return {
        "greedy_output_match": all(
            dynamic[name]["token_ids"] == token_ids
            for name, token_ids in references.items()
        ),
        "request_b_submitted_while_a_active": bool(
            a_first is not None
            and a_finished is not None
            and a_first <= request_b_submitted_at < a_finished
            and request_a["first_event_finished"] is False
        ),
        "request_b_scheduled_before_a_finished": bool(
            b_first is not None
            and a_finished is not None
            and request_b_submitted_at <= b_first < a_finished
        ),
    }


async def _run(args: argparse.Namespace) -> int:
    import torch
    from vllm import AsyncEngineArgs, SamplingParams
    from vllm.sampling_params import RequestOutputKind
    from vllm.v1.engine.async_llm import AsyncLLM

    prompts = _prompt_map()
    sampling_params = SamplingParams(
        temperature=1.0,
        top_k=1,
        max_tokens=args.max_new_tokens,
        ignore_eos=True,
        output_kind=RequestOutputKind.CUMULATIVE,
    )
    engine_args = AsyncEngineArgs(
        model=args.model,
        dtype="float16",
        load_format="auto",
        enforce_eager=True,
        max_model_len=128,
        max_num_seqs=8,
        max_num_batched_tokens=128,
        gpu_memory_utilization=0.85,
        async_scheduling=True,
        stream_interval=1,
        disable_log_stats=True,
        enable_log_requests=False,
    )

    init_started = time.perf_counter()
    engine = AsyncLLM.from_engine_args(engine_args)
    init_elapsed = time.perf_counter() - init_started
    try:
        references: dict[str, list[int]] = {}
        reference_elapsed: dict[str, float] = {}
        for name, prompt in prompts.items():
            reference = await _collect_request(
                engine,
                prompt,
                sampling_params,
                request_id=f"reference-{name}",
            )
            references[name] = reference["token_ids"]
            reference_elapsed[name] = (
                reference["finished_at"] - reference["started_at"]
            )

        dynamic_started_at = time.perf_counter()
        first_token_ready = asyncio.Event()
        request_a_task = asyncio.create_task(
            _collect_request(
                engine,
                prompts["request_a"],
                sampling_params,
                request_id="dynamic-request-a",
                first_token_ready=first_token_ready,
            )
        )
        await asyncio.wait_for(
            first_token_ready.wait(), timeout=args.first_token_timeout_s
        )
        request_b_submitted_at = time.perf_counter()
        request_b_task = asyncio.create_task(
            _collect_request(
                engine,
                prompts["request_b"],
                sampling_params,
                request_id="dynamic-request-b",
            )
        )
        request_a, request_b = await asyncio.gather(
            request_a_task,
            request_b_task,
        )
        dynamic = {"request_a": request_a, "request_b": request_b}
        checks = _evaluate_dynamic_batch(
            references,
            dynamic,
            request_b_submitted_at,
        )
        complete = all(
            len(row["token_ids"]) == args.max_new_tokens
            for row in dynamic.values()
        )
        cmix_nofc_disabled = (
            os.environ.get("VLLM_RWKV7_DISABLE_CMIX_NOFC") == "1"
        )
        worker_multiproc_method = os.environ.get("VLLM_WORKER_MULTIPROC_METHOD")
        c500_spawn_enabled = worker_multiproc_method == "spawn"
        passed = (
            complete
            and cmix_nofc_disabled
            and c500_spawn_enabled
            and all(checks.values())
        )

        def relative(timestamp: object) -> float | None:
            if timestamp is None:
                return None
            return float(timestamp) - dynamic_started_at

        result: dict[str, object] = {
            "schema_version": 1,
            "status": "pass" if passed else "fail",
            "track": "vllm",
            "public_api": "vllm.v1.engine.async_llm.AsyncLLM",
            "gate": "continuous_arrival_dynamic_batch",
            "model": Path(args.model).name,
            "gpu": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "maca": getattr(torch.version, "maca", None),
            "python": platform.python_version(),
            "prompt_lengths": {
                name: len(prompt["prompt_token_ids"])
                for name, prompt in prompts.items()
            },
            "max_new_tokens": args.max_new_tokens,
            "scheduler": {
                "async_scheduling": True,
                "stream_interval": 1,
                "max_num_batched_tokens": 128,
                "max_num_seqs": 8,
            },
            **checks,
            "complete_output_lengths": complete,
            "rwkv7_cmix_nofc_disabled": cmix_nofc_disabled,
            "worker_multiproc_method": worker_multiproc_method,
            "c500_spawn_enabled": c500_spawn_enabled,
            "init_elapsed_s": init_elapsed,
            "reference_elapsed_s": reference_elapsed,
            "dynamic_elapsed_s": max(
                request_a["finished_at"], request_b["finished_at"]
            )
            - dynamic_started_at,
            "timeline_s": {
                "request_a_started": relative(request_a["started_at"]),
                "request_a_first_event": relative(request_a["first_event_at"]),
                "request_b_submitted": relative(request_b_submitted_at),
                "request_b_started": relative(request_b["started_at"]),
                "request_b_first_event": relative(request_b["first_event_at"]),
                "request_a_finished": relative(request_a["finished_at"]),
                "request_b_finished": relative(request_b["finished_at"]),
            },
            "stream_event_counts": {
                name: row["event_count"] for name, row in dynamic.items()
            },
            "output_ids_by_request": references,
        }
        _write_json(args.output, result)
        print(json.dumps(result, sort_keys=True))
        return 0 if passed else 1
    finally:
        engine.shutdown()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--first-token-timeout-s", type=float, default=120.0)
    args = parser.parse_args()

    if args.max_new_tokens < 2:
        parser.error("--max-new-tokens must be at least two")
    if args.first_token_timeout_s <= 0:
        parser.error("--first-token-timeout-s must be positive")
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
