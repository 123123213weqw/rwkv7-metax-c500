#!/usr/bin/env python3
"""Validate SGLang RWKV recurrent prefix-state caching on MetaX C500."""

from __future__ import annotations

import argparse
import gc
import json
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any


COMMON_PREFIX_LENGTH = 128
SHARED_REQUESTS = 4


def _tokens(length: int, seed: int) -> list[int]:
    return [((position * 47 + seed) % 65000) + 1 for position in range(length)]


def _prompts() -> tuple[list[int], list[list[int]]]:
    common = _tokens(COMMON_PREFIX_LENGTH, 37)
    suffix_lengths = (17, 19, 23, 29)
    requests = [
        common + _tokens(length, 1009 + request_index * 101)
        for request_index, length in enumerate(suffix_lengths)
    ]
    requests.append(_tokens(151, 7001))
    return common, requests


def _rows(value: Any) -> list[dict[str, Any]]:
    return value if isinstance(value, list) else [value]


def _write_json(path: Path, result: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


def _run_worker(args: argparse.Namespace) -> int:
    import torch
    import sglang as sgl

    cache_enabled = args.worker_mode == "cache"
    common, requests = _prompts()
    init_started = time.perf_counter()
    engine = sgl.Engine(
        model_path=args.model,
        dtype="float16",
        skip_tokenizer_init=True,
        attention_backend="triton",
        sampling_backend="pytorch",
        disable_cuda_graph=True,
        disable_piecewise_cuda_graph=True,
        disable_overlap_schedule=True,
        disable_radix_cache=not cache_enabled,
        mamba_scheduler_strategy="no_buffer",
        chunked_prefill_size=args.chunked_prefill_size,
        random_seed=333,
        max_running_requests=8,
        context_length=256,
        mem_fraction_static=0.7,
    )
    init_elapsed = time.perf_counter() - init_started
    try:
        one_token = {
            "temperature": 1.0,
            "top_k": 1,
            "max_new_tokens": 1,
            "ignore_eos": True,
        }
        engine.generate(
            input_ids=[_tokens(COMMON_PREFIX_LENGTH, 31001)],
            sampling_params=one_token,
        )
        engine.flush_cache()
        warm_started = time.perf_counter()
        warm_output = engine.generate(
            input_ids=[common],
            sampling_params=one_token,
        )
        warm_elapsed = time.perf_counter() - warm_started
        generate_started = time.perf_counter()
        output = engine.generate(
            input_ids=requests,
            sampling_params={
                "temperature": 1.0,
                "top_k": 1,
                "max_new_tokens": args.max_new_tokens,
                "ignore_eos": True,
            },
        )
        generate_elapsed = time.perf_counter() - generate_started
        free_memory, total_memory = torch.cuda.mem_get_info()
    finally:
        engine.shutdown()

    output_rows = _rows(output)
    metadata = [row.get("meta_info", {}) for row in output_rows]
    output_ids = [list(row["output_ids"]) for row in output_rows]
    complete = (
        len(_rows(warm_output)) == 1
        and len(_rows(warm_output)[0].get("output_ids", [])) == 1
        and len(output_rows) == len(requests)
        and all(len(value) == args.max_new_tokens for value in output_ids)
    )
    worker_result = {
        "status": "pass" if complete else "fail",
        "mode": args.worker_mode,
        "cache_enabled": cache_enabled,
        "model": Path(args.model).name,
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "maca": getattr(torch.version, "maca", None),
        "python": platform.python_version(),
        "prompt_lengths": [len(request) for request in requests],
        "init_elapsed_s": init_elapsed,
        "warm_elapsed_s": warm_elapsed,
        "generate_elapsed_s": generate_elapsed,
        "request_e2e_latency_s": [
            meta.get("e2e_latency") for meta in metadata
        ],
        "gpu_memory_used_mib": (total_memory - free_memory) / 1024**2,
        "cached_tokens": [
            int(meta.get("cached_tokens") or 0) for meta in metadata
        ],
        "output_ids": output_ids,
    }
    _write_json(args.worker_output, worker_result)
    print(json.dumps(worker_result, sort_keys=True))
    del engine
    gc.collect()
    return 0 if worker_result["status"] == "pass" else 1


def _worker_command(
    args: argparse.Namespace,
    mode: str,
    worker_output: Path,
) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "--model",
        args.model,
        "--output",
        str(args.output),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--chunked-prefill-size",
        str(args.chunked_prefill_size),
        "--worker-mode",
        mode,
        "--worker-output",
        str(worker_output),
    ]


def _median(values: list[float | None]) -> float | None:
    concrete = [float(value) for value in values if value is not None]
    return statistics.median(concrete) if concrete else None


def _run_comparison(args: argparse.Namespace) -> int:
    with tempfile.TemporaryDirectory(prefix="rwkv7-sglang-state-cache-") as tmp:
        results: dict[str, dict[str, Any]] = {}
        for mode in ("baseline", "cache"):
            worker_output = Path(tmp) / f"{mode}.json"
            subprocess.run(
                _worker_command(args, mode, worker_output),
                check=True,
            )
            results[mode] = json.loads(
                worker_output.read_text(encoding="utf-8")
            )

    baseline = results["baseline"]
    cache = results["cache"]
    cached_tokens = cache["cached_tokens"]
    shared_hits = [value >= COMMON_PREFIX_LENGTH for value in cached_tokens[:-1]]
    control_miss = cached_tokens[-1] == 0
    request_hit_rate = sum(value > 0 for value in cached_tokens) / len(cached_tokens)
    token_hit_rate = sum(cached_tokens) / sum(cache["prompt_lengths"])
    greedy_output_match = baseline["output_ids"] == cache["output_ids"]
    baseline_latency = _median(baseline["request_e2e_latency_s"][:-1])
    cache_latency = _median(cache["request_e2e_latency_s"][:-1])
    latency_ratio = (
        cache_latency / baseline_latency
        if baseline_latency is not None
        and cache_latency is not None
        and baseline_latency > 0
        else None
    )
    passed = (
        baseline["status"] == "pass"
        and cache["status"] == "pass"
        and greedy_output_match
        and all(shared_hits)
        and control_miss
        and request_hit_rate >= 0.8
    )
    result = {
        "schema_version": 1,
        "status": "pass" if passed else "fail",
        "track": "sglang",
        "public_api": "sglang.Engine",
        "gate": "rwkv_recurrent_prefix_state_cache",
        "model": cache["model"],
        "gpu": cache["gpu"],
        "torch": cache["torch"],
        "maca": cache["maca"],
        "python": cache["python"],
        "common_prefix_length": COMMON_PREFIX_LENGTH,
        "shared_request_count": SHARED_REQUESTS,
        "control_request_count": 1,
        "prompt_lengths": cache["prompt_lengths"],
        "max_new_tokens": args.max_new_tokens,
        "chunked_prefill_size": args.chunked_prefill_size,
        "cached_tokens": cached_tokens,
        "shared_prefix_hits": shared_hits,
        "control_cache_miss": control_miss,
        "request_hit_rate": request_hit_rate,
        "token_hit_rate": token_hit_rate,
        "greedy_output_match": greedy_output_match,
        "baseline_generate_elapsed_s": baseline["generate_elapsed_s"],
        "cache_generate_elapsed_s": cache["generate_elapsed_s"],
        "baseline_shared_e2e_latency_median_s": baseline_latency,
        "cache_shared_e2e_latency_median_s": cache_latency,
        "cache_to_baseline_shared_e2e_latency_ratio": latency_ratio,
        "baseline_gpu_memory_used_mib": baseline["gpu_memory_used_mib"],
        "cache_gpu_memory_used_mib": cache["gpu_memory_used_mib"],
        "baseline_output_ids": baseline["output_ids"],
        "output_ids": cache["output_ids"],
    }
    _write_json(args.output, result)
    print(json.dumps(result, sort_keys=True))
    return 0 if passed else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--chunked-prefill-size", type=int, default=64)
    parser.add_argument("--worker-mode", choices=("baseline", "cache"))
    parser.add_argument("--worker-output", type=Path)
    args = parser.parse_args()
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be positive")
    if args.chunked_prefill_size < 1:
        parser.error("--chunked-prefill-size must be positive")
    if args.worker_mode:
        if args.worker_output is None:
            parser.error("--worker-output is required in worker mode")
        return _run_worker(args)
    if args.worker_output is not None:
        parser.error("--worker-output is private to worker mode")
    return _run_comparison(args)


if __name__ == "__main__":
    raise SystemExit(main())
