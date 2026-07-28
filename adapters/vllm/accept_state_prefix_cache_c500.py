#!/usr/bin/env python3
"""Validate RWKV recurrent prefix-state caching on MetaX C500."""

from __future__ import annotations

import argparse
import json
import os
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


def _prompts() -> tuple[dict[str, list[int]], list[dict[str, list[int]]]]:
    common = _tokens(COMMON_PREFIX_LENGTH, 37)
    warmup = {"prompt_token_ids": common + _tokens(9, 401)}
    shared_suffix_lengths = (17, 19, 23, 29)
    requests = [
        {
            "prompt_token_ids": common
            + _tokens(suffix_len, 1009 + request_index * 101)
        }
        for request_index, suffix_len in enumerate(shared_suffix_lengths)
    ]
    requests.append({"prompt_token_ids": _tokens(151, 7001)})
    return warmup, requests


def _write_json(path: Path, result: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


def _first_token_latency(output: Any) -> float | None:
    metrics = getattr(output, "metrics", None)
    value = getattr(metrics, "first_token_latency", None)
    return float(value) if value is not None else None


def _run_worker(args: argparse.Namespace) -> int:
    import torch
    from vllm import LLM, SamplingParams

    cache_enabled = args.worker_mode == "cache"
    warmup, prompts = _prompts()
    sampling_params = SamplingParams(
        temperature=1.0,
        top_k=1,
        max_tokens=args.max_new_tokens,
        ignore_eos=True,
    )
    engine_kwargs: dict[str, Any] = {
        "model": args.model,
        "dtype": "float16",
        "load_format": "auto",
        "enforce_eager": True,
        "enable_chunked_prefill": True,
        "enable_prefix_caching": cache_enabled,
        "max_model_len": 256,
        "max_num_seqs": 8,
        "max_num_batched_tokens": 512,
        "gpu_memory_utilization": 0.85,
        "disable_log_stats": False,
    }
    if cache_enabled:
        engine_kwargs["additional_config"] = {
            "rwkv_state_prefix_cache_capacity": args.cache_capacity,
            "rwkv_state_prefix_cache_block_size": args.cache_block_size,
        }

    torch.cuda.reset_peak_memory_stats()
    init_started = time.perf_counter()
    llm = LLM(**engine_kwargs)
    init_elapsed = time.perf_counter() - init_started

    llm.generate(
        warmup,
        SamplingParams(
            temperature=1.0,
            top_k=1,
            max_tokens=1,
            ignore_eos=True,
        ),
        use_tqdm=False,
    )
    torch.cuda.synchronize()

    generate_started = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
    torch.cuda.synchronize()
    generate_elapsed = time.perf_counter() - generate_started

    output_ids = [list(row.outputs[0].token_ids) for row in outputs]
    cached_tokens = [int(row.num_cached_tokens or 0) for row in outputs]
    first_token_latencies = [_first_token_latency(row) for row in outputs]
    complete = len(outputs) == len(prompts) and all(
        len(token_ids) == args.max_new_tokens for token_ids in output_ids
    )
    cmix_nofc_disabled = os.environ.get("VLLM_RWKV7_DISABLE_CMIX_NOFC") == "1"
    worker_result: dict[str, Any] = {
        "status": "pass" if complete and cmix_nofc_disabled else "fail",
        "mode": args.worker_mode,
        "model": Path(args.model).name,
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "maca": getattr(torch.version, "maca", None),
        "python": platform.python_version(),
        "cache_enabled": cache_enabled,
        "cache_capacity": args.cache_capacity if cache_enabled else None,
        "cache_block_size": args.cache_block_size if cache_enabled else None,
        "prompt_lengths": [len(row["prompt_token_ids"]) for row in prompts],
        "init_elapsed_s": init_elapsed,
        "generate_elapsed_s": generate_elapsed,
        "peak_memory_mib": torch.cuda.max_memory_allocated() / 1024**2,
        "cached_tokens": cached_tokens,
        "first_token_latency_s": first_token_latencies,
        "output_ids": output_ids,
        "rwkv7_cmix_nofc_disabled": cmix_nofc_disabled,
    }
    _write_json(args.worker_output, worker_result)
    print(json.dumps(worker_result, sort_keys=True))
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
        "--cache-capacity",
        str(args.cache_capacity),
        "--cache-block-size",
        str(args.cache_block_size),
        "--worker-mode",
        mode,
        "--worker-output",
        str(worker_output),
    ]


def _median(values: list[float | None]) -> float | None:
    concrete = [value for value in values if value is not None]
    return statistics.median(concrete) if concrete else None


def _run_comparison(args: argparse.Namespace) -> int:
    with tempfile.TemporaryDirectory(prefix="rwkv7-vllm-state-cache-") as temp_dir:
        temp = Path(temp_dir)
        results: dict[str, dict[str, Any]] = {}
        for mode in ("baseline", "cache"):
            worker_output = temp / f"{mode}.json"
            subprocess.run(_worker_command(args, mode, worker_output), check=True)
            results[mode] = json.loads(worker_output.read_text(encoding="utf-8"))

    baseline = results["baseline"]
    cache = results["cache"]
    cached_tokens = cache["cached_tokens"]
    shared_hits = [value >= COMMON_PREFIX_LENGTH for value in cached_tokens[:-1]]
    control_miss = cached_tokens[-1] == 0
    request_hits = sum(value > 0 for value in cached_tokens)
    request_hit_rate = request_hits / len(cached_tokens)
    total_prompt_tokens = sum(cache["prompt_lengths"])
    token_hit_rate = sum(cached_tokens) / total_prompt_tokens
    greedy_output_match = baseline["output_ids"] == cache["output_ids"]
    baseline_shared_ttft = _median(baseline["first_token_latency_s"][:-1])
    cache_shared_ttft = _median(cache["first_token_latency_s"][:-1])
    ttft_ratio = (
        cache_shared_ttft / baseline_shared_ttft
        if cache_shared_ttft is not None
        and baseline_shared_ttft is not None
        and baseline_shared_ttft > 0
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

    result: dict[str, Any] = {
        "schema_version": 1,
        "status": "pass" if passed else "fail",
        "track": "vllm",
        "public_api": "vllm.LLM",
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
        "cache_capacity": args.cache_capacity,
        "cache_block_size": args.cache_block_size,
        "cached_tokens": cached_tokens,
        "shared_prefix_hits": shared_hits,
        "control_cache_miss": control_miss,
        "request_hit_rate": request_hit_rate,
        "token_hit_rate": token_hit_rate,
        "greedy_output_match": greedy_output_match,
        "baseline_generate_elapsed_s": baseline["generate_elapsed_s"],
        "cache_generate_elapsed_s": cache["generate_elapsed_s"],
        "baseline_shared_ttft_median_s": baseline_shared_ttft,
        "cache_shared_ttft_median_s": cache_shared_ttft,
        "cache_to_baseline_shared_ttft_ratio": ttft_ratio,
        "baseline_peak_memory_mib": baseline["peak_memory_mib"],
        "cache_peak_memory_mib": cache["peak_memory_mib"],
        "rwkv7_cmix_nofc_disabled": cache["rwkv7_cmix_nofc_disabled"],
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
    parser.add_argument("--cache-capacity", type=int, default=16)
    parser.add_argument("--cache-block-size", type=int, default=64)
    parser.add_argument("--worker-mode", choices=("baseline", "cache"))
    parser.add_argument("--worker-output", type=Path)
    args = parser.parse_args()

    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be positive")
    if args.cache_capacity < 3:
        parser.error("--cache-capacity must hold the warmup boundaries")
    if args.cache_block_size <= 0:
        parser.error("--cache-block-size must be positive")
    if args.worker_mode:
        if args.worker_output is None:
            parser.error("--worker-output is required in worker mode")
        return _run_worker(args)
    if args.worker_output is not None:
        parser.error("--worker-output is private to worker mode")
    return _run_comparison(args)


if __name__ == "__main__":
    raise SystemExit(main())
