#!/usr/bin/env python3
"""Compare forced chunked prefill with an unchunked public vLLM baseline."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path


PROMPT_LENGTHS = (129, 193)


def _prompts() -> list[dict[str, list[int]]]:
    return [
        {
            "prompt_token_ids": [
                ((position * 31 + seed) % 65000) + 1 for position in range(length)
            ]
        }
        for seed, length in zip((17, 101), PROMPT_LENGTHS, strict=True)
    ]


def _write_json(path: Path, result: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


def _run_worker(args: argparse.Namespace) -> int:
    import torch
    from vllm import LLM, SamplingParams

    budget = args.baseline_budget if args.worker_mode == "baseline" else args.chunk_size
    prompts = _prompts()
    torch.cuda.reset_peak_memory_stats()
    init_started = time.perf_counter()
    llm = LLM(
        model=args.model,
        dtype="float16",
        load_format="auto",
        enforce_eager=True,
        enable_chunked_prefill=True,
        max_model_len=256,
        max_num_seqs=8,
        max_num_batched_tokens=budget,
        gpu_memory_utilization=0.85,
    )
    init_elapsed = time.perf_counter() - init_started

    generate_started = time.perf_counter()
    outputs = llm.generate(
        prompts,
        SamplingParams(
            temperature=1.0,
            top_k=1,
            max_tokens=args.max_new_tokens,
            ignore_eos=True,
        ),
    )
    torch.cuda.synchronize()
    generate_elapsed = time.perf_counter() - generate_started

    output_ids = [list(row.outputs[0].token_ids) for row in outputs]
    cmix_nofc_disabled = os.environ.get("VLLM_RWKV7_DISABLE_CMIX_NOFC") == "1"
    complete = len(output_ids) == len(prompts) and all(
        len(token_ids) == args.max_new_tokens for token_ids in output_ids
    )
    result: dict[str, object] = {
        "schema_version": 1,
        "status": "pass" if complete and cmix_nofc_disabled else "fail",
        "track": "vllm",
        "public_api": "vllm.LLM",
        "mode": args.worker_mode,
        "model": Path(args.model).name,
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "maca": getattr(torch.version, "maca", None),
        "python": platform.python_version(),
        "batch_size": len(prompts),
        "prompt_lengths": list(PROMPT_LENGTHS),
        "max_new_tokens": args.max_new_tokens,
        "enable_chunked_prefill": True,
        "max_num_batched_tokens": budget,
        "forced_chunk_boundary": max(PROMPT_LENGTHS) > budget,
        "minimum_prefill_steps_from_budget": math.ceil(sum(PROMPT_LENGTHS) / budget),
        "init_elapsed_s": init_elapsed,
        "generate_elapsed_s": generate_elapsed,
        "peak_memory_mib": torch.cuda.max_memory_allocated() / 1024**2,
        "rwkv7_cmix_nofc_disabled": cmix_nofc_disabled,
        "output_ids": output_ids,
    }
    _write_json(args.worker_output, result)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] == "pass" else 1


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
        "--baseline-budget",
        str(args.baseline_budget),
        "--chunk-size",
        str(args.chunk_size),
        "--worker-mode",
        mode,
        "--worker-output",
        str(worker_output),
    ]


def _run_comparison(args: argparse.Namespace) -> int:
    with tempfile.TemporaryDirectory(prefix="rwkv7-vllm-chunk-") as temp_dir:
        temp = Path(temp_dir)
        results: dict[str, dict[str, object]] = {}
        for mode in ("baseline", "chunked"):
            worker_output = temp / f"{mode}.json"
            subprocess.run(
                _worker_command(args, mode, worker_output),
                check=True,
            )
            results[mode] = json.loads(worker_output.read_text(encoding="utf-8"))

    baseline = results["baseline"]
    chunked = results["chunked"]
    outputs_match = baseline["output_ids"] == chunked["output_ids"]
    passed = (
        baseline["status"] == "pass"
        and chunked["status"] == "pass"
        and not baseline["forced_chunk_boundary"]
        and chunked["forced_chunk_boundary"]
        and outputs_match
    )
    result: dict[str, object] = {
        "schema_version": 1,
        "status": "pass" if passed else "fail",
        "track": "vllm",
        "public_api": "vllm.LLM",
        "gate": "chunked_prefill_equivalence",
        "model": Path(args.model).name,
        "gpu": chunked["gpu"],
        "torch": chunked["torch"],
        "maca": chunked["maca"],
        "python": chunked["python"],
        "batch_size": chunked["batch_size"],
        "prompt_lengths": chunked["prompt_lengths"],
        "max_new_tokens": args.max_new_tokens,
        "baseline_max_num_batched_tokens": args.baseline_budget,
        "chunked_max_num_batched_tokens": args.chunk_size,
        "forced_chunk_boundary": chunked["forced_chunk_boundary"],
        "minimum_chunked_prefill_steps_from_budget": chunked[
            "minimum_prefill_steps_from_budget"
        ],
        "greedy_output_match": outputs_match,
        "rwkv7_cmix_nofc_disabled": chunked["rwkv7_cmix_nofc_disabled"],
        "baseline_init_elapsed_s": baseline["init_elapsed_s"],
        "baseline_generate_elapsed_s": baseline["generate_elapsed_s"],
        "baseline_peak_memory_mib": baseline["peak_memory_mib"],
        "chunked_init_elapsed_s": chunked["init_elapsed_s"],
        "chunked_generate_elapsed_s": chunked["generate_elapsed_s"],
        "chunked_peak_memory_mib": chunked["peak_memory_mib"],
        "output_ids": chunked["output_ids"],
    }
    _write_json(args.output, result)
    print(json.dumps(result, sort_keys=True))
    return 0 if passed else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--baseline-budget", type=int, default=512)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--worker-mode", choices=("baseline", "chunked"))
    parser.add_argument("--worker-output", type=Path)
    args = parser.parse_args()

    if args.baseline_budget < sum(PROMPT_LENGTHS):
        parser.error("--baseline-budget must fit both prompts in one scheduler step")
    if args.chunk_size >= max(PROMPT_LENGTHS):
        parser.error("--chunk-size must force at least one prompt across a boundary")
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be positive")
    if args.worker_mode:
        if args.worker_output is None:
            parser.error("--worker-output is required in worker mode")
        return _run_worker(args)
    if args.worker_output is not None:
        parser.error("--worker-output is private to worker mode")
    return _run_comparison(args)


if __name__ == "__main__":
    raise SystemExit(main())
