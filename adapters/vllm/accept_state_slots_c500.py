#!/usr/bin/env python3
"""Validate public vLLM per-request RWKV state isolation on MetaX C500."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path


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


def _generate(llm, prompts: list[dict[str, list[int]]], max_new_tokens: int):
    from vllm import SamplingParams

    return llm.generate(
        prompts,
        SamplingParams(
            temperature=1.0,
            top_k=1,
            max_tokens=max_new_tokens,
            ignore_eos=True,
        ),
    )


def _token_rows(outputs) -> list[list[int]]:
    return [list(row.outputs[0].token_ids) for row in outputs]


def _run_worker(args: argparse.Namespace) -> int:
    import torch
    from vllm import LLM

    prompts = _prompt_map()
    torch.cuda.reset_peak_memory_stats()
    init_started = time.perf_counter()
    llm = LLM(
        model=args.model,
        dtype="float16",
        load_format="auto",
        enforce_eager=True,
        max_model_len=128,
        max_num_seqs=8,
        max_num_batched_tokens=128,
        gpu_memory_utilization=0.85,
    )
    init_elapsed = time.perf_counter() - init_started

    generate_started = time.perf_counter()
    if args.worker_mode == "solo":
        outputs = {
            name: _token_rows(_generate(llm, [prompt], args.max_new_tokens))[0]
            for name, prompt in prompts.items()
        }
        schedules: dict[str, dict[str, list[int]]] = {"solo": outputs}
    else:
        names = list(prompts)
        forward_rows = _token_rows(
            _generate(llm, [prompts[name] for name in names], args.max_new_tokens)
        )
        reverse_names = list(reversed(names))
        reverse_rows = _token_rows(
            _generate(
                llm,
                [prompts[name] for name in reverse_names],
                args.max_new_tokens,
            )
        )
        schedules = {
            "batch_ab": dict(zip(names, forward_rows, strict=True)),
            "batch_ba": dict(zip(reverse_names, reverse_rows, strict=True)),
        }
    torch.cuda.synchronize()
    generate_elapsed = time.perf_counter() - generate_started

    cmix_nofc_disabled = os.environ.get("VLLM_RWKV7_DISABLE_CMIX_NOFC") == "1"
    complete = all(
        len(token_ids) == args.max_new_tokens
        for schedule in schedules.values()
        for token_ids in schedule.values()
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
        "prompt_lengths": {
            name: len(prompt["prompt_token_ids"]) for name, prompt in prompts.items()
        },
        "max_new_tokens": args.max_new_tokens,
        "init_elapsed_s": init_elapsed,
        "generate_elapsed_s": generate_elapsed,
        "peak_memory_mib": torch.cuda.max_memory_allocated() / 1024**2,
        "rwkv7_cmix_nofc_disabled": cmix_nofc_disabled,
        "schedules": schedules,
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
        "--worker-mode",
        mode,
        "--worker-output",
        str(worker_output),
    ]


def _run_comparison(args: argparse.Namespace) -> int:
    with tempfile.TemporaryDirectory(prefix="rwkv7-vllm-slots-") as temp_dir:
        temp = Path(temp_dir)
        results: dict[str, dict[str, object]] = {}
        for mode in ("solo", "batched"):
            worker_output = temp / f"{mode}.json"
            subprocess.run(_worker_command(args, mode, worker_output), check=True)
            results[mode] = json.loads(worker_output.read_text(encoding="utf-8"))

    solo = results["solo"]
    batched = results["batched"]
    solo_outputs = solo["schedules"]["solo"]
    batch_ab_outputs = batched["schedules"]["batch_ab"]
    batch_ba_outputs = batched["schedules"]["batch_ba"]
    forward_match = solo_outputs == batch_ab_outputs
    reverse_match = solo_outputs == batch_ba_outputs
    passed = (
        solo["status"] == "pass"
        and batched["status"] == "pass"
        and forward_match
        and reverse_match
    )
    result: dict[str, object] = {
        "schema_version": 1,
        "status": "pass" if passed else "fail",
        "track": "vllm",
        "public_api": "vllm.LLM",
        "gate": "per_request_state_slot_isolation",
        "model": Path(args.model).name,
        "gpu": batched["gpu"],
        "torch": batched["torch"],
        "maca": batched["maca"],
        "python": batched["python"],
        "prompt_lengths": batched["prompt_lengths"],
        "max_new_tokens": args.max_new_tokens,
        "solo_vs_batch_ab_match": forward_match,
        "solo_vs_batch_ba_match": reverse_match,
        "post_completion_slot_reuse_smoke": reverse_match,
        "rwkv7_cmix_nofc_disabled": batched["rwkv7_cmix_nofc_disabled"],
        "solo_init_elapsed_s": solo["init_elapsed_s"],
        "solo_generate_elapsed_s": solo["generate_elapsed_s"],
        "solo_peak_memory_mib": solo["peak_memory_mib"],
        "batched_init_elapsed_s": batched["init_elapsed_s"],
        "batched_generate_elapsed_s": batched["generate_elapsed_s"],
        "batched_peak_memory_mib": batched["peak_memory_mib"],
        "output_ids_by_request": solo_outputs,
    }
    _write_json(args.output, result)
    print(json.dumps(result, sort_keys=True))
    return 0 if passed else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--worker-mode", choices=("solo", "batched"))
    parser.add_argument("--worker-output", type=Path)
    args = parser.parse_args()

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
