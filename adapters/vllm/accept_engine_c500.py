#!/usr/bin/env python3
"""Run the public vLLM LLM gate for RWKV-7 on MetaX C500."""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path

import torch


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-new-tokens", type=int, default=4)
    args = parser.parse_args()

    from vllm import LLM, SamplingParams

    prompts = [
        {"prompt_token_ids": [1, 2, 3, 4]},
        {"prompt_token_ids": [5, 6, 7, 8, 9]},
    ]
    torch.cuda.reset_peak_memory_stats()
    init_started = time.perf_counter()
    llm = LLM(
        model=args.model,
        dtype="float16",
        load_format="auto",
        enforce_eager=True,
        max_model_len=512,
        max_num_seqs=8,
        max_num_batched_tokens=512,
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
        ),
    )
    torch.cuda.synchronize()
    generate_elapsed = time.perf_counter() - generate_started

    output_ids = [list(row.outputs[0].token_ids) for row in outputs]
    cmix_nofc_disabled = os.environ.get("VLLM_RWKV7_DISABLE_CMIX_NOFC") == "1"
    passed = (
        cmix_nofc_disabled
        and len(output_ids) == len(prompts)
        and all(len(token_ids) == args.max_new_tokens for token_ids in output_ids)
    )
    result = {
        "schema_version": 1,
        "status": "pass" if passed else "fail",
        "track": "vllm",
        "public_api": "vllm.LLM",
        "model": Path(args.model).name,
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "maca": getattr(torch.version, "maca", None),
        "python": platform.python_version(),
        "batch_size": len(prompts),
        "prompt_lengths": [len(row["prompt_token_ids"]) for row in prompts],
        "max_new_tokens": args.max_new_tokens,
        "init_elapsed_s": init_elapsed,
        "generate_elapsed_s": generate_elapsed,
        "generated_tokens": sum(len(row) for row in output_ids),
        "peak_memory_mib": torch.cuda.max_memory_allocated() / 1024**2,
        "rwkv7_cmix_nofc_disabled": cmix_nofc_disabled,
        "output_ids": output_ids,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
