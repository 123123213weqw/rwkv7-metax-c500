#!/usr/bin/env python3
"""Run the public SGLang Engine gate for RWKV-7 on MetaX C500."""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import torch


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=4)
    args = parser.parse_args()

    import sglang as sgl

    prompts = [[1, 2, 3, 4], [5, 6, 7, 8, 9]]
    started = time.perf_counter()
    engine = sgl.Engine(
        model_path=args.model,
        dtype="float16",
        skip_tokenizer_init=True,
        attention_backend="triton",
        sampling_backend="pytorch",
        disable_cuda_graph=True,
        disable_piecewise_cuda_graph=True,
        disable_overlap_schedule=True,
        disable_radix_cache=True,
        max_running_requests=8,
        context_length=512,
        mem_fraction_static=0.7,
    )
    try:
        outputs = engine.generate(
            input_ids=prompts,
            sampling_params={
                "temperature": 0,
                "max_new_tokens": args.max_new_tokens,
            },
        )
    finally:
        engine.shutdown()
    elapsed = time.perf_counter() - started

    rows = outputs if isinstance(outputs, list) else [outputs]
    output_ids = [list(row["output_ids"]) for row in rows]
    passed = len(output_ids) == len(prompts) and all(
        len(token_ids) == args.max_new_tokens for token_ids in output_ids
    )
    result = {
        "schema_version": 1,
        "status": "pass" if passed else "fail",
        "track": "sglang",
        "public_api": "sglang.Engine",
        "model": Path(args.model).name,
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "maca": getattr(torch.version, "maca", None),
        "python": platform.python_version(),
        "batch_size": len(prompts),
        "prompt_lengths": [len(prompt) for prompt in prompts],
        "max_new_tokens": args.max_new_tokens,
        "elapsed_s": elapsed,
        "output_ids": output_ids,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
