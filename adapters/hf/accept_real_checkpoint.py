#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import shutil
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("RWKV7_NATIVE_MODEL_BACKEND", "eager")
os.environ.setdefault("RWKV7_NATIVE_MODEL_JIT", "0")
os.environ.setdefault("RWKV7_FAST_PREFILL", "0")

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


def cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(
        F.cosine_similarity(left.float().flatten(), right.float().flatten(), dim=0).item()
    )


def load_model(model_path: str, *, device: str, dtype: torch.dtype):
    return AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        dtype=dtype,
        device_map=device if device == "cuda" else None,
    ).eval()


def main() -> int:
    parser = argparse.ArgumentParser(description="Real-checkpoint HF acceptance on MetaX C500")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--reload-dir", required=True)
    parser.add_argument("--prompt", default="User: Hello!\n\nAssistant:")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    args = parser.parse_args()

    if not torch.cuda.is_available() or "c500" not in torch.cuda.get_device_name(0).lower():
        raise RuntimeError("this acceptance requires a visible MetaX C500")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    encoded = tokenizer(args.prompt, return_tensors="pt")
    input_ids = encoded.input_ids
    attention_mask = encoded.attention_mask

    cpu_model = load_model(args.model, device="cpu", dtype=torch.float32)
    with torch.inference_mode():
        cpu_logits = cpu_model(
            input_ids, attention_mask=attention_mask, use_cache=True
        ).logits.detach().float().cpu()
    cpu_last = cpu_logits[:, -1]
    del cpu_model, cpu_logits
    gc.collect()

    model = load_model(args.model, device="cuda", dtype=torch.float16)
    ids = input_ids.cuda()
    mask = attention_mask.cuda()
    batch_ids = ids.repeat(8, 1)
    batch_mask = mask.repeat(8, 1)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        model(ids, attention_mask=mask, use_cache=True)
        torch.cuda.synchronize()
        started = time.perf_counter()
        full = model(ids, attention_mask=mask, use_cache=True)
        torch.cuda.synchronize()
        forward_ms = (time.perf_counter() - started) * 1000.0

        midpoint = max(1, ids.shape[1] // 2)
        first = model(ids[:, :midpoint], use_cache=True)
        split = model(
            ids[:, midpoint:], past_key_values=first.past_key_values, use_cache=True
        )

        chunk_cache = None
        chunked = None
        for start in range(0, ids.shape[1], 3):
            chunked = model(
                ids[:, start : start + 3],
                past_key_values=chunk_cache,
                use_cache=True,
            )
            chunk_cache = chunked.past_key_values
        assert chunked is not None

        batched = model(batch_ids, attention_mask=batch_mask, use_cache=True)
        selected = batched.past_key_values.select_batch(
            torch.tensor([7, 0, 3], device="cuda"), inplace=False
        )
        generated = model.generate(
            ids,
            attention_mask=mask,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            use_cache=True,
        )
        torch.cuda.synchronize()

    gpu_last = full.logits[:, -1].detach().float().cpu()
    split_last = split.logits[:, -1].detach().float().cpu()
    chunk_last = chunked.logits[:, -1].detach().float().cpu()
    cpu_cosine = cosine(gpu_last, cpu_last)
    split_cosine = cosine(split_last, gpu_last)
    chunk_cosine = cosine(chunk_last, gpu_last)
    generated_ids = generated[0, ids.shape[1] :].detach().cpu().tolist()
    peak_memory = int(torch.cuda.max_memory_allocated())
    selected_batch_size = int(selected.get_batch_size())

    reload_dir = Path(args.reload_dir)
    if reload_dir.exists():
        shutil.rmtree(reload_dir)
    model.save_pretrained(reload_dir, safe_serialization=True)
    tokenizer.save_pretrained(reload_dir)
    del model, full, first, split, chunked, chunk_cache, batched, selected, generated
    gc.collect()
    torch.cuda.empty_cache()

    reloaded = load_model(str(reload_dir), device="cuda", dtype=torch.float16)
    with torch.inference_mode():
        reload_logits = (
            reloaded(ids, attention_mask=mask, use_cache=True)
            .logits[:, -1]
            .detach()
            .float()
            .cpu()
        )
    reload_cosine = cosine(reload_logits, gpu_last)
    backend_getter = getattr(reloaded, "rwkv7_last_fast_token_backend", None)
    backend = backend_getter() if callable(backend_getter) else None

    passed = bool(
        torch.isfinite(gpu_last).all()
        and cpu_cosine >= 0.999
        and split_cosine >= 0.99999
        and chunk_cosine >= 0.99999
        and reload_cosine >= 0.99999
        and selected_batch_size == 3
        and len(generated_ids) == args.max_new_tokens
    )
    result: dict[str, Any] = {
        "schema": "rwkv7-metax-c500-hf-real-checkpoint-v1",
        "status": "pass" if passed else "fail",
        "source_commit": "9d7dbc9213f0fb6b021d8dd0e3a828dad5fcd4af",
        "model": Path(args.model).name,
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "python": platform.python_version(),
        "dtype": "fp16",
        "prompt_tokens": int(ids.shape[1]),
        "batch_sizes": [1, 8],
        "forward_ms": forward_ms,
        "forward_cosine_vs_cpu_fp32": cpu_cosine,
        "split_prefill_cosine": split_cosine,
        "chunked_prefill_cosine": chunk_cosine,
        "save_reload_cosine": reload_cosine,
        "selected_cache_batch_size": selected_batch_size,
        "generated_token_ids": generated_ids,
        "generated_text": tokenizer.decode(generated_ids, skip_special_tokens=True),
        "peak_memory_bytes": peak_memory,
        "requested_backend": os.environ["RWKV7_NATIVE_MODEL_BACKEND"],
        "last_observed_fast_token_backend": backend,
    }
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
