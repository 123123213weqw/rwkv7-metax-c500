#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("RWKV7_NATIVE_MODEL_BACKEND", "eager")
os.environ.setdefault("RWKV7_NATIVE_MODEL_JIT", "0")

import torch
import torch.nn.functional as F

from rwkv7_hf.native_model import NativeRWKV7Config, NativeRWKV7ForCausalLM


DTYPES = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


def build_model() -> NativeRWKV7ForCausalLM:
    torch.manual_seed(20260726)
    config = NativeRWKV7Config(
        vocab_size=127,
        hidden_size=32,
        num_hidden_layers=2,
        head_dim=8,
        intermediate_size=64,
        decay_low_rank_dim=8,
        gate_low_rank_dim=8,
        a_low_rank_dim=8,
        v_low_rank_dim=8,
        use_cache=True,
    )
    return NativeRWKV7ForCausalLM(config)


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(F.cosine_similarity(left.float().flatten(), right.float().flatten(), dim=0).item())


def run_dtype(
    *,
    name: str,
    dtype: torch.dtype,
    state_dict: dict[str, torch.Tensor],
    input_ids: torch.Tensor,
    cpu_logits: torch.Tensor,
) -> dict[str, Any]:
    model = build_model()
    model.load_state_dict(state_dict)
    model.to(device="cuda", dtype=dtype).eval()
    ids = input_ids.to("cuda")

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        model(ids, use_cache=True)
        torch.cuda.synchronize()
        started = time.perf_counter()
        full = model(ids, use_cache=True)
        torch.cuda.synchronize()
        forward_ms = (time.perf_counter() - started) * 1000.0
        prefill = model(ids[:, :-1], use_cache=True)
        decoded = model(ids[:, -1:], past_key_values=prefill.past_key_values, use_cache=True)
        selected = full.past_key_values.select_batch(torch.tensor([1], device="cuda"), inplace=False)
        generated = model.generate(ids[:1, :4], max_new_tokens=4, do_sample=False, use_cache=True)

    logits = full.logits.detach().float().cpu()
    decode_logits = decoded.logits[:, -1].detach().float().cpu()
    full_last = logits[:, -1]
    forward_cosine = _cosine(logits, cpu_logits)
    cache_cosine = _cosine(decode_logits, full_last)
    peak_memory = int(torch.cuda.max_memory_allocated())

    model.train()
    model.zero_grad(set_to_none=True)
    train_out = model(ids, labels=ids, use_cache=False)
    train_out.loss.backward()
    finite_grad_count = sum(
        int(parameter.grad is not None and bool(torch.isfinite(parameter.grad).all()))
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    trainable_count = sum(int(parameter.requires_grad) for parameter in model.parameters())
    threshold = 0.99999 if name == "fp32" else 0.999
    passed = bool(
        torch.isfinite(full.logits).all()
        and forward_cosine >= threshold
        and cache_cosine >= 0.99999
        and selected.get_batch_size() == 1
        and generated.shape == (1, 8)
        and torch.isfinite(train_out.loss)
        and finite_grad_count == trainable_count
    )
    return {
        "dtype": name,
        "status": "pass" if passed else "fail",
        "batch_size": int(ids.shape[0]),
        "sequence_length": int(ids.shape[1]),
        "forward_ms": forward_ms,
        "forward_cosine_vs_cpu_fp32": forward_cosine,
        "cache_handoff_cosine": cache_cosine,
        "cache_batch_select_size": selected.get_batch_size(),
        "generated_token_ids": generated[0, 4:].detach().cpu().tolist(),
        "training_loss": float(train_out.loss.detach().float().cpu().item()),
        "finite_grad_count": finite_grad_count,
        "trainable_parameter_tensor_count": trainable_count,
        "peak_memory_bytes": peak_memory,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Native/no-FLA RWKV-7 smoke on MetaX C500")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("this acceptance requires exactly one visible C500")
    if "c500" not in torch.cuda.get_device_name(0).lower():
        raise RuntimeError(f"expected MetaX C500, got {torch.cuda.get_device_name(0)!r}")

    base = build_model().eval()
    state_dict = {name: tensor.detach().clone() for name, tensor in base.state_dict().items()}
    input_ids = torch.tensor(
        [[1, 2, 3, 4, 5, 6, 7, 8], [8, 7, 6, 5, 4, 3, 2, 1]],
        dtype=torch.long,
    )
    with torch.inference_mode():
        cpu_logits = base(input_ids, use_cache=True).logits.detach().float().cpu()

    rows = [
        run_dtype(
            name=name,
            dtype=dtype,
            state_dict=state_dict,
            input_ids=input_ids,
            cpu_logits=cpu_logits,
        )
        for name, dtype in DTYPES.items()
    ]
    result = {
        "schema": "rwkv7-metax-c500-hf-native-smoke-v1",
        "status": "pass" if all(row["status"] == "pass" for row in rows) else "fail",
        "source_commit": "9d7dbc9213f0fb6b021d8dd0e3a828dad5fcd4af",
        "backend": "native_eager_no_fla",
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "python": platform.python_version(),
        "rows": rows,
    }
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
