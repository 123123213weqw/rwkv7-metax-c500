#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import platform
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load


def cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(
        F.cosine_similarity(left.float().flatten(), right.float().flatten(), dim=0).item()
    )


def reference(
    query_start_loc: torch.Tensor,
    slot_indices: torch.Tensor,
    state: torch.Tensor,
    r: torch.Tensor,
    w: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    result = torch.empty_like(r)
    out_state = state.clone()
    head_count = state.shape[1]
    head_size = state.shape[-1]
    for request, slot in enumerate(slot_indices.tolist()):
        start = int(query_start_loc[request])
        end = int(query_start_loc[request + 1])
        for token in range(start, end):
            for head in range(head_count):
                offset = head * head_size
                sl = slice(offset, offset + head_size)
                decay = torch.exp2(-0.8750387749145276 / (1.0 + torch.exp2(-1.4426950408889634 * w[token, sl])))
                current = out_state[slot, head]
                current = (
                    current * decay.unsqueeze(0)
                    + (current @ a[token, sl]).unsqueeze(1) * b[token, sl].unsqueeze(0)
                    + v[token, sl].unsqueeze(1) * k[token, sl].unsqueeze(0)
                )
                out_state[slot, head] = current
                result[token, sl] = current @ r[token, sl]
    return result, out_state


def main() -> int:
    parser = argparse.ArgumentParser(description="Compile and validate the RWKV WKV extension on C500")
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--build-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available() or "c500" not in torch.cuda.get_device_name(0).lower():
        raise RuntimeError("this probe requires a visible MetaX C500")

    args.build_dir.mkdir(parents=True, exist_ok=True)
    sources = [
        str(args.source_dir / "rwkv7_wkv_fp32_v2.cpp"),
        str(args.source_dir / "rwkv7_wkv_fp32_v2.cu"),
    ]
    compile_started = time.perf_counter()
    load(
        name="rwkv7_wkv_fp32_v2_c500",
        sources=sources,
        build_directory=str(args.build_dir),
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3"],
        with_cuda=True,
        is_python_module=False,
        verbose=True,
    )
    compile_seconds = time.perf_counter() - compile_started

    torch.manual_seed(20260726)
    query_start_loc = torch.tensor([0, 2, 5], dtype=torch.int32, device="cuda")
    slot_indices = torch.tensor([2, 0], dtype=torch.int32, device="cuda")
    state = torch.randn((4, 1, 64, 64), dtype=torch.float32, device="cuda") * 0.01
    tensors = [torch.randn((5, 64), dtype=torch.float32, device="cuda") * 0.05 for _ in range(6)]
    r, w, k, v, a, b = tensors
    expected_y, expected_state = reference(
        query_start_loc.cpu(),
        slot_indices.cpu(),
        state.cpu(),
        r.cpu(),
        w.cpu(),
        k.cpu(),
        v.cpu(),
        a.cpu(),
        b.cpu(),
    )
    output = torch.empty_like(r)
    torch.ops.rwkv7_wkv_fp32_v2.wkv(
        query_start_loc, slot_indices, state, r, w, k, v, a, b, output
    )
    torch.cuda.synchronize()
    output_cpu = output.cpu()
    state_cpu = state.cpu()
    output_cosine = cosine(output_cpu, expected_y)
    state_cosine = cosine(state_cpu, expected_state)
    output_max_abs = float((output_cpu - expected_y).abs().max().item())
    state_max_abs = float((state_cpu - expected_state).abs().max().item())
    passed = bool(
        math.isfinite(output_cosine)
        and output_cosine >= 0.999999
        and state_cosine >= 0.999999
        and output_max_abs <= 1e-5
        and state_max_abs <= 1e-5
    )
    result = {
        "schema": "rwkv7-metax-c500-vllm-wkv-extension-v1",
        "status": "pass" if passed else "fail",
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "python": platform.python_version(),
        "compile_seconds": compile_seconds,
        "requests": 2,
        "packed_tokens": 5,
        "state_slots": 4,
        "output_cosine": output_cosine,
        "state_cosine": state_cosine,
        "output_max_abs": output_max_abs,
        "state_max_abs": state_max_abs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
