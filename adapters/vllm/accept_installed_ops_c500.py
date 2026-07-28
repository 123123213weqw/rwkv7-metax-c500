#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import platform
from pathlib import Path

import torch
import torch.nn.functional as F


NEXP_HALF_LOG2_E = -0.8750387749145276
NLOG2_E = -1.4426950408889634
ROT1 = 2654435769
TWO_NEG_41 = 2.0**-41


def cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(
        F.cosine_similarity(left.float().flatten(), right.float().flatten(), dim=0).item()
    )


def rotator(phase: torch.Tensor) -> torch.Tensor:
    bits = (phase.to(torch.int64) * ROT1) & 0xFFFFFFFF
    signed = torch.where(bits >= 0x80000000, bits - 0x100000000, bits)
    return signed.to(torch.float32) * TWO_NEG_41


def reference_fp32_state(
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
    output_state = state.clone()
    for request, slot in enumerate(slot_indices.tolist()):
        start = int(query_start_loc[request])
        end = int(query_start_loc[request + 1])
        for token in range(start, end):
            current = output_state[slot, 0]
            decay = torch.exp2(
                NEXP_HALF_LOG2_E
                / (1.0 + torch.exp2(NLOG2_E * w[token].float()))
            )
            current = (
                current * decay.unsqueeze(0)
                + (current @ a[token].float()).unsqueeze(1) * b[token].float().unsqueeze(0)
                + v[token].float().unsqueeze(1) * k[token].float().unsqueeze(0)
            )
            output_state[slot, 0] = current
            result[token] = (current @ r[token].float()).to(result.dtype)
    return result, output_state


def reference_fp16_state(
    query_start_loc: torch.Tensor,
    slot_indices: torch.Tensor,
    elapsed: torch.Tensor,
    state: torch.Tensor,
    r: torch.Tensor,
    w: torch.Tensor,
    w0: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    result = torch.empty_like(r)
    output_state = state.clone()
    channels = torch.arange(state.shape[-1], device=state.device)
    for request, slot in enumerate(slot_indices.tolist()):
        start = int(query_start_loc[request])
        end = int(query_start_loc[request + 1])
        for offset, token in enumerate(range(start, end)):
            current = output_state[slot, 0]
            w_value = w[token].float() + w0.float()
            delta = (
                torch.exp2(
                    NEXP_HALF_LOG2_E
                    / (1.0 + torch.exp2(NLOG2_E * w_value))
                )
                - 1.0
                + rotator(elapsed[slot].to(torch.int64) + channels + offset)
            ).half()
            current = current * delta.unsqueeze(0) + current
            current = current + v[token].unsqueeze(1) * k[token].unsqueeze(0)
            output_state[slot, 0] = current
            result[token] = current @ r[token]
    return result, output_state


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate installed RWKV vLLM C500 operators")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    if not torch.cuda.is_available() or "c500" not in torch.cuda.get_device_name(0).lower():
        raise RuntimeError("this acceptance probe requires a visible MetaX C500")

    import vllm._rapid_sampling as rapid_sampling
    import vllm.rwkv7_ops  # noqa: F401

    torch.manual_seed(20260727)
    device = torch.device("cuda")
    query_start_loc = torch.tensor([0, 2, 5], dtype=torch.int32, device=device)
    slot_indices = torch.tensor([2, 0], dtype=torch.int32, device=device)

    state32 = torch.randn((4, 1, 64, 64), device=device) * 0.01
    tensors32 = [
        (torch.randn((5, 64), device=device) * 0.05).half() for _ in range(6)
    ]
    r32, w32, k32, v32, a32, b32 = tensors32
    expected_y32, expected_state32 = reference_fp32_state(
        query_start_loc.cpu(),
        slot_indices.cpu(),
        state32,
        r32,
        w32,
        k32,
        v32,
        a32,
        b32,
    )
    actual_state32 = state32.clone()
    actual_y32 = torch.empty_like(r32)
    torch.ops.rwkv7_wkv_fp32_v2.wkv(
        query_start_loc,
        slot_indices,
        actual_state32,
        r32,
        w32,
        k32,
        v32,
        a32,
        b32,
        actual_y32,
    )

    state16 = (torch.randn((4, 1, 64, 64), device=device) * 0.01).half()
    tensors16 = [
        (torch.randn((5, 64), device=device) * 0.05).half() for _ in range(5)
    ]
    r16, w16, k16, v16, unused = tensors16
    del unused
    w0 = (torch.randn((64,), device=device) * 0.02).half()
    zeros = torch.zeros_like(r16)
    elapsed = torch.tensor([11, 0, 23, 0], dtype=torch.int32, device=device)
    expected_y16, expected_state16 = reference_fp16_state(
        query_start_loc.cpu(),
        slot_indices.cpu(),
        elapsed,
        state16,
        r16,
        w16,
        w0,
        k16,
        v16,
    )
    actual_state16 = state16.clone()
    actual_y16 = torch.empty_like(r16)
    torch.ops.rwkv7_wkv_fp16_v2.wkv(
        query_start_loc,
        slot_indices,
        actual_state16,
        r16,
        w16,
        w0,
        k16,
        v16,
        zeros,
        zeros,
        actual_y16,
        elapsed,
    )

    pinned = torch.arange(8, dtype=torch.int32, pin_memory=True)
    mapped = torch.ops._C.get_cuda_view_from_cpu_tensor(pinned)
    mapped_matches = bool(torch.equal(mapped.cpu(), pinned))
    random_state = rapid_sampling.setup_rand(20260727, 2)
    torch.cuda.synchronize()

    fp32_output_cosine = cosine(actual_y32, expected_y32)
    fp32_state_cosine = cosine(actual_state32, expected_state32)
    fp16_output_cosine = cosine(actual_y16, expected_y16)
    fp16_state_cosine = cosine(actual_state16, expected_state16)
    result = {
        "schema": "rwkv7-metax-c500-vllm-installed-ops-v1",
        "device": torch.cuda.get_device_name(0),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "vllm": importlib.metadata.version("vllm"),
        "packed_requests": 2,
        "packed_tokens": 5,
        "state_slots": 4,
        "fp32io16_output_cosine": fp32_output_cosine,
        "fp32io16_state_cosine": fp32_state_cosine,
        "fp32io16_output_max_abs": float((actual_y32 - expected_y32).abs().max().item()),
        "fp32io16_state_max_abs": float(
            (actual_state32 - expected_state32).abs().max().item()
        ),
        "fp16_output_cosine": fp16_output_cosine,
        "fp16_state_cosine": fp16_state_cosine,
        "fp16_output_max_abs": float((actual_y16 - expected_y16).abs().max().item()),
        "fp16_state_max_abs": float(
            (actual_state16 - expected_state16).abs().max().item()
        ),
        "cuda_view_matches": mapped_matches,
        "rapid_sampling_state_shape": list(random_state.shape),
        "rapid_sampling_state_device": str(random_state.device),
    }
    passed = bool(
        fp32_output_cosine >= 0.99999
        and fp32_state_cosine >= 0.99999
        and fp16_output_cosine >= 0.999
        and fp16_state_cosine >= 0.999
        and mapped_matches
        and random_state.is_cuda
        and all(math.isfinite(value) for key, value in result.items() if key.endswith(("cosine", "max_abs")))
    )
    result["status"] = "pass" if passed else "fail"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
