from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Sequence


SAFE_ENV_KEYS = (
    "MACA_PATH",
    "CUCC_PATH",
    "CUDA_PATH",
    "MXLOG_LEVEL",
    "MACA_GRAPH_LAUNCH_MODE",
    "MCPTI_ENABLED",
)

PACKAGE_NAMES = (
    "torch",
    "transformers",
    "accelerate",
    "peft",
    "trl",
    "deepspeed",
    "vllm",
    "sglang",
    "triton",
)

COMMANDS: tuple[tuple[str, ...], ...] = (
    ("mx-smi", "--version"),
    ("mx-smi",),
    ("mxcc", "--version"),
    ("cucc", "--version"),
)


def _run_command(command: Sequence[str], timeout: int = 15) -> dict[str, Any]:
    executable = shutil.which(command[0])
    if executable is None:
        return {"command": list(command), "status": "missing"}
    try:
        completed = subprocess.run(
            [executable, *command[1:]],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {
            "command": list(command),
            "status": "error",
            "error": type(exc).__name__,
        }
    output = "\n".join(part.strip() for part in (completed.stdout, completed.stderr) if part.strip())
    return {
        "command": list(command),
        "status": "pass" if completed.returncode == 0 else "fail",
        "exit_code": completed.returncode,
        "output": output[-12000:],
    }


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in PACKAGE_NAMES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _timed_matmul(torch: Any, dtype: Any) -> dict[str, Any]:
    try:
        generator = torch.Generator(device="cpu").manual_seed(20260726)
        left_cpu = torch.randn((256, 256), generator=generator, dtype=torch.float32)
        right_cpu = torch.randn((256, 256), generator=generator, dtype=torch.float32)
        reference = left_cpu @ right_cpu
        left = left_cpu.to(device="cuda", dtype=dtype)
        right = right_cpu.to(device="cuda", dtype=dtype)
        torch.cuda.synchronize()
        started = time.perf_counter()
        output = left @ right
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        output_cpu = output.float().cpu()
        cosine = float(torch.nn.functional.cosine_similarity(output_cpu.flatten(), reference.flatten(), dim=0))
        max_abs = float((output_cpu - reference).abs().max())
        return {
            "status": "pass" if bool(torch.isfinite(output).all()) else "fail",
            "dtype": str(dtype).replace("torch.", ""),
            "shape": [256, 256, 256],
            "elapsed_ms": elapsed_ms,
            "cosine": cosine,
            "max_abs_error": max_abs,
        }
    except Exception as exc:  # Hardware/runtime errors belong in the evidence.
        return {
            "status": "error",
            "dtype": str(dtype).replace("torch.", ""),
            "error_type": type(exc).__name__,
            "error": str(exc)[-2000:],
        }


def _torch_probe(run_smoke: bool) -> dict[str, Any]:
    try:
        import torch
    except Exception as exc:
        return {"status": "error", "error_type": type(exc).__name__, "error": str(exc)[-2000:]}

    result: dict[str, Any] = {
        "status": "pass",
        "version": torch.__version__,
        "cuda_version": getattr(torch.version, "cuda", None),
        "cuda_available": bool(torch.cuda.is_available()),
        "device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        "devices": [],
        "smoke": [],
    }
    if not torch.cuda.is_available():
        return result

    for index in range(torch.cuda.device_count()):
        properties = torch.cuda.get_device_properties(index)
        result["devices"].append(
            {
                "index": index,
                "name": torch.cuda.get_device_name(index),
                "total_memory_bytes": int(properties.total_memory),
                "major": int(properties.major),
                "minor": int(properties.minor),
                "multi_processor_count": int(properties.multi_processor_count),
            }
        )
    if run_smoke:
        result["smoke"].append(_timed_matmul(torch, torch.float16))
        if hasattr(torch, "bfloat16"):
            result["smoke"].append(_timed_matmul(torch, torch.bfloat16))
    return result


def collect_probe(*, run_smoke: bool = False) -> dict[str, Any]:
    commands = [_run_command(command) for command in COMMANDS]
    torch_result = _torch_probe(run_smoke)
    searchable = "\n".join(str(row.get("output", "")) for row in commands).lower()
    searchable += "\n" + "\n".join(str(row.get("name", "")) for row in torch_result.get("devices", [])).lower()
    return {
        "schema": "rwkv7-metax-c500-probe-v1",
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python": platform.python_version(),
        },
        "environment": {key: os.environ.get(key) for key in SAFE_ENV_KEYS if os.environ.get(key)},
        "packages": _package_versions(),
        "commands": commands,
        "torch": torch_result,
        "hardware_detected": "c500" in searchable or "metax" in searchable or "maca" in searchable,
    }


def write_probe(path: str | Path, probe: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(probe, indent=2, sort_keys=True) + "\n", encoding="utf-8")
