from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ProbeValidation:
    passed: bool
    failures: tuple[str, ...]


def validate_probe(probe: dict[str, Any], *, require_smoke: bool = True) -> ProbeValidation:
    failures: list[str] = []
    if probe.get("schema") != "rwkv7-metax-c500-probe-v1":
        failures.append("unexpected or missing schema")
    if probe.get("platform", {}).get("system") != "Linux":
        failures.append("C500 acceptance must run on Linux")
    if not probe.get("hardware_detected"):
        failures.append("MetaX C500/MXMACA hardware was not detected")
    torch_result = probe.get("torch", {})
    if torch_result.get("status") != "pass":
        failures.append("PyTorch import failed")
    if not torch_result.get("cuda_available"):
        failures.append("MXMACA torch.cuda compatibility path is unavailable")
    if int(torch_result.get("device_count", 0)) < 1:
        failures.append("no accelerator device is visible to PyTorch")
    if require_smoke:
        smoke = torch_result.get("smoke", [])
        for dtype in ("float16", "bfloat16"):
            row = next((item for item in smoke if item.get("dtype") == dtype), None)
            if row is None:
                failures.append(f"missing {dtype} matmul smoke")
            elif row.get("status") != "pass":
                failures.append(f"{dtype} matmul smoke did not pass")
            elif float(row.get("cosine", 0.0)) < 0.999:
                failures.append(f"{dtype} matmul cosine is below 0.999")
    return ProbeValidation(passed=not failures, failures=tuple(failures))
