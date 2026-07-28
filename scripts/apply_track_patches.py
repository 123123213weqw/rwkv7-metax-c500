#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def _git(source: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=source,
        check=check,
        capture_output=True,
        text=True,
    )


def apply_patchset(*, repository: Path, source: Path, track: str) -> list[str]:
    adapter = repository / "adapters" / track
    manifest_path = adapter / "patchset.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "rwkv7-metax-patchset-v1":
        raise ValueError(f"unsupported patchset schema in {manifest_path}")

    expected = str(manifest["commit"])
    actual = _git(source, "rev-parse", "HEAD").stdout.strip()
    if actual != expected:
        raise RuntimeError(f"{track}: expected upstream {expected}, got {actual}")

    applied: list[str] = []
    for relative in manifest["patches"]:
        patch = (adapter / relative).resolve()
        reverse = _git(source, "apply", "--reverse", "--check", str(patch), check=False)
        if reverse.returncode == 0:
            continue
        _git(source, "apply", "--check", str(patch))
        _git(source, "apply", str(patch))
        applied.append(str(relative))
    return applied


def main() -> int:
    parser = argparse.ArgumentParser(description="Apply a locked C500 adapter patchset")
    parser.add_argument("track", choices=("hf", "vllm", "sglang"))
    parser.add_argument("source", type=Path)
    args = parser.parse_args()
    repository = Path(__file__).resolve().parents[1]
    applied = apply_patchset(repository=repository, source=args.source.resolve(), track=args.track)
    for patch in applied:
        print(f"applied {patch}")
    if not applied:
        print("patchset already applied")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
