#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
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


def apply_patchset(
    *,
    repository: Path,
    source: Path,
    track: str,
    manifest_name: str = "patchset.json",
) -> list[str]:
    adapter = repository / "adapters" / track
    if Path(manifest_name).name != manifest_name or not manifest_name.endswith(".json"):
        raise ValueError(f"invalid manifest name: {manifest_name}")
    manifest_path = adapter / manifest_name
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "rwkv7-metax-patchset-v1":
        raise ValueError(f"unsupported patchset schema in {manifest_path}")

    expected = str(manifest["commit"])
    actual = _git(source, "rev-parse", "HEAD").stdout.strip()
    if actual != expected:
        raise RuntimeError(f"{track}: expected upstream {expected}, got {actual}")

    patch_entries = [
        (relative, (adapter / relative).resolve())
        for relative in manifest["patches"]
    ]
    git_dir = Path(_git(source, "rev-parse", "--git-dir").stdout.strip())
    if not git_dir.is_absolute():
        git_dir = source / git_dir
    marker_dir = git_dir.resolve() / "rwkv7-metax-patchsets"

    applied: list[str] = []
    for relative, patch in patch_entries:
        patch_digest = hashlib.sha256(
            relative.encode("utf-8") + b"\0" + patch.read_bytes()
        ).hexdigest()
        marker = marker_dir / f"{patch_digest}.applied"
        if marker.is_file():
            continue

        reverse = _git(source, "apply", "--reverse", "--check", str(patch), check=False)
        if reverse.returncode == 0:
            marker_dir.mkdir(parents=True, exist_ok=True)
            marker.write_text(relative + "\n", encoding="utf-8")
            continue
        _git(source, "apply", "--check", str(patch))
        _git(source, "apply", str(patch))
        marker_dir.mkdir(parents=True, exist_ok=True)
        marker.write_text(relative + "\n", encoding="utf-8")
        applied.append(str(relative))
    return applied


def main() -> int:
    parser = argparse.ArgumentParser(description="Apply a locked C500 adapter patchset")
    parser.add_argument("track", choices=("hf", "vllm", "sglang"))
    parser.add_argument("source", type=Path)
    parser.add_argument("--manifest", default="patchset.json")
    args = parser.parse_args()
    repository = Path(__file__).resolve().parents[1]
    applied = apply_patchset(
        repository=repository,
        source=args.source.resolve(),
        track=args.track,
        manifest_name=args.manifest,
    )
    for patch in applied:
        print(f"applied {patch}")
    if not applied:
        print("patchset already applied")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
