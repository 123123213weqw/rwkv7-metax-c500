#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def run(command: list[str], cwd: Path | None = None) -> None:
    subprocess.run(command, cwd=cwd, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Clone exact upstream revisions into an ignored workspace")
    parser.add_argument("--lock", default="upstreams.lock.json")
    parser.add_argument("--root", default="worktrees")
    args = parser.parse_args()
    lock = json.loads(Path(args.lock).read_text(encoding="utf-8"))
    root = Path(args.root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    for name, upstream in lock["upstreams"].items():
        target = root / name
        if not target.exists():
            run(["git", "clone", "--filter=blob:none", "--no-checkout", upstream["url"], str(target)])
        run(["git", "fetch", "origin", upstream["commit"], "--depth=1"], cwd=target)
        run(["git", "checkout", "--detach", upstream["commit"]], cwd=target)
        actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=target, text=True).strip()
        if actual != upstream["commit"]:
            raise RuntimeError(f"{name}: expected {upstream['commit']}, got {actual}")
        print(f"{name}: {actual}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
