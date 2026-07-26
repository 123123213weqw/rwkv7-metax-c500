#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from rwkv7_metax.evidence import validate_probe


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify a MetaX C500 probe artifact")
    parser.add_argument("probe")
    parser.add_argument("--allow-no-smoke", action="store_true")
    args = parser.parse_args()
    probe = json.loads(Path(args.probe).read_text(encoding="utf-8"))
    result = validate_probe(probe, require_smoke=not args.allow_no_smoke)
    print(json.dumps({"passed": result.passed, "failures": result.failures}, indent=2))
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
