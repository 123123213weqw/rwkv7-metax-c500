#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from rwkv7_metax.probe import collect_probe, write_probe


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect a redacted MetaX C500 environment probe")
    parser.add_argument("--run-smoke", action="store_true", help="run fp16 and bf16 matrix operations")
    parser.add_argument("--output", help="optional JSON output path")
    args = parser.parse_args()
    probe = collect_probe(run_smoke=args.run_smoke)
    if args.output:
        write_probe(args.output, probe)
    print(json.dumps(probe, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
