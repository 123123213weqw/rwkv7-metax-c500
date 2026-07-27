from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def assert_patchset(track: str) -> None:
    manifest = json.loads(
        (ROOT / f"adapters/{track}/patchset.json").read_text(encoding="utf-8")
    )
    lock = json.loads((ROOT / "upstreams.lock.json").read_text(encoding="utf-8"))
    assert manifest["schema"] == "rwkv7-metax-patchset-v1"
    assert manifest["commit"] == lock["upstreams"][manifest["upstream"]]["commit"]
    assert manifest["patches"]
    for relative in manifest["patches"]:
        patch = ROOT / "adapters" / track / relative
        assert patch.is_file()
        assert patch.read_text(encoding="utf-8").startswith("diff --git ")


def test_hf_patchset_is_locked_and_complete():
    assert_patchset("hf")


def test_vllm_patchset_is_locked_and_complete():
    assert_patchset("vllm")
