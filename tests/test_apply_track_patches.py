import json
import subprocess
from pathlib import Path

from scripts.apply_track_patches import apply_patchset


def _git(source: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=source,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def test_patchset_reapply_handles_overlapping_patches(tmp_path: Path) -> None:
    repository = tmp_path / "adapter"
    source = tmp_path / "upstream"
    patch_dir = repository / "adapters" / "vllm" / "patches"
    patch_dir.mkdir(parents=True)
    source.mkdir()

    _git(source, "init")
    _git(source, "config", "user.name", "Test User")
    _git(source, "config", "user.email", "test@example.com")
    (source / "value.txt").write_text("value=1\n", encoding="utf-8")
    _git(source, "add", "value.txt")
    _git(source, "commit", "-m", "base")
    commit = _git(source, "rev-parse", "HEAD")

    (patch_dir / "0001.patch").write_text(
        """diff --git a/value.txt b/value.txt
--- a/value.txt
+++ b/value.txt
@@ -1 +1 @@
-value=1
+value=2
""",
        encoding="utf-8",
    )
    (patch_dir / "0002.patch").write_text(
        """diff --git a/value.txt b/value.txt
--- a/value.txt
+++ b/value.txt
@@ -1 +1 @@
-value=2
+value=3
""",
        encoding="utf-8",
    )
    manifest = {
        "schema": "rwkv7-metax-patchset-v1",
        "upstream": "test",
        "commit": commit,
        "patches": ["patches/0001.patch", "patches/0002.patch"],
    }
    (repository / "adapters" / "vllm" / "patchset.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )

    assert apply_patchset(
        repository=repository,
        source=source,
        track="vllm",
    ) == ["patches/0001.patch", "patches/0002.patch"]
    assert (source / "value.txt").read_text(encoding="utf-8") == "value=3\n"

    assert apply_patchset(
        repository=repository,
        source=source,
        track="vllm",
    ) == []
    assert (source / "value.txt").read_text(encoding="utf-8") == "value=3\n"
