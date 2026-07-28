import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "rwkv7_metax_vllm_state_prefix_cache_gate",
    ROOT / "adapters" / "vllm" / "accept_state_prefix_cache_c500.py",
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_state_cache_gate_uses_a_shared_prefix_and_control_miss() -> None:
    warmup, prompts = MODULE._prompts()
    common = warmup["prompt_token_ids"][: MODULE.COMMON_PREFIX_LENGTH]

    assert [len(row["prompt_token_ids"]) for row in prompts] == [
        145,
        147,
        151,
        157,
        151,
    ]
    assert all(
        row["prompt_token_ids"][: MODULE.COMMON_PREFIX_LENGTH] == common
        for row in prompts[: MODULE.SHARED_REQUESTS]
    )
    assert prompts[-1]["prompt_token_ids"][: MODULE.COMMON_PREFIX_LENGTH] != common
    assert all(
        1 <= token <= 65000
        for row in [warmup, *prompts]
        for token in row["prompt_token_ids"]
    )


def test_state_cache_gate_helpers_are_deterministic() -> None:
    assert MODULE._prompts() == MODULE._prompts()
    assert MODULE._median([0.3, 0.1, None, 0.2]) == 0.2
    assert MODULE._median([None]) is None


def test_committed_state_cache_result_meets_the_exact_cell_contract() -> None:
    result = json.loads(
        (
            ROOT
            / "evidence"
            / "c500_vllm_state_prefix_cache_20260728"
            / "result.json"
        ).read_text(encoding="utf-8")
    )

    assert result["status"] == "pass"
    assert result["track"] == "vllm"
    assert result["public_api"] == "vllm.LLM"
    assert result["gpu"] == "MetaX C500"
    assert result["cached_tokens"] == [128, 128, 128, 128, 0]
    assert result["shared_prefix_hits"] == [True, True, True, True]
    assert result["control_cache_miss"] is True
    assert result["request_hit_rate"] >= 0.8
    assert result["greedy_output_match"] is True
    assert result["cache_to_baseline_shared_ttft_ratio"] < 1.0
