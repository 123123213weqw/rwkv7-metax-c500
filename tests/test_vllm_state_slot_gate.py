import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "rwkv7_metax_vllm_state_slot_gate",
    ROOT / "adapters" / "vllm" / "accept_state_slots_c500.py",
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_state_slot_gate_uses_distinct_deterministic_prompts() -> None:
    first = MODULE._prompt_map()
    second = MODULE._prompt_map()

    assert first == second
    assert list(first) == ["request_a", "request_b"]
    assert [len(row["prompt_token_ids"]) for row in first.values()] == [33, 47]
    assert first["request_a"] != first["request_b"]


def test_state_slot_gate_tokens_are_in_checkpoint_vocabulary() -> None:
    prompts = MODULE._prompt_map()

    assert all(
        1 <= token <= 65000
        for row in prompts.values()
        for token in row["prompt_token_ids"]
    )
