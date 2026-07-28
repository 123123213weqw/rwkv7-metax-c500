import importlib.util
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "rwkv7_metax_vllm_chunked_gate",
    ROOT / "adapters" / "vllm" / "accept_chunked_prefill_c500.py",
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_chunked_gate_prompts_force_multiple_scheduler_steps() -> None:
    prompts = MODULE._prompts()
    prompt_lengths = [len(row["prompt_token_ids"]) for row in prompts]

    assert prompt_lengths == list(MODULE.PROMPT_LENGTHS)
    assert all(
        1 <= token <= 65000
        for row in prompts
        for token in row["prompt_token_ids"]
    )
    assert sum(prompt_lengths) <= 512
    assert max(prompt_lengths) > 64
    assert math.ceil(sum(prompt_lengths) / 64) == 6


def test_chunked_gate_prompts_are_deterministic_and_distinct() -> None:
    first = MODULE._prompts()
    second = MODULE._prompts()

    assert first == second
    first_prompt = first[0]["prompt_token_ids"]
    assert first_prompt != first[1]["prompt_token_ids"][: len(first_prompt)]
