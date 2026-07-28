from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_acceptance_contract_covers_full_model_and_shape_matrix():
    contract = json.loads((ROOT / "acceptance_matrix.json").read_text(encoding="utf-8"))

    assert contract["schema"] == "rwkv7-metax-acceptance-v1"
    assert [model["label"] for model in contract["models"]] == [
        "0.1b",
        "0.4b",
        "1.5b",
        "2.9b",
        "7.2b",
        "13.3b",
    ]
    assert contract["inference_shapes"] == {
        "batch_sizes": [1, 2, 4, 8],
        "prompt_tokens": [128, 512, 2048],
        "decode_tokens": [1, 128],
    }
    assert contract["quantization"]["modes"] == ["w8", "w4"]
    assert contract["quantization"]["all_models_and_inference_shapes_required"] is True
    assert contract["speculative_decoding"]["all_target_models_required"] is True


def test_acceptance_contract_does_not_promote_partial_evidence():
    contract = json.loads((ROOT / "acceptance_matrix.json").read_text(encoding="utf-8"))
    policy = contract["claim_policy"]

    assert policy["missing_evidence_is_pass"] is False
    assert policy["smoke_implies_performance"] is False
    assert policy["microbenchmark_implies_end_to_end"] is False
    assert policy["single_model_implies_all_models"] is False
    assert policy["single_shape_implies_all_shapes"] is False
    assert contract["performance"]["minimum_speed_ratio"] == 1.0
    assert contract["performance"]["maximum_peak_memory_ratio"] == 1.0
    assert contract["cache"]["minimum_eligible_hit_rate_after_warmup"] == 0.8


def test_each_engine_contract_includes_stateful_serving_and_parallelism():
    contract = json.loads((ROOT / "acceptance_matrix.json").read_text(encoding="utf-8"))

    for track in ("vllm", "sglang"):
        required = set(contract["tracks"][track]["required"])
        assert {
            "continuous_dynamic_batching",
            "chunked_prefill",
            "per_request_state_slots",
            "preemption_restore",
            "TP",
            "PP",
        } <= required

    training = set(contract["tracks"]["hf"]["training"])
    assert {"PEFT_LoRA_save_load_merge", "TRL_SFT", "TRL_DPO", "TRL_GRPO"} <= training
    assert {"DeepSpeed_Zero2", "DeepSpeed_Zero3"} <= training
