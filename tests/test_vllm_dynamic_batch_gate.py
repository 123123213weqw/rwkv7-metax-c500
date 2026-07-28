import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "rwkv7_metax_vllm_dynamic_batch_gate",
    ROOT / "adapters" / "vllm" / "accept_dynamic_batch_c500.py",
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _row(token_ids, started_at, first_event_at, finished_at, *, final=False):
    return {
        "token_ids": token_ids,
        "started_at": started_at,
        "first_event_at": first_event_at,
        "first_event_finished": final,
        "finished_at": finished_at,
    }


def test_dynamic_batch_gate_accepts_overlapping_matching_requests() -> None:
    references = {"request_a": [1, 2], "request_b": [3, 4]}
    dynamic = {
        "request_a": _row([1, 2], 0.0, 1.0, 5.0),
        "request_b": _row([3, 4], 1.1, 2.0, 6.0),
    }

    assert all(MODULE._evaluate_dynamic_batch(references, dynamic, 1.1).values())


def test_dynamic_batch_gate_rejects_serial_scheduling() -> None:
    references = {"request_a": [1, 2], "request_b": [3, 4]}
    dynamic = {
        "request_a": _row([1, 2], 0.0, 1.0, 5.0),
        "request_b": _row([3, 4], 1.1, 5.1, 6.0),
    }

    checks = MODULE._evaluate_dynamic_batch(references, dynamic, 1.1)

    assert checks["greedy_output_match"]
    assert checks["request_b_submitted_while_a_active"]
    assert not checks["request_b_scheduled_before_a_finished"]


def test_dynamic_batch_gate_rejects_output_mismatch() -> None:
    references = {"request_a": [1, 2], "request_b": [3, 4]}
    dynamic = {
        "request_a": _row([1, 9], 0.0, 1.0, 5.0),
        "request_b": _row([3, 4], 1.1, 2.0, 6.0),
    }

    checks = MODULE._evaluate_dynamic_batch(references, dynamic, 1.1)

    assert not checks["greedy_output_match"]
    assert checks["request_b_scheduled_before_a_finished"]
