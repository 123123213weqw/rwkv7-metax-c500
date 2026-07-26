from rwkv7_metax.evidence import validate_probe


def valid_probe():
    return {
        "schema": "rwkv7-metax-c500-probe-v1",
        "platform": {"system": "Linux"},
        "hardware_detected": True,
        "torch": {
            "status": "pass",
            "cuda_available": True,
            "device_count": 1,
            "smoke": [
                {"dtype": "float16", "status": "pass", "cosine": 0.9999},
                {"dtype": "bfloat16", "status": "pass", "cosine": 0.9998},
            ],
        },
    }


def test_valid_probe_passes():
    result = validate_probe(valid_probe())
    assert result.passed
    assert result.failures == ()


def test_probe_fails_closed_without_hardware_or_smoke():
    probe = valid_probe()
    probe["hardware_detected"] = False
    probe["torch"]["smoke"] = []
    result = validate_probe(probe)
    assert not result.passed
    assert "MetaX C500/MXMACA hardware was not detected" in result.failures
    assert "missing float16 matmul smoke" in result.failures
    assert "missing bfloat16 matmul smoke" in result.failures


def test_probe_can_validate_inventory_without_smoke():
    probe = valid_probe()
    probe["torch"]["smoke"] = []
    assert validate_probe(probe, require_smoke=False).passed
