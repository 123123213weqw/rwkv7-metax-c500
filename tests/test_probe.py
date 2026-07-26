from rwkv7_metax import probe


def test_missing_command_is_structured(monkeypatch):
    monkeypatch.setattr(probe.shutil, "which", lambda _: None)
    assert probe._run_command(("mx-smi",)) == {
        "command": ["mx-smi"],
        "status": "missing",
    }


def test_collect_probe_does_not_copy_secrets(monkeypatch):
    monkeypatch.setattr(probe, "_run_command", lambda command: {"command": list(command), "status": "missing"})
    monkeypatch.setattr(probe, "_torch_probe", lambda run_smoke: {"status": "pass", "devices": []})
    monkeypatch.setenv("MACA_PATH", "/opt/maca")
    monkeypatch.setenv("SSH_PASSWORD", "must-not-leak")
    result = probe.collect_probe()
    assert result["environment"] == {"MACA_PATH": "/opt/maca"}
    assert "SSH_PASSWORD" not in result["environment"]
