"""Offline checks for fresh-machine preparation; no packages or models downloaded."""
from pathlib import Path

from scripts.bootstrap_local import build_plan, executable
from scripts import run_portable


def test_fresh_plan_creates_both_environments_and_pinned_models(tmp_path):
    commands = build_plan(tmp_path)
    creation = [command for command in commands if command[1:3] == ["-m", "venv"]]
    assert len(creation) == 2
    assert any("requirements.lock.txt" in command for command in commands)
    assert any("requirements.objects.txt" in command for command in commands)
    assert ["scripts/setup_architecture_models.py", "--pose", "--objects"] in [command[1:] for command in commands]
    assert commands[-1][1:] == ["scripts/setup_local_runtime.py"]


def test_manual_runtime_still_downloads_matching_public_models(tmp_path):
    commands = build_plan(tmp_path, skip_runtime=True)
    assert commands[-1][1:] == ["scripts/setup_local_runtime.py", "--skip-runtime"]


def test_reusing_models_does_not_trigger_any_model_installers(tmp_path):
    commands = build_plan(tmp_path, skip_downloads=True)
    assert not any(any("scripts/setup_" in part for part in command) for command in commands)


def test_explicit_missing_executable_is_rejected(tmp_path):
    import pytest
    with pytest.raises(ValueError, match="Missing"):
        executable(str(tmp_path / "missing-ffmpeg.exe"), "ffmpeg")


def test_start_without_bootstrap_has_actionable_error(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(run_portable, "ROOT", tmp_path)
    assert run_portable.main() == 2
    assert "bootstrap_local.py" in capsys.readouterr().err
