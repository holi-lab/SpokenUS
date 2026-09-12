import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import synthesize


def test_default_reference_dir_points_to_speech_accent_archive_files():
    assert synthesize.DEFAULT_REFERENCE_DIR == Path(
        "datasets/SpeechAccentArchive/recordings/recordings"
    )


def test_cli_help_without_optional_dependencies(tmp_path):
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [sys.executable, str(root / "src/synthesize.py"), "--help"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(root / "src")},
    )
    assert result.returncode == 0, result.stderr
    assert "--text" in result.stdout
    assert "--base-url" in result.stdout


def test_create_synthesizer_uses_self_hosted_vllm_by_default(monkeypatch):
    captured = {}

    class FakeVllm:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(synthesize, "VllmOmniSynthesizer", FakeVllm)
    args = SimpleNamespace(
        base_url="http://tts.internal:8000",
        model_id="Qwen/test",
        timeout=45.0,
    )

    synthesize.create_synthesizer(args)

    assert captured == {
        "base_url": "http://tts.internal:8000",
        "model": "Qwen/test",
        "timeout": 45.0,
    }
