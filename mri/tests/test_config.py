"""Tests for deterministic MRI model resolution."""

from pathlib import Path

import pytest

from mri import config


@pytest.mark.parametrize("variable", ["CLINIQ_MRI_MODEL_PATH", "MRI_MODEL_PATH"])
def test_configured_model_path_has_priority(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, variable: str) -> None:
    configured_model = tmp_path / "configured.keras"
    configured_model.touch()
    monkeypatch.setenv(variable, str(configured_model))
    monkeypatch.delenv("MRI_MODEL_PATH" if variable == "CLINIQ_MRI_MODEL_PATH" else "CLINIQ_MRI_MODEL_PATH", raising=False)

    assert config.find_model_path() == configured_model


def test_exact_default_model_path_is_used_when_not_configured(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    default_model = tmp_path / config.DEFAULT_CHECKPOINT_NAME
    default_model.touch()
    monkeypatch.delenv("CLINIQ_MRI_MODEL_PATH", raising=False)
    monkeypatch.delenv("MRI_MODEL_PATH", raising=False)
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path)

    assert config.find_model_path() == default_model


def test_unrelated_model_files_are_not_selected(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    (tmp_path / "unrelated.keras").touch()
    (tmp_path / "legacy.h5").touch()
    monkeypatch.delenv("CLINIQ_MRI_MODEL_PATH", raising=False)
    monkeypatch.delenv("MRI_MODEL_PATH", raising=False)
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path)

    assert config.find_model_path() is None


def test_configured_class_mapping_path_has_priority(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    configured_mapping = tmp_path / "training-class-names.json"
    monkeypatch.setenv("CLINIQ_MRI_CLASS_NAMES_PATH", str(configured_mapping))

    assert config.find_class_names_path() == configured_mapping
