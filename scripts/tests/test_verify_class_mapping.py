"""Tests for the class-mapping verification CLI."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from PIL import Image

import scripts.verify_class_mapping as verifier
from mri.service import InvalidImageError


def _write_image(path: Path) -> None:
    Image.new("RGB", (2, 2)).save(path, format="PNG")


def _prepare_data(tmp_path: Path, include_corrupt: bool = False) -> tuple[Path, Path]:
    model_path = tmp_path / "model.keras"
    model_path.touch()
    data_path = tmp_path / "testing"
    for folder_name in ("glioma", "meningioma"):
        folder = data_path / folder_name
        folder.mkdir(parents=True)
        _write_image(folder / f"{folder_name}.png")
    if include_corrupt:
        (data_path / "glioma" / "corrupt.png").write_bytes(b"not an image")
    return model_path, data_path


def _set_args(monkeypatch: pytest.MonkeyPatch, model_path: Path, data_path: Path) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["verify_class_mapping.py", "--model", str(model_path), "--data", str(data_path)],
    )


def test_matching_mapping_returns_zero(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    model_path, data_path = _prepare_data(tmp_path)

    class FakePredictor:
        class_names = {0: "glioma", 1: "meningioma"}
        is_ready = True

        def __init__(self, model_path: Path) -> None:
            self.unavailable_message = ""

        def predict_file(self, image_path: Path) -> dict[str, int]:
            return {"class_index": 0 if image_path.parent.name == "glioma" else 1}

    monkeypatch.setattr(verifier, "MRIPredictor", FakePredictor)
    _set_args(monkeypatch, model_path, data_path)

    assert verifier.main() == 0


def test_swapped_mapping_returns_three(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    model_path, data_path = _prepare_data(tmp_path)

    class FakePredictor:
        class_names = {0: "glioma", 1: "meningioma"}
        is_ready = True

        def __init__(self, model_path: Path) -> None:
            self.unavailable_message = ""

        def predict_file(self, image_path: Path) -> dict[str, int]:
            return {"class_index": 1 if image_path.parent.name == "glioma" else 0}

    monkeypatch.setattr(verifier, "MRIPredictor", FakePredictor)
    _set_args(monkeypatch, model_path, data_path)

    assert verifier.main() == 3


def test_corrupt_image_is_skipped(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    model_path, data_path = _prepare_data(tmp_path, include_corrupt=True)

    class FakePredictor:
        class_names = {0: "glioma", 1: "meningioma"}
        is_ready = True

        def __init__(self, model_path: Path) -> None:
            self.unavailable_message = ""

        def predict_file(self, image_path: Path) -> dict[str, int]:
            if image_path.name == "corrupt.png":
                raise InvalidImageError("invalid image")
            return {"class_index": 0 if image_path.parent.name == "glioma" else 1}

    monkeypatch.setattr(verifier, "MRIPredictor", FakePredictor)
    _set_args(monkeypatch, model_path, data_path)

    assert verifier.main() == 0
    captured = capsys.readouterr()
    assert "Warning: skipped corrupt.png." in captured.err
    assert "Skipped: 1" in captured.out
