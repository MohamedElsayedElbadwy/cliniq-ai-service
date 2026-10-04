"""Shared MRI test fixtures."""

from pathlib import Path

import pytest


@pytest.fixture
def class_names_path(tmp_path: Path) -> Path:
    """Write metadata matching the mocked four-output model."""
    path = tmp_path / "class_names.json"
    path.write_text(
        '{"0": "glioma", "1": "meningioma", "2": "notumor", "3": "pituitary"}',
        encoding="utf-8",
    )
    return path
