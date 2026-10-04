"""Configuration for the MRI module.

All paths are portable: model weights can be supplied through an environment
variable or stored locally beside the lightweight model metadata.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


MODULE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODULE_DIR.parent
MODELS_DIR = PROJECT_ROOT / "models"
DEFAULT_CHECKPOINT_NAME = "mri_xception_selective_finetuning_best.keras"
MAX_UPLOAD_SIZE_BYTES = 10 * 1024 * 1024
ALLOWED_IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png"})
ALLOWED_IMAGE_CONTENT_TYPES = frozenset({"image/jpeg", "image/png"})


@dataclass(frozen=True)
class MRISettings:
    """Runtime settings for the MRI classifier."""

    model_path: Path | None
    max_upload_size_bytes: int = MAX_UPLOAD_SIZE_BYTES


def find_model_path() -> Path | None:
    """Return the configured checkpoint or the exact default checkpoint, if present."""
    configured_path = os.getenv("CLINIQ_MRI_MODEL_PATH") or os.getenv("MRI_MODEL_PATH")
    if configured_path:
        candidate = Path(configured_path).expanduser()
        if candidate.is_file():
            return candidate

    expected_path = MODELS_DIR / DEFAULT_CHECKPOINT_NAME
    if expected_path.is_file():
        return expected_path

    return None


def get_settings() -> MRISettings:
    """Create settings at call time so environment changes are respected."""
    return MRISettings(model_path=find_model_path())
