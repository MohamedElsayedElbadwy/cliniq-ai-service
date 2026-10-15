"""Portable environment configuration, following MRI's frozen dataclass pattern."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ARTIFACTS_DIR=Path(__file__).resolve().parent/'artifacts'/'six-class'
CLASS_NAMES=('Acne', 'Eczema', 'Psoriasis', 'Rosacea', 'Ringworm', 'Normal')


@dataclass(frozen=True)
class SkinSettings:
    model_path: Path=ARTIFACTS_DIR/'skin_six.keras'
    manifest_path: Path=ARTIFACTS_DIR/'model_manifest.json'
    max_upload_bytes: int=5_242_880
    max_pixels: int=20_000_000
    top_k: int=3
    abstain_threshold: float | None=None


def get_settings() -> SkinSettings:
    model=Path(os.getenv('SKIN_MODEL_PATH') or str(ARTIFACTS_DIR/'skin_six.keras')).expanduser()
    settings=SkinSettings(
        model_path=model,
        manifest_path=Path(os.getenv('SKIN_MANIFEST_PATH') or str(model.parent/'model_manifest.json')).expanduser(),
        max_upload_bytes=int(os.getenv('SKIN_MAX_UPLOAD_BYTES','5242880')),
        max_pixels=int(os.getenv('SKIN_MAX_PIXELS','20000000')),
        top_k=int(os.getenv('SKIN_TOP_K','3')),
        abstain_threshold=float(os.environ['SKIN_ABSTAIN_THRESHOLD']) if os.getenv('SKIN_ABSTAIN_THRESHOLD') else None)
    if settings.max_upload_bytes<=0 or settings.max_pixels<=0 or settings.top_k<=0:
        raise ValueError('Invalid skin resource limits')
    if settings.abstain_threshold is not None and not 0<=settings.abstain_threshold<=1:
        raise ValueError('Invalid skin abstention threshold')
    return settings
