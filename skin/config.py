"""Portable environment configuration, following MRI's frozen dataclass pattern."""
from __future__ import annotations

import os
import math
from dataclasses import dataclass
from pathlib import Path

ARTIFACTS_DIR=Path(__file__).resolve().parent/'artifacts'/'six-class'
CLASS_NAMES=('Acne', 'Eczema', 'Psoriasis', 'Rosacea', 'Ringworm', 'Normal')
DEFAULT_MAX_UPLOAD_BYTES=10*1024*1024
DEFAULT_MAX_PIXELS=120_000_000
PRETRAINED_DIR=Path(__file__).resolve().parents[1]/'workspaces'/'pretrained-skin'/'1b4fccab2c8b83bf6964e394c40f0dcdd21b1d1c'
MAKE_DIR=Path(__file__).resolve().parents[1]/'workspaces'/'pretrained-make'/'5efa1915f1d0128451ddd61242aad8475bfb8e1a'


@dataclass(frozen=True)
class SkinSettings:
    model_path: Path=ARTIFACTS_DIR/'skin_six.keras'
    manifest_path: Path=ARTIFACTS_DIR/'model_manifest.json'
    max_upload_bytes: int=DEFAULT_MAX_UPLOAD_BYTES
    max_pixels: int=DEFAULT_MAX_PIXELS
    top_k: int=3
    abstain_threshold: float | None=None
    backend: str='keras'
    pretrained_dir: Path=PRETRAINED_DIR
    make_dir: Path=MAKE_DIR
    pretrained_min_confidence: float=.70
    pretrained_min_margin: float=.15
    pretrained_threads: int=2


def get_settings() -> SkinSettings:
    model=Path(os.getenv('SKIN_MODEL_PATH') or str(ARTIFACTS_DIR/'skin_six.keras')).expanduser()
    settings=SkinSettings(
        model_path=model,
        manifest_path=Path(os.getenv('SKIN_MANIFEST_PATH') or str(model.parent/'model_manifest.json')).expanduser(),
        max_upload_bytes=int(os.getenv('SKIN_MAX_UPLOAD_BYTES',str(DEFAULT_MAX_UPLOAD_BYTES))),
        max_pixels=int(os.getenv('SKIN_MAX_PIXELS',str(DEFAULT_MAX_PIXELS))),
        top_k=int(os.getenv('SKIN_TOP_K','3')),
        abstain_threshold=float(os.environ['SKIN_ABSTAIN_THRESHOLD']) if os.getenv('SKIN_ABSTAIN_THRESHOLD') else None,
        backend=(os.getenv('SKIN_BACKEND') or 'keras').strip().lower(),
        pretrained_dir=Path(os.getenv('SKIN_PRETRAINED_DIR') or str(PRETRAINED_DIR)).expanduser(),
        make_dir=Path(os.getenv('SKIN_MAKE_DIR') or str(MAKE_DIR)).expanduser(),
        pretrained_min_confidence=float(os.getenv('SKIN_PRETRAINED_MIN_CONFIDENCE') or '.70'),
        pretrained_min_margin=float(os.getenv('SKIN_PRETRAINED_MIN_MARGIN') or '.15'),
        pretrained_threads=int(os.getenv('SKIN_PRETRAINED_THREADS') or '2'))
    if settings.max_upload_bytes<=0 or settings.max_pixels<=0 or settings.top_k<=0:
        raise ValueError('Invalid skin resource limits')
    if settings.abstain_threshold is not None and not 0<=settings.abstain_threshold<=1:
        raise ValueError('Invalid skin abstention threshold')
    if settings.backend not in {'keras','pretrained','make'} or settings.pretrained_threads<=0:
        raise ValueError('Invalid skin backend or CPU thread count')
    for value in [settings.pretrained_min_confidence,settings.pretrained_min_margin]:
        if not math.isfinite(value) or not 0<=value<=1:
            raise ValueError('Invalid pretrained rejection threshold')
    return settings
