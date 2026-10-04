"""
Unit tests for the MRI inference and preprocessing service.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from mri.service import (
    DEFAULT_CLASS_NAMES,
    DEFAULT_TARGET_SIZE,
    MRIPredictor,
    load_image_to_pil,
    normalize_xception,
    pad_and_resize,
    preprocess_mri_batch,
    preprocess_mri_image,
)


class MockKerasModel:
    """Simulates Keras model output without TensorFlow dependency."""

    def __init__(self, probabilities: list[float] | None = None) -> None:
        self.probabilities = probabilities or [0.10, 0.70, 0.10, 0.10]

    def predict(self, batch: np.ndarray, verbose: int = 0) -> np.ndarray:
        assert batch.shape == (len(batch), 299, 299, 3)
        return np.tile(self.probabilities, (len(batch), 1))


# ------------------------------------------------------------------------------
# 1. Preprocessing Unit Tests
# ------------------------------------------------------------------------------

def test_load_image_to_pil_rgb() -> None:
    img = Image.new("RGB", (100, 100), color=(255, 0, 0))
    result = load_image_to_pil(img)
    assert result.mode == "RGB"
    assert result.size == (100, 100)


def test_load_image_to_pil_grayscale() -> None:
    img = Image.new("L", (80, 80), color=128)
    result = load_image_to_pil(img)
    assert result.mode == "RGB"
    assert result.getpixel((0, 0)) == (128, 128, 128)


def test_load_image_to_pil_rgba() -> None:
    img = Image.new("RGBA", (50, 50), color=(255, 128, 0, 200))
    result = load_image_to_pil(img)
    assert result.mode == "RGB"


def test_load_image_to_pil_raw_bytes() -> None:
    img = Image.new("RGB", (64, 64), color=(0, 255, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    result = load_image_to_pil(buf.getvalue())
    assert result.mode == "RGB"
    assert result.size == (64, 64)


def test_load_image_to_pil_numpy_array() -> None:
    arr = np.zeros((120, 120, 3), dtype=np.uint8)
    arr[:, :] = [50, 100, 150]
    result = load_image_to_pil(arr)
    assert result.mode == "RGB"
    assert result.size == (120, 120)


def test_load_image_to_pil_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="Image input cannot be None"):
        load_image_to_pil(None)

    with pytest.raises(ValueError, match="Image byte buffer is empty"):
        load_image_to_pil(b"")

    with pytest.raises(ValueError, match="Failed to decode image"):
        load_image_to_pil(b"not_valid_image_bytes")

    with pytest.raises(FileNotFoundError):
        load_image_to_pil("non_existent_file_path.jpg")


def test_pad_and_resize_aspect_ratio_preservation() -> None:
    wide_img = Image.new("RGB", (400, 200), color=(200, 200, 200))
    padded = pad_and_resize(wide_img, target_size=(299, 299))
    assert padded.size == (299, 299)
    assert padded.mode == "RGB"
    # Border margins should be black (0, 0, 0)
    assert padded.getpixel((149, 5)) == (0, 0, 0)
    # Center should contain original image color
    assert padded.getpixel((149, 149)) == (200, 200, 200)


def test_normalize_xception() -> None:
    test_arr = np.array([0.0, 127.5, 255.0], dtype=np.float32)
    norm = normalize_xception(test_arr)
    np.testing.assert_allclose(norm, np.array([-1.0, 0.0, 1.0], dtype=np.float32), atol=1e-5)


def test_preprocess_mri_image() -> None:
    img = Image.new("RGB", (256, 256), color=(50, 50, 50))
    tensor = preprocess_mri_image(img)
    assert tensor.shape == (1, 299, 299, 3)
    assert tensor.dtype == np.float32
    assert -1.0 <= tensor.min() <= tensor.max() <= 1.0


def test_preprocess_mri_batch() -> None:
    img1 = Image.new("RGB", (100, 100), color=(10, 10, 10))
    img2 = Image.new("RGB", (200, 150), color=(20, 20, 20))
    batch = preprocess_mri_batch([img1, img2])
    assert batch.shape == (2, 299, 299, 3)


# ------------------------------------------------------------------------------
# 2. Predictor Service Tests
# ------------------------------------------------------------------------------

def test_mri_predictor_success() -> None:
    # 0: glioma, 1: meningioma, 2: notumor, 3: pituitary
    mock_model = MockKerasModel(probabilities=[0.05, 0.85, 0.05, 0.05])
    predictor = MRIPredictor(model_instance=mock_model)

    dummy_img = Image.new("RGB", (200, 200), color=(60, 60, 60))
    result = predictor.predict(dummy_img)

    assert result["success"] is True
    assert result["prediction"] == "meningioma"
    assert result["class_index"] == 1
    assert result["confidence"] == pytest.approx(0.85, abs=1e-3)
    assert sum(result["probabilities"].values()) == pytest.approx(1.0, abs=1e-3)
    assert result["error"] is None


def test_mri_predictor_predict_bytes() -> None:
    mock_model = MockKerasModel(probabilities=[0.10, 0.10, 0.70, 0.10])
    predictor = MRIPredictor(model_instance=mock_model)

    buf = io.BytesIO()
    Image.new("RGB", (64, 64), color=(30, 30, 30)).save(buf, format="JPEG")
    result = predictor.predict_bytes(buf.getvalue())

    assert result["success"] is True
    assert result["prediction"] == "notumor"
    assert result["class_index"] == 2
    assert result["confidence"] == pytest.approx(0.70, abs=1e-3)


def test_mri_predictor_batch() -> None:
    mock_model = MockKerasModel(probabilities=[0.90, 0.04, 0.03, 0.03])
    predictor = MRIPredictor(model_instance=mock_model)

    imgs = [Image.new("RGB", (50, 50)), Image.new("RGB", (60, 60))]
    results = predictor.predict_batch(imgs)

    assert len(results) == 2
    for res in results:
        assert res["success"] is True
        assert res["prediction"] == "glioma"


def test_mri_predictor_unweighted_error() -> None:
    predictor = MRIPredictor(model_path="non_existent_weights_file.keras")
    dummy_img = Image.new("RGB", (50, 50))
    result = predictor.predict(dummy_img)

    assert result["success"] is False
    assert result["prediction"] is None
    assert "MRI model weights are unavailable" in result["error"]


def test_mri_predictor_corrupt_payload_graceful_error() -> None:
    mock_model = MockKerasModel()
    predictor = MRIPredictor(model_instance=mock_model)

    result = predictor.predict(b"corrupted_invalid_data")
    assert result["success"] is False
    assert result["prediction"] is None
    assert "Image processing or inference failed" in result["error"]
