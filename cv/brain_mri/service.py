"""Model loading and inference service for Brain MRI image classification.

This layer owns MRI inference. It returns only dataset-class probabilities and
never adds diagnosis, treatment, triage, or clinical interpretation.
"""

from __future__ import annotations

import json
import io
import logging
from pathlib import Path
from typing import Any, Optional, Sequence, Union

import numpy as np
from PIL import Image, ImageOps

from .config import DEFAULT_CHECKPOINT_NAME, MODELS_DIR, find_model_path


logger = logging.getLogger(__name__)
DEFAULT_CLASS_NAMES = {0: "glioma", 1: "meningioma", 2: "notumor", 3: "pituitary"}
DEFAULT_TARGET_SIZE = (299, 299)
SUPPORTED_IMAGE_TYPES = Union[str, Path, bytes, bytearray, io.BytesIO, Image.Image, np.ndarray]


def load_image_to_pil(image_input: SUPPORTED_IMAGE_TYPES) -> Image.Image:
    """Decode supported image input and explicitly convert it to RGB."""
    if image_input is None:
        raise ValueError("Image input cannot be None.")
    if isinstance(image_input, Image.Image):
        return image_input.convert("RGB")
    if isinstance(image_input, np.ndarray):
        if image_input.size == 0:
            raise ValueError("Input numpy array is empty.")
        if image_input.dtype != np.uint8:
            if image_input.max() <= 1.0 and image_input.min() >= 0.0:
                image_input = (image_input * 255.0).astype(np.uint8)
            else:
                image_input = np.clip(image_input, 0, 255).astype(np.uint8)
        return Image.fromarray(image_input).convert("RGB")
    if isinstance(image_input, (bytes, bytearray, io.BytesIO)):
        if isinstance(image_input, (bytes, bytearray)) and not image_input:
            raise ValueError("Image byte buffer is empty (0 bytes).")
        stream = io.BytesIO(image_input) if isinstance(image_input, (bytes, bytearray)) else image_input
        try:
            stream.seek(0)
            with Image.open(stream) as image:
                return image.convert("RGB")
        except Exception as error:
            raise ValueError(f"Failed to decode image from byte stream: {error}") from error
    if isinstance(image_input, (str, Path)):
        path = Path(image_input)
        if not path.exists():
            raise FileNotFoundError(f"Image file does not exist at path: {path}")
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"Image file is invalid or empty: {path}")
        try:
            with Image.open(path) as image:
                return image.convert("RGB")
        except Exception as error:
            raise ValueError(f"Failed to open image file '{path}': {error}") from error
    raise TypeError(f"Unsupported input type '{type(image_input).__name__}'.")


def pad_and_resize(
    image: Image.Image,
    target_size: tuple[int, int] = DEFAULT_TARGET_SIZE,
    resample: Image.Resampling = Image.Resampling.LANCZOS,
    fill_color: tuple[int, int, int] = (0, 0, 0),
) -> Image.Image:
    """Preserve aspect ratio, resize, and pad to the model input size."""
    return ImageOps.pad(image.convert("RGB"), size=target_size, method=resample, color=fill_color, centering=(0.5, 0.5))


def normalize_xception(image_array: np.ndarray) -> np.ndarray:
    """Map image pixels from [0, 255] to the Xception range [-1, 1]."""
    return (image_array.astype(np.float32) / 127.5) - 1.0


def preprocess_mri_image(
    image_input: SUPPORTED_IMAGE_TYPES, target_size: tuple[int, int] = DEFAULT_TARGET_SIZE
) -> np.ndarray:
    """Decode, RGB-convert, aspect-pad, and Xception-normalize one image."""
    image = pad_and_resize(load_image_to_pil(image_input), target_size=target_size)
    return np.expand_dims(normalize_xception(np.asarray(image, dtype=np.float32)), axis=0)


def preprocess_mri_batch(
    images: Sequence[SUPPORTED_IMAGE_TYPES], target_size: tuple[int, int] = DEFAULT_TARGET_SIZE
) -> np.ndarray:
    """Preprocess a non-empty sequence of MRI images."""
    if not images:
        raise ValueError("Cannot preprocess an empty batch of images.")
    return np.concatenate([preprocess_mri_image(image, target_size=target_size) for image in images], axis=0)


class MRIPredictor:
    """Load the saved MRI model once and classify individual images or batches."""

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        class_names_path: Optional[Union[str, Path]] = None,
        model_instance: Any = None,
        **_deprecated_options: Any,
    ) -> None:
        self.class_names = self._load_class_names(class_names_path)
        self.model: Any = model_instance
        self.model_path = Path("in_memory_instance") if model_instance is not None else (
            Path(model_path) if model_path else find_model_path()
        )
        if model_instance is None and self.model_path and self.model_path.is_file():
            self._load_model()

    @staticmethod
    def _error(message: str) -> dict[str, Any]:
        return {"success": False, "prediction": None, "class_index": None, "confidence": 0.0, "probabilities": {}, "error": message}

    @staticmethod
    def _load_class_names(class_names_path: Optional[Union[str, Path]]) -> dict[int, str]:
        path = Path(class_names_path) if class_names_path else MODELS_DIR / "class_names.json"
        if path.is_file():
            try:
                with path.open(encoding="utf-8") as file:
                    parsed = {int(index): str(name) for index, name in json.load(file).items()}
                if set(parsed) == set(DEFAULT_CLASS_NAMES):
                    return parsed
            except (OSError, ValueError, TypeError) as error:
                logger.warning("Unable to read class mapping at %s: %s", path, error)
        return DEFAULT_CLASS_NAMES.copy()

    def _load_model(self) -> None:
        try:
            import tensorflow as tf
            self.model = tf.keras.models.load_model(self.model_path, compile=False)
        except Exception as error:
            raise RuntimeError(f"Unable to load MRI model weights from {self.model_path}: {error}") from error

    def predict(self, image_input: SUPPORTED_IMAGE_TYPES) -> dict[str, Any]:
        """Return one class prediction and the model's softmax outputs."""
        if self.model is None:
            return self._error(
                "MRI model weights are unavailable. Set CLINIQ_MRI_MODEL_PATH or place "
                f"'{DEFAULT_CHECKPOINT_NAME}' in '{MODELS_DIR}'."
            )
        try:
            tensor = preprocess_mri_image(image_input, target_size=DEFAULT_TARGET_SIZE)
            output = np.asarray(self.model.predict(tensor, verbose=0), dtype=np.float32)
            if output.shape != (1, len(self.class_names)):
                raise ValueError(f"Unexpected model output shape {output.shape}; expected (1, {len(self.class_names)}).")
            values = output[0]
            if not np.all(np.isfinite(values)) or np.any(values < 0):
                raise ValueError("Model returned invalid probability values.")
            total = float(values.sum())
            if not np.isclose(total, 1.0, atol=1e-3):
                raise ValueError(f"Model output does not sum to 1 (received {total:.6f}).")
            class_index = int(np.argmax(values))
            return {
                "success": True,
                "prediction": self.class_names[class_index],
                "class_index": class_index,
                "confidence": float(values[class_index]),
                "probabilities": {self.class_names[index]: float(value) for index, value in enumerate(values)},
                "error": None,
            }
        except Exception as error:
            logger.warning("MRI inference failed: %s", error)
            return self._error(f"Image processing or inference failed: {error}")

    def predict_batch(self, images: Sequence[SUPPORTED_IMAGE_TYPES]) -> list[dict[str, Any]]:
        return [self.predict(image) for image in images]

    def predict_file(self, file_path: Union[str, Path]) -> dict[str, Any]:
        return self.predict(file_path)

    def predict_bytes(self, raw_bytes: Union[bytes, bytearray]) -> dict[str, Any]:
        return self.predict(raw_bytes)


_GLOBAL_PREDICTOR: Optional[MRIPredictor] = None


def get_mri_predictor(model_path: Optional[Union[str, Path]] = None, reload_model: bool = False, **deprecated_options: Any) -> MRIPredictor:
    """Return a cached predictor so a web service does not reload weights per request."""
    global _GLOBAL_PREDICTOR
    if _GLOBAL_PREDICTOR is None or reload_model:
        _GLOBAL_PREDICTOR = MRIPredictor(model_path=model_path, **deprecated_options)
    return _GLOBAL_PREDICTOR


def predict_mri(image_input: SUPPORTED_IMAGE_TYPES, model_path: Optional[Union[str, Path]] = None) -> dict[str, Any]:
    """Convenience function for one MRI image classification."""
    return get_mri_predictor(model_path=model_path).predict(image_input)
