"""Model loading and inference service for Brain MRI image classification.

This layer owns MRI inference. It returns only dataset-class probabilities and
never adds diagnosis, treatment, triage, or clinical interpretation.
"""

from __future__ import annotations

import io
import json
import logging
from pathlib import Path
from typing import Any, Optional, Sequence, Union

import numpy as np
from PIL import Image, ImageOps

from .config import DEFAULT_CHECKPOINT_NAME, MODELS_DIR, find_class_names_path, find_model_path


logger = logging.getLogger(__name__)
DEFAULT_TARGET_SIZE = (299, 299)
SUPPORTED_IMAGE_TYPES = Union[str, Path, bytes, bytearray, io.BytesIO, Image.Image, np.ndarray]


class ModelUnavailableError(RuntimeError):
    """Raised when the MRI predictor has no usable model instance."""


class InvalidImageError(ValueError):
    """Raised when an image cannot be processed for MRI inference."""


class ModelInferenceError(RuntimeError):
    """Raised when an available model cannot produce a valid prediction."""


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
    ) -> None:
        self.model: Any = model_instance
        self.class_names: dict[int, str] = {}
        self.load_error: str | None = None
        self.model_path = Path("in_memory_instance") if model_instance is not None else (
            Path(model_path) if model_path else find_model_path()
        )
        self.class_names_path = Path(class_names_path) if class_names_path else find_class_names_path()

        try:
            self.class_names = self._load_class_names(self.class_names_path)
        except ValueError as error:
            self.load_error = str(error)

        if self.load_error is None and model_instance is None and self.model_path and self.model_path.is_file():
            self._load_model()
        elif self.load_error is None and model_instance is None:
            self.load_error = self._missing_model_message()

        if self.load_error is None and self.model is not None:
            self._validate_output_size()

    @property
    def is_ready(self) -> bool:
        """Whether a model instance is available for inference."""
        return self.model is not None and bool(self.class_names) and self.load_error is None

    @staticmethod
    def _load_class_names(path: Path) -> dict[int, str]:
        """Load and validate training-run class metadata without a fallback."""
        if not path.is_file():
            raise ValueError(f"Class mapping metadata '{path.name}' is missing.")
        try:
            with path.open(encoding="utf-8") as file:
                raw_mapping = json.load(file)
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"Class mapping metadata '{path.name}' is invalid.") from error

        if not isinstance(raw_mapping, dict) or not raw_mapping:
            raise ValueError(f"Class mapping metadata '{path.name}' must be a non-empty JSON object.")
        try:
            parsed = {int(index): name for index, name in raw_mapping.items()}
        except (TypeError, ValueError) as error:
            raise ValueError(f"Class mapping metadata '{path.name}' has non-integer indexes.") from error
        if len(parsed) != len(raw_mapping) or set(parsed) != set(range(len(parsed))):
            raise ValueError(f"Class mapping metadata '{path.name}' must use consecutive indexes starting at 0.")
        if any(not isinstance(name, str) or not name.strip() for name in parsed.values()):
            raise ValueError(f"Class mapping metadata '{path.name}' contains an empty class name.")
        if len(set(parsed.values())) != len(parsed):
            raise ValueError(f"Class mapping metadata '{path.name}' contains duplicate class names.")
        return parsed

    def _load_model(self) -> None:
        try:
            import tensorflow as tf
            self.model = tf.keras.models.load_model(self.model_path, compile=False)
        except Exception as error:
            logger.warning("Unable to load MRI model weights from %s: %s", self.model_path, error)
            self.load_error = f"Unable to load MRI model weights '{self.model_path.name}'."

    def _validate_output_size(self) -> None:
        """Ensure training metadata and model output dimensions agree."""
        try:
            output_size = self.model.output_shape[-1]
        except (AttributeError, IndexError, TypeError) as error:
            self.load_error = "MRI model output shape is unavailable."
            logger.warning("Unable to determine MRI model output shape: %s", error)
            return
        if not isinstance(output_size, (int, np.integer)) or output_size != len(self.class_names):
            self.load_error = "MRI model output size does not match class mapping metadata."
            logger.warning("MRI model output size %s does not match %s classes.", output_size, len(self.class_names))

    def _missing_model_message(self) -> str:
        return (
            "MRI model weights are unavailable. Set CLINIQ_MRI_MODEL_PATH or place "
            f"'{DEFAULT_CHECKPOINT_NAME}' in '{MODELS_DIR.name}'."
        )

    @property
    def unavailable_message(self) -> str:
        """Readable reason why the predictor cannot currently serve requests."""
        return self.load_error or self._missing_model_message()

    def predict(self, image_input: SUPPORTED_IMAGE_TYPES) -> dict[str, Any]:
        """Return one class prediction and the model's softmax outputs."""
        if not self.is_ready:
            raise ModelUnavailableError(self.unavailable_message)
        try:
            tensor = preprocess_mri_image(image_input, target_size=DEFAULT_TARGET_SIZE)
        except Exception as error:
            logger.warning("MRI image preprocessing failed: %s", error)
            raise InvalidImageError(f"Image processing failed: {error}") from error

        try:
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
            raise ModelInferenceError("MRI model inference failed.") from error

    def predict_batch(self, images: Sequence[SUPPORTED_IMAGE_TYPES]) -> list[dict[str, Any]]:
        return [self.predict(image) for image in images]

    def predict_file(self, file_path: Union[str, Path]) -> dict[str, Any]:
        return self.predict(file_path)

    def predict_bytes(self, raw_bytes: Union[bytes, bytearray]) -> dict[str, Any]:
        return self.predict(raw_bytes)


_GLOBAL_PREDICTOR: Optional[MRIPredictor] = None


def get_mri_predictor(model_path: Optional[Union[str, Path]] = None, reload_model: bool = False) -> MRIPredictor:
    """Return a cached predictor so a web service does not reload weights per request."""
    global _GLOBAL_PREDICTOR
    if _GLOBAL_PREDICTOR is None or reload_model:
        _GLOBAL_PREDICTOR = MRIPredictor(model_path=model_path)
    return _GLOBAL_PREDICTOR


def predict_mri(image_input: SUPPORTED_IMAGE_TYPES, model_path: Optional[Union[str, Path]] = None) -> dict[str, Any]:
    """Convenience function for one MRI image classification."""
    return get_mri_predictor(model_path=model_path).predict(image_input)
