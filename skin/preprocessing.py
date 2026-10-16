"""One preprocessing path for training, evaluation and production inference."""
from __future__ import annotations

import io
import warnings
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from .config import DEFAULT_MAX_PIXELS

PREPROCESSING_SPEC='exif-rgb-direct-resize-float32-0-255-v1'
TARGET_SIZE=(224,224)


class InvalidImageError(ValueError):
    pass


class ImageTooLargeError(InvalidImageError):
    pass


def _pixel_limit_message(max_pixels):
    limit=f'{max_pixels/1_000_000:g} megapixels' if max_pixels>=1_000_000 else f'{max_pixels} pixels'
    return f'Image dimensions exceed the allowed pixel limit of {limit}. Resize the image and try again.'


def decode_image(source, max_pixels=DEFAULT_MAX_PIXELS) -> Image.Image:
    """Decode all pixels with EXIF orientation before metadata removal."""
    try:
        with warnings.catch_warnings():
            # Enforce our configured cap before decoding; Pillow's warning
            # threshold is lower than supported phone-camera resolutions.
            # Keep its hard-error protection and global settings unchanged.
            warnings.simplefilter('ignore',Image.DecompressionBombWarning)
            if isinstance(source,Image.Image):
                im=source
                if im.width*im.height>max_pixels:
                    raise ImageTooLargeError(_pixel_limit_message(max_pixels))
                result=ImageOps.exif_transpose(im).convert('RGB')
            else:
                stream=io.BytesIO(source) if isinstance(source,(bytes,bytearray)) else source
                with Image.open(stream) as im:
                    if im.width*im.height>max_pixels:
                        raise ImageTooLargeError(_pixel_limit_message(max_pixels))
                    im.load()
                    result=ImageOps.exif_transpose(im).convert('RGB')
            result.info.clear()
            return result
    except (Image.DecompressionBombError,Image.DecompressionBombWarning) as error:
        raise ImageTooLargeError(_pixel_limit_message(max_pixels)) from error
    except ImageTooLargeError:
        raise
    except Exception as error:
        raise InvalidImageError('The uploaded image cannot be decoded.') from error


def preprocess_skin_image(source, max_pixels=DEFAULT_MAX_PIXELS) -> np.ndarray:
    """224×224 NHWC float32 [0,255]; EfficientNet owns normalization."""
    with decode_image(source,max_pixels=max_pixels) as image:
        return np.asarray(image.resize(TARGET_SIZE,Image.Resampling.BILINEAR),dtype=np.float32)[None,...]
