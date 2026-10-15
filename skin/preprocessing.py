"""One preprocessing path for training, evaluation and production inference."""
from __future__ import annotations

import io
import warnings
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

PREPROCESSING_SPEC='exif-rgb-direct-resize-float32-0-255-v1'
TARGET_SIZE=(224,224)


class InvalidImageError(ValueError):
    pass


class ImageTooLargeError(InvalidImageError):
    pass


def decode_image(source, max_pixels=20_000_000) -> Image.Image:
    """Decode all pixels with EXIF orientation before metadata removal."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error',Image.DecompressionBombWarning)
            if isinstance(source,Image.Image):
                im=source
                if im.width*im.height>max_pixels:
                    raise ImageTooLargeError('Image dimensions exceed the allowed pixel limit.')
                result=ImageOps.exif_transpose(im).convert('RGB').copy()
            else:
                stream=io.BytesIO(source) if isinstance(source,(bytes,bytearray)) else source
                with Image.open(stream) as im:
                    if im.width*im.height>max_pixels:
                        raise ImageTooLargeError('Image dimensions exceed the allowed pixel limit.')
                    im.load()
                    result=ImageOps.exif_transpose(im).convert('RGB').copy()
            result.info.clear()
            return result
    except (Image.DecompressionBombError,Image.DecompressionBombWarning) as error:
        raise ImageTooLargeError('Image dimensions exceed the allowed pixel limit.') from error
    except ImageTooLargeError:
        raise
    except Exception as error:
        raise InvalidImageError('The uploaded image cannot be decoded.') from error


def preprocess_skin_image(source, max_pixels=20_000_000) -> np.ndarray:
    """224×224 NHWC float32 [0,255]; EfficientNet owns normalization."""
    image=decode_image(source,max_pixels=max_pixels)
    return np.asarray(image.resize(TARGET_SIZE,Image.Resampling.BILINEAR),dtype=np.float32)[None,...]
