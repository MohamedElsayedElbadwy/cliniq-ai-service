"""Optional cached Keras inference service, with verified artifact metadata."""
from __future__ import annotations

import hashlib
import json
import logging
import threading
import time

import numpy as np

from .config import SkinSettings, get_settings
from .preprocessing import PREPROCESSING_SPEC, preprocess_skin_image
from .schemas import DISCLAIMER

logger=logging.getLogger(__name__)


class ModelUnavailableError(RuntimeError):
    pass


class ModelInferenceError(RuntimeError):
    pass


def calibrated_probabilities(logits, temperature):
    values=np.asarray(logits,dtype=np.float64)
    if values.ndim!=1 or not np.all(np.isfinite(values)) or not np.isfinite(temperature) or temperature<=0:
        raise ValueError('Invalid logits or temperature')
    values=(values-values.max())/temperature
    scores=np.exp(values)
    return scores/scores.sum()


def _load_keras(path):
    import tensorflow as tf
    return tf.keras.models.load_model(path,compile=False,safe_mode=True)


class SkinPredictor:
    def __init__(self,settings: SkinSettings,model_loader=None):
        self.settings=settings
        self.model=None
        self.class_names=[]
        self.model_version=None
        self.load_error=None
        self._semaphore=threading.BoundedSemaphore(1)
        try:
            raw=json.loads(settings.manifest_path.read_text(encoding='utf-8'))
            if not isinstance(raw,dict):
                raise ValueError('Invalid manifest')
            mapping=raw['classNames']
            if not isinstance(mapping,dict) or not mapping or set(mapping)!=set(map(str,range(len(mapping)))):
                raise ValueError('Invalid class map')
            names=[mapping[str(i)] for i in range(len(mapping))]
            if any(not isinstance(name,str) or not name.strip() for name in names) or len(set(names))!=len(names):
                raise ValueError('Invalid class labels')
            if raw['inputSize']!=[224,224] or raw['preprocessing']!=PREPROCESSING_SPEC or raw['outputType']!='logits':
                raise ValueError('Incompatible preprocessing/output')
            self.temperature=float(raw['temperature'])
            self.threshold=settings.abstain_threshold if settings.abstain_threshold is not None else float(raw['abstainThreshold'])
            if not np.isfinite(self.temperature) or self.temperature<=0 or not np.isfinite(self.threshold) or not 0<=self.threshold<=1:
                raise ValueError('Invalid calibration')
            version=raw['modelVersion']
            if not isinstance(version,str) or not version.strip():
                raise ValueError('Invalid version')
            with settings.model_path.open('rb') as stream:
                digest=hashlib.file_digest(stream,'sha256').hexdigest()
            if digest!=raw['modelSha256']:
                raise ValueError('Artifact checksum mismatch')
            model=(model_loader or _load_keras)(settings.model_path)
            if tuple(model.input_shape[1:])!=(224,224,3) or model.output_shape[-1]!=len(names):
                raise ValueError('Incompatible model shape')
            warm=np.asarray(model(np.zeros((1,224,224,3),dtype=np.float32),training=False))
            if warm.shape!=(1,len(names)) or not np.all(np.isfinite(warm)):
                raise ValueError('Invalid warmup output')
            self.model=model
            self.class_names=names
            self.model_version=version
            logger.info('Skin model loaded modelVersion=%s classCount=%d',version,len(names))
        except Exception as error:
            self.model=None
            self.load_error='Skin model is unavailable. Verify the artifact and manifest, then restart the service.'
            logger.warning('Skin initialization failed (%s)',type(error).__name__)

    @property
    def is_ready(self):
        return self.model is not None and self.load_error is None

    @property
    def unavailable_message(self):
        return self.load_error or 'Skin model is unavailable.'

    def predict_bytes(self,payload):
        if not self.is_ready:
            raise ModelUnavailableError(self.unavailable_message)
        tensor=preprocess_skin_image(payload,max_pixels=self.settings.max_pixels)
        started=time.perf_counter()
        try:
            with self._semaphore:
                output=np.asarray(self.model(tensor,training=False))
            if output.shape!=(1,len(self.class_names)):
                raise ValueError('Unexpected output shape')
            probabilities=calibrated_probabilities(output[0],self.temperature)
            indexes=np.argsort(-probabilities,kind='stable')[:min(self.settings.top_k,len(self.class_names))]
            best=int(indexes[0])
            abstained=bool(probabilities[best]<self.threshold)
            return dict(predicted_class=None if abstained else self.class_names[best],
                confidence=float(probabilities[best]),abstained=abstained,
                abstain_reason='low_confidence' if abstained else None,
                top_predictions=[dict(label=self.class_names[int(i)],probability=float(probabilities[i])) for i in indexes],
                probabilities={name:float(p) for name,p in zip(self.class_names,probabilities)},
                model_version=self.model_version,inference_ms=round((time.perf_counter()-started)*1000,3),
                disclaimer=DISCLAIMER)
        except Exception as error:
            logger.warning('Skin inference failed (%s)',type(error).__name__)
            raise ModelInferenceError('Skin model inference failed.') from error


_GLOBAL_PREDICTOR=None
_PREDICTOR_LOCK=threading.Lock()


def get_skin_predictor():
    global _GLOBAL_PREDICTOR
    with _PREDICTOR_LOCK:
        if _GLOBAL_PREDICTOR is None:
            try:
                settings=get_settings()
                _GLOBAL_PREDICTOR=SkinPredictor(settings)
            except Exception as error:
                # Bad optional config must also leave Voice/MRI startup intact.
                _GLOBAL_PREDICTOR=SkinPredictor(SkinSettings())
                _GLOBAL_PREDICTOR.model=None
                _GLOBAL_PREDICTOR.load_error='Skin configuration is invalid. Correct SKIN settings and restart.'
                logger.warning('Skin configuration failed (%s)',type(error).__name__)
    return _GLOBAL_PREDICTOR
