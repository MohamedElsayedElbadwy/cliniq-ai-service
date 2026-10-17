"""Optional CPU ViT backend, official processor, native labels, uncalibrated scores."""
from __future__ import annotations

import logging
import os
import threading
import time

import numpy as np

from .preprocessing import decode_image,ImageTooLargeError
from .pretrained_artifact import LABELS,MODEL_ID,REVISION,verify_artifact
from .service import ModelInferenceError,ModelUnavailableError,calibrated_probabilities

logger=logging.getLogger(__name__)
PRETRAINED_DISCLAIMER=('Educational image classification only. Not a medical diagnosis or prescription. '
    'Scores are uncalibrated; confident errors and unsupported conditions can be missed. '
    'Unknown Normal does not confirm healthy skin.')


class OfficialViTRuntime:
    def __init__(self,directory,threads):
        # This flag only disables Transformers' optional TF integration; MRI still
        # imports TensorFlow normally. No torchvision/accelerate/GPU runtime needed.
        os.environ.setdefault('USE_TF','0')
        import torch
        from transformers import ViTImageProcessor,ViTForImageClassification
        torch.set_num_threads(threads)
        self.torch=torch
        self.processor=ViTImageProcessor.from_pretrained(str(directory),local_files_only=True)
        self.model,info=ViTForImageClassification.from_pretrained(str(directory),local_files_only=True,
            use_safetensors=True,trust_remote_code=False,output_loading_info=True)
        if any(info.get(key) for key in ['missing_keys','unexpected_keys','mismatched_keys','error_msgs']):
            raise ValueError('Checkpoint did not load all expected ViT parameters')
        self.model.to('cpu').eval().requires_grad_(False)
        self.class_names=[self.model.config.id2label[i] for i in range(self.model.config.num_labels)]

    def predict(self,image):
        inputs=self.processor(images=image,return_tensors='pt')
        with self.torch.inference_mode():logits=self.model(**inputs).logits
        return logits[0].float().cpu().numpy()


class PretrainedSkinPredictor:
    labels=LABELS
    model_id=MODEL_ID
    revision=REVISION
    directory_setting='pretrained_dir'
    runtime_type=OfficialViTRuntime
    artifact_verifier=staticmethod(verify_artifact)
    disclaimer=PRETRAINED_DISCLAIMER
    rejected_labels={'Unknown Normal':'unknown_or_normal'}

    def __init__(self,settings,runtime_loader=None):
        self.settings=settings;self.model=None;self.class_names=[];self.model_version=None;self.load_error=None
        self._semaphore=threading.BoundedSemaphore(1)
        try:
            directory=getattr(settings,self.directory_setting)
            self.artifact_verifier(directory)
            runtime=(runtime_loader or self.runtime_type)(directory,settings.pretrained_threads)
            if tuple(runtime.class_names)!=self.labels:raise ValueError('Loaded model class map mismatch')
            self.model=runtime;self.class_names=list(self.labels)
            self.model_version=self.model_id+'@'+self.revision
            self.threshold=settings.abstain_threshold if settings.abstain_threshold is not None else settings.pretrained_min_confidence
            self.margin=settings.pretrained_min_margin
            if not np.isfinite(self.threshold) or not 0<=self.threshold<=1 or not np.isfinite(self.margin) or not 0<=self.margin<=1:
                raise ValueError('Invalid pretrained uncertainty thresholds')
            logger.info('Pretrained Skin loaded modelVersion=%s classCount=%d device=cpu',self.model_version,len(self.labels))
        except Exception as error:
            self.model=None;self.class_names=[];self.model_version=None
            self.load_error='Pretrained Skin model unavailable. Install optional dependencies, verify the pinned local download, and restart. No model was substituted.'
            logger.warning('Pretrained Skin initialization failed (%s)',type(error).__name__)

    @property
    def is_ready(self):return self.model is not None and self.load_error is None

    @property
    def unavailable_message(self):return self.load_error or 'Pretrained Skin model unavailable.'

    def predict_bytes(self,payload,*,max_pixels=None):
        if not self.is_ready:raise ModelUnavailableError(self.unavailable_message)
        started=time.perf_counter()
        with self._semaphore:
            with decode_image(payload,max_pixels=self.settings.max_pixels if max_pixels is None else max_pixels) as image:
                try:
                    logits=np.asarray(self.model.predict(image))
                    if logits.shape!=(len(self.labels),):raise ValueError('Unexpected model output shape')
                    # Temperature1 is ordinary softmax, with no fitted calibration.
                    scores=calibrated_probabilities(logits,1.)
                    order=np.argsort(-scores,kind='stable');best=int(order[0])
                    reason=None
                    if self.labels[best] in self.rejected_labels:reason=self.rejected_labels[self.labels[best]]
                    elif scores[best]<self.threshold:reason='low_confidence'
                    elif scores[best]-scores[order[1]]<self.margin:reason='ambiguous_prediction'
                    return dict(predicted_class=None if reason else self.labels[best],confidence=float(scores[best]),
                        abstained=reason is not None,abstain_reason=reason,
                        top_predictions=[dict(label=self.labels[int(i)],probability=float(scores[i])) for i in order[:min(self.settings.top_k,len(self.labels))]],
                        probabilities={name:float(p) for name,p in zip(self.labels,scores)},model_version=self.model_version,
                        inference_ms=round((time.perf_counter()-started)*1000,3),disclaimer=self.disclaimer)
                except ImageTooLargeError:
                    raise
                except Exception as error:
                    logger.warning('Pretrained Skin inference failed (%s)',type(error).__name__)
                    raise ModelInferenceError('Pretrained Skin inference failed.') from error
