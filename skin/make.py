"""Local, fixed-taxonomy zero-shot MAKE inference; no training or network calls."""
import os

from .make_artifact import LABELS,MODEL_ID,REVISION,PROMPTS,LOGIT_SCALE,verify_artifact
from .pretrained import PretrainedSkinPredictor
from .config import DEFAULT_MAX_PIXELS
from .preprocessing import ImageTooLargeError


class OfficialMakeRuntime:
    def __init__(self,directory,threads):
        os.environ.setdefault('USE_TF','0')
        import torch
        import open_clip
        torch.set_num_threads(threads)
        self.torch=torch
        # Official local-dir loader uses strict checkpoint loading, the pinned
        # safetensors filename and its official eval preprocessing. No HF calls.
        local='local-dir:'+str(directory.resolve())
        self.model,_,self.processor=open_clip.create_model_and_transforms(local,device='cpu')
        self.model.eval().requires_grad_(False)
        tokenizer=open_clip.get_tokenizer(local)
        with torch.inference_mode():
            text=self.model.encode_text(tokenizer(list(PROMPTS)))
            self.text_features=text/text.norm(dim=-1,keepdim=True)
        if not torch.isfinite(self.text_features).all():raise ValueError('Invalid MAKE text embeddings')
        self.class_names=list(LABELS)

    def predict(self,image):
        # Shortest-side resizing can expand a tiny but extremely thin image to
        # billions of pixels before center crop. Check with integers first.
        short,long=sorted(image.size)
        if 224*((224*long)//short)>DEFAULT_MAX_PIXELS:
            raise ImageTooLargeError('Image aspect ratio makes the model resize exceed the 120 megapixel intermediate limit. Crop the image and try again.')
        inputs=self.processor(image).unsqueeze(0)
        with self.torch.inference_mode():
            features=self.model.encode_image(inputs)
            features=features/features.norm(dim=-1,keepdim=True)
            logits=LOGIT_SCALE*features@self.text_features.T
        return logits[0].float().cpu().numpy()


class MakeSkinPredictor(PretrainedSkinPredictor):
    labels=LABELS
    model_id=MODEL_ID
    revision=REVISION
    directory_setting='make_dir'
    runtime_type=OfficialMakeRuntime
    artifact_verifier=staticmethod(verify_artifact)
    rejected_labels={'Normal':'normal_unverified'}
    disclaimer=('Educational zero-shot image classification only. Not a medical diagnosis or prescription. '
        'Scores are uncalibrated and conditional on six fixed candidate descriptions. '
        'Unsupported conditions and non-skin images can receive confident scores. '
        'Normal is not clinically verified and is always withheld.')
