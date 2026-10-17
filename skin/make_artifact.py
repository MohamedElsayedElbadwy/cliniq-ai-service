"""Immutable MAKE checkpoint and fixed prospective zero-shot task definition."""
import json
from .config import CLASS_NAMES
from .pretrained_artifact import file_matches

MODEL_ID='xieji-x/MAKE'
REVISION='5efa1915f1d0128451ddd61242aad8475bfb8e1a'
LABELS=CLASS_NAMES
PROMPTS=tuple('This is a skin image of '+name for name in ('acne','eczema','psoriasis','rosacea','ringworm','healthy skin'))
LOGIT_SCALE=100.
ARTIFACT_FILES={
    'README.md':(1092,'d394de27b0adde63568e20517a9db0a04efed36138e693dcab988ccb7eae6842'),
    'open_clip_config.json':(532,'274003b2ce3821652bddd63ad34b01a33f7a249172e2981aa25da4819a005ca5'),
    'open_clip_model.safetensors':(598516980,'4ed233070183558b9f9e4572bdb2676f7da0f2c463d9b4d436b169e62700bf47')}
MODEL_CONFIG={
    'model_cfg':{'embed_dim':512,'vision_cfg':{'image_size':224,'layers':12,'width':768,'patch_size':16},
                 'text_cfg':{'context_length':77,'vocab_size':49408,'width':512,'heads':8,'layers':12}},
    'preprocess_cfg':{'mean':[.48145466,.4578275,.40821073],'std':[.26862954,.26130258,.27577711],
                      'interpolation':'bicubic','resize_mode':'shortest'}}


def verify_artifact(directory):
    for name,spec in ARTIFACT_FILES.items():
        if not file_matches(directory/name,spec):raise ValueError('Missing or mismatched pinned MAKE artifact: '+name)
    if json.loads((directory/'open_clip_config.json').read_text(encoding='utf-8'))!=MODEL_CONFIG:
        raise ValueError('Unexpected MAKE architecture or preprocessing')
