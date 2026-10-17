"""Pinned public checkpoint identity; verification performs no network requests."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

MODEL_ID='LaurianeMD/vit-skin-disease'
REVISION='1b4fccab2c8b83bf6964e394c40f0dcdd21b1d1c'
LABELS=('Acne','Actinic Keratosis','Benign Tumors','Bullous','Candidiasis','Drug Eruption',
        'Eczema','Infestations Bites','Lichen','Lupus','Moles','Psoriasis','Rosacea',
        'Seborrh Keratoses','Skin Cancer','Sun Sunlight Damage','Tinea','Unknown Normal',
        'Vascular Tumors','Vasculitis','Vitiligo','Warts')
ARTIFACT_FILES={
    'config.json':(1628,'345165a41f7c7e025da20fa40e8d6d173143fa9c12934881d32731a6bc6817a6'),
    'preprocessor_config.json':(351,'445c77049d082004aa07593344d5e1f521f1198228bf586196f70ce7ae021414'),
    'README.md':(7754,'9d74859ce042dd24aef4b36e223370239108061ca18a10324a186af8d0b7ac4e'),
    'model.safetensors':(343285496,'3d85c0ef41fc65937401ae1fa5cc37a8ec139754ef5e1e21539603417c6ae29c'),
}
PROCESSOR_SPEC={'do_convert_rgb':None,'do_normalize':True,'do_rescale':True,'do_resize':True,
    'image_mean':[.5,.5,.5],'image_processor_type':'ViTImageProcessor','image_std':[.5,.5,.5],
    'resample':2,'rescale_factor':1/255,'size':{'height':224,'width':224}}


def file_matches(path:Path,spec):
    if not path.is_file() or path.stat().st_size!=spec[0]:return False
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()==spec[1]


def verify_artifact(directory:Path):
    directory=Path(directory)
    for name,spec in ARTIFACT_FILES.items():
        if not file_matches(directory/name,spec):raise ValueError(f'Pinned pretrained artifact missing or checksum mismatch: {name}')
    config=json.loads((directory/'config.json').read_text(encoding='utf-8'))
    if config.get('model_type')!='vit' or config.get('architectures')!=['ViTForImageClassification']:
        raise ValueError('Pretrained architecture mismatch')
    if config.get('id2label')!={str(i):name for i,name in enumerate(LABELS)}:
        raise ValueError('Pretrained native class map mismatch')
    processor=json.loads((directory/'preprocessor_config.json').read_text(encoding='utf-8'))
    if processor!=PROCESSOR_SPEC:raise ValueError('Pretrained preprocessing mismatch')
    return config
