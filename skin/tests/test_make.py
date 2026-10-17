"""Fixed MAKE taxonomy, official runtime integration, and fail-closed behavior."""
import hashlib
import io
import json

import numpy as np
import pytest
from PIL import Image

from skin.config import SkinSettings,get_settings


def png():
    stream=io.BytesIO();Image.new('RGB',(31,79),(140,80,60)).save(stream,format='PNG')
    return stream.getvalue()


@pytest.fixture
def make_artifact(tmp_path,monkeypatch):
    from skin import make_artifact as module
    files={'README.md':b'MIT research model','open_clip_config.json':json.dumps(module.MODEL_CONFIG).encode(),
           'open_clip_model.safetensors':b'unit-test runtime injection only'}
    monkeypatch.setattr(module,'ARTIFACT_FILES',{name:(len(data),hashlib.sha256(data).hexdigest()) for name,data in files.items()})
    for name,data in files.items():(tmp_path/name).write_bytes(data)
    return tmp_path


def predictor(directory,logits):
    from skin.make import MakeSkinPredictor
    from skin.make_artifact import LABELS
    class Runtime:
        class_names=list(LABELS)
        def predict(self,image):
            assert image.mode=='RGB' and image.size==(31,79)
            return np.asarray(logits)
    return MakeSkinPredictor(SkinSettings(backend='make',make_dir=directory),runtime_loader=lambda *_:Runtime())


def test_make_selection_is_explicit_and_cached(monkeypatch):
    from skin import make,service
    monkeypatch.setenv('SKIN_BACKEND','make');assert get_settings().backend=='make'
    created=[]
    class Stub:
        def __init__(self,settings):created.append(settings)
    monkeypatch.setattr(make,'MakeSkinPredictor',Stub);monkeypatch.setattr(service,'_GLOBAL_PREDICTOR',None)
    first=service.get_skin_predictor();assert service.get_skin_predictor() is first
    assert len(created)==1


def test_make_fixed_mapping_and_uncalibrated_output(make_artifact):
    result=predictor(make_artifact,[0.,12.,0.,0.,0.,0.]).predict_bytes(png())
    assert result['predicted_class']=='Eczema' and not result['abstained']
    assert list(result['probabilities'])==['Acne','Eczema','Psoriasis','Rosacea','Ringworm','Normal']
    assert 'MAKE@5efa1915' in result['model_version']
    assert 'zero-shot' in result['disclaimer'] and 'uncalibrated' in result['disclaimer']
    assert len(result['top_predictions'])==3


@pytest.mark.parametrize('logits,reason',[([0.]*6,'low_confidence'),([0.]*5+[15.],'normal_unverified')])
def test_make_abstains_for_uncertainty_and_unverified_normal(make_artifact,logits,reason):
    result=predictor(make_artifact,logits).predict_bytes(png())
    assert result['predicted_class'] is None and result['abstained']
    assert result['abstain_reason']==reason


def test_make_tampered_weights_never_fall_back(make_artifact):
    from skin.make import MakeSkinPredictor
    (make_artifact/'open_clip_model.safetensors').write_bytes(b'changed')
    called=[]
    model=MakeSkinPredictor(SkinSettings(make_dir=make_artifact),runtime_loader=lambda *_:called.append(1))
    assert not model.is_ready and not called and not model.class_names


def test_make_protocol_is_frozen_not_runtime_user_prompts():
    from skin.make_artifact import LABELS,PROMPTS,LOGIT_SCALE
    assert len(LABELS)==len(PROMPTS)==6 and LOGIT_SCALE==100.
    assert PROMPTS==tuple('This is a skin image of '+name for name in ['acne','eczema','psoriasis','rosacea','ringworm','healthy skin'])


def test_extreme_ratio_is_rejected_before_official_resize():
    from skin.make import OfficialMakeRuntime
    from skin.preprocessing import ImageTooLargeError
    runtime=OfficialMakeRuntime.__new__(OfficialMakeRuntime)
    def forbidden(_):pytest.fail('Unsafe intermediate image reached the official resize')
    runtime.processor=forbidden
    with Image.new('RGB',(100000,1)) as image:
        with pytest.raises(ImageTooLargeError,match='resize'):runtime.predict(image)


def test_predictor_preserves_resource_error_as_413(make_artifact):
    from skin.preprocessing import ImageTooLargeError
    model=predictor(make_artifact,[15.,0.,0.,0.,0.,0.])
    def too_large(_):raise ImageTooLargeError('Model resize exceeds pixel budget')
    model.model.predict=too_large
    with pytest.raises(ImageTooLargeError):model.predict_bytes(png())


@pytest.mark.real_model
def test_real_make_checkpoint_matches_evaluated_scores():
    from skin.config import MAKE_DIR
    from skin.make import MakeSkinPredictor
    if not (MAKE_DIR/'open_clip_model.safetensors').is_file():pytest.skip('MAKE checkpoint absent; real inference unverified')
    model=MakeSkinPredictor(SkinSettings(backend='make'))
    assert model.is_ready,model.unavailable_message
    from pathlib import Path
    from skin.training.v2 import load_protocol
    cohort=Path(__file__).resolve().parents[2]/'workspaces/skin-six-v2-reviewed'
    evaluated=Path(__file__).resolve().parents[2]/'workspaces/make-skin-evaluation-2026-10-17-run2/native_scores.npy'
    if not evaluated.is_file():pytest.skip('Locked comparison cache absent')
    row=load_protocol(cohort,verify_images=False)['rows']['test'][0]
    result=model.predict_bytes((cohort/row['path']).read_bytes())
    assert np.allclose(list(result['probabilities'].values()),np.load(evaluated,allow_pickle=False)[0],rtol=1e-5,atol=1e-6)
