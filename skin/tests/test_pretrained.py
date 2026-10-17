from __future__ import annotations

import hashlib
import io
import json

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from skin.config import SkinSettings, get_settings


def png():
    stream=io.BytesIO();Image.new('RGB',(40,60),(120,70,50)).save(stream,format='PNG')
    return stream.getvalue()


@pytest.fixture
def artifact(tmp_path,monkeypatch):
    from skin import pretrained_artifact as module
    contents={'config.json':json.dumps({'model_type':'vit','architectures':['ViTForImageClassification'],
        'id2label':{str(i):label for i,label in enumerate(module.LABELS)}}).encode(),
        'preprocessor_config.json':json.dumps(module.PROCESSOR_SPEC).encode(),
        'README.md':b'Apache-2.0 declared; educational only',
        'model.safetensors':b'injected unit runtime only; not a real checkpoint'}
    specs={name:(len(data),hashlib.sha256(data).hexdigest()) for name,data in contents.items()}
    monkeypatch.setattr(module,'ARTIFACT_FILES',specs)
    for name,data in contents.items():(tmp_path/name).write_bytes(data)
    return tmp_path


def make_predictor(artifact,logits=None,**kwargs):
    from skin.pretrained import PretrainedSkinPredictor
    from skin.pretrained_artifact import LABELS
    class Runtime:
        class_names=list(LABELS)
        def predict(self,image):
            assert image.mode=='RGB' and image.size==(40,60)
            return np.asarray(logits if logits is not None else [10.]+[0.]*21)
    settings=SkinSettings(backend='pretrained',pretrained_dir=artifact,**kwargs)
    return PretrainedSkinPredictor(settings,runtime_loader=lambda *_:Runtime())


def test_pretrained_configuration_is_explicit_and_validated(monkeypatch):
    monkeypatch.delenv('SKIN_BACKEND',raising=False)
    assert get_settings().backend=='keras'
    monkeypatch.setenv('SKIN_BACKEND','pretrained')
    assert get_settings().backend=='pretrained'
    for name,value in [('SKIN_BACKEND','anything'),('SKIN_PRETRAINED_MIN_MARGIN','nan'),('SKIN_PRETRAINED_THREADS','0')]:
        with monkeypatch.context() as patch:
            patch.setenv(name,value)
            with pytest.raises(ValueError):get_settings()


def test_native_mapping_and_existing_response_fields(artifact):
    predictor=make_predictor(artifact)
    assert predictor.is_ready and len(predictor.class_names)==22
    result=predictor.predict_bytes(png())
    assert result['predicted_class']=='Acne' and not result['abstained']
    assert len(result['top_predictions'])==3
    assert list(result['probabilities'])==predictor.class_names
    assert sum(result['probabilities'].values())==pytest.approx(1.)
    assert 'LaurianeMD/vit-skin-disease@' in result['model_version']
    assert 'uncalibrated' in result['disclaimer']


@pytest.mark.parametrize('logits,reason',[
    ([0.]*22,'low_confidence'),
    ([4.,3.9]+[-10.]*20,'low_confidence'),
    ([2.,1.5]+[-10.]*20,'low_confidence'),
    ([8. if i==17 else 0. for i in range(22)],'unknown_or_normal'),
])
def test_conservative_abstention(artifact,logits,reason):
    result=make_predictor(artifact,logits).predict_bytes(png())
    assert result['abstained'] and result['predicted_class'] is None
    assert result['abstain_reason']==reason


def test_margin_abstention_is_distinct(artifact):
    result=make_predictor(artifact,[1.,.9]+[-20.]*20,abstain_threshold=.4).predict_bytes(png())
    assert result['abstain_reason']=='ambiguous_prediction'


def test_missing_and_tampered_artifacts_never_load(artifact):
    from skin.pretrained import PretrainedSkinPredictor
    loaded=[]
    (artifact/'model.safetensors').write_bytes(b'changed')
    predictor=PretrainedSkinPredictor(SkinSettings(pretrained_dir=artifact),runtime_loader=lambda *_:loaded.append(1))
    assert not predictor.is_ready and not loaded
    (artifact/'model.safetensors').unlink()
    assert not PretrainedSkinPredictor(SkinSettings(pretrained_dir=artifact)).is_ready


def test_bad_shape_or_nonfinite_output_is_inference_error(artifact):
    from skin.service import ModelInferenceError
    for logits in [[0.]*21,[float('nan')]*22]:
        with pytest.raises(ModelInferenceError):make_predictor(artifact,logits).predict_bytes(png())


def test_pretrained_route_keeps_contract_and_upload_errors(artifact):
    from main import app
    from skin.router import get_skin_service
    predictor=make_predictor(artifact)
    app.dependency_overrides[get_skin_service]=lambda:predictor
    try:
        with TestClient(app) as client:
            assert client.get('/api/ai/skin/health').json()['classCount']==22
            response=client.post('/api/ai/skin/predict',files={'file':('skin.png',png(),'image/png')})
            assert response.status_code==200
            result=response.json();assert result['predictedClass']=='Acne' and len(result['probabilities'])==22
            for name,payload,code in [('bad.txt',b'text',415),('bad.jpg',b'\xff\xd8\xffbad',400)]:
                assert client.post('/api/ai/skin/predict',files={'file':(name,payload,'image/jpeg')}).status_code==code
            assert client.get('/docs').status_code==200
            assert client.get('/api/ai/mri/health').status_code==200
            assert client.post('/api/ai/voice/turn').status_code==422
    finally:app.dependency_overrides.clear()


def test_factory_loads_pretrained_once_and_no_keras_fallback(monkeypatch):
    from skin import service,pretrained
    monkeypatch.setenv('SKIN_BACKEND','pretrained')
    monkeypatch.setattr(service,'_GLOBAL_PREDICTOR',None)
    created=[]
    class Unavailable:
        is_ready=False
        def __init__(self,settings):created.append(settings)
    monkeypatch.setattr(pretrained,'PretrainedSkinPredictor',Unavailable)
    first=service.get_skin_predictor()
    assert service.get_skin_predictor() is first and len(created)==1
    assert not first.is_ready


def test_invalid_backend_configuration_does_not_load_backup_weights(monkeypatch):
    from skin import service
    monkeypatch.setenv('SKIN_BACKEND','invalid')
    monkeypatch.setattr(service,'_GLOBAL_PREDICTOR',None)
    def forbidden(_):pytest.fail('Invalid pretrained configuration attempted to load backup weights')
    monkeypatch.setattr(service,'SkinPredictor',forbidden)
    predictor=service.get_skin_predictor()
    assert not predictor.is_ready and predictor.class_names==[]
    assert predictor.model_version is None


def test_reference_scoring_does_not_hide_outside_labels():
    from skin.evaluate_pretrained import score_native
    result=score_native(['Acne','Eczema','Ringworm'],['Acne','Skin Cancer','Tinea'],[True,False,True])
    assert result['correct']==2 and result['incorrect']==1
    assert result['accuracy']==pytest.approx(2/3)
    assert result['confusionMatrix'][1][-1]==1
    assert result['perClass']['Eczema']['recall']==0
    assert result['acceptedAccuracy']==1 and result['coverage']==pytest.approx(2/3)
    assert result['normalClinicallyVerified'] is False


def test_download_refuses_to_overwrite_existing_weights(artifact):
    from skin.download_pretrained import download
    target=artifact/'model.safetensors';target.write_bytes(b'preserve different existing model')
    with pytest.raises(ValueError,match='Existing file differs'):download(artifact)
    assert target.read_bytes()==b'preserve different existing model'


def test_loaded_runtime_cannot_substitute_a_different_label_map(artifact):
    from skin.pretrained import PretrainedSkinPredictor
    class WrongRuntime:
        class_names=['wrong']*22
    predictor=PretrainedSkinPredictor(SkinSettings(pretrained_dir=artifact),runtime_loader=lambda *_:WrongRuntime())
    assert not predictor.is_ready


@pytest.mark.real_model
def test_real_pretrained_artifact():
    from skin.config import PRETRAINED_DIR
    from skin.pretrained import PretrainedSkinPredictor
    if not (PRETRAINED_DIR/'model.safetensors').is_file():
        pytest.skip('Pinned pretrained checkpoint absent; actual inference not verified')
    predictor=PretrainedSkinPredictor(SkinSettings(backend='pretrained'))
    assert predictor.is_ready,predictor.unavailable_message
    result=predictor.predict_bytes(png())
    assert len(result['probabilities'])==22
    assert sum(result['probabilities'].values())==pytest.approx(1.)
    assert result['predicted_class'] is None if result['abstained'] else result['predicted_class'] in predictor.class_names
