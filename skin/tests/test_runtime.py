from __future__ import annotations

import hashlib
import io
import json
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from main import app
from skin.config import SkinSettings
from skin.preprocessing import InvalidImageError, decode_image, preprocess_skin_image
from skin.router import get_skin_service
from skin.service import SkinPredictor, calibrated_probabilities


class FakeModel:
    input_shape = (None,224,224,3)
    output_shape = (None,3)

    def __call__(self, batch, training=False):
        return np.tile([0.,1.,3.],(len(batch),1))


def image_bytes(mode='RGB', fmt='PNG', size=(80,100)):
    stream=io.BytesIO()
    Image.new(mode,size).save(stream,format=fmt)
    return stream.getvalue()


@pytest.fixture
def artifact(tmp_path):
    model=tmp_path/'test.keras'
    model.write_bytes(b'fake artifact for injected model only')
    manifest=tmp_path/'model_manifest.json'
    raw=dict(modelVersion='unit-test',architecture='test',inputSize=[224,224],
        preprocessing='exif-rgb-direct-resize-float32-0-255-v1',classNames={'0':'A','1':'B','2':'C'},
        temperature=1.,abstainThreshold=.5,modelSha256=hashlib.sha256(model.read_bytes()).hexdigest(),outputType='logits')
    manifest.write_text(json.dumps(raw))
    return model,manifest,raw


@pytest.fixture
def predictor(artifact):
    model,manifest,_=artifact
    return SkinPredictor(SkinSettings(model_path=model,manifest_path=manifest),model_loader=lambda _:FakeModel())


@pytest.fixture
def client(predictor):
    app.dependency_overrides[get_skin_service]=lambda:predictor
    # Context manager triggers actual optional startup of both modules.
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


def test_success_contract_and_health(client):
    health=client.get('/api/ai/skin/health').json()
    assert health['modelAvailable'] and health['classCount']==3
    response=client.post('/api/ai/skin/predict',files={'file':('skin.png',image_bytes(),'image/png')})
    assert response.status_code==200
    result=response.json()
    assert result['predictedClass']=='C' and not result['abstained']
    assert result['modelVersion']=='unit-test'
    assert len(result['topPredictions'])==3
    assert sum(result['probabilities'].values())==pytest.approx(1.)
    assert all(0<=v<=1 for v in result['probabilities'].values())
    assert 'Not a medical diagnosis' in result['disclaimer']


@pytest.mark.parametrize('name,data,content_type,code',[
    ('bad.txt',b'text','text/plain',415),('fake.jpg',b'text','image/jpeg',415),
    ('empty.jpg',b'','image/jpeg',400),('truncated.jpg',b'\xff\xd8\xffbad','image/jpeg',400),
    ('bad.gif',image_bytes(fmt='GIF'),'image/gif',415),
    ('bad.jpg',image_bytes(),'image/jpeg',415),
])
def test_bad_uploads(client,name,data,content_type,code):
    response=client.post('/api/ai/skin/predict',files={'file':(name,data,content_type)})
    assert response.status_code==code
    assert set(response.json())=={'detail'}
    assert 'Traceback' not in response.text and 'D:\\' not in response.text


def test_size_pixel_caps_and_form_field(client,monkeypatch):
    monkeypatch.setenv('SKIN_MAX_UPLOAD_BYTES','100')
    assert client.post('/api/ai/skin/predict',files={'file':('a.jpg',b'x'*101,'image/jpeg')}).status_code==413
    monkeypatch.setenv('SKIN_MAX_UPLOAD_BYTES','5242880')
    monkeypatch.setenv('SKIN_MAX_PIXELS','100')
    assert client.post('/api/ai/skin/predict',files={'file':('a.png',image_bytes(),'image/png')}).status_code==413
    assert client.post('/api/ai/skin/predict',files={'image':('a.png',image_bytes(),'image/png')}).status_code==422


def test_missing_skin_does_not_break_app(tmp_path,monkeypatch):
    import skin.service as service
    monkeypatch.setenv('SKIN_MODEL_PATH',str(tmp_path/'missing.keras'))
    monkeypatch.setenv('SKIN_MANIFEST_PATH',str(tmp_path/'missing.json'))
    monkeypatch.setattr(service,'_GLOBAL_PREDICTOR',None)
    with TestClient(app) as c:
        assert c.get('/').status_code==200
        assert c.get('/docs').status_code==200
        assert c.get('/api/ai/mri/health').status_code==200
        assert c.post('/api/ai/voice/turn',files={'audio':('a.txt',b'x','text/plain')}).status_code==400
        assert c.get('/api/ai/skin/health').json()['modelAvailable'] is False
        response=c.post('/api/ai/skin/predict',files={'file':('a.png',image_bytes(),'image/png')})
        assert response.status_code==503
        assert 'missing.keras' not in response.text


@pytest.mark.parametrize('mode',['RGB','L','RGBA','P','CMYK'])
def test_preprocess_modes(mode):
    im=Image.new(mode,(20,30))
    arr=preprocess_skin_image(im)
    assert arr.shape==(1,224,224,3) and arr.dtype==np.float32
    assert 0<=arr.min()<=arr.max()<=255


def test_exif_orientation_and_parity():
    im=Image.new('RGB',(30,50),'red')
    exif=Image.Exif()
    exif[274]=6
    stream=io.BytesIO()
    im.save(stream,format='JPEG',exif=exif)
    decoded=decode_image(stream.getvalue())
    assert decoded.size==(50,30) and not decoded.info
    assert np.array_equal(preprocess_skin_image(stream.getvalue()),preprocess_skin_image(decoded))


def test_calibration_and_abstention(predictor):
    probs=calibrated_probabilities(np.array([0,1,3]),2.)
    assert probs.sum()==pytest.approx(1.) and probs.max()<calibrated_probabilities(np.array([0,1,3]),1.).max()
    predictor.threshold=.99
    result=predictor.predict_bytes(image_bytes())
    assert result['abstained'] and result['predicted_class'] is None
    assert result['abstain_reason']=='low_confidence'


@pytest.mark.parametrize('key,value',[('modelSha256','wrong'),('classNames',{'0':'A','2':'B'}),
    ('classNames',{'0':'A','1':'A','2':'C'}),('temperature',0),('temperature',float('nan')),
    ('abstainThreshold',1.5),('inputSize',[128,128]),('outputType','probabilities')])
def test_manifest_failure_sanitized(artifact,key,value):
    model,manifest,raw=artifact
    raw[key]=value
    manifest.write_text(json.dumps(raw))
    predictor=SkinPredictor(SkinSettings(model_path=model,manifest_path=manifest),model_loader=lambda _:FakeModel())
    assert not predictor.is_ready
    assert str(model.parent) not in predictor.unavailable_message


def test_model_shape_mismatch_and_bad_inference(artifact,predictor):
    model,manifest,_=artifact
    class Wrong(FakeModel):
        output_shape=(None,4)
    assert not SkinPredictor(SkinSettings(model_path=model,manifest_path=manifest),model_loader=lambda _:Wrong()).is_ready
    predictor.model=lambda *a,**kw:np.array([[np.nan,0,1]])
    from skin.service import ModelInferenceError
    with pytest.raises(ModelInferenceError):
        predictor.predict_bytes(image_bytes())


def test_parallel_requests_and_responsive_health(client):
    def request(_):
        return client.post('/api/ai/skin/predict',files={'file':('a.png',image_bytes(),'image/png')}).status_code
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures=[pool.submit(request,i) for i in range(8)]
        assert client.get('/api/ai/skin/health').status_code==200
        assert all(f.result()==200 for f in futures)


def test_singleton_created_once(monkeypatch,predictor):
    import skin.service as service
    calls=[]
    monkeypatch.setattr(service,'_GLOBAL_PREDICTOR',None)
    monkeypatch.setattr(service,'SkinPredictor',lambda settings:(calls.append(1) or predictor))
    with ThreadPoolExecutor(max_workers=8) as pool:
        results=list(pool.map(lambda _:service.get_skin_predictor(),range(16)))
    assert len(calls)==1 and all(result is predictor for result in results)


def test_inference_queue_does_not_exhaust_shared_workers(predictor):
    import httpx
    import anyio
    entered=threading.Event()
    release=threading.Event()
    class Slow(FakeModel):
        def __call__(self,*a,**kw):
            entered.set()
            assert release.wait(5)
            return super().__call__(*a,**kw)
    predictor.model=Slow()
    app.dependency_overrides[get_skin_service]=lambda:predictor
    async def run():
        limiter=anyio.to_thread.current_default_thread_limiter()
        original=limiter.total_tokens
        limiter.total_tokens=4
        tasks=[]
        try:
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url='http://test') as c:
                tasks=[asyncio.create_task(c.post('/api/ai/skin/predict',files={'file':('a.png',image_bytes(),'image/png')})) for _ in range(12)]
                for _ in range(100):
                    if entered.is_set():
                        break
                    await asyncio.sleep(.01)
                assert entered.is_set()
                await asyncio.sleep(.1)
                health=await asyncio.wait_for(c.get('/api/ai/skin/health'),timeout=1.)
                assert health.status_code==200
                release.set()
                assert all(r.status_code==200 for r in await asyncio.gather(*tasks))
        finally:
            release.set()
            await asyncio.gather(*tasks,return_exceptions=True)
            limiter.total_tokens=original
    try:
        asyncio.run(run())
    finally:
        app.dependency_overrides.clear()


def test_valid_large_upload_stays_in_memory(client,monkeypatch):
    import starlette.formparsers as parsers
    original=parsers.SpooledTemporaryFile
    opened=[]
    def track(*a,**kw):
        stream=original(*a,**kw)
        opened.append(stream)
        return stream
    monkeypatch.setattr(parsers,'SpooledTemporaryFile',track)
    buffer=io.BytesIO()
    Image.fromarray(np.random.default_rng(42).integers(0,256,(800,800,3),dtype=np.uint8)).save(buffer,format='PNG')
    assert len(buffer.getvalue())>1024*1024
    response=client.post('/api/ai/skin/predict',files={'file':('a.png',buffer.getvalue(),'image/png')})
    assert response.status_code==200
    assert opened and all(not stream._rolled for stream in opened)


def test_chunked_body_rejected_before_full_multipart_ingestion(predictor,monkeypatch):
    import httpx
    monkeypatch.setenv('SKIN_MAX_UPLOAD_BYTES','100')
    app.dependency_overrides[get_skin_service]=lambda:predictor
    consumed=[]
    async def chunks():
        yield b'--BOUNDARY\r\nContent-Disposition: form-data; name="file"; filename="a.jpg"\r\nContent-Type: image/jpeg\r\n\r\n'
        for _ in range(30):
            consumed.append(4096)
            yield b'x'*4096
        yield b'\r\n--BOUNDARY--\r\n'
    async def run():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url='http://test') as c:
            response=await c.post('/api/ai/skin/predict',content=chunks(),headers={'Content-Type':'multipart/form-data; boundary=BOUNDARY'})
            assert response.status_code==413
            assert sum(consumed)<100000
    try:
        asyncio.run(run())
    finally:
        app.dependency_overrides.clear()


@pytest.mark.real_model
def test_real_model_artifact():
    from skin.service import get_skin_predictor
    predictor=get_skin_predictor()
    if not predictor.is_ready:
        pytest.skip('Real skin checkpoint/manifest absent or invalid; real inference NOT verified')
    result=predictor.predict_bytes(image_bytes())
    assert sum(result['probabilities'].values())==pytest.approx(1.)
