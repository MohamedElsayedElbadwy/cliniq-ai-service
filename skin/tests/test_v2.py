import json

import numpy as np
import pytest


def test_v2_rejects_tied_and_unsupported_top_labels():
    from skin.data.prepare_v2 import choose_label
    assert choose_label({'Eczema':.5,'Psoriasis':.5}) is None
    assert choose_label({'Urticaria':.7,'Eczema':.3}) is None
    assert choose_label({'Eczema':.3,'Acne':.2}) is None
    assert choose_label({'Tinea':.8,'Eczema':.2})==('Ringworm',.8)


def test_v2_all_images_of_frozen_cases_are_excluded():
    from skin.data.prepare_v2 import select_cases
    import hashlib
    cases={'held':{'case_id':'held','image_1_path':'dataset/images/a.png','image_2_path':'dataset/images/b.png'},
           'new':{'case_id':'new','image_1_path':'dataset/images/c.png'}}
    labels=[{'case_id':k,'weighted_skin_condition_label':"{'Acne': 1.0}"} for k in cases]
    excluded={hashlib.sha256(b'held').hexdigest()}
    selected=select_cases(cases,labels,excluded)
    assert len(selected)==1 and selected[0]['case_id']=='new'
    assert [p['source_path'] for p in selected[0]['images']]==['dataset/images/c.png']


def test_v2_group_split_is_deterministic_and_weak_labels_are_training_only():
    from skin.data.prepare_v2 import split_cases
    cases=[{'case_id':f'{label}-{i}','label':label,'weight':.9,'group_id':f'{label}-{i//2}'}
        for label in ['Acne','Eczema','Psoriasis','Rosacea','Ringworm'] for i in range(12)]
    cases.append({'case_id':'weak','label':'Acne','weight':.45,'group_id':'weak'})
    assignments=split_cases(cases)
    assert assignments==split_cases(list(reversed(cases)))
    assert assignments['weak']=='train'
    for label in ['Acne','Eczema','Psoriasis','Rosacea','Ringworm']:
        assert {assignments[f'{label}-{i}'] for i in range(12)}=={'train','val','test'}
        assert all(assignments[f'{label}-{i}']==assignments[f'{label}-{i+1}'] for i in range(0,12,2))


def test_domain_selection_gives_equal_weight_to_original_and_scin():
    from skin.training.v2 import domain_selection
    labels=np.array([0,1,2,3,4,5]*10+[0,1,2,3,4])
    logits=np.eye(6)[labels]*10
    logits[-5:]=np.eye(6)[[1,2,3,4,0]]*10
    assert domain_selection(labels,logits,['original']*60+['scin']*5)==pytest.approx(.5)


def test_v2_protocol_detects_changed_image_and_cross_split_case(tmp_path):
    from skin.training.v2 import verify_rows
    import hashlib
    image=tmp_path/'images'/'a.png';image.parent.mkdir();image.write_bytes(b'original')
    row={'path':'images/a.png','label':'Acne','label_idx':'0','sha256':hashlib.sha256(b'original').hexdigest(),
         'pixel_sha256':'pixels','cluster_id':'cluster','source':'scin','case_id':'case'}
    with pytest.raises(ValueError,match='case|leak'):
        verify_rows(tmp_path,{'train':[row],'val':[dict(row,path='images/b.png')]},verify_images=False)
    image.write_bytes(b'changed')
    with pytest.raises(ValueError,match='changed'):
        verify_rows(tmp_path,{'train':[row]},verify_images=True)


def test_external_gate_cannot_pass_without_normal_support():
    from skin.training.v2 import external_gate
    per={name:{'recall':1.,'support':10} for name in ['Acne','Eczema','Psoriasis','Rosacea','Ringworm']}
    result=external_gate({'accuracy':1.,'macroF1':1.,'perClass':per})
    assert result['diseaseTargetsPassed'] is True
    assert result['sixClassExternalGatePassed'] is False


def test_external_gate_requires_all_diseases_and_verified_normal():
    from skin.training.v2 import external_gate
    assert external_gate({'accuracy':1.,'macroF1':1.,'perClass':{'Acne':{'recall':1.,'support':10}}})['diseaseTargetsPassed'] is False
    per={name:{'recall':1.,'support':10} for name in ['Acne','Eczema','Psoriasis','Rosacea','Ringworm','Normal']}
    assert external_gate({'accuracy':1.,'macroF1':1.,'perClass':per})['sixClassExternalGatePassed'] is False


def test_atlas_distribution_must_match_original_author_image_identity(tmp_path):
    from skin.data.atlas_v2 import verified_atlas_images
    import hashlib,zipfile
    key=hashlib.md5(b'author image').hexdigest()
    archive=tmp_path/'candidate.zip'
    with zipfile.ZipFile(archive,'w') as zf:zf.writestr('images/'+key+'.jpg',b'different image')
    rows,stats=verified_atlas_images(archive,[{'md5hash':key,'label':'rosacea','qc':'1 Diagnostic',
        'url':'https://www.dermaamin.com/site/images/rosacea1.jpg'}],tmp_path/'cache')
    assert rows==[]
    assert stats['author_identity_mismatch']==1


def test_third_validation_domain_has_predeclared_weight():
    from skin.training.v2 import domain_selection
    labels=np.array([0,1,2,3,4,5]+[0,1,2,3,4]+[0,1,2,3])
    logits=np.eye(6)[labels]*10
    logits[-4:]=np.eye(6)[[1,2,3,0]]*10
    assert domain_selection(labels,logits,['original']*6+['scin']*5+['atlas']*4)==pytest.approx(.8)


def test_atlas_accepts_verified_bytes_and_rejects_flagged_labels(tmp_path):
    from skin.data.atlas_v2 import verified_atlas_images
    from PIL import Image
    import hashlib,io,zipfile
    stream=io.BytesIO();Image.fromarray(np.random.default_rng(42).integers(0,256,(80,80,3),dtype=np.uint8)).save(stream,format='JPEG')
    payload=stream.getvalue();key=hashlib.md5(payload).hexdigest()
    archive=tmp_path/'raw.zip'
    with zipfile.ZipFile(archive,'w') as zf:zf.writestr('images/'+key+'.jpg',payload)
    record={'md5hash':key,'label':'rosacea','qc':'1 Diagnostic','url':'https://www.dermaamin.com/site/images/rosacea1.jpg'}
    rows,stats=verified_atlas_images(archive,[record],tmp_path/'cache')
    assert len(rows)==1 and rows[0]['source_md5']==key and stats['verified']==1
    assert verified_atlas_images(archive,[dict(record,qc='3 Wrongly Labelled')],tmp_path/'cache')[0]==[]
    cached=tmp_path/'cache'/'atlas'/'Rosacea'/(key+'.jpg')
    cached.write_bytes(b'poisoned cache')
    rows,_=verified_atlas_images(archive,[record],tmp_path/'cache')
    assert len(rows)==1 and cached.read_bytes()==payload


def test_promotion_requires_every_predeclared_condition():
    from skin.training.compare_v2 import promotion_rule
    baseline={'auditAccuracy':.48,'auditMacroF1':.48,'blindAccuracy':.6,'originalMacroF1':.83}
    candidate={'auditAccuracy':.54,'auditMacroF1':.54,'blindAccuracy':.61,'originalMacroF1':.82}
    assert promotion_rule(baseline,candidate)['promoteEducationalCandidate'] is True
    for key,value in [('auditAccuracy',.52),('auditMacroF1',.52),('blindAccuracy',.59),('originalMacroF1',.80)]:
        assert promotion_rule(baseline,dict(candidate,**{key:value}))['promoteEducationalCandidate'] is False


def test_protection_covers_transitive_duplicates_and_case_links():
    from skin.data.prepare_v2 import protected_case_ids
    def row(key,value,case):
        return dict(sha256=key,pixel_sha256=key,phash=value,dhash=value,case_id=case)
    reference=row('protected',0,'held')
    candidates=[row('bridge',15,'a'),row('relative',255,'b'),row('second-photo',65535,'b'),row('case-relative',1048575,'c')]
    assert protected_case_ids(candidates,[reference])=={'a','b','c'}


def test_finalization_rejects_changed_training_protocol():
    from skin.training.v2 import validate_training_identity
    with pytest.raises(ValueError,match='protocol'):
        validate_training_identity({'splitFileSha256s':{'train':'A'}},{'splitFileSha256s':{'train':'B'}})


def test_export_lock_rejects_calibration_changes(tmp_path):
    from skin.training.v2 import verify_export_lock
    from skin.data.prepare_six import sha256
    manifest=tmp_path/'model_manifest.json';manifest.write_text('{"temperature":1}')
    lock={'manifestSha256':sha256(manifest),'modelSha256':'model','splitFileSha256s':{'test':'test'}}
    manifest.write_text('{"temperature":2}')
    with pytest.raises(ValueError,match='manifest|calibration'):
        verify_export_lock(tmp_path,lock,'model',{'test':'test'})


def test_recovery_requires_same_protocol_and_pinned_v2_checkpoint(tmp_path):
    from skin.training.recovery_v2 import validate_recovery_source
    from skin.data.prepare_six import sha256
    checkpoint=tmp_path/'best.keras';checkpoint.write_bytes(b'known six-class checkpoint')
    config={'protocol':'skin-v2','architecture':'efficientnetb2','initialization':'fresh ImageNet; no skin checkpoint',
            'splitFileSha256s':{'train':'A'},'testPredictionsRead':False}
    provenance={'splitFileSha256s':{'train':'A'}}
    validate_recovery_source(config,provenance,checkpoint,sha256(checkpoint))
    with pytest.raises(ValueError,match='protocol'):
        validate_recovery_source(config,{'splitFileSha256s':{'train':'B'}},checkpoint,sha256(checkpoint))
    with pytest.raises(ValueError,match='checkpoint'):
        validate_recovery_source(config,provenance,checkpoint,'wrong')
    with pytest.raises(ValueError,match='v2|test'):
        validate_recovery_source(dict(config,testPredictionsRead=True),provenance,checkpoint,sha256(checkpoint))
