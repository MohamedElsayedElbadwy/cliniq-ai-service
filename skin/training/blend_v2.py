"""Validation-only selection of six-class models and calibrated logit blends."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from skin.config import CLASS_NAMES, SkinSettings
from skin.data.prepare_six import dump_json, sha256
from skin.preprocessing import preprocess_skin_image
from skin.service import SkinPredictor
from skin.training.compare_v2 import BASELINE_SHA
from skin.training.metrics import classification_metrics, fit_temperature
from skin.training.six_class import configure_tensorflow
from skin.training.v2 import load_protocol, verify_rows, validate_training_identity, domain_selection


def choose_blend(labels,baseline,candidate,sources):
    labels=np.asarray(labels);baseline=np.asarray(baseline);candidate=np.asarray(candidate);sources=np.asarray(sources)
    if baseline.shape!=(len(labels),6) or candidate.shape!=baseline.shape or not np.isfinite(baseline).all() or not np.isfinite(candidate).all():
        raise ValueError('Invalid or nonfinite blend logits')
    if sources.shape!=labels.shape or labels.ndim!=1:raise ValueError('Invalid validation cohort')
    original=sources=='original'
    original_baseline=classification_metrics(labels[original],baseline[original],CLASS_NAMES)['macroF1']
    trials=[]
    for weight in [0.,.25,.5,.75,1.]:
        logits=baseline*np.float32(1-weight)+candidate*np.float32(weight)
        original_f1=classification_metrics(labels[original],logits[original],CLASS_NAMES)['macroF1']
        trials.append(dict(candidateWeight=weight,selectionScore=domain_selection(labels,logits,sources),
            originalMacroF1=original_f1,eligible=original_f1>=original_baseline-.02-1e-12))
    eligible=[trial for trial in trials if trial['eligible']]
    best=max(eligible,key=lambda trial:(trial['selectionScore'],-int(trial['candidateWeight'] not in {0.,1.}),-trial['candidateWeight']))
    return dict(best,baselineOriginalMacroF1=original_baseline,originalMacroF1Floor=original_baseline-.02,trials=trials)


def build_blend(tf,baseline,candidate,weight,temperatures):
    """Use serializable built-in layers only; both branches are inference-only."""
    baseline.trainable=False;candidate.trainable=False
    inputs=tf.keras.Input((224,224,3),dtype='float32',name='skin_image')
    a=tf.keras.layers.Rescaling(1/temperatures[0],name='baseline_temperature')(baseline(inputs,training=False))
    b=tf.keras.layers.Rescaling(1/temperatures[1],name='candidate_temperature')(candidate(inputs,training=False))
    a=tf.keras.layers.Rescaling(1-weight,name='baseline_weight')(a)
    b=tf.keras.layers.Rescaling(weight,name='candidate_weight')(b)
    return tf.keras.Model(inputs,tf.keras.layers.Add(name='class_logits')([a,b]),name='skin_six_v2_blend')


def select(args):
    protocol=load_protocol(args.workspace,verify_images=False)
    if (args.workspace/'selection_lock.json').exists():raise ValueError('Cannot select another candidate after external finalization')
    rows=protocol['rows']['val'];workspace=protocol['workspace']
    verify_rows(workspace,{'val':rows},verify_images=True)
    source_config=json.loads((args.candidate_run/'train_config.json').read_text())
    validate_training_identity(source_config,protocol['provenance'])
    completed=json.loads((args.candidate_run/'completed.json').read_text())
    if completed.get('testPredictionsRead') is not False:raise ValueError('Candidate must be selected before external tests')
    candidate_hash=sha256(args.candidate_run/'best.keras')
    if candidate_hash!=completed['modelSha256']:raise ValueError('Recovered checkpoint changed')
    if sha256(args.baseline/'skin_six.keras')!=BASELINE_SHA:raise ValueError('Historical six-class baseline changed')
    args.run.mkdir(parents=True,exist_ok=False)
    tf=configure_tensorflow(require_gpu=False)
    predictor=SkinPredictor(SkinSettings(model_path=args.baseline/'skin_six.keras',manifest_path=args.baseline/'model_manifest.json'))
    if not predictor.is_ready:raise RuntimeError('Historical baseline failed production loading')
    baseline=predictor.model
    candidate=tf.keras.models.load_model(args.candidate_run/'best.keras',compile=False,safe_mode=True)
    if tuple(candidate.input_shape[1:])!=(224,224,3) or candidate.output_shape[-1]!=6:raise ValueError('Incompatible recovered model')
    old_parts=[];new_parts=[]
    for start in range(0,len(rows),args.batch_size):
        batch=np.concatenate([preprocess_skin_image(workspace/r['path']) for r in rows[start:start+args.batch_size]])
        old_parts.append(np.asarray(baseline(batch,training=False)))
        new_parts.append(np.asarray(candidate(batch,training=False)))
        if start//args.batch_size%20==0:print('Validation blend selection images:',start,'/',len(rows),flush=True)
    old=np.concatenate(old_parts);new=np.concatenate(new_parts)
    labels=np.array([int(r['label_idx']) for r in rows]);sources=np.array([r['source'] for r in rows])
    temperatures=[fit_temperature(old,labels),fit_temperature(new,labels)]
    scaled_old=old*np.float32(1/temperatures[0]);scaled_new=new*np.float32(1/temperatures[1])
    chosen=choose_blend(labels,scaled_old,scaled_new,sources);weight=chosen['candidateWeight']
    if weight==0:
        model=baseline;logits=old;architecture='efficientnetb2-flip-average'
    elif weight==1:
        model=candidate;logits=new;architecture='efficientnetb2'
    else:
        model=build_blend(tf,baseline,candidate,weight,temperatures)
        logits=scaled_old*np.float32(1-weight)+scaled_new*np.float32(weight)
        architecture='efficientnetb2-calibrated-logit-blend'
        representatives={}
        for i,row in enumerate(rows):representatives.setdefault((row['source'],row['label']),i)
        indexes=list(representatives.values())
        batch=np.concatenate([preprocess_skin_image(workspace/rows[i]['path']) for i in indexes])
        actual=np.asarray(model(batch,training=False))
        if not np.allclose(actual,logits[indexes],rtol=1e-4,atol=2e-4):raise RuntimeError('Serialized blend graph differs from validation calculation')
    model.save(args.run/'best.keras')
    np.save(args.run/'validation_logits.npy',logits)
    dump_json(args.run/'blend_selection.json',dict(chosen,temperatures=temperatures,testPredictionsRead=False))
    config=dict(source_config,protocol='skin-v2-validation-selection',architecture=architecture,
        initialization='Validation-selected combination of ImageNet-origin six-class runs; no legacy 22-class weights',
        validationBlendSelection=chosen,blendTemperatures=temperatures,
        sourceModelSha256s={'baseline':BASELINE_SHA,'recovered':candidate_hash},
        baselineManifestSha256=sha256(args.baseline/'model_manifest.json'),
        recoveredConfigSha256=sha256(args.candidate_run/'train_config.json'),testPredictionsRead=False)
    dump_json(args.run/'train_config.json',config)
    dump_json(args.run/'validation.json',dict(selectionScore=chosen['selectionScore'],originalMacroF1=chosen['originalMacroF1'],
        metrics=classification_metrics(labels,logits,CLASS_NAMES)))
    dump_json(args.run/'completed.json',dict(modelSha256=sha256(args.run/'best.keras'),
        selectionScore=chosen['selectionScore'],candidateWeight=weight,testPredictionsRead=False))
    print(json.dumps(chosen,indent=2),flush=True)
    print('Validation selection complete; no blind or historical candidate predictions were read.',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['workspace','candidate-run','baseline','run']:parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--batch-size',type=int,default=16)
    args=parser.parse_args()
    if args.batch_size<1:parser.error('Batch size must be positive')
    select(args)


if __name__=='__main__':main()
