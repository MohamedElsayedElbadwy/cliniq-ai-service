"""Evaluate the fixed pretrained backend on the existing cohort without tuning."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
import os
from pathlib import Path
import time

import numpy as np

from .config import CLASS_NAMES,SkinSettings,PRETRAINED_DIR
from .data.prepare_six import sha256
from .pretrained import PretrainedSkinPredictor
from .pretrained_artifact import MODEL_ID,REVISION,ARTIFACT_FILES,LABELS
from .training.metrics import softmax
from .training.v2 import load_protocol,verify_rows

REFERENCE_LABELS=CLASS_NAMES[:-1]


def score_native(truth,native_predictions,accepted):
    if not (len(truth)==len(native_predictions)==len(accepted)) or not truth:
        raise ValueError('Empty or inconsistent scoring inputs')
    matrix=np.zeros((5,6),dtype=np.int64)
    mapped=['Ringworm' if label=='Tinea' else label for label in native_predictions]
    for true,predicted in zip(truth,mapped):
        if true not in REFERENCE_LABELS:raise ValueError('Unsupported reference label')
        column=REFERENCE_LABELS.index(predicted) if predicted in REFERENCE_LABELS else 5
        matrix[REFERENCE_LABELS.index(true),column]+=1
    per_class={}
    for i,name in enumerate(REFERENCE_LABELS):
        correct=int(matrix[i,i]);support=int(matrix[i].sum());predicted=int(matrix[:,i].sum())
        precision=correct/predicted if predicted else 0.;recall=correct/support if support else 0.
        per_class[name]=dict(precision=precision,recall=recall,f1=2*precision*recall/(precision+recall) if precision+recall else 0.,
                             correct=correct,support=support)
    correct=np.asarray([a==b for a,b in zip(truth,mapped)],dtype=bool)
    accepted=np.asarray(accepted,dtype=bool);supported=[v for v in per_class.values() if v['support']]
    return dict(support=len(truth),correct=int(correct.sum()),incorrect=int((~correct).sum()),accuracy=float(correct.mean()),
        macroPrecision=float(np.mean([v['precision'] for v in supported])),macroRecall=float(np.mean([v['recall'] for v in supported])),
        macroF1=float(np.mean([v['f1'] for v in supported])),perClass=per_class,
        confusionRows=list(REFERENCE_LABELS),confusionColumns=[*REFERENCE_LABELS,'Other native label'],confusionMatrix=matrix.tolist(),
        accepted=int(accepted.sum()),coverage=float(accepted.mean()),
        acceptedCorrect=int(correct[accepted].sum()),acceptedIncorrect=int((~correct[accepted]).sum()),
        acceptedAccuracy=float(correct[accepted].mean()) if accepted.any() else None,normalClinicallyVerified=False,
        outsideReferencePredictions=dict(Counter(p for p in native_predictions if p not in REFERENCE_LABELS and p!='Tinea')))


def evaluate(args):
    protocol=load_protocol(args.workspace,verify_images=False)
    rows=protocol['rows']['test'];verify_rows(protocol['workspace'],{'test':rows},verify_images=True)
    baseline=json.loads((args.baseline_comparison/'comparison.json').read_text())
    if baseline['dataset']['splitFileSha256s']!=protocol['provenance']['splitFileSha256s']:
        raise ValueError('Historical baseline uses a different cohort')
    baseline_manifest=json.loads((args.baseline/'model_manifest.json').read_text())
    if sha256(args.baseline/'skin_six.keras')!=baseline['models']['v1']:
        raise ValueError('Preserved baseline artifact changed')
    old_logits=np.load(args.baseline_comparison/'v1_blind_logits.npy',allow_pickle=False)
    if old_logits.shape!=(len(rows),6) or not np.isfinite(old_logits).all():raise ValueError('Invalid baseline predictions')
    old_names=[CLASS_NAMES[i] for i in old_logits.argmax(axis=1)]
    old_accepted=softmax(old_logits,baseline_manifest['temperature']).max(axis=1)>=baseline_manifest['abstainThreshold']
    truth=[r['label'] for r in rows]
    old=score_native(truth,old_names,old_accepted)
    if abs(old['accuracy']-baseline['results']['v1']['blind']['accuracy'])>1e-12 or abs(old['macroF1']-baseline['results']['v1']['blind']['macroF1'])>1e-12:
        raise ValueError('Cached baseline predictions do not reproduce recorded metrics')
    args.output.mkdir(parents=True,exist_ok=False)
    settings=SkinSettings(backend='pretrained',pretrained_dir=args.model_dir)
    print(json.dumps({'processId':os.getpid(),'model':MODEL_ID,'revision':REVISION,'images':len(rows),'threshold':.70,'margin':.15}),flush=True)
    started=time.perf_counter();predictor=PretrainedSkinPredictor(settings)
    if not predictor.is_ready:raise RuntimeError(predictor.unavailable_message)
    load_seconds=time.perf_counter()-started
    responses=[];logits=[];latencies=[]
    for index,row in enumerate(rows):
        result=predictor.predict_bytes((protocol['workspace']/row['path']).read_bytes())
        responses.append(result);latencies.append(result['inference_ms'])
        logits.append([result['probabilities'][name] for name in LABELS])
        if (index+1)%25==0:print('Images evaluated:',index+1,'/',len(rows),flush=True)
    names=[r['top_predictions'][0]['label'] for r in responses];accepted=[not r['abstained'] for r in responses]
    candidate=score_native(truth,names,accepted)
    exact_indexes=[i for i,label in enumerate(truth) if label!='Ringworm']
    subset=lambda values:[values[i] for i in exact_indexes]
    results=dict(baseline=old,pretrained=candidate,
        exactFourLabelSubset=dict(baseline=score_native(subset(truth),subset(old_names),subset(old_accepted)),
                                 pretrained=score_native(subset(truth),subset(names),subset(accepted))),
        bySource={source:dict(baseline=score_native([truth[i] for i in indexes],[old_names[i] for i in indexes],[bool(old_accepted[i]) for i in indexes]),
                             pretrained=score_native([truth[i] for i in indexes],[names[i] for i in indexes],[accepted[i] for i in indexes]))
            for source in sorted({r['source'] for r in rows})
            for indexes in [[i for i,r in enumerate(rows) if r['source']==source]]})
    report=dict(model=MODEL_ID,revision=REVISION,weightSha256=ARTIFACT_FILES['model.safetensors'][1],nativeLabels=list(LABELS),
        datasetSplitSha256s=protocol['provenance']['splitFileSha256s'],images=len(rows),baselineModelSha256=baseline['models']['v1'],
        baselineCachedLogitsSha256=sha256(args.baseline_comparison/'v1_blind_logits.npy'),
        selection='Requested primary usable checkpoint; not selected/tuned from this evaluation',trainingPerformed=False,
        calibrated=False,abstention=dict(minConfidence=.70,minMargin=.15,alwaysReject=['Unknown Normal'],tunedOnCohort=False),
        loadSeconds=load_seconds,medianInferenceMs=float(np.median(latencies)),p95InferenceMs=float(np.percentile(latencies,95)),
        results=results,caveats=[
            'The pretrained publisher training images/splits are unavailable; overlap with this cohort cannot be excluded.',
            'Tinea is mapped approximately to reference Ringworm; the separate254image comparison excludes this alias.',
            'Unknown Normal combines unknown and normal; it is always rejected and never verifies healthy skin.',
            'Reference SCIN labels are dermatologist differentials, not confirmed diagnoses.',
            'Scores and abstention heuristics are uncalibrated and cannot identify all unsupported diseases.',
            'No diagnosis, prescription, clinical approval, or patient independence is established.'])
    (args.output/'evaluation.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    np.save(args.output/'native_scores.npy',np.asarray(logits))
    (args.output/'responses.json').write_text(json.dumps(responses,indent=2),encoding='utf-8')
    with (args.output/'image_results.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=['path','source','truth','nativeTop1','confidence','abstained','reason'])
        writer.writeheader()
        for row,result in zip(rows,responses):writer.writerow(dict(path=row['path'],source=row['source'],truth=row['label'],
            nativeTop1=result['top_predictions'][0]['label'],confidence=result['confidence'],abstained=result['abstained'],reason=result['abstain_reason']))
    print(json.dumps({'baseline':old['accuracy'],'pretrained':candidate['accuracy'],'macroF1':candidate['macroF1'],
                     'acceptedAccuracy':candidate['acceptedAccuracy'],'coverage':candidate['coverage'],'output':str(args.output)},indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace',type=Path,default=Path('workspaces/skin-six-v2-reviewed'))
    parser.add_argument('--model-dir',type=Path,default=PRETRAINED_DIR)
    parser.add_argument('--baseline',type=Path,default=Path('skin/artifacts/six-class'))
    parser.add_argument('--baseline-comparison',type=Path,default=Path('workspaces/skin-v2-comparison'))
    parser.add_argument('--output',type=Path,required=True)
    evaluate(parser.parse_args())


if __name__=='__main__':main()
