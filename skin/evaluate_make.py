"""Evaluate the shipped fixed MAKE backend; never train or tune on this cohort."""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from .config import SkinSettings,MAKE_DIR
from .evaluate_pretrained import score_native
from .make import MakeSkinPredictor
from .make_artifact import MODEL_ID,REVISION,LABELS,PROMPTS,LOGIT_SCALE,ARTIFACT_FILES
from .training.v2 import load_protocol,verify_rows


def evaluate(args):
    protocol=load_protocol(args.workspace,verify_images=False)
    rows=protocol['rows']['test'];verify_rows(protocol['workspace'],{'test':rows},verify_images=True)
    args.output.mkdir(parents=True,exist_ok=False)
    fixed=dict(model=MODEL_ID,revision=REVISION,labels=list(LABELS),prompts=list(PROMPTS),logitScale=LOGIT_SCALE,
        minConfidence=.70,minMargin=.15,alwaysReject=['Normal'],trainingPerformed=False,tunedOnCohort=False,
        cohortUse='Exploratory model selection; not an independent final holdout',publisherTrainingOverlapUnknown=True,
        datasetSplitSha256s=protocol['provenance']['splitFileSha256s'],weightSha256=ARTIFACT_FILES['open_clip_model.safetensors'][1])
    (args.output/'protocol-before-evaluation.json').write_text(json.dumps(fixed,indent=2),encoding='utf-8')
    started=time.perf_counter();predictor=MakeSkinPredictor(SkinSettings(backend='make',make_dir=args.model_dir))
    if not predictor.is_ready:raise RuntimeError(predictor.unavailable_message)
    load_seconds=time.perf_counter()-started
    scores=[];responses=[];names=[];accepted=[];times=[]
    for i,row in enumerate(rows):
        result=predictor.predict_bytes((protocol['workspace']/row['path']).read_bytes())
        responses.append(result);names.append(result['top_predictions'][0]['label']);accepted.append(not result['abstained'])
        scores.append([result['probabilities'][label] for label in LABELS]);times.append(result['inference_ms'])
        if (i+1)%25==0:print('MAKE images evaluated:',i+1,'/',len(rows),flush=True)
    scores=np.asarray(scores)
    if args.reference_scores is not None:
        expected=np.load(args.reference_scores,allow_pickle=False)
        if expected.shape!=scores.shape or not np.allclose(scores,expected,rtol=1e-5,atol=1e-6):
            raise ValueError('Shipped backend differs from frozen candidate evaluation')
    truth=[row['label'] for row in rows]
    report={**fixed,'results':score_native(truth,names,accepted),'images':len(rows),'failures':0,'skipped':0,
        'loadSeconds':load_seconds,'medianInferenceMs':float(np.median(times)),'p95InferenceMs':float(np.percentile(times,95)),
        'matchesReferenceScores':args.reference_scores is not None,
        'bySource':{source:score_native([truth[i] for i in ix],[names[i] for i in ix],[accepted[i] for i in ix])
            for source in sorted({r['source'] for r in rows}) for ix in [[i for i,r in enumerate(rows) if r['source']==source]]}}
    (args.output/'evaluation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    (args.output/'responses.json').write_text(json.dumps(responses,indent=2),encoding='utf-8')
    np.save(args.output/'native_scores.npy',scores)
    print(json.dumps({'accuracy':report['results']['accuracy'],'macroF1':report['results']['macroF1'],
                      'matchesReferenceScores':report['matchesReferenceScores'],'output':str(args.output)}),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace',type=Path,default=Path('workspaces/skin-six-v2-reviewed'))
    parser.add_argument('--model-dir',type=Path,default=MAKE_DIR)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--reference-scores',type=Path)
    evaluate(parser.parse_args())


if __name__=='__main__':main()
