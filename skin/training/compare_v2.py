"""Locked v1/v2 comparison; evaluation cannot select a different candidate."""
from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath

import numpy as np

from skin.config import CLASS_NAMES, SkinSettings
from skin.data.prepare_six import dump_json, sha256
from skin.data.prepare_v2 import _fetch, SCIN_BASE
from skin.service import SkinPredictor
from skin.training.metrics import classification_metrics, softmax
from skin.training.v2 import load_protocol, verify_rows, predict_rows, external_gate

AUDIT_SHA='9ef5efb5e367467e80b5a09be99d12fddf28071fc7036d370db4a33979f2b27a'
BASELINE_SHA='d3aa8ee9417f57e323465d5d36f4162a6f5b23bc57f26064942f8852c033a375'


def promotion_rule(baseline,candidate):
    checks={
        'auditAccuracyImprovesAtLeastFivePoints':candidate['auditAccuracy']-baseline['auditAccuracy']>=.05-1e-12,
        'auditMacroF1ImprovesAtLeastFivePoints':candidate['auditMacroF1']-baseline['auditMacroF1']>=.05-1e-12,
        'blindAccuracyDoesNotFall':candidate['blindAccuracy']>=baseline['blindAccuracy']-1e-12,
        'originalValidationMacroF1DropsAtMostTwoPoints':candidate['originalMacroF1']>=baseline['originalMacroF1']-.02-1e-12}
    return dict(promoteEducationalCandidate=all(checks.values()),checks=checks,
                clinicalApproval=False,normalClinicallyVerified=False)


def score(labels,logits,manifest):
    labels=np.asarray(labels,dtype=int)
    metrics=classification_metrics(labels,logits,CLASS_NAMES)
    metrics['macroF1']=float(np.mean([v['f1'] for v in metrics['perClass'].values() if v['support']]))
    metrics['macroRecall']=float(np.mean([v['recall'] for v in metrics['perClass'].values() if v['support']]))
    n=len(labels);p=metrics['accuracy'];z=1.959963984540054;den=1+z*z/n
    center=(p+z*z/(2*n))/den;radius=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    metrics['imageLevelWilson95']=[float(center-radius),float(center+radius)]
    probs=softmax(logits,manifest['temperature']);accepted=probs.max(axis=1)>=manifest['abstainThreshold']
    metrics['abstention']=dict(threshold=manifest['abstainThreshold'],answered=int(accepted.sum()),
        coverage=float(accepted.mean()),acceptedAccuracy=float((probs.argmax(axis=1)[accepted]==labels[accepted]).mean()) if accepted.any() else None)
    metrics['gate']=external_gate(metrics)
    return metrics


def compare(args):
    # A portable evaluation bundle omits training photographs. Validate all split
    # identities, and verify the actual bytes for every image scored below.
    protocol=load_protocol(args.workspace,verify_images=False)
    verify_rows(protocol['workspace'],{k:protocol['rows'][k] for k in ['val','test']},verify_images=True)
    evaluation=json.loads((args.candidate/'evaluation.json').read_text())
    candidate_manifest=json.loads((args.candidate/'model_manifest.json').read_text())
    baseline_manifest=json.loads((args.baseline/'model_manifest.json').read_text())
    if sha256(args.candidate/'skin_six.keras')!=evaluation['exportSha256']:raise ValueError('Locked candidate changed')
    if evaluation['identity']!=json.loads((args.workspace/'selection_lock.json').read_text()):raise ValueError('Selection lock changed')
    if sha256(args.baseline/'skin_six.keras')!=BASELINE_SHA:raise ValueError('Historical baseline changed')
    if sha256(args.audit)!=AUDIT_SHA:raise ValueError('Frozen audit changed')
    args.output.mkdir(parents=True,exist_ok=False)
    audit=json.loads(args.audit.read_text())['retained'];audit_rows=[]
    for row in audit:
        source=PurePosixPath(row['sourcePath'])
        if source.parts[:2]!=('dataset','images') or len(source.parts)!=3:raise ValueError('Unsafe audit source path')
        path=_fetch(SCIN_BASE+row['sourcePath'],args.cache/source.name,row['fingerprint']['sha256'])
        audit_rows.append(dict(path=str(path.resolve()),label_idx=CLASS_NAMES.index(row['label'])))
    blind=protocol['rows']['test'];original=[r for r in protocol['rows']['val'] if r['source']=='original']
    results={}
    # Both artifacts were fixed before these evaluations. No retraining or retuning.
    for name,folder,manifest in [('v1',args.baseline,baseline_manifest),('v2',args.candidate,candidate_manifest)]:
        predictor=SkinPredictor(SkinSettings(model_path=folder/'skin_six.keras',manifest_path=folder/'model_manifest.json'))
        if not predictor.is_ready:raise RuntimeError(f'{name} failed production loading')
        result={}
        for split,rows,root in [('blind',blind,protocol['workspace']),('originalValidation',original,protocol['workspace']),('historicalAudit',audit_rows,Path('/'))]:
            logits=predict_rows(predictor.model,root,rows,args.batch_size)
            np.save(args.output/f'{name}_{split}_logits.npy',logits)
            labels=[int(r['label_idx']) for r in rows]
            result[split]=score(labels,logits,manifest)
            if split=='blind':
                sources=np.array([r['source'] for r in rows]);result['blindBySource']={}
                for source in sorted(set(sources)):
                    mask=sources==source
                    result['blindBySource'][source]=score(np.array(labels)[mask],logits[mask],manifest)
            print(f'{name} {split}: accuracy={result[split]["accuracy"]:.4f}, F1={result[split]["macroF1"]:.4f}',flush=True)
        results[name]=result
        del predictor
    def summary(result):
        return dict(auditAccuracy=result['historicalAudit']['accuracy'],auditMacroF1=result['historicalAudit']['macroF1'],
                    blindAccuracy=result['blind']['accuracy'],originalMacroF1=result['originalValidation']['macroF1'])
    report=dict(models={'v1':BASELINE_SHA,'v2':evaluation['exportSha256']},results=results,
        promotion=promotion_rule(summary(results['v1']),summary(results['v2'])),
        dataset=protocol['provenance'],auditManifestSha256=AUDIT_SHA,
        caveats=['SCIN differential labels are not definitive clinical diagnoses.',
                 'No verified Normal external controls; six-class clinical/generalization gate cannot pass.',
                 'Case and duplicate grouping cannot establish patient independence.',
                 'Wilson intervals describe image-level sampling uncertainty only; source/label bias remains.'])
    dump_json(args.output/'comparison.json',report)
    print(json.dumps(report['promotion'],indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['workspace','candidate','baseline','audit','output','cache']:parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--batch-size',type=int,default=32)
    args=parser.parse_args()
    if args.batch_size<1:parser.error('Batch size must be positive')
    compare(args)


if __name__=='__main__':main()
