"""Fresh v2 training: case-safe data, domain-balanced selection, locked evaluation."""
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

from skin.config import CLASS_NAMES
from skin.data.prepare_six import dump_json, sha256
from skin.preprocessing import PREPROCESSING_SPEC, preprocess_skin_image
from skin.training.metrics import classification_metrics, fit_temperature, choose_abstention
from skin.training.six_class import configure_tensorflow, epoch_order


def verify_rows(workspace,splits,verify_images=True):
    workspace=Path(workspace).resolve();owners={}
    for split,rows in splits.items():
        for row in rows:
            target=(workspace/row['path']).resolve()
            if Path(row['path']).is_absolute() or not target.is_relative_to(workspace/'images'):
                raise ValueError('Image path escapes workspace')
            if row['label'] not in CLASS_NAMES or int(row['label_idx'])!=CLASS_NAMES.index(row['label']):raise ValueError('Label/index mismatch')
            quality=float(row.get('quality',1))
            if not np.isfinite(quality) or not 0<quality<=1:raise ValueError('Invalid label quality')
            if row['source'] not in {'original','scin','atlas'}:raise ValueError('Unknown data source')
            if row['source']=='scin' and split!='train' and quality<.5:raise ValueError('Weak evaluation label')
            for key in ['case_id','sha256','pixel_sha256','cluster_id']:
                value=(key,row[key])
                if owners.setdefault(value,split)!=split:raise ValueError(f'Cross-split case/image leakage: {key}')
            if verify_images and sha256(target)!=row['sha256']:raise ValueError('Image bytes changed')


def load_protocol(workspace,verify_images=True):
    workspace=Path(workspace).resolve()
    provenance=json.loads((workspace/'provenance.json').read_text())
    if provenance['protocol']!='skin-v2-scin-case-groups':raise ValueError('Wrong data protocol')
    names=json.loads((workspace/'class_names.json').read_text())
    if names!={str(i):v for i,v in enumerate(CLASS_NAMES)} or sha256(workspace/'class_names.json')!=provenance['classNamesSha256']:
        raise ValueError('Class map changed')
    rows={}
    for split in ['train','val','test']:
        path=workspace/'splits'/f'{split}.csv'
        if sha256(path)!=provenance['splitFileSha256s'][split]:raise ValueError('Split file changed')
        rows[split]=list(csv.DictReader(path.open(encoding='utf-8',newline='')))
        expected=set(CLASS_NAMES) if split!='test' else set(CLASS_NAMES[:-1])
        if {r['label'] for r in rows[split]}!=expected:raise ValueError('Missing required class support')
    verify_rows(workspace,rows,verify_images=verify_images)
    return dict(workspace=workspace,rows=rows,provenance=provenance)


def domain_selection(labels,logits,sources):
    labels=np.asarray(labels);logits=np.asarray(logits);sources=np.asarray(sources)
    domains=[('original',CLASS_NAMES,.5),('scin',CLASS_NAMES[:-1],.5)]
    if 'atlas' in sources:domains=[('original',CLASS_NAMES,.4),('scin',CLASS_NAMES[:-1],.4),('atlas',CLASS_NAMES[:4],.2)]
    scores=[]
    for source,names,weight in domains:
        mask=sources==source
        if not mask.any():raise ValueError('Both validation domains are required')
        metrics=classification_metrics(labels[mask],logits[mask],CLASS_NAMES)
        if any(metrics['perClass'][name]['support']==0 for name in names):raise ValueError('Validation domain lacks class support')
        scores.append(weight*np.mean([metrics['perClass'][name]['f1'] for name in names]))
    return float(sum(scores))


def external_gate(metrics,normal_verified=False):
    classes=metrics['perClass'];supported=[v for v in classes.values() if v['support']>0]
    complete=all(classes.get(name,{}).get('support',0)>0 for name in CLASS_NAMES[:-1])
    passed=bool(complete and supported and metrics['accuracy']>=.85 and metrics['macroF1']>=.85 and min(v['recall'] for v in supported)>=.7)
    normal=classes.get('Normal',{}).get('support',0)>0
    return dict(diseaseTargetsPassed=passed,sixClassExternalGatePassed=passed and normal and normal_verified,
        normalSupported=normal,normalClinicallyVerified=normal_verified,minAccuracy=.85,minMacroF1=.85,minPerClassRecall=.7)


def validate_training_identity(config,provenance):
    if config.get('splitFileSha256s')!=provenance['splitFileSha256s']:
        raise ValueError('Training run and evaluation data protocol differ')


def verify_export_lock(folder,lock,model_hash,split_hashes):
    if lock.get('manifestSha256')!=sha256(Path(folder)/'model_manifest.json'):
        raise ValueError('Export manifest/calibration changed after lock')
    if lock.get('modelSha256')!=model_hash or lock.get('splitFileSha256s')!=split_hashes:
        raise ValueError('Locked model or data protocol changed')


def predict_rows(model,workspace,rows,batch_size):
    parts=[]
    for start in range(0,len(rows),batch_size):
        batch=np.concatenate([preprocess_skin_image(workspace/r['path']) for r in rows[start:start+batch_size]])
        parts.append(np.asarray(model(batch,training=False)))
    return np.concatenate(parts)


def sample_weights(rows):
    totals=defaultdict(float);domains=defaultdict(set)
    for row in rows:
        totals[(row['label'],row['source'])]+=float(row['quality']);domains[row['label']].add(row['source'])
    return np.array([len(rows)/(6*len(domains[r['label']]))*float(r['quality'])/totals[(r['label'],r['source'])] for r in rows],dtype=np.float32)


def train(args):
    protocol=load_protocol(args.workspace)
    if (args.workspace/'selection_lock.json').exists():raise ValueError('This protocol was already finalized')
    args.run.mkdir(parents=True,exist_ok=False)
    tf=configure_tensorflow(require_gpu=True)
    rows=protocol['rows'];workspace=protocol['workspace']
    config=dict(protocol='skin-v2',architecture='efficientnetb2',initialization='fresh ImageNet; no skin checkpoint',
        seed=42,batchSize=args.batch_size,headEpochs=args.head_epochs,fineTuneEpochs=args.finetune_epochs,
        selection='validation macro-F1: original6 .4, SCIN5 .4, atlas4 .2 when atlas present; otherwise .5/.5',
        splitFileSha256s=protocol['provenance']['splitFileSha256s'],normalClinicallyVerified=False,
        weighting='equal class mass, equal source mass within class; SCIN differential weight divided by case photo count',
        tensorflowVersion=tf.__version__,kerasVersion=tf.keras.__version__,testPredictionsRead=False)
    dump_json(args.run/'train_config.json',config)
    weights=sample_weights(rows['train']);epoch=0
    def samples():
        nonlocal epoch
        order=epoch_order(len(rows['train']),epoch);epoch+=1
        for index in order:
            row=rows['train'][index]
            yield preprocess_skin_image(workspace/row['path'])[0],np.eye(6,dtype=np.float32)[int(row['label_idx'])],weights[index]
    dataset=tf.data.Dataset.from_generator(samples,output_signature=(tf.TensorSpec((224,224,3),tf.float32),
        tf.TensorSpec((6,),tf.float32),tf.TensorSpec((),tf.float32)))
    dataset=dataset.apply(tf.data.experimental.assert_cardinality(len(rows['train']))).batch(args.batch_size).prefetch(1)
    backbone=tf.keras.applications.EfficientNetB2(include_top=False,weights='imagenet',input_shape=(224,224,3),pooling='avg')
    backbone.trainable=False
    inputs=tf.keras.Input((224,224,3),dtype='float32')
    augmented=tf.keras.Sequential([tf.keras.layers.RandomFlip('horizontal',seed=42),
        tf.keras.layers.RandomRotation(.06,fill_mode='reflect',seed=43),
        tf.keras.layers.RandomZoom(.15,fill_mode='reflect',seed=44),
        tf.keras.layers.RandomContrast(.2,seed=45),
        tf.keras.layers.RandomBrightness(.12,value_range=(0,255),seed=46)],name='train_augmentation')(inputs)
    outputs=tf.keras.layers.Dense(6,dtype='float32',name='class_logits')(
        tf.keras.layers.Dropout(.4,seed=47)(backbone(augmented,training=False)))
    model=tf.keras.Model(inputs,outputs,name='skin_six_v2')
    labels=np.array([int(r['label_idx']) for r in rows['val']]);sources=[r['source'] for r in rows['val']]
    class Selection(tf.keras.callbacks.Callback):
        def __init__(self):super().__init__();self.best=-1.;self.stage='head';self.history=[]
        def on_epoch_end(self,epoch,logs=None):
            logits=predict_rows(self.model,workspace,rows['val'],args.batch_size)
            score=domain_selection(labels,logits,sources);logs['val_domain_macro_f1']=score
            metrics=classification_metrics(labels,logits,CLASS_NAMES)
            original=np.array(sources)=='original'
            original_f1=classification_metrics(labels[original],logits[original],CLASS_NAMES)['macroF1']
            record=dict(stage=self.stage,epoch=epoch+1,selectionScore=score,originalMacroF1=original_f1,
                validationAccuracy=metrics['accuracy'],**{k:float(v) for k,v in logs.items()})
            self.history.append(record);dump_json(args.run/'history.json',self.history)
            print(f'\nV2 {self.stage} epoch {epoch+1}: domain macro-F1={score:.4f}, original F1={original_f1:.4f}',flush=True)
            if score>self.best+1e-6:
                self.best=score
                pending=args.run/'.saving.keras';self.model.save(pending);os.replace(pending,args.run/'best.keras')
                np.save(args.run/'validation_logits.npy',logits)
                dump_json(args.run/'validation.json',dict(selectionScore=score,originalMacroF1=original_f1,metrics=metrics))
    selection=Selection()
    def stage(epochs,lr):
        model.compile(optimizer=tf.keras.optimizers.Adam(lr,clipnorm=5.),
            loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True,label_smoothing=.03),metrics=['accuracy'])
        model.fit(dataset,epochs=epochs,verbose=2,callbacks=[selection,
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_domain_macro_f1',mode='max',patience=2,factor=.5,min_lr=1e-7),
            tf.keras.callbacks.EarlyStopping(monitor='val_domain_macro_f1',mode='max',patience=4)])
    stage(args.head_epochs,1e-3)
    model=tf.keras.models.load_model(args.run/'best.keras',compile=False,safe_mode=True)
    backbone=model.get_layer('efficientnetb2');backbone.trainable=True
    for i,layer in enumerate(backbone.layers):layer.trainable=i>=int(.65*len(backbone.layers)) and not isinstance(layer,tf.keras.layers.BatchNormalization)
    selection.stage='finetune';stage(args.finetune_epochs,2e-5)
    model=tf.keras.models.load_model(args.run/'best.keras',compile=False,safe_mode=True)
    logits=predict_rows(model,workspace,rows['val'],args.batch_size)
    score=domain_selection(labels,logits,sources)
    if abs(score-selection.best)>1e-6:raise RuntimeError('Selected checkpoint did not reproduce validation')
    dump_json(args.run/'completed.json',dict(modelSha256=sha256(args.run/'best.keras'),selectionScore=score,testPredictionsRead=False))
    print('V2 training complete; test images remain unscored.',flush=True)


def finalize(args):
    """Export and calibrate before reading new test images; no automatic promotion."""
    protocol=load_protocol(args.workspace)
    completed=json.loads((args.run/'completed.json').read_text())
    training_config=json.loads((args.run/'train_config.json').read_text())
    validate_training_identity(training_config,protocol['provenance'])
    if sha256(args.run/'best.keras')!=completed['modelSha256']:raise ValueError('Selected model changed')
    identity=dict(checkpointSha256=completed['modelSha256'],splitFileSha256s=protocol['provenance']['splitFileSha256s'])
    lock=args.workspace/'selection_lock.json'
    if lock.exists():
        if json.loads(lock.read_text())!=identity:raise ValueError('Different model already selected')
    tf=configure_tensorflow(require_gpu=False)
    model=tf.keras.models.load_model(args.run/'best.keras',compile=False,safe_mode=True)
    val=protocol['rows']['val'];val_logits=predict_rows(model,protocol['workspace'],val,args.batch_size)
    labels=np.array([int(r['label_idx']) for r in val])
    actual_selection=domain_selection(labels,val_logits,[r['source'] for r in val])
    if abs(actual_selection-completed['selectionScore'])>1e-6:raise ValueError('Selected checkpoint validation score changed')
    blend=training_config.get('validationBlendSelection')
    if blend:
        original=np.array([r['source']=='original' for r in val])
        original_f1=classification_metrics(labels[original],val_logits[original],CLASS_NAMES)['macroF1']
        if original_f1<blend['originalMacroF1Floor']-1e-12:raise ValueError('Selected blend violates original validation retention')
    temperature=fit_temperature(val_logits,labels);abstention=choose_abstention(val_logits,labels,temperature)
    try:
        with lock.open('x',encoding='utf-8') as stream:json.dump(identity,stream,indent=2)
    except FileExistsError:
        if json.loads(lock.read_text())!=identity:raise ValueError('Different model already selected')
    args.output.mkdir(parents=True,exist_ok=False);model.save(args.output/'skin_six.keras')
    digest=sha256(args.output/'skin_six.keras')
    architecture=training_config['architecture']
    manifest=dict(modelVersion='skin-six-v2-'+architecture+'-'+digest[:12],architecture=architecture,modelFile='skin_six.keras',
        modelSha256=digest,inputSize=[224,224],preprocessing=PREPROCESSING_SPEC,outputType='logits',
        classNames={str(i):name for i,name in enumerate(CLASS_NAMES)},temperature=temperature,
        abstainThreshold=abstention['threshold'],abstentionValidation=abstention,
        qualityGate={'passed':False,'reason':'Full six-class external validation unavailable; awaiting locked comparisons'},
        normalClinicallyVerified=False,clinicalValidation=False,trainingConfig=training_config,
        provenance=protocol['provenance'])
    dump_json(args.output/'model_manifest.json',manifest);dump_json(args.output/'class_names.json',manifest['classNames'])
    export_lock=dict(checkpointSha256=completed['modelSha256'],modelSha256=digest,
        manifestSha256=sha256(args.output/'model_manifest.json'),splitFileSha256s=identity['splitFileSha256s'])
    with (args.output/'export_lock.json').open('x',encoding='utf-8') as stream:json.dump(export_lock,stream,indent=2)
    # The exported artifact/hash and calibration are now fixed before test predictions.
    test=protocol['rows']['test'];logits=predict_rows(model,protocol['workspace'],test,args.batch_size)
    np.save(args.output/'blind_test_logits.npy',logits)
    raw=classification_metrics([int(r['label_idx']) for r in test],logits,CLASS_NAMES)
    raw['macroPrecision']=float(np.mean([v['precision'] for v in raw['perClass'].values() if v['support']]))
    raw['macroF1']=float(np.mean([v['f1'] for v in raw['perClass'].values() if v['support']]))
    raw['macroRecall']=float(np.mean([v['recall'] for v in raw['perClass'].values() if v['support']]))
    dump_json(args.output/'evaluation.json',dict(blindTest=raw,gate=external_gate(raw),identity=identity,
        exportSha256=digest,selection=completed,normalClinicallyVerified=False,testUsedForTraining=False))
    from skin.config import SkinSettings
    from skin.service import SkinPredictor
    predictor=SkinPredictor(SkinSettings(model_path=args.output/'skin_six.keras',manifest_path=args.output/'model_manifest.json'))
    if not predictor.is_ready:raise RuntimeError('Export cannot be loaded by production predictor')
    dump_json(args.output/'runtime_smoke.json',predictor.predict_bytes((protocol['workspace']/val[0]['path']).read_bytes()))
    print(json.dumps(dict(blindTest=raw,gate=external_gate(raw),output=str(args.output)),indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    for name in ['train','finalize']:
        p=sub.add_parser(name);p.add_argument('--workspace',type=Path,required=True);p.add_argument('--run',type=Path,required=True)
        p.add_argument('--batch-size',type=int,default=32)
        if name=='train':p.add_argument('--head-epochs',type=int,default=3);p.add_argument('--finetune-epochs',type=int,default=8)
        else:p.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.batch_size<1:parser.error('Batch size must be positive')
    if args.command=='train':
        if min(args.head_epochs,args.finetune_epochs)<1:parser.error('Epoch counts must be positive')
        train(args)
    else:finalize(args)


if __name__=='__main__':main()
