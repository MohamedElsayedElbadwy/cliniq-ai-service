"""Continue a verified, unfinalized v2 checkpoint when its GPU session is lost."""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np

from skin.config import CLASS_NAMES
from skin.data.prepare_six import dump_json, sha256
from skin.preprocessing import preprocess_skin_image
from skin.training.metrics import classification_metrics
from skin.training.six_class import configure_tensorflow, epoch_order
from skin.training.v2 import load_protocol, validate_training_identity, domain_selection, predict_rows, sample_weights


def validate_recovery_source(config,provenance,checkpoint,expected_sha):
    validate_training_identity(config,provenance)
    if (config.get('protocol')!='skin-v2' or config.get('architecture')!='efficientnetb2'
        or config.get('initialization')!='fresh ImageNet; no skin checkpoint' or config.get('testPredictionsRead') is not False):
        raise ValueError('Recovery requires an untested v2 ImageNet-origin training checkpoint')
    if sha256(checkpoint)!=expected_sha:raise ValueError('Recovery checkpoint checksum differs')


def recover(args):
    protocol=load_protocol(args.workspace)
    if (args.workspace/'selection_lock.json').exists():raise ValueError('Cannot resume after test finalization')
    source_config=json.loads(args.source_config.read_text())
    validate_recovery_source(source_config,protocol['provenance'],args.checkpoint,args.checkpoint_sha256)
    args.run.mkdir(parents=True,exist_ok=False)
    tf=configure_tensorflow(require_gpu=False)
    model=tf.keras.models.load_model(args.checkpoint,compile=True,safe_mode=True)
    if model.name!='skin_six_v2' or tuple(model.input_shape[1:])!=(224,224,3) or model.output_shape[-1]!=6:
        raise ValueError('Wrong recovery architecture')
    if model.optimizer is None:raise ValueError('Compiled optimizer state is required for recovery')
    iterations=int(model.optimizer.iterations.numpy())
    if iterations!=args.expected_iterations:raise ValueError('Recovered optimizer iteration differs')
    rows=protocol['rows'];workspace=protocol['workspace']
    previous_steps=math.ceil(len(rows['train'])/source_config['batchSize'])
    if iterations%previous_steps:raise ValueError('Recovery requires a complete fine-tuning epoch checkpoint')
    fine_epochs=iterations//previous_steps
    if not 0<fine_epochs<=source_config['fineTuneEpochs']:raise ValueError('Invalid recovered fine-tuning progress')
    config=dict(source_config,batchSize=args.batch_size,headEpochs=0,fineTuneEpochs=args.epochs,
        recovery=dict(checkpointSha256=args.checkpoint_sha256,sourceConfigSha256=sha256(args.source_config),
            optimizerIterations=iterations,completedFineTuneEpochs=fine_epochs,originalBatchSize=source_config['batchSize'],
            sourceHeadEpochs=source_config['headEpochs'],optimizerRestored=True,
            note='CPU continuation from a saved intermediate GPU checkpoint; different batch size, not bitwise Colab replay.'),
        tensorflowVersion=tf.__version__,kerasVersion=tf.keras.__version__,testPredictionsRead=False)
    dump_json(args.run/'train_config.json',config)
    weights=sample_weights(rows['train']);epoch=source_config['headEpochs']+fine_epochs
    def samples():
        nonlocal epoch
        order=epoch_order(len(rows['train']),epoch);epoch+=1
        for i in order:
            row=rows['train'][i]
            yield preprocess_skin_image(workspace/row['path'])[0],np.eye(6,dtype=np.float32)[int(row['label_idx'])],weights[i]
    dataset=tf.data.Dataset.from_generator(samples,output_signature=(tf.TensorSpec((224,224,3),tf.float32),
        tf.TensorSpec((6,),tf.float32),tf.TensorSpec((),tf.float32)))
    dataset=dataset.apply(tf.data.experimental.assert_cardinality(len(rows['train']))).batch(args.batch_size).prefetch(1)
    labels=np.array([int(r['label_idx']) for r in rows['val']]);sources=np.array([r['source'] for r in rows['val']])
    class Selection(tf.keras.callbacks.Callback):
        def __init__(self):super().__init__();self.best=-1.;self.history=[];self.epoch=0
        def measure(self,epoch,logs):
            logits=predict_rows(model,workspace,rows['val'],args.batch_size)
            score=domain_selection(labels,logits,sources)
            original=sources=='original'
            f1=classification_metrics(labels[original],logits[original],CLASS_NAMES)['macroF1']
            self.history.append(dict(epoch=epoch,selectionScore=score,originalMacroF1=f1,
                optimizerIterations=int(model.optimizer.iterations.numpy()),**{k:float(v) for k,v in logs.items()}))
            dump_json(args.run/'history.json',self.history)
            print(f'Recovery validation epoch={epoch}: domain F1={score:.4f}; original F1={f1:.4f}',flush=True)
            if score>self.best+1e-6:
                self.best=score
                pending=args.run/'.saving.keras';model.save(pending);os.replace(pending,args.run/'best.keras')
                np.save(args.run/'validation_logits.npy',logits)
                dump_json(args.run/'validation.json',dict(selectionScore=score,originalMacroF1=f1,
                    metrics=classification_metrics(labels,logits,CLASS_NAMES)))
            return score
        def on_epoch_begin(self,epoch,logs=None):self.epoch=epoch+1;self.started=time.perf_counter()
        def on_train_batch_end(self,batch,logs=None):
            if (batch+1)%50==0:
                status=dict(epoch=self.epoch,batches=batch+1,totalBatches=math.ceil(len(rows['train'])/args.batch_size),
                    elapsedSeconds=round(time.perf_counter()-self.started,1),testPredictionsRead=False)
                dump_json(args.run/'progress.json',status);print(json.dumps(status),flush=True)
        def on_epoch_end(self,epoch,logs=None):
            logs['val_domain_macro_f1']=self.measure(epoch+1,logs)
    selection=Selection()
    print('Verifying recovered checkpoint on validation only; blind cases remain unscored.',flush=True)
    selection.measure(0,{})
    model.fit(dataset,epochs=args.epochs,verbose=2,callbacks=[selection,
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_domain_macro_f1',mode='max',patience=2,factor=.5,min_lr=1e-7),
        tf.keras.callbacks.EarlyStopping(monitor='val_domain_macro_f1',mode='max',patience=4)])
    best=tf.keras.models.load_model(args.run/'best.keras',compile=False,safe_mode=True)
    score=domain_selection(labels,predict_rows(best,workspace,rows['val'],args.batch_size),sources)
    if abs(score-selection.best)>1e-6:raise RuntimeError('Recovery checkpoint failed validation reproduction')
    dump_json(args.run/'completed.json',dict(modelSha256=sha256(args.run/'best.keras'),selectionScore=score,
        recovery=True,testPredictionsRead=False))
    print('Recovery training completed. Finalization and blind evaluation have not run.',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['workspace','run','checkpoint','source-config']:parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--checkpoint-sha256',required=True);parser.add_argument('--expected-iterations',type=int,required=True)
    parser.add_argument('--batch-size',type=int,default=8);parser.add_argument('--epochs',type=int,default=3)
    args=parser.parse_args()
    if min(args.batch_size,args.epochs,args.expected_iterations)<1:parser.error('Counts must be positive')
    recover(args)


if __name__=='__main__':main()
