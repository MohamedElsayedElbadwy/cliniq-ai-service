"""Fresh TensorFlow GPU training for six classes; independent of all legacy runs.

train reads train/val images only. finalize freezes selection, calibrates on val,
then evaluates the untouched test set once and exports a ClinIQ-compatible bundle.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from skin.data.prepare_six import CLASS_NAMES, DATASET_ID, RIGHTS_NOTE, dump_json, sha256
from skin.preprocessing import PREPROCESSING_SPEC, preprocess_skin_image
from .metrics import calibration_metrics, choose_abstention, classification_metrics, fit_temperature


def load_protocol(workspace):
    workspace = Path(workspace).resolve()
    provenance = json.loads((workspace / 'provenance.json').read_text())
    if sha256(workspace / 'class_names.json') != provenance['classNamesSha256']:
        raise ValueError('Class map changed since preparation')
    mapping = json.loads((workspace / 'class_names.json').read_text())
    if mapping != {str(i): name for i, name in enumerate(CLASS_NAMES)}:
        raise ValueError('Protocol must contain exactly the six agreed classes in exported index order')
    rows = {}
    owners = {}
    for split in ('train', 'val', 'test'):
        path = workspace / 'splits' / f'{split}.csv'
        if sha256(path) != provenance['splitFileSha256s'][split]:
            raise ValueError(f'{split} split changed since preparation')
        with path.open(encoding='utf-8', newline='') as stream:
            rows[split] = list(csv.DictReader(stream))
        if not rows[split] or {r['label'] for r in rows[split]} != set(CLASS_NAMES):
            raise ValueError(f'{split}: every class needs support')
        for row in rows[split]:
            if row['split'] != split or int(row['label_idx']) not in range(6) or CLASS_NAMES[int(row['label_idx'])] != row['label']:
                raise ValueError('Split label/index mismatch')
            target = (workspace / row['path']).resolve()
            if Path(row['path']).is_absolute() or not target.is_relative_to(workspace / 'images'):
                raise ValueError('Image path escapes prepared workspace')
            for key in ('sha256', 'pixel_sha256', 'cluster_id'):
                if owners.setdefault((key, row[key]), split) != split:
                    raise ValueError(f'Cross-split leakage: {key}')
    return dict(workspace=workspace, provenance=provenance, names=list(CLASS_NAMES), rows=rows)


def create_run(path, config):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    dump_json(path / 'train_config.json', config)


def freeze_selection(workspace, identity):
    """Exclusive creation prevents two finalizers choosing different test models."""
    path = Path(workspace) / 'finalization_lock.json'
    try:
        with path.open('x', encoding='utf-8') as stream:
            json.dump(identity, stream, indent=2)
    except FileExistsError:
        if json.loads(path.read_text(encoding='utf-8')) != identity:
            raise ValueError('Test protocol already locked to another model or quality threshold')


def image_batches(protocol, split, batch_size, verify=True):
    rows = protocol['rows'][split]
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        images = []
        for row in batch:
            path = protocol['workspace'] / row['path']
            if verify and sha256(path) != row['sha256']:
                raise ValueError('Prepared image bytes changed')
            images.append(preprocess_skin_image(path, max_pixels=100_000_000)[0])
        yield np.stack(images)


def predict_split(model, protocol, split, batch_size):
    return np.concatenate([np.asarray(model(batch, training=False))
                           for batch in image_batches(protocol, split, batch_size)])


def epoch_order(count, epoch):
    """Globally mix lightweight indexes before decoding class-sorted CSV rows."""
    return np.random.default_rng(np.random.SeedSequence([42, epoch])).permutation(count)


def configure_tensorflow(require_gpu):
    os.environ.setdefault('KERAS_BACKEND', 'tensorflow')
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    gpus = tf.config.list_physical_devices('GPU')
    if require_gpu and not gpus:
        raise RuntimeError('No TensorFlow GPU. Select a GPU runtime in Colab/Kaggle before training.')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    tf.keras.utils.set_random_seed(42)
    tf.config.experimental.enable_op_determinism()
    return tf


def train(args):
    protocol = load_protocol(args.workspace)
    if (Path(args.workspace) / 'finalization_lock.json').exists():
        raise ValueError('Test protocol is locked. No further model selection on this dataset split.')
    if args.run.exists():
        raise FileExistsError('Use a new run directory; old checkpoints are never reused or overwritten')
    tf = configure_tensorflow(require_gpu=not args.allow_cpu)
    provenance = protocol['provenance']
    if not provenance.get('officialArchiveVerified') and not args.allow_cpu:
        raise ValueError('Training requires the verified pinned source archive')
    config = dict(classCount=6, classNames=protocol['names'], architecture=args.architecture,
                  initialization='random weights (synthetic CPU smoke only)' if args.allow_cpu else 'ImageNet (fresh; no previous skin checkpoint)', seed=42,
                  batchSize=args.batch_size, headEpochs=args.head_epochs, fineTuneEpochs=args.finetune_epochs,
                  selectionMetric='validation macro-F1', splitFileSha256s=provenance['splitFileSha256s'],
                  dataset=DATASET_ID, revision=provenance['revision'],
                  sourceArchiveSha256=provenance['sourceArchiveSha256'],
                  sourceManifestSha256=provenance['sourceManifestSha256'],
                  tensorflowVersion=tf.__version__, kerasVersion=tf.keras.__version__,
                  augmentation='train only: horizontal flip, rotation .04, zoom .1, contrast .1',
                  trainingOrder='global seeded index permutation per epoch; validation/test retain manifest order',
                  labelSmoothing=.05, testImagesRead=False, productionEligible=not args.allow_cpu,
                  note='--allow-cpu is an offline smoke test and cannot produce a deployable export' if args.allow_cpu else '')
    create_run(args.run, config)
    # Check only train/val bytes. Held-out test images stay unopened until finalization.
    for split in ('train', 'val'):
        for row in protocol['rows'][split]:
            if sha256(protocol['workspace'] / row['path']) != row['sha256']:
                raise ValueError('Prepared image bytes changed')

    def dataset(split):
        rows = protocol['rows'][split]
        epoch = 0

        def samples():
            nonlocal epoch
            order = epoch_order(len(rows), epoch) if split == 'train' else range(len(rows))
            epoch += 1
            for index in order:
                row = rows[index]
                image = preprocess_skin_image(protocol['workspace'] / row['path'], max_pixels=100_000_000)[0]
                yield image, np.eye(6, dtype=np.float32)[int(row['label_idx'])]

        result = tf.data.Dataset.from_generator(samples, output_signature=(
            tf.TensorSpec((224, 224, 3), tf.float32), tf.TensorSpec((6,), tf.float32)))
        result = result.apply(tf.data.experimental.assert_cardinality(len(rows)))
        return result.batch(args.batch_size).prefetch(1)

    factory = {'efficientnetb0': tf.keras.applications.EfficientNetB0,
               'efficientnetb2': tf.keras.applications.EfficientNetB2}[args.architecture]
    # The CPU smoke test explicitly avoids downloading ImageNet and is labelled ineligible.
    backbone = factory(include_top=False, weights=None if args.allow_cpu else 'imagenet',
                       input_shape=(224, 224, 3), pooling='avg')
    backbone.trainable = False
    inputs = tf.keras.Input((224, 224, 3), dtype='float32')
    augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip('horizontal', seed=42),
        tf.keras.layers.RandomRotation(.04, fill_mode='reflect', seed=43),
        tf.keras.layers.RandomZoom(.1, fill_mode='reflect', seed=44),
        tf.keras.layers.RandomContrast(.1, seed=45),
    ], name='train_augmentation')
    features = backbone(augmentation(inputs), training=False)  # Keep BatchNorm statistics frozen.
    outputs = tf.keras.layers.Dense(6, dtype='float32', name='class_logits')(
        tf.keras.layers.Dropout(.35, seed=46)(features))
    model = tf.keras.Model(inputs, outputs, name=f'skin_six_{args.architecture}')
    counts = np.bincount([int(r['label_idx']) for r in protocol['rows']['train']], minlength=6)
    class_weights = {i: float(counts.sum() / (6 * count)) for i, count in enumerate(counts)}
    config['classWeights'] = class_weights
    dump_json(args.run / 'train_config.json', config)
    labels = np.array([int(r['label_idx']) for r in protocol['rows']['val']])

    class ValidationSelection(tf.keras.callbacks.Callback):
        def __init__(self):
            super().__init__()
            self.best = -1.
            self.stage = 'head'
            self.history = []

        def on_epoch_end(self, epoch, logs=None):
            logits = predict_split(self.model, protocol, 'val', args.batch_size)
            metrics = classification_metrics(labels, logits, protocol['names'])
            logs['val_macro_f1'] = metrics['macroF1']
            record = dict(stage=self.stage, epoch=epoch + 1,
                          **{key: float(value) for key, value in logs.items()})
            self.history.append(record)
            dump_json(args.run / 'history.json', self.history)
            print(f'\n{self.stage} validation accuracy={metrics["accuracy"]:.4f} macro-F1={metrics["macroF1"]:.4f}', flush=True)
            if metrics['macroF1'] > self.best + 1e-6:
                self.best = metrics['macroF1']
                self.model.save(args.run / 'best.keras')
                dump_json(args.run / 'validation.json', metrics)
                np.save(args.run / 'validation_logits.npy', logits)

    selection = ValidationSelection()

    def fit_stage(epochs, learning_rate):
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate, clipnorm=5.),
                      loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=.05),
                      metrics=[tf.keras.metrics.CategoricalAccuracy(name='accuracy')])
        model.fit(dataset('train'), validation_data=dataset('val'), epochs=epochs, shuffle=False,
                  class_weight=class_weights, callbacks=[selection,
                      tf.keras.callbacks.ReduceLROnPlateau(monitor='val_macro_f1', mode='max', patience=2, factor=.5, min_lr=1e-7),
                      tf.keras.callbacks.EarlyStopping(monitor='val_macro_f1', mode='max', patience=5)], verbose=2)

    fit_stage(args.head_epochs, 1e-3)
    model = tf.keras.models.load_model(args.run / 'best.keras', compile=False, safe_mode=True)
    backbone = model.get_layer(backbone.name)
    backbone.trainable = True
    first_trainable = int(len(backbone.layers) * .65)
    for index, layer in enumerate(backbone.layers):
        layer.trainable = index >= first_trainable and not isinstance(layer, tf.keras.layers.BatchNormalization)
    selection.stage = 'finetune'
    fit_stage(args.finetune_epochs, 2e-5)
    selected = tf.keras.models.load_model(args.run / 'best.keras', compile=False, safe_mode=True)
    logits = predict_split(selected, protocol, 'val', args.batch_size)
    validation = classification_metrics(labels, logits, protocol['names'])
    if abs(validation['macroF1'] - selection.best) > 1e-6:
        raise RuntimeError('Saved checkpoint does not reproduce selected validation score')
    dump_json(args.run / 'completed.json', dict(modelSha256=sha256(args.run / 'best.keras'),
                                              validation=validation, testImagesRead=False))
    print(json.dumps(validation, indent=2), flush=True)


def finalize(args):
    protocol = load_protocol(args.workspace)
    config = json.loads((args.run / 'train_config.json').read_text())
    completed = json.loads((args.run / 'completed.json').read_text())
    if not config.get('productionEligible'):
        raise ValueError('Smoke-test models cannot be finalized/deployed')
    if config['splitFileSha256s'] != protocol['provenance']['splitFileSha256s']:
        raise ValueError('Run belongs to a different prepared dataset')
    model_path = args.run / 'best.keras'
    if sha256(model_path) != completed['modelSha256']:
        raise ValueError('Selected model changed since training')
    identity = dict(modelSha256=completed['modelSha256'], splitFileSha256s=config['splitFileSha256s'],
                    classNames=protocol['names'], qualityThresholds=dict(accuracy=args.min_accuracy,
                        macroF1=args.min_macro_f1, perClassRecall=args.min_recall))
    freeze_selection(args.workspace, identity)
    if args.output.exists():
        raise FileExistsError('Export directory exists. Retain it; do not silently overwrite test results/artifacts.')
    tf = configure_tensorflow(require_gpu=False)
    model = tf.keras.models.load_model(model_path, compile=False, safe_mode=True)
    val_logits = predict_split(model, protocol, 'val', args.batch_size)
    val_labels = np.array([int(r['label_idx']) for r in protocol['rows']['val']])
    val_metrics = classification_metrics(val_labels, val_logits, protocol['names'])
    temperature = fit_temperature(val_logits, val_labels)
    abstention = choose_abstention(val_logits, val_labels, temperature)
    # Selection and calibration are now frozen. All examples count in primary metrics.
    test_logits = predict_split(model, protocol, 'test', args.batch_size)
    test_labels = np.array([int(r['label_idx']) for r in protocol['rows']['test']])
    metrics = classification_metrics(test_labels, test_logits, protocol['names'])
    calibration = calibration_metrics(test_logits, test_labels, temperature)
    from .metrics import softmax
    probs = softmax(test_logits, temperature)
    answered = probs.max(axis=1) >= abstention['threshold']
    gate = dict(minAccuracy=args.min_accuracy, minMacroF1=args.min_macro_f1,
                minPerClassRecall=args.min_recall, passed=bool(
                    metrics['accuracy'] >= args.min_accuracy and metrics['macroF1'] >= args.min_macro_f1
                    and min(r['recall'] for r in metrics['perClass'].values()) >= args.min_recall))
    report = dict(test=metrics, testCalibration=calibration, qualityGate=gate,
                  selective=dict(coverage=float(answered.mean()), answered=int(answered.sum()),
                                 accuracy=float((probs.argmax(axis=1)[answered] == test_labels[answered]).mean()) if answered.any() else None),
                  selection='validation macro-F1 only; primary test accuracy includes all images',
                  patientIndependence=False, clinicalValidation=False, provenance=protocol['provenance'])
    args.output.mkdir(parents=True, exist_ok=False)
    model_name = 'skin_six.keras'
    shutil.copy2(model_path, args.output / model_name)
    created = datetime.now(timezone.utc).isoformat(timespec='seconds')
    manifest = dict(modelVersion=f'skin-six-{config["architecture"]}-{completed["modelSha256"][:12]}',
                    architecture=config['architecture'], modelFile=model_name, modelSha256=completed['modelSha256'],
                    inputSize=[224, 224], preprocessing=PREPROCESSING_SPEC, outputType='logits',
                    classNames={str(i): name for i, name in enumerate(protocol['names'])}, temperature=temperature,
                    abstainThreshold=abstention['threshold'], abstentionValidation=abstention,
                    validationMetrics=val_metrics, testMetrics=metrics, qualityGate=gate,
                    splitFileSha256s=config['splitFileSha256s'], trainingConfig=config,
                    dataset=DATASET_ID, datasetReference=protocol['provenance']['sourceUrl'],
                    datasetRevision=protocol['provenance']['revision'], license=RIGHTS_NOTE,
                    createdAtUtc=created, createdDate=created[:10], externalGeneralizationValidated=False)
    dump_json(args.output / 'model_manifest.json', manifest)
    dump_json(args.output / 'class_names.json', manifest['classNames'])
    dump_json(args.output / 'evaluation.json', report)
    dump_json(args.output / 'train_config.json', config)
    # Verify reload with the EXACT production service and preprocessing contract.
    from skin.config import SkinSettings
    from skin.service import SkinPredictor
    predictor = SkinPredictor(SkinSettings(model_path=args.output / model_name,
                                           manifest_path=args.output / 'model_manifest.json'))
    if not predictor.is_ready:
        raise RuntimeError('Export failed production SkinPredictor readiness')
    sample = protocol['workspace'] / protocol['rows']['val'][0]['path']
    prediction = predictor.predict_bytes(sample.read_bytes())
    dump_json(args.output / 'runtime_smoke.json', prediction)
    lines = ['# Six-class held-out evaluation', '',
             f'Accuracy: {metrics["accuracy"]:.2%}; macro-F1: {metrics["macroF1"]:.4f}; test images: {metrics["support"]}.', '',
             f'Quality gate passed: {gate["passed"]}. No clinical or patient-independent validation.', '',
             '| Class | Precision | Recall | F1 | Images |', '|---|---:|---:|---:|---:|']
    lines += [f'| {name} | {m["precision"]:.3f} | {m["recall"]:.3f} | {m["f1"]:.3f} | {m["support"]} |'
              for name, m in metrics['perClass'].items()]
    (args.output / 'evaluation.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps(dict(accuracy=metrics['accuracy'], macroF1=metrics['macroF1'], qualityGate=gate,
                          export=str(args.output)), indent=2), flush=True)
    if not gate['passed']:
        raise SystemExit('Measured quality gate failed; report/artifact retained for inspection, not recommended for deployment.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    train_parser = commands.add_parser('train')
    train_parser.add_argument('--workspace', type=Path, required=True)
    train_parser.add_argument('--run', type=Path, required=True)
    train_parser.add_argument('--architecture', choices=['efficientnetb0', 'efficientnetb2'], default='efficientnetb2')
    train_parser.add_argument('--batch-size', type=int, default=24)
    train_parser.add_argument('--head-epochs', type=int, default=8)
    train_parser.add_argument('--finetune-epochs', type=int, default=25)
    train_parser.add_argument('--allow-cpu', action='store_true', help='Synthetic offline smoke test only; random weights; export disabled')
    final_parser = commands.add_parser('finalize')
    final_parser.add_argument('--workspace', type=Path, required=True)
    final_parser.add_argument('--run', type=Path, required=True)
    final_parser.add_argument('--output', type=Path, required=True)
    final_parser.add_argument('--batch-size', type=int, default=24)
    final_parser.add_argument('--min-accuracy', type=float, default=.85)
    final_parser.add_argument('--min-macro-f1', type=float, default=.85)
    final_parser.add_argument('--min-recall', type=float, default=.70)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error('Batch size must be positive')
    if args.command == 'train':
        if args.head_epochs < 1 or args.finetune_epochs < 1:
            parser.error('Both training stages require at least one epoch')
        train(args)
    else:
        if any(not 0 <= value <= 1 for value in (args.min_accuracy, args.min_macro_f1, args.min_recall)):
            parser.error('Quality thresholds must be between zero and one')
        finalize(args)


if __name__ == '__main__':
    main()
