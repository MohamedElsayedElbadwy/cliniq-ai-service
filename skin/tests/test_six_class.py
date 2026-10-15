"""Fresh six-class data protocol; real archives/images, no network in tests."""
import csv
import hashlib
import io
import json
import zipfile
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image


def test_epoch_order_mixes_class_sorted_manifest_without_dropping_images():
    from skin.training.six_class import epoch_order
    labels = np.repeat(np.arange(6), [805, 615, 696, 548, 450, 647])
    first = epoch_order(len(labels), 0)
    assert sorted(first) == list(range(len(labels)))
    assert set(labels[first[:512]]) == set(range(6))
    assert set(labels[first[-512:]]) == set(range(6))
    assert np.array_equal(first, epoch_order(len(labels), 0))
    assert not np.array_equal(first, epoch_order(len(labels), 1))


def test_flip_candidate_cannot_bypass_locked_test_protocol(tmp_path):
    from skin.training.six_class import make_flip_candidate
    workspace = tmp_path / 'workspace'
    workspace.mkdir()
    (workspace / 'finalization_lock.json').write_text('{}')
    args = SimpleNamespace(workspace=workspace, source_run=tmp_path / 'missing', run=tmp_path / 'new')
    with pytest.raises(ValueError, match='locked'):
        make_flip_candidate(args)
    assert not args.run.exists()


def test_flip_candidate_rejects_synthetic_source_before_loading_weights(tmp_path):
    from skin.data.prepare_six import prepare
    from skin.training.six_class import make_flip_candidate
    archive, manifest = make_archive(tmp_path, [
        (f'{label}/{i}.png', png(100 * k + i))
        for k, label in enumerate(['acne', 'Eczema', 'Psoriasis', 'Rosacea', 'Ringworm', 'normal'])
        for i in range(6)
    ])
    workspace = tmp_path / 'workspace'
    prepare(archive, manifest, workspace)
    source = tmp_path / 'source'
    source.mkdir()
    (source / 'train_config.json').write_text(json.dumps(dict(productionEligible=False)))
    (source / 'completed.json').write_text('{}')
    args = SimpleNamespace(workspace=workspace, source_run=source, run=tmp_path / 'candidate')
    with pytest.raises(ValueError, match='eligible six-class'):
        make_flip_candidate(args)
    assert not args.run.exists()


def make_archive(tmp_path, entries):
    archive = tmp_path / 'source.zip'
    manifest = tmp_path / 'manifest.csv'
    with zipfile.ZipFile(archive, 'w') as zf, manifest.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['path', 'class_name', 'size_bytes', 'sha256'])
        writer.writeheader()
        for path, payload in entries:
            zf.writestr(path, payload)
            writer.writerow(dict(path=path, class_name=path.split('/')[0],
                                 size_bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest()))
    return archive, manifest


def png(seed):
    image = Image.fromarray(np.random.default_rng(seed).integers(0, 256, (72, 72, 3), dtype=np.uint8))
    stream = io.BytesIO()
    image.save(stream, format='PNG')
    return stream.getvalue()


def test_only_six_classes_and_original_images_are_extracted(tmp_path):
    from skin.data.prepare_six import extract_verified
    archive, manifest = make_archive(tmp_path, [
        ('acne/a.png', png(1)), ('normal/b.png', png(2)),
        ('acne/aug_1_a.png', png(3)), ('Herpes/c.png', png(4)),
    ])
    report = extract_verified(archive, manifest, tmp_path / 'images')
    assert report['selectedOriginals'] == 2
    assert report['excludedPreAugmented'] == 1
    assert sorted(p.relative_to(tmp_path / 'images').as_posix()
                  for p in (tmp_path / 'images').rglob('*.png')) == ['acne/a.png', 'normal/b.png']


def test_archive_content_must_match_publisher_sha256(tmp_path):
    from skin.data.prepare_six import extract_verified
    archive, manifest = make_archive(tmp_path, [('acne/a.png', png(1))])
    text = manifest.read_text().replace(hashlib.sha256(png(1)).hexdigest(), '0' * 64)
    manifest.write_text(text)
    with pytest.raises(ValueError, match='checksum'):
        extract_verified(archive, manifest, tmp_path / 'images')


@pytest.mark.parametrize('path', ['acne/../../escape.png', 'acne/..\\escape.png', 'acne/C:escape.png'])
def test_unsafe_manifest_paths_are_rejected(tmp_path, path):
    from skin.data.prepare_six import extract_verified
    archive, manifest = make_archive(tmp_path, [(path, png(1))])
    with pytest.raises(ValueError, match='path'):
        extract_verified(archive, manifest, tmp_path / 'images')
    assert not (tmp_path / 'escape.png').exists()


def test_preparation_removes_conflicts_and_freezes_portable_splits(tmp_path):
    from skin.data.prepare_six import prepare
    entries = [(f'{label}/{i}.png', png(100 * k + i))
               for k, label in enumerate(['acne', 'Eczema', 'Psoriasis', 'Rosacea', 'Ringworm', 'normal'])
               for i in range(12)]
    # Same pixels with contradictory diagnoses must disappear from every split.
    entries += [('acne/conflict.png', png(999)), ('Eczema/conflict.png', png(999))]
    archive, manifest = make_archive(tmp_path, entries)
    workspace = tmp_path / 'six'
    report = prepare(archive, manifest, workspace)
    assert report['retained'] == 72
    assert report['removedReasons']['cross_class_duplicate_cluster'] == 2
    seen = {}
    counts = Counter()
    for split in ['train', 'val', 'test']:
        with (workspace / 'splits' / f'{split}.csv').open() as stream:
            rows = list(csv.DictReader(stream))
        assert len(rows) > 0
        for row in rows:
            assert not Path(row['path']).is_absolute()
            assert (workspace / row['path']).is_file()
            for key in ['sha256', 'pixel_sha256', 'cluster_id']:
                assert seen.setdefault((key, row[key]), split) == split
            counts[row['label']] += 1
    assert counts == {'Acne': 12, 'Eczema': 12, 'Psoriasis': 12,
                      'Rosacea': 12, 'Ringworm': 12, 'Normal': 12}
    assert json.loads((workspace / 'class_names.json').read_text()) == {
        '0': 'Acne', '1': 'Eczema', '2': 'Psoriasis', '3': 'Rosacea', '4': 'Ringworm', '5': 'Normal'}
    with pytest.raises(FileExistsError):
        prepare(archive, manifest, workspace)


def test_training_rejects_tampered_or_cross_split_duplicate_manifests(tmp_path):
    from skin.training.six_class import load_protocol
    from skin.data.prepare_six import prepare
    entries = [(f'{label}/{i}.png', png(100 * k + i))
               for k, label in enumerate(['acne', 'Eczema', 'Psoriasis', 'Rosacea', 'Ringworm', 'normal'])
               for i in range(12)]
    archive, manifest = make_archive(tmp_path, entries)
    workspace = tmp_path / 'six'
    prepare(archive, manifest, workspace)
    protocol = load_protocol(workspace)
    assert len(protocol['names']) == 6
    assert set(protocol['rows']) == {'train', 'val', 'test'}
    with (workspace / 'splits' / 'test.csv').open('a') as stream:
        stream.write('tampered\n')
    with pytest.raises(ValueError, match='changed'):
        load_protocol(workspace)


def test_new_training_run_never_reuses_old_22_class_experiment(tmp_path):
    from skin.training.six_class import create_run
    run = tmp_path / 'six-run'
    create_run(run, {'classCount': 6})
    (run / 'old.keras').write_bytes(b'keep')
    with pytest.raises(FileExistsError):
        create_run(run, {'classCount': 22})
    assert (run / 'old.keras').read_bytes() == b'keep'


def test_cross_split_pixel_leakage_is_rejected_even_with_updated_csv_checksum(tmp_path):
    from skin.training.six_class import load_protocol
    from skin.data.prepare_six import prepare, sha256
    entries = [(f'{label}/{i}.png', png(100 * k + i))
               for k, label in enumerate(['acne', 'Eczema', 'Psoriasis', 'Rosacea', 'Ringworm', 'normal'])
               for i in range(12)]
    archive, manifest = make_archive(tmp_path, entries)
    workspace = tmp_path / 'six'
    prepare(archive, manifest, workspace)
    with (workspace / 'splits' / 'train.csv').open() as stream:
        train_rows = list(csv.DictReader(stream))
    test_path = workspace / 'splits' / 'test.csv'
    with test_path.open() as stream:
        test_rows = list(csv.DictReader(stream))
    test_rows[0]['pixel_sha256'] = train_rows[0]['pixel_sha256']
    with test_path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(test_rows[0]))
        writer.writeheader()
        writer.writerows(test_rows)
    provenance_path = workspace / 'provenance.json'
    provenance = json.loads(provenance_path.read_text())
    provenance['splitFileSha256s']['test'] = sha256(test_path)
    provenance_path.write_text(json.dumps(provenance))
    with pytest.raises(ValueError, match='leakage'):
        load_protocol(workspace)


def test_synthetic_smoke_model_cannot_be_finalized_or_exported(tmp_path):
    from skin.training.six_class import finalize
    from skin.data.prepare_six import prepare
    entries = [(f'{label}/{i}.png', png(100 * k + i))
               for k, label in enumerate(['acne', 'Eczema', 'Psoriasis', 'Rosacea', 'Ringworm', 'normal'])
               for i in range(12)]
    archive, manifest = make_archive(tmp_path, entries)
    workspace = tmp_path / 'six'
    prepare(archive, manifest, workspace)
    run = tmp_path / 'smoke'
    run.mkdir()
    (run / 'train_config.json').write_text('{"productionEligible":false}')
    (run / 'completed.json').write_text('{}')
    output = tmp_path / 'export'
    with pytest.raises(ValueError, match='Smoke-test'):
        finalize(SimpleNamespace(workspace=workspace, run=run, output=output))
    assert not output.exists()
    assert not (workspace / 'finalization_lock.json').exists()


def test_test_selection_lock_cannot_be_replaced_by_another_model_or_threshold(tmp_path):
    from skin.training.six_class import freeze_selection
    identity = {'modelSha256': 'chosen', 'qualityThresholds': {'accuracy': .85}}
    freeze_selection(tmp_path, identity)
    freeze_selection(tmp_path, identity)  # An interrupted run may resume the SAME protocol.
    with pytest.raises(ValueError, match='locked'):
        freeze_selection(tmp_path, {'modelSha256': 'different'})
    with pytest.raises(ValueError, match='locked'):
        freeze_selection(tmp_path, {'modelSha256': 'chosen', 'qualityThresholds': {'accuracy': .1}})
    assert json.loads((tmp_path / 'finalization_lock.json').read_text()) == identity
