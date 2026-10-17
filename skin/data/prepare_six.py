"""Download, verify and freeze a NEW six-class dataset, isolated from legacy runs.

python -m skin.data.prepare_six --download --workspace workspaces/skin-six
Only original images are retained; supplied aug_* images are excluded BEFORE splitting.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import urllib.request
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath

from skin.config import CLASS_NAMES
from .inspect_dataset import assign_splits, find_clusters, inspect_image, write_csv

DATASET_ID = 'Cuebot/skin-assessment-13-class-filtered'
REVISION = '41c48b0b0a3b9f36e645db494a0563f7cc770ebd'
ARCHIVE_NAME = 'skin-assessment-13-classes.zip'
ARCHIVE_SHA256 = 'd0c5d8019d960ceccabf7dcf1dddf66ebacc0c1d7393b7340b0dbdee6b5259b1'
MANIFEST_SHA256 = '1105026c7d9b1af8939862de7e5783d883f910af94410e8bf0c436cf9da529d4'
SOURCE_LABELS = dict(zip(('acne', 'Eczema', 'Psoriasis', 'Rosacea', 'Ringworm', 'normal'), CLASS_NAMES))
RIGHTS_NOTE = ('Research/technical evaluation only. Inherited folder labels are not independently '
              'clinically verified. Source Kaggle CC0 declaration does not verify upstream image '
              'rights; the filtered release asserts no repository-wide license.')


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def dump_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name('.' + path.name + '.tmp')
    pending.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
    pending.replace(path)


def download(cache):
    cache.mkdir(parents=True, exist_ok=True)
    for name in ('README.md', 'manifest.csv', ARCHIVE_NAME):
        target = cache / name
        if not target.exists():
            partial = target.with_suffix(target.suffix + '.partial')
            url = f'https://huggingface.co/datasets/{DATASET_ID}/resolve/{REVISION}/{name}'
            print(f'Downloading pinned {name}', flush=True)
            with urllib.request.urlopen(url, timeout=120) as response, partial.open('wb') as stream:
                received = 0
                while block := response.read(1024 * 1024):
                    stream.write(block)
                    received += len(block)
                    if received % (32 * 1024 * 1024) == 0:
                        print(f'{name}: {received // (1024 * 1024)} MiB downloaded', flush=True)
            partial.replace(target)
    if sha256(cache / ARCHIVE_NAME) != ARCHIVE_SHA256:
        raise ValueError('Source archive checksum mismatch; no images extracted')
    if sha256(cache / 'manifest.csv') != MANIFEST_SHA256:
        raise ValueError('Source manifest checksum mismatch; no images extracted')
    return cache / ARCHIVE_NAME, cache / 'manifest.csv'


def extract_verified(archive, manifest, destination):
    """Select exact source folders; verify bytes before writing safe, new paths."""
    destination = Path(destination).resolve()
    counts = Counter()
    selected = []
    seen = set()
    with Path(manifest).open(encoding='utf-8-sig', newline='') as stream:
        for row in csv.DictReader(stream):
            if row['class_name'] not in SOURCE_LABELS:
                continue
            name = row['path']
            path = PurePosixPath(name)
            if (len(path.parts) != 2 or path.parts[0] != row['class_name']
                    or '\\' in name or ':' in name or '..' in path.parts
                    or path.is_absolute() or name != path.as_posix()):
                raise ValueError(f'Unsafe manifest path: {name}')
            if name.casefold() in seen:
                raise ValueError('Duplicate manifest path')
            seen.add(name.casefold())
            counts['selectedFiles'] += 1
            if re.match(r'^aug_', path.name, flags=re.IGNORECASE):
                counts['excludedPreAugmented'] += 1
                continue
            selected.append(row)
    if not selected:
        raise ValueError('No original six-class images in manifest')
    with zipfile.ZipFile(archive) as zf:
        members = Counter(zf.namelist())
        for row in selected:
            name = row['path']
            if members[name] != 1:
                raise ValueError(f'Missing/duplicate archive path: {name}')
            info = zf.getinfo(name)
            if info.file_size != int(row['size_bytes']) or info.file_size > 25 * 1024 * 1024:
                raise ValueError(f'Image size mismatch: {name}')
            payload = zf.read(info)  # Includes ZIP CRC validation.
            if hashlib.sha256(payload).hexdigest() != row['sha256']:
                raise ValueError(f'Image checksum mismatch: {name}')
            target = destination.joinpath(*PurePosixPath(name).parts)
            if not target.resolve().is_relative_to(destination):
                raise ValueError('Unsafe extraction path')
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open('xb') as output:
                output.write(payload)
    counts['selectedOriginals'] = len(selected)
    return dict(counts)


def prepare(archive, manifest, workspace):
    """One immutable run folder; no reads/writes of legacy data or skin/artifacts."""
    workspace = Path(workspace).resolve()
    workspace.mkdir(parents=True, exist_ok=False)
    selected = extract_verified(archive, manifest, workspace / 'images')
    images_dir = workspace / 'images'
    # WindowsPath sorting folds case; use explicit POSIX strings for the same
    # ordering/group IDs on Windows, Colab and Kaggle.
    files = sorted((p for p in images_dir.rglob('*') if p.is_file()),
                   key=lambda p: p.relative_to(images_dir).as_posix())
    with ThreadPoolExecutor(max_workers=4) as pool:
        rows = list(pool.map(inspect_image, files))
    removed = [dict(r, reason='unreadable') for r in rows if r['error']]
    valid = [r for r in rows if not r['error']]
    clusters = find_clusters(valid)
    groups = defaultdict(list)
    for row, cluster in zip(valid, clusters):
        row['label'] = SOURCE_LABELS[row['label']]
        row['cluster_id'] = cluster
        row['path'] = Path(row['path']).relative_to(workspace).as_posix()
        groups[cluster].append(row)
    conflicts = {cluster for cluster, members in groups.items() if len({r['label'] for r in members}) > 1}
    retained = []
    seen_bytes, seen_pixels = set(), set()
    for row in valid:
        reason = None
        if row['cluster_id'] in conflicts:
            reason = 'cross_class_duplicate_cluster'
        elif row['sha256'] in seen_bytes or row['pixel_sha256'] in seen_pixels:
            reason = 'exact_duplicate'
        if reason:
            removed.append(dict(row, reason=reason))
            continue
        retained.append(row)
        seen_bytes.add(row['sha256'])
        seen_pixels.add(row['pixel_sha256'])
    if {r['label'] for r in retained} != set(CLASS_NAMES):
        raise ValueError('Every one of the six classes must retain original images')
    assignments = assign_splits(retained, seed=42)
    for row, split in zip(retained, assignments):
        row.update(split=split, label_idx=CLASS_NAMES.index(row['label']))
    fields = ['path', 'label', 'label_idx', 'split', 'sha256', 'pixel_sha256', 'cluster_id']
    split_hashes = {}
    for split in ('train', 'val', 'test'):
        target = workspace / 'splits' / f'{split}.csv'
        write_csv(target, [r for r in retained if r['split'] == split], fields)
        split_hashes[split] = sha256(target)
    for key in ('sha256', 'pixel_sha256', 'cluster_id'):
        owners = {}
        for row in retained:
            if owners.setdefault(row[key], row['split']) != row['split']:
                raise ValueError(f'Cross-split leakage: {key}')
    names = {str(i): name for i, name in enumerate(CLASS_NAMES)}
    dump_json(workspace / 'class_names.json', names)
    archive_hash = sha256(archive)
    manifest_hash = sha256(manifest)
    report = dict(dataset=DATASET_ID, revision=REVISION, sourceUrl=f'https://huggingface.co/datasets/{DATASET_ID}',
                  sourceArchiveSha256=archive_hash, sourceManifestSha256=manifest_hash,
                  officialArchiveVerified=archive_hash == ARCHIVE_SHA256 and manifest_hash == MANIFEST_SHA256,
                  classNames=names, classCount=6, selection=selected, retained=len(retained),
                  removedReasons=dict(Counter(r['reason'] for r in removed)),
                  perClassSplits={name: dict(Counter(r['split'] for r in retained if r['label'] == name))
                                  for name in CLASS_NAMES},
                  splitFileSha256s=split_hashes, classNamesSha256=sha256(workspace / 'class_names.json'),
                  seed=42, splitPolicy='70/15/15 stratified image-similarity clusters; all aug_* excluded first',
                  phashRadius=4, dhashRadius=6, detectedOverlap=0,
                  patientIndependence='Not established: no reliable patient identifiers', rights=RIGHTS_NOTE)
    write_csv(workspace / 'reports' / 'removed.csv', removed, ['path', 'label', 'reason', 'sha256', 'cluster_id'])
    dump_json(workspace / 'provenance.json', report)
    print(json.dumps(report, indent=2), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download', action='store_true')
    parser.add_argument('--cache', type=Path, default=Path('.skin-six-cache'))
    parser.add_argument('--archive', type=Path)
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--workspace', type=Path, default=Path('workspaces/skin-six'))
    args = parser.parse_args()
    if args.workspace.exists():
        parser.error('Workspace exists. Keep frozen splits and choose a NEW workspace for a new protocol.')
    if args.download:
        archive, manifest = download(args.cache)
    elif args.archive and args.manifest:
        archive, manifest = args.archive, args.manifest
        if sha256(archive) != ARCHIVE_SHA256:
            parser.error('Local archive does not match the pinned release')
        if sha256(manifest) != MANIFEST_SHA256:
            parser.error('Local manifest does not match the pinned release')
    else:
        parser.error('Use --download or supply both --archive and --manifest')
    prepare(archive, manifest, args.workspace)


if __name__ == '__main__':
    main()
