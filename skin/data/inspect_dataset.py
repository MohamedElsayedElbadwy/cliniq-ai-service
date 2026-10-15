"""Read-only image audit, duplicate clustering and deterministic group splits.

Run from repo root: python -m skin.data.inspect_dataset --data ../SkinDisease/SkinDisease
No source files are modified. Hashes cover encoded bytes AND decoded RGB pixels.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


class BKTree:
    """Hamming distance index for perceptual hash candidates."""
    def __init__(self):
        self.root = None

    def add(self, value, index):
        node = (value, [index], {})
        if self.root is None:
            self.root = node
            return
        cur = self.root
        while True:
            distance = (value ^ cur[0]).bit_count()
            if distance == 0:
                cur[1].append(index)
                return
            if distance not in cur[2]:
                cur[2][distance] = node
                return
            cur = cur[2][distance]

    def query(self, value, radius):
        stack = [self.root] if self.root else []
        while stack:
            key, indexes, children = stack.pop()
            distance = (value ^ key).bit_count()
            if distance <= radius:
                yield from indexes
            stack.extend(child for d, child in children.items() if distance-radius <= d <= distance+radius)


def find_clusters(rows, phash_radius=4, dhash_radius=6):
    parent = list(range(len(rows)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a, b):
        a, b = root(a), root(b)
        if a != b:
            parent[max(a,b)] = min(a,b)

    tree = BKTree()
    exact = {}
    for i, row in enumerate(rows):
        for kind in ['sha256', 'pixel_sha256']:
            key = (kind, row[kind])
            if key in exact:
                union(i, exact[key])
            exact[key] = i
        for j in tree.query(row['phash'], phash_radius):
            if (row['dhash'] ^ rows[j]['dhash']).bit_count() <= dhash_radius:
                union(i, j)
        tree.add(row['phash'], i)
    return [f'cluster-{root(i):06d}' for i in range(len(rows))]


def assign_splits(rows, seed=42):
    """Largest-first stratified allocation of indivisible same-class clusters."""
    grouped = defaultdict(list)
    for i, row in enumerate(rows):
        grouped[(row['label'], row['cluster_id'])].append(i)
    by_class = defaultdict(list)
    for (label, _), indexes in grouped.items():
        by_class[label].append(indexes)
    result = [None]*len(rows)
    rng = random.Random(seed)
    for label in sorted(by_class):
        groups = by_class[label]
        if len(groups) < 3:
            raise ValueError(f'{label}: fewer than 3 independent groups; cannot produce honest 3-way split')
        rng.shuffle(groups)
        groups.sort(key=len, reverse=True)
        n = sum(map(len, groups))
        targets = np.array([.7, .15, .15])*n
        counts = np.zeros(3)
        for indexes in groups:
            # Choose smallest squared-error allocation relative to target counts.
            scores = []
            for k in range(3):
                proposed = counts.copy()
                proposed[k] += len(indexes)
                scores.append(float(np.sum((proposed-targets)**2)))
            selected = int(np.argmin(scores))
            counts[selected] += len(indexes)
            for i in indexes:
                result[i] = ['train','val','test'][selected]
        if any(c == 0 for c in counts):
            raise ValueError(f'{label}: empty split; review group imbalance')
    return result


_DCT = np.cos(np.pi/32*(np.arange(32)+.5)[None,:]*np.arange(8)[:,None])


def bits_to_int(bits):
    return int.from_bytes(np.packbits(bits.flatten()).tobytes(), 'big')


def inspect_image(path):
    row = dict(path=str(path.resolve()), label=path.parent.name, original_split=path.parent.parent.name,
               size_bytes=path.stat().st_size)
    try:
        data = path.read_bytes()
        row['sha256'] = hashlib.sha256(data).hexdigest()
        with Image.open(path) as im:
            im.verify()
        with Image.open(path) as im:
            row.update(format=im.format, mode=im.mode, width=im.width, height=im.height,
                       has_exif=bool(im.info.get('exif')))
            rgb = ImageOps.exif_transpose(im).convert('RGB')
            rgb.load()
            row['pixel_sha256'] = hashlib.sha256(str(rgb.size).encode()+rgb.tobytes()).hexdigest()
            gray = rgb.convert('L')
            small = np.asarray(gray.resize((32,32), Image.Resampling.LANCZOS),dtype=float)
            coeff = _DCT @ small @ _DCT.T
            row['phash'] = bits_to_int(coeff > np.median(coeff.flatten()[1:]))
            diff = np.asarray(gray.resize((9,8), Image.Resampling.LANCZOS))
            row['dhash'] = bits_to_int(diff[:,1:] > diff[:,:-1])
            row['brightness'] = round(float(small.mean()),2)
            row['contrast'] = round(float(small.std()),2)
            row['error'] = None
    except Exception as error:
        row['error'] = type(error).__name__
    return row


def contact_sheets(rows, output):
    output.mkdir(parents=True, exist_ok=True)
    rng = random.Random(42)
    for label in sorted({r['label'] for r in rows}):
        samples = sorted((r for r in rows if r['label']==label), key=lambda r:r['path'])
        samples = rng.sample(samples, min(20,len(samples)))
        sheet = Image.new('RGB',(5*190,4*166),'white')
        draw = ImageDraw.Draw(sheet)
        for i, row in enumerate(samples):
            with Image.open(row['path']) as im:
                tile = ImageOps.contain(ImageOps.exif_transpose(im).convert('RGB'),(186,140))
            x,y = (i%5)*190,(i//5)*166
            sheet.paste(tile,(x,y))
            draw.text((x,y+142),f"{i}: {Path(row['path']).name[:25]}", fill='black')
        sheet.save(output/f'{label}.jpg',quality=85)
        write_csv(output/f'{label}.csv',samples,['path','label','sha256'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output',type=Path,default=Path('.'))
    parser.add_argument('--license-note',default='Kaggle metadata: CC0: Public Domain (2026-10-13); original image-source rights not verified.')
    args = parser.parse_args()
    out = args.output
    reports = out/'reports'
    reports.mkdir(parents=True,exist_ok=True)
    files = sorted(p for p in args.data.rglob('*') if p.is_file())
    with ThreadPoolExecutor(max_workers=6) as pool:
        raw = []
        for i,row in enumerate(pool.map(inspect_image,files),1):
            raw.append(row)
            if i%1000==0:
                print(f'Decoded {i}/{len(files)}',flush=True)
    valid = [r for r in raw if r['error'] is None]
    classes = sorted({r['label'] for r in valid})
    print(f'Clustering {len(valid)} valid images',flush=True)
    clusters = find_clusters(valid)
    groups = defaultdict(list)
    for row,cluster in zip(valid,clusters):
        row['cluster_id'] = cluster
        groups[cluster].append(row)
    conflicting = {key for key,rows in groups.items() if len({r['label'] for r in rows})>1}
    removed = [dict(r,reason='unreadable:'+r['error']) for r in raw if r['error']]
    cleaned = []
    seen_bytes, seen_pixels = set(),set()
    for row in valid:
        if row['cluster_id'] in conflicting:
            removed.append(dict(row,reason='cross_class_duplicate_cluster'))
        elif row['sha256'] in seen_bytes or row['pixel_sha256'] in seen_pixels:
            removed.append(dict(row,reason='exact_duplicate'))
        else:
            cleaned.append(row)
            seen_bytes.add(row['sha256'])
            seen_pixels.add(row['pixel_sha256'])
    names = {str(i):label for i,label in enumerate(classes)}
    split_names = assign_splits(cleaned)
    for row,split in zip(cleaned,split_names):
        row.update(split=split,label_idx=classes.index(row['label']))
    fields = ['path','label','label_idx','split','sha256','pixel_sha256','cluster_id']
    for split in ['train','val','test']:
        write_csv(out/'splits'/f'{split}.csv',[r for r in cleaned if r['split']==split],fields)
    # Assert disjoint bytes, decoded content AND transitive near-duplicate clusters.
    for key in ['sha256','pixel_sha256','cluster_id']:
        for a,b in [('train','val'),('train','test'),('val','test')]:
            assert not ({r[key] for r in cleaned if r['split']==a} & {r[key] for r in cleaned if r['split']==b})
    assert len(cleaned)+len(removed)==len(raw)
    write_csv(reports/'removed_files.csv', removed,['path','label','reason','sha256','cluster_id'])
    audit_fields = list(dict.fromkeys(k for row in raw for k in row))
    write_csv(reports/'image_inventory.csv',raw,audit_fields)
    (out/'skin'/'artifacts').mkdir(parents=True,exist_ok=True)
    (out/'skin'/'artifacts'/'class_names.json').write_text(json.dumps(names,indent=2),encoding='utf-8')
    counts = {label:{s:sum(r['label']==label and r['split']==s for r in cleaned) for s in ['train','val','test']} for label in classes}
    original_overlap = sum(len({r['original_split'] for r in members})>1 for members in groups.values())
    summary = dict(raw_count=len(raw),valid_count=len(valid),cleaned_count=len(cleaned),
        class_count=len(classes),removed_count=len(removed),removal_reasons=dict(Counter(r['reason'] for r in removed)),
        original_counts={s:dict(Counter(r['label'] for r in raw if r['original_split']==s)) for s in ['train','test']},
        original_split_overlapping_clusters=original_overlap,conflict_clusters=len(conflicting),
        formats=dict(Counter(r['format'] for r in valid)),modes=dict(Counter(r['mode'] for r in valid)),
        dimension_quantiles={key:np.quantile([r[key] for r in valid],[0,.05,.5,.95,1]).tolist() for key in ['width','height','size_bytes']},
        tiny_images=sum(min(r['width'],r['height'])<64 for r in valid),
        extreme_aspect=sum(max(r['width']/r['height'],r['height']/r['width'])>4 for r in valid),
        per_class_splits=counts,hash_overlap=0,pixel_hash_overlap=0,cluster_overlap=0,seed=42,
        phash_radius=4,dhash_radius=6,license=args.license_note,
        patient_ids='No reliable patient identifiers available; grouping only protects detected image similarity.',
        split_policy='Replace original train/test (no validation) with deterministic 70/15/15 grouped split; test frozen now.')
    (reports/'dataset_inspection.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    text = '# Dataset inspection\n\n'+json.dumps(summary,indent=2)+'\n\nThreshold: pHash Hamming <=4 AND dHash <=6; conservative detection, not proof of patient independence. Cross-class transitive clusters are excluded in full. Raw files unchanged. Contact sheets: reports/contact_sheets (20 seeded images/class). Manual review pending; no junk removed without evidence. Absolute file paths in local manifests are private local artifacts, excluded from Git.\n'
    (reports/'dataset_inspection.md').write_text(text,encoding='utf-8')
    print('Building 22 contact sheets',flush=True)
    contact_sheets(cleaned,reports/'contact_sheets')
    print(json.dumps(summary,indent=2),flush=True)


if __name__ == '__main__':
    main()
