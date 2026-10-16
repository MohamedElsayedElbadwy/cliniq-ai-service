"""Audited SCIN additions with case-grouped splits and protected v1 holdouts."""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import shutil
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath
from urllib.error import HTTPError

from skin.config import CLASS_NAMES
from skin.data.inspect_dataset import BKTree, find_clusters, inspect_image, write_csv
from skin.data.prepare_six import dump_json, sha256
from skin.training.six_class import load_protocol
from skin.data.atlas_v2 import verified_atlas_images, METADATA_SHA256 as ATLAS_METADATA_SHA256

SCIN_BASE='https://storage.googleapis.com/dx-scin-public-data/'
METADATA={'scin_cases.csv':'7923ee9ba9775af413ca115c4587dc1bd32259d7936aa97fa80d88dd616ca3c3',
          'scin_labels.csv':'616ad03c24c304f5f7ef01b4116e4d3274f1c71c9b19a0e3d5620698b081a9fe'}
LABELS={'Acne':'Acne','Eczema':'Eczema','Psoriasis':'Psoriasis','Rosacea':'Rosacea','Tinea':'Ringworm'}
FROZEN=Path(__file__).with_name('frozen_scin_cases.json')


def choose_label(weighted,min_weight=.4):
    if not isinstance(weighted,dict) or not weighted:return None
    if any(not isinstance(v,(int,float)) or not math.isfinite(v) or not 0<=v<=1 for v in weighted.values()):return None
    ranked=sorted(weighted.items(),key=lambda item:-item[1])
    name,weight=ranked[0]
    if name not in LABELS or weight<min_weight:return None
    if len(ranked)>1 and abs(weight-ranked[1][1])<1e-12:return None
    return LABELS[name],float(weight)


def image_paths(case):
    images=[]
    for i in range(1,4):
        value=case.get(f'image_{i}_path')
        if not value:continue
        path=PurePosixPath(value)
        if (path.parts[:2]!=('dataset','images') or len(path.parts)!=3 or '\\' in value
            or ':' in value or '..' in path.parts or path.suffix.lower() not in {'.png','.jpg','.jpeg'}):
            raise ValueError('Unsafe SCIN image path')
        images.append({'source_path':value,'shot_type':case.get(f'image_{i}_shot_type','')})
    return images


def select_cases(cases,labels,excluded_ids):
    selected=[]
    for row in labels:
        case_id=row['case_id']
        if hashlib.sha256(case_id.encode()).hexdigest() in excluded_ids:continue
        chosen=choose_label(ast.literal_eval(row.get('weighted_skin_condition_label') or '{}'))
        if chosen is None or case_id not in cases:continue
        images=image_paths(cases[case_id])
        if images:selected.append(dict(case_id=case_id,label=chosen[0],weight=chosen[1],images=images))
    return selected


def split_cases(records):
    """Assign duplicate-linked cases atomically; uncertain groups are train-only."""
    groups=defaultdict(list)
    for row in records:groups[(row['label'],row['group_id'])].append(row)
    assignment={}
    for label in sorted({r['label'] for r in records}):
        eligible=[(key,members) for key,members in groups.items() if key[0]==label and min(r['weight'] for r in members)>=.5]
        eligible.sort(key=lambda pair:hashlib.sha256(('skin-v2-42/'+str(pair[0])).encode()).hexdigest())
        if len(eligible)<3:raise ValueError(f'{label}: too few independent eligible groups')
        n=max(1,round(.15*len(eligible)))
        if len(eligible)>=6:n=max(2,n)
        n=min(n,(len(eligible)-1)//2)
        for i,(_,members) in enumerate(eligible):
            split='test' if i<n else ('val' if i<2*n else 'train')
            assignment.update({r['case_id']:split for r in members})
    for row in records:assignment.setdefault(row['case_id'],'train')
    return assignment


def protected_case_ids(candidates,references):
    """Close duplicate AND whole-case links before removing protected images."""
    clusters=find_clusters(references+candidates)
    parent={cluster:cluster for cluster in clusters}
    def root(key):
        while parent[key]!=key:parent[key]=parent[parent[key]];key=parent[key]
        return key
    seen={}
    for row,cluster in zip(candidates,clusters[len(references):]):
        case=row['case_id']
        if case in seen:
            a,b=root(cluster),root(seen[case]);parent[max(a,b)]=min(a,b)
        seen[case]=cluster
    protected={root(cluster) for cluster in clusters[:len(references)]}
    return {row['case_id'] for row,cluster in zip(candidates,clusters[len(references):]) if root(cluster) in protected}


def _fetch(url,target,expected=None):
    if not target.exists():
        target.parent.mkdir(parents=True,exist_ok=True)
        pending=target.with_suffix(target.suffix+'.partial')
        with urllib.request.urlopen(url,timeout=90) as response,pending.open('wb') as stream:
            shutil.copyfileobj(response,stream)
        pending.replace(target)
    if expected and sha256(target)!=expected:raise ValueError('Source metadata checksum changed')
    return target


def _case_fingerprints(case,cache):
    images=[]
    for image in case['images']:
        target=cache/'pictures'/PurePosixPath(image['source_path']).name
        try:_fetch(SCIN_BASE+image['source_path'],target)
        except HTTPError as error:
            if error.code==404:continue
            raise
        if target.stat().st_size>15*1024*1024:continue
        info=inspect_image(target)
        if info['error'] or info['format'] not in {'JPEG','PNG'}:continue
        if info['width']*info['height']>120_000_000:continue
        if case['label']!='protected' and (min(info['width'],info['height'])<64 or info['contrast']<3):continue
        images.append(dict(info,case_id=case['case_id'],label=case['label'],weight=case['weight'],
            source_path=image['source_path'],shot_type=image['shot_type'],source='scin'))
    return images


def prepare(args):
    workspace=args.workspace.resolve()
    if workspace.exists():raise FileExistsError('Use a new workspace; frozen protocols are immutable')
    original=load_protocol(args.original)
    cache=args.cache.resolve();cache.mkdir(parents=True,exist_ok=True)
    metadata={name:_fetch(SCIN_BASE+'dataset/'+name,cache/name,digest) for name,digest in METADATA.items()}
    cases={r['case_id']:r for r in csv.DictReader(metadata['scin_cases.csv'].open(encoding='utf-8'))}
    labels=list(csv.DictReader(metadata['scin_labels.csv'].open(encoding='utf-8')))
    frozen=json.loads(FROZEN.read_text());excluded_ids=set(frozen['caseIdSha256s'])
    selected=select_cases(cases,labels,excluded_ids)
    print('Eligible unused SCIN cases:',dict(Counter(r['label'] for r in selected)),flush=True)
    protected=[dict(case_id=k,label='protected',weight=1.,images=image_paths(v)) for k,v in cases.items()
        if hashlib.sha256(k.encode()).hexdigest() in excluded_ids]
    if len(protected)!=len(excluded_ids):raise ValueError('Frozen SCIN case set no longer matches source')
    # Reuse cached original fingerprints only after checking every encoded identity.
    reference_cache=cache/'original_fingerprints.json'
    old_rows=[r for split in original['rows'].values() for r in split]
    if reference_cache.exists():
        references=json.loads(reference_cache.read_text())
        if {r['sha256'] for r in references}!={r['sha256'] for r in old_rows}:raise ValueError('Reference cache differs')
    else:
        with ThreadPoolExecutor(max_workers=4) as pool:
            references=list(pool.map(lambda r:inspect_image(original['workspace']/r['path']),old_rows))
        if any(r['error'] for r in references):raise ValueError('Original source image became unreadable')
        dump_json(reference_cache,references)
    with ThreadPoolExecutor(max_workers=4) as pool:
        for images in pool.map(lambda r:_case_fingerprints(r,cache),protected):references.extend(images)
    tree=BKTree();encoded=set();pixels=set()
    for i,row in enumerate(references):
        tree.add(row['phash'],i);encoded.add(row['sha256']);pixels.add(row['pixel_sha256'])
    candidates=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i,images in enumerate(pool.map(lambda r:_case_fingerprints(r,cache),selected),1):
            candidates.extend(images)
            if i%100==0:print('Audited new SCIN cases:',i,flush=True)
    atlas=[];atlas_stats={}
    if getattr(args,'atlas_zip',None):
        if not args.atlas_metadata or sha256(args.atlas_metadata)!=ATLAS_METADATA_SHA256:raise ValueError('Atlas author metadata changed')
        atlas,atlas_stats=verified_atlas_images(args.atlas_zip,list(csv.DictReader(args.atlas_metadata.open(encoding='utf-8'))),cache)
    blocked=protected_case_ids(candidates+atlas,references)
    rejected={key:'protected_duplicate_component' for key in blocked}
    by_case=defaultdict(list)
    for row in candidates:
        by_case[row['case_id']].append(row)
        overlap=(row['sha256'] in encoded or row['pixel_sha256'] in pixels or
            any((row['dhash']^references[j]['dhash']).bit_count()<=6 for j in tree.query(row['phash'],4)))
        if overlap:rejected[row['case_id']]='overlap_with_original_or_frozen_audit'
    all_scin_candidates=list(candidates)
    candidates=[r for r in candidates if r['case_id'] not in rejected]
    clusters=find_clusters(candidates)
    parent={r['case_id']:r['case_id'] for r in candidates}
    def owner(key):
        while parent[key]!=key:parent[key]=parent[parent[key]];key=parent[key]
        return key
    seen={}
    for row,cluster in zip(candidates,clusters):
        key=row['case_id']
        if cluster in seen:
            a,b=owner(key),owner(seen[cluster]);parent[max(a,b)]=min(a,b)
        seen[cluster]=key
    group_labels=defaultdict(set)
    for row in candidates:group_labels[owner(row['case_id'])].add(row['label'])
    for row in candidates:
        if len(group_labels[owner(row['case_id'])])>1:rejected[row['case_id']]='conflicting_duplicate_group'
    candidates=[r for r in candidates if r['case_id'] not in rejected]
    records=[dict(r,group_id=owner(r['case_id'])) for r in selected if r['case_id'] in parent and r['case_id'] not in rejected]
    assignments=split_cases(records)
    # Pick one fixed image for each evaluation case; all its other photos stay out of training.
    candidate_cases=defaultdict(list)
    for row in candidates:candidate_cases[row['case_id']].append(row)
    retained=[]
    for key,images in candidate_cases.items():
        split=assignments[key]
        images.sort(key=lambda r:(r['shot_type']!='CLOSE_UP',r['source_path']))
        images=images if split=='train' else images[:1]
        unique={r['sha256']:r for r in images}
        for row in unique.values():retained.append(dict(row,split=split,cluster_id='scin-'+owner(key),quality=row['weight']/len(unique)))
    if getattr(args,'atlas_zip',None):
        # Exclude atlas overlaps with every original, protected and new SCIN image,
        # including SCIN images not chosen for validation/test representation.
        protected_images=references+all_scin_candidates
        atlas_blocked=protected_case_ids(atlas,protected_images)|blocked
        clean=[]
        for row in atlas:
            overlap=row['case_id'] in atlas_blocked
            if overlap:atlas_stats['protected_overlap']=atlas_stats.get('protected_overlap',0)+1
            else:clean.append(row)
        atlas_clusters=find_clusters(clean);atlas_labels=defaultdict(set)
        for row,cluster in zip(clean,atlas_clusters):atlas_labels[cluster].add(row['label'])
        clean=[dict(row,group_id=cluster,cluster_id='atlas-'+cluster) for row,cluster in zip(clean,atlas_clusters) if len(atlas_labels[cluster])==1]
        atlas_assignment=split_cases(clean)
        retained.extend(dict(row,split=atlas_assignment[row['case_id']]) for row in clean)
        atlas_stats['retained']=len(clean)
        print('Verified atlas audit:',atlas_stats,flush=True)
    workspace.mkdir(parents=True)
    output=[];technical_exclusions=0
    for split in ['train','val']:
        for row in original['rows'][split]:
            source=original['workspace']/row['path']
            # Cached identities also carry the source quality audit.
            # Lookup is prepared once below, so no repeated full image decoding is needed.
            output.append(dict(row,source='original',case_id='original-'+row['cluster_id'],quality=1.,_source=str(source)))
    original_info={r['sha256']:r for r in references if r.get('source')!='scin'}
    filtered=[]
    for row in output:
        info=original_info[row['sha256']]
        if min(info['width'],info['height'])<64 or info['contrast']<3:
            technical_exclusions+=1;continue
        filtered.append(row)
    output=filtered+retained
    fields=['path','label','label_idx','split','sha256','pixel_sha256','cluster_id','source','case_id','quality']
    for row in output:
        source=Path(row.pop('_source',row.get('path','')))
        extension='.jpg' if row.get('format',original_info.get(row['sha256'],{}).get('format'))=='JPEG' else '.png'
        rel=Path('images')/row['source']/row['label']/(row['sha256']+extension)
        target=workspace/rel;target.parent.mkdir(parents=True,exist_ok=True)
        if sha256(source)!=row['sha256']:raise ValueError('Image bytes changed before copy')
        if not target.exists():shutil.copyfile(source,target)
        row.update(path=rel.as_posix(),label_idx=CLASS_NAMES.index(row['label']))
    hashes={}
    for split in ['train','val','test']:
        rows=[r for r in output if r['split']==split]
        expected=set(CLASS_NAMES) if split!='test' else set(CLASS_NAMES[:-1])
        if {r['label'] for r in rows}!=expected:raise ValueError(f'{split}: missing required label support')
        target=workspace/'splits'/f'{split}.csv';write_csv(target,rows,fields);hashes[split]=sha256(target)
    names={str(i):name for i,name in enumerate(CLASS_NAMES)}
    dump_json(workspace/'class_names.json',names)
    provenance=dict(protocol='skin-v2-scin-case-groups',seed=42,classNames=names,splitFileSha256s=hashes,
        classNamesSha256=sha256(workspace/'class_names.json'),metadataSha256s=METADATA,
        originalProvenanceSha256=sha256(args.original/'provenance.json'),frozenAuditCases=len(excluded_ids),
        originalAndFrozenImagesCompared=len(references),rejectedCaseReasons=dict(Counter(rejected.values())),
        technicalOriginalExclusions=technical_exclusions,counts={split:dict(Counter(r['label'] for r in output if r['split']==split)) for split in hashes},
        sourceCounts={split:dict(Counter(r['source'] for r in output if r['split']==split)) for split in hashes},
        originalDataRights=original['provenance']['rights'],scinLicense='https://github.com/google-research-datasets/scin/blob/main/LICENSE',
        normalClinicallyVerified=False,patientIndependence=False,
        labels='SCIN unique top differential >=0.4 for training, >=0.5 validation/test; Tinea -> Ringworm; original labels inherited',
        grouping='All case photos and exact/near duplicate-linked cases stay in one partition; frozen audit cases wholly excluded',
        testPredictionsRead=False)
    if atlas_stats:
        provenance['atlas']=dict(stats=atlas_stats,metadataSha256=ATLAS_METADATA_SHA256,
            source='https://github.com/mattgroh/fitzpatrick17k',distribution='asosenge/fitzpatrick17k',
            license='CC-BY-NC-SA-3.0; educational noncommercial research only',
            identity='Each admitted encoded image matches original-author MD5; QC/path labels filtered',
            labels='Acne/acne vulgaris, eczema, psoriasis, rosacea; no healthy or Ringworm labels')
    dump_json(workspace/'provenance.json',provenance)
    dump_json(workspace/'case_exclusions.json',rejected)
    print(json.dumps(provenance,indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original',type=Path,required=True)
    parser.add_argument('--workspace',type=Path,required=True)
    parser.add_argument('--cache',type=Path,default=Path('workspaces/skin-v2-cache'))
    parser.add_argument('--atlas-zip',type=Path)
    parser.add_argument('--atlas-metadata',type=Path)
    prepare(parser.parse_args())


if __name__=='__main__':main()
