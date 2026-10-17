"""Verify clinical-atlas image identities against the original authors' metadata."""
import hashlib
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath

from skin.data.inspect_dataset import inspect_image

METADATA_SHA256='851dfaabf1791c2a8b1c2957619f3e0673cf0002f4ce924a5c02d185c7d4a1cd'
MAPPING={'acne':'Acne','acne vulgaris':'Acne','eczema':'Eczema','psoriasis':'Psoriasis','rosacea':'Rosacea'}


def verified_atlas_images(archive,records,cache):
    eligible={r['md5hash']:r for r in records if r['label'] in MAPPING and r.get('qc','') in {'','1 Diagnostic','2 Characteristic'}}
    rows=[];counts=Counter();seen=set()
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            identity=PurePosixPath(member.filename).stem
            if identity not in eligible or identity in seen:continue
            record=eligible[identity];label=MAPPING[record['label']]
            if member.file_size>15*1024*1024:counts['oversized']+=1;continue
            markers={'Acne':['acne'],'Eczema':['eczema','eczematous'],'Psoriasis':['psoria'],'Rosacea':['rosacea']}[label]
            url=record['url'].lower()
            if 'dermaamin.com' in url and not any(marker in url for marker in markers):
                counts['path_label_mismatch']+=1;continue
            payload=bundle.read(member)
            if hashlib.md5(payload).hexdigest()!=identity:
                counts['author_identity_mismatch']+=1;continue
            seen.add(identity)
            target=Path(cache)/'atlas'/label/(identity+'.jpg');target.parent.mkdir(parents=True,exist_ok=True)
            if not target.exists() or hashlib.md5(target.read_bytes()).hexdigest()!=identity:target.write_bytes(payload)
            audit=inspect_image(target)
            if audit['error'] or audit['format'] not in {'JPEG','PNG'} or min(audit['width'],audit['height'])<64 or audit['contrast']<3:
                counts['technical_quality']+=1;continue
            rows.append(dict(audit,label=label,case_id='atlas-'+identity,weight=1.,quality=1.,source='atlas',source_md5=identity))
            counts['verified']+=1
    return rows,dict(counts)
