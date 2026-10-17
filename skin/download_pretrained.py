"""Download a pinned public safetensors artifact once; never overwrite existing files."""
from __future__ import annotations

import argparse
import os
import uuid
from pathlib import Path

import httpx

from .config import PRETRAINED_DIR
from .pretrained_artifact import ARTIFACT_FILES,MODEL_ID,REVISION,file_matches,verify_artifact


def download(directory:Path,*,artifact_files=None,model_id=MODEL_ID,revision=REVISION,verifier=verify_artifact):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    specs=ARTIFACT_FILES if artifact_files is None else artifact_files
    with httpx.Client(timeout=httpx.Timeout(120,connect=30),follow_redirects=True,trust_env=False) as client:
        for name,spec in specs.items():
            target=directory/name
            if target.exists():
                if not file_matches(target,spec):raise ValueError(f'Existing file differs; preserve it and choose a new directory: {name}')
                print('Verified cached',name,flush=True);continue
            pending=directory/(name+'.'+uuid.uuid4().hex+'.part')
            try:
                with client.stream('GET',f'https://huggingface.co/{model_id}/resolve/{revision}/{name}') as response:
                    response.raise_for_status();total=0;last=0
                    with pending.open('xb') as stream:
                        for chunk in response.iter_bytes(1024*1024):
                            total+=len(chunk)
                            if total>spec[0]:raise ValueError('Downloaded file exceeds pinned size')
                            stream.write(chunk)
                            if total-last>=32*1024*1024:
                                print(f'{name}: {total}/{spec[0]} bytes',flush=True);last=total
                if not file_matches(pending,spec):raise ValueError(f'Download checksum mismatch: {name}')
                # Windows rename refuses to replace a concurrently created target.
                if os.name=='nt':pending.rename(target)
                else:
                    os.link(pending,target);pending.unlink()
                print('Downloaded and verified',name,flush=True)
            finally:
                if pending.exists():pending.unlink()
    verifier(directory)
    print('Ready for offline CPU inference:',directory.resolve(),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=PRETRAINED_DIR)
    download(parser.parse_args().output)


if __name__=='__main__':main()
