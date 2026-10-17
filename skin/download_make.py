"""Acquire the selected public MAKE checkpoint without replacing other models."""
import argparse
from pathlib import Path
from .config import MAKE_DIR
from .download_pretrained import download
from .make_artifact import ARTIFACT_FILES,MODEL_ID,REVISION,verify_artifact


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=MAKE_DIR)
    download(parser.parse_args().output,artifact_files=ARTIFACT_FILES,model_id=MODEL_ID,revision=REVISION,verifier=verify_artifact)


if __name__=='__main__':main()
