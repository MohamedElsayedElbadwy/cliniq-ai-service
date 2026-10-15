"""Package ONLY six-class source for Colab/Kaggle. No images, secrets or weights.

python scripts/package_skin_six.py
Upload skin-six-source.zip next to the training notebook in Colab or Kaggle.
"""
from pathlib import Path
import zipfile


ROOT = Path(__file__).resolve().parents[1]
FILES = (
    'skin/__init__.py', 'skin/config.py', 'skin/preprocessing.py', 'skin/service.py', 'skin/schemas.py',
    'skin/data/__init__.py', 'skin/data/inspect_dataset.py', 'skin/data/prepare_six.py',
    'skin/training/__init__.py', 'skin/training/metrics.py', 'skin/training/six_class.py',
    'skin/training/requirements-cloud.txt', 'docs/skin/SIX_CLASS_TRAINING.md',
)


def main():
    destination = ROOT / 'skin-six-source.zip'
    with zipfile.ZipFile(destination, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for name in FILES:
            archive.write(ROOT / name, name)
    print(f'{destination} ({destination.stat().st_size:,} bytes; source only)')


if __name__ == '__main__':
    main()
