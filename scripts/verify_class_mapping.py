"""Verify training class metadata against a labeled MRI testing directory."""

from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mri.service import MRIPredictor, ModelUnavailableError


IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png"})


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for class-mapping verification."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path, help="Path to the saved .keras model.")
    parser.add_argument("--data", required=True, type=Path, help="Testing directory with one folder per true class.")
    parser.add_argument("--per-class", default=50, type=int, help="Maximum images to evaluate from each class folder.")
    return parser.parse_args()


def main() -> int:
    """Evaluate a saved model and print metadata consistency information."""
    args = parse_args()
    if args.per_class < 1:
        print("--per-class must be at least 1.", file=sys.stderr)
        return 2
    if not args.model.is_file():
        print(f"Model file not found: {args.model.name}", file=sys.stderr)
        return 2
    if not args.data.is_dir():
        print("Testing directory not found.", file=sys.stderr)
        return 2

    predictor = MRIPredictor(model_path=args.model)
    if not predictor.is_ready:
        print(f"Predictor unavailable: {predictor.unavailable_message}", file=sys.stderr)
        return 1

    inferred_counts: dict[int, Counter[str]] = defaultdict(Counter)
    correct = 0
    total = 0
    class_folders = sorted(path for path in args.data.iterdir() if path.is_dir())
    for class_folder in class_folders:
        image_paths = sorted(path for path in class_folder.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
        for image_path in image_paths[: args.per_class]:
            try:
                result = predictor.predict_file(image_path)
            except ModelUnavailableError as error:
                print(f"Predictor unavailable: {error}", file=sys.stderr)
                return 1
            inferred_counts[result["class_index"]][class_folder.name] += 1
            correct += predictor.class_names[result["class_index"]] == class_folder.name
            total += 1

    if total == 0:
        print("No JPG, JPEG, or PNG images found in class folders.", file=sys.stderr)
        return 2

    inferred_mapping = {
        index: counts.most_common(1)[0][0]
        for index, counts in sorted(inferred_counts.items())
        if counts
    }
    metadata_mapping = predictor.class_names

    print("Inferred index -> folder mapping:")
    for index in sorted(metadata_mapping):
        print(f"  {index} -> {inferred_mapping.get(index, '<no predictions>')}")
    print("class_names.json mapping:")
    for index, name in sorted(metadata_mapping.items()):
        print(f"  {index} -> {name}")
    print(f"Mappings identical: {inferred_mapping == metadata_mapping}")
    print(f"Accuracy: {correct / total:.2%} ({correct}/{total})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
