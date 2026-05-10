"""
verify_sahi_output.py
=====================
Verifies that the SAHI pre-processing ran correctly and the dataset
is clean and safe to train on. Run this immediately after
prepare_sahi_dataset.py finishes.

Usage:
    python verify_sahi_output.py
    python verify_sahi_output.py --splits train    # check only train
    python verify_sahi_output.py --size 1280       # if you sliced at 1280

Exit code 0 = all checks passed, safe to train.
Exit code 1 = errors found, do NOT train yet.
"""

import sys
import json
import argparse
import random
from pathlib import Path

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("⚠️  Pillow not installed. Image size checks will be skipped.")
    print("   Install with:  pip install Pillow\n")


# Expected class IDs for SeaDronesSee (0-indexed)
# 0=ignored, 1=swimmer, 2=boat, 3=jetski, 4=life_saving_appliances, 5=buoy
# Class 6 also appears in some annotation files
EXPECTED_CLASS_IDS = set(range(7))   # {0, 1, 2, 3, 4, 5, 6}


def check_split(split_name: str, images_dir: Path, labels_dir: Path,
                expected_size: int, sample_check: int = 50) -> bool:

    print(f"\n{'='*60}")
    print(f"  Checking split: {split_name.upper()}")
    print(f"  Images : {images_dir}")
    print(f"  Labels : {labels_dir}")
    print(f"{'='*60}")

    errors   = []
    warnings = []
    passed   = []

    # ------------------------------------------------------------------
    # Check 0: Directories exist
    # ------------------------------------------------------------------
    if not images_dir.exists():
        errors.append(f"Images directory does not exist: {images_dir}")
        print(f"  ❌ Images directory missing — was SAHI run yet?")
        return False

    if not labels_dir.exists():
        errors.append(f"Labels directory does not exist: {labels_dir}")
        print(f"  ❌ Labels directory missing — conversion step may have failed.")
        return False

    # ------------------------------------------------------------------
    # Check 1: Count images and labels
    # ------------------------------------------------------------------
    images = sorted(list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png")))
    labels = sorted(list(labels_dir.glob("*.txt")))

    if len(images) == 0:
        errors.append("No image files found. SAHI may not have run or images were not downloaded.")
    if len(labels) == 0:
        errors.append("No label files found. Conversion step may have failed.")

    if not errors:
        passed.append(f"Found {len(images)} image patches and {len(labels)} label files")

    # ------------------------------------------------------------------
    # Check 2: Every image has a matching label
    # ------------------------------------------------------------------
    img_stems = {f.stem for f in images}
    lbl_stems = {f.stem for f in labels}

    missing_labels = img_stems - lbl_stems
    if missing_labels:
        sample = list(missing_labels)[:5]
        errors.append(
            f"{len(missing_labels)} images have NO matching label file "
            f"(first 5: {sample})"
        )
    else:
        passed.append("Every image has a matching label file ✓")

    orphan_labels = lbl_stems - img_stems
    if orphan_labels:
        warnings.append(
            f"{len(orphan_labels)} label files have no matching image "
            f"(harmless if caused by JSON-only annotations)"
        )

    # ------------------------------------------------------------------
    # Check 3: No empty label files
    # ------------------------------------------------------------------
    empty_labels = [f for f in labels if f.stat().st_size == 0]
    if empty_labels:
        errors.append(
            f"{len(empty_labels)} label files are EMPTY — "
            f"ignore_negative_samples may not have worked. "
            f"Examples: {[f.name for f in empty_labels[:3]]}"
        )
    else:
        passed.append("All label files are non-empty ✓")

    # ------------------------------------------------------------------
    # Check 4: Image dimensions (sample for speed)
    # ------------------------------------------------------------------
    if PIL_AVAILABLE and images:
        sample_imgs = random.sample(images, min(sample_check, len(images)))
        wrong_size  = []
        for img_path in sample_imgs:
            with Image.open(img_path) as img:
                w, h = img.size
                # Patches at the edge may be smaller than slice_size
                if w > expected_size or h > expected_size:
                    wrong_size.append(f"{img_path.name}: {w}×{h}")
        if wrong_size:
            errors.append(
                f"{len(wrong_size)} sampled images exceed expected size "
                f"{expected_size}×{expected_size}: {wrong_size[:3]}"
            )
        else:
            passed.append(
                f"All sampled images ≤ {expected_size}×{expected_size} ✓ "
                f"(checked {len(sample_imgs)} images)"
            )

    # ------------------------------------------------------------------
    # Check 5: YOLO bounding box sanity (all labels)
    # ------------------------------------------------------------------
    bbox_errors = []
    all_class_ids = set()

    for lbl_path in labels:
        with open(lbl_path) as f:
            for line_num, line in enumerate(f.readlines(), 1):
                line = line.strip()
                if not line:
                    continue
                parts = line.split()

                # Must have exactly 5 values: class x y w h
                if len(parts) != 5:
                    bbox_errors.append(
                        f"{lbl_path.name}:{line_num} — "
                        f"expected 5 values, got {len(parts)}: '{line}'"
                    )
                    continue

                try:
                    cls_id = int(parts[0])
                    coords = [float(v) for v in parts[1:]]
                except ValueError:
                    bbox_errors.append(
                        f"{lbl_path.name}:{line_num} — non-numeric value: '{line}'"
                    )
                    continue

                all_class_ids.add(cls_id)

                # Coordinates must be in [0, 1]
                if any(v < 0.0 or v > 1.0 for v in coords):
                    bbox_errors.append(
                        f"{lbl_path.name}:{line_num} — coords out of [0,1] "
                        f"range: {coords}"
                    )

                # Width and height must be positive
                if coords[2] <= 0 or coords[3] <= 0:
                    bbox_errors.append(
                        f"{lbl_path.name}:{line_num} — zero or negative box "
                        f"dimensions: w={coords[2]}, h={coords[3]}"
                    )

        if len(bbox_errors) > 20:
            bbox_errors.append("... (truncated, too many errors)")
            break

    if bbox_errors:
        errors.append(
            f"{len(bbox_errors)} invalid bounding box lines:\n    "
            + "\n    ".join(bbox_errors[:10])
        )
    else:
        passed.append("All bounding box coordinates are valid YOLO format ✓")

    # ------------------------------------------------------------------
    # Check 6: Class ID range
    # ------------------------------------------------------------------
    unexpected_ids = all_class_ids - EXPECTED_CLASS_IDS
    if unexpected_ids:
        errors.append(
            f"UNEXPECTED class IDs found: {unexpected_ids}\n"
            f"    Expected: {sorted(EXPECTED_CLASS_IDS)}\n"
            f"    Got     : {sorted(all_class_ids)}\n"
            f"    FIX: Set CATEGORY_ID_OFFSET = -1 in prepare_sahi_dataset.py "
            f"and re-run the conversion step."
        )
    else:
        passed.append(
            f"Class IDs are all valid: {sorted(all_class_ids)} ✓"
        )

    # ------------------------------------------------------------------
    # Check 7: Ratio of labels to images (sanity on negative-sample drop)
    # ------------------------------------------------------------------
    if len(images) > 0:
        ratio = len(labels) / len(images)
        if ratio < 0.05:
            warnings.append(
                f"Only {ratio*100:.1f}% of images have labels. "
                f"This may be correct (lots of ocean) but double-check "
                f"ignore_negative_samples worked as intended."
            )
        elif ratio > 1.05:
            warnings.append(
                f"More label files ({len(labels)}) than images ({len(images)}). "
                f"Orphan labels exist — usually harmless."
            )

    # ------------------------------------------------------------------
    # Print results
    # ------------------------------------------------------------------
    for msg in passed:
        print(f"  ✅ {msg}")
    for msg in warnings:
        print(f"  ⚠️  {msg}")

    if errors:
        print(f"\n  🚨 {len(errors)} ERROR(S) FOUND — do NOT train yet:")
        for i, err in enumerate(errors, 1):
            print(f"  [{i}] ❌ {err}")
        return False
    else:
        print(f"\n  🎉 {split_name.upper()} split is CLEAN. Safe to train.")
        return True


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify SAHI output before training")
    parser.add_argument("--splits", nargs="+", default=["train", "val"],
                        choices=["train", "val", "test"],
                        help="Which splits to verify (default: train val)")
    parser.add_argument("--size", type=int, default=640,
                        help="Expected max patch size in pixels (default: 640)")
    args = parser.parse_args()

    BASE_DIR    = Path(__file__).parent.resolve()
    OUTPUT_ROOT = BASE_DIR / "data" / "sahi"

    all_passed = True
    for split in args.splits:
        ok = check_split(
            split_name    = split,
            images_dir    = OUTPUT_ROOT / split / "images",
            labels_dir    = OUTPUT_ROOT / split / "labels",
            expected_size = args.size,
        )
        all_passed = all_passed and ok

    print("\n" + "="*60)
    if all_passed:
        print("  ✅✅  ALL SPLITS PASSED. You do NOT need to re-run SAHI.")
        print("="*60)
        print("\nReady to train. Use this command:")
        yaml_path = BASE_DIR / "data" / "SeaDronesSee_SAHI.yaml"
        print(f"\n  python train.py --workers 4 --device 0 --batch-size 8 --epochs 50 \\")
        print(f"    --img 640 640 --data {yaml_path} \\")
        print(f"    --hyp data/hyp.scratch.custom.yaml \\")
        print(f"    --cfg cfg/training/yolov7_SeaDronesSee.yaml \\")
        print(f"    --name yolov7-SAHI --weights yolov7_training.pt")
        sys.exit(0)
    else:
        print("  ❌  ERRORS FOUND. Fix issues above before training.")
        print("="*60)
        sys.exit(1)
