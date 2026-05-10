"""
prepare_sahi_dataset.py
=======================
Slices the SeaDronesSee OD dataset into 640x640 patches using SAHI,
drops empty ocean patches, and converts the output to YOLO .txt labels.

Directory layout expected BEFORE running this script:
    yolov7-main/
    └── data/
        ├── annotations/
        │   ├── instances_train.json
        │   └── instances_val.json
        ├── train/
        │   └── images/      ← place your train images here first
        ├── val/
        │   └── images/      ← place your val images here first
        └── test/
            └── images/      ← place your test images here first

Directory layout AFTER running this script:
    yolov7-main/
    └── data/
        └── sahi/
            ├── train/
            │   ├── images/  ← 640x640 patches (object patches only)
            │   └── labels/  ← matching YOLO .txt files
            └── val/
                ├── images/
                └── labels/

Usage:
    python prepare_sahi_dataset.py
    python prepare_sahi_dataset.py --sample 0.25   # use 25% of patches for a fast test run
    python prepare_sahi_dataset.py --slice 1280    # use 1280x1280 patches instead
    python prepare_sahi_dataset.py --overlap 0.2   # increase overlap to 20%

Install requirement first:
    pip install sahi
"""

import os
import json
import random
import argparse
import shutil
from pathlib import Path

# ---------------------------------------------------------------------------
# SeaDronesSee class mapping
# COCO JSON uses category_id starting at 0 for this dataset.
# Verify with: check first annotation in instances_train.json
# If you see category_ids like {1,2,3,4,5,6} (not starting at 0),
# set CATEGORY_ID_OFFSET = -1 below.
# ---------------------------------------------------------------------------
CATEGORY_ID_OFFSET = 0   # change to -1 if class IDs in JSON start at 1

# ---------------------------------------------------------------------------
# Default SAHI parameters (change via CLI args or edit here)
# ---------------------------------------------------------------------------
DEFAULT_SLICE_SIZE = 640
DEFAULT_OVERLAP    = 0.1      # 10% overlap between adjacent patches
DEFAULT_MIN_AREA   = 0.1      # drop boxes that are more than 90% cut off
DEFAULT_SAMPLE     = 1.0      # keep 100% of patches (use 0.25 for quick tests)


def convert_sahi_coco_to_yolo(sahi_json_path: Path, output_labels_dir: Path):
    """
    Convert the COCO JSON that SAHI produces into YOLO .txt label files.
    One .txt file is created per image slice, named identically to the image.
    """
    output_labels_dir.mkdir(parents=True, exist_ok=True)

    with open(sahi_json_path, "r") as f:
        data = json.load(f)

    # Build image_id → image_info lookup for fast access
    images = {img["id"]: img for img in data["images"]}

    print(f"    Converting {len(data['annotations'])} annotations "
          f"across {len(data['images'])} slices to YOLO format...")

    for ann in data["annotations"]:
        img_info = images[ann["image_id"]]

        # COCO bbox: [x_min, y_min, width, height] — all in pixels
        x_min, y_min, w_box, h_box = ann["bbox"]
        w_img = img_info["width"]
        h_img = img_info["height"]

        # Convert to YOLO normalised centre format
        x_centre = (x_min + w_box / 2.0) / w_img
        y_centre  = (y_min + h_box / 2.0) / h_img
        w_norm    = w_box / w_img
        h_norm    = h_box / h_img

        # Guard: clamp to [0, 1] in case of floating-point edge drift
        x_centre = max(0.0, min(1.0, x_centre))
        y_centre  = max(0.0, min(1.0, y_centre))
        w_norm    = max(0.0, min(1.0, w_norm))
        h_norm    = max(0.0, min(1.0, h_norm))

        class_id = ann["category_id"] + CATEGORY_ID_OFFSET

        # Label file named exactly like the image slice (minus extension)
        stem     = Path(img_info["file_name"]).stem
        txt_path = output_labels_dir / f"{stem}.txt"

        with open(txt_path, "a") as f_out:
            f_out.write(
                f"{class_id} {x_centre:.6f} {y_centre:.6f} "
                f"{w_norm:.6f} {h_norm:.6f}\n"
            )


def apply_sampling(images_dir: Path, labels_dir: Path, sample_ratio: float, seed: int = 42):
    """
    Randomly keep only `sample_ratio` fraction of the patches.
    Deletes both the image and its matching label for dropped patches.
    """
    if sample_ratio >= 1.0:
        return  # nothing to do

    all_images = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
    random.seed(seed)
    n_keep    = max(1, int(len(all_images) * sample_ratio))
    keep_set  = set(random.sample(all_images, n_keep))
    n_dropped = 0

    for img_path in all_images:
        if img_path not in keep_set:
            img_path.unlink(missing_ok=True)
            txt_path = labels_dir / (img_path.stem + ".txt")
            txt_path.unlink(missing_ok=True)
            n_dropped += 1

    print(f"    Sampling: kept {n_keep}/{len(all_images)} patches "
          f"({sample_ratio*100:.0f}%), dropped {n_dropped}")


def prepare_split(
    split_name:   str,
    coco_json:    Path,
    images_dir:   Path,
    output_root:  Path,
    slice_size:   int,
    overlap:      float,
    min_area:     float,
    sample_ratio: float,
):
    from sahi.slicing import slice_coco

    out_images = output_root / split_name / "images"
    out_labels = output_root / split_name / "labels"
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Processing split: {split_name.upper()}")
    print(f"  Source images : {images_dir}")
    print(f"  COCO JSON     : {coco_json}")
    print(f"  Slice size    : {slice_size}x{slice_size}")
    print(f"  Overlap       : {overlap*100:.0f}%")
    print(f"  Min area ratio: {min_area}")
    print(f"  Sample ratio  : {sample_ratio*100:.0f}%")
    print(f"{'='*60}")

    if not images_dir.exists() or not any(images_dir.iterdir()):
        print(f"  ⚠️  WARNING: {images_dir} is empty or does not exist.")
        print(f"  ⚠️  Download and place your {split_name} images there first, then re-run.")
        return

    if not coco_json.exists():
        print(f"  ❌ ERROR: {coco_json} not found. Cannot process {split_name}.")
        return

    # --- Step 1: SAHI slicing ---
    print(f"\n  [1/3] Slicing images with SAHI...")
    _, sahi_json_path = slice_coco(
        coco_annotation_file_path=str(coco_json),
        image_dir=str(images_dir),
        output_coco_annotation_file_name=f"sliced_{split_name}",
        ignore_negative_samples=True,       # drop empty ocean patches — critical for speed
        output_dir=str(out_images),
        slice_height=slice_size,
        slice_width=slice_size,
        overlap_height_ratio=overlap,
        overlap_width_ratio=overlap,
        min_area_ratio=min_area,
        verbose=True,
    )

    # SAHI saves the JSON inside the images output dir
    sahi_json_path = out_images / f"sliced_{split_name}_coco.json"
    if not sahi_json_path.exists():
        # fallback: SAHI sometimes names it without the split suffix
        candidates = list(out_images.glob("*.json"))
        if candidates:
            sahi_json_path = candidates[0]
        else:
            print(f"  ❌ Could not find SAHI output JSON in {out_images}. Aborting.")
            return

    # --- Step 2: Convert to YOLO .txt ---
    print(f"\n  [2/3] Converting SAHI COCO JSON → YOLO .txt labels...")
    convert_sahi_coco_to_yolo(sahi_json_path, out_labels)

    # --- Step 3: Optional sampling ---
    print(f"\n  [3/3] Applying sampling ({sample_ratio*100:.0f}%)...")
    apply_sampling(out_images, out_labels, sample_ratio)

    # --- Summary ---
    n_images = len(list(out_images.glob("*.jpg"))) + len(list(out_images.glob("*.png")))
    n_labels = len(list(out_labels.glob("*.txt")))
    print(f"\n  ✅ {split_name.upper()} done.")
    print(f"     Images : {n_images}  →  {out_images}")
    print(f"     Labels : {n_labels}  →  {out_labels}")


def write_yaml(output_root: Path, base_dir: Path):
    """Write a ready-to-use SeaDronesSee_SAHI.yaml for YOLOv7 training."""
    yaml_path = base_dir / "data" / "SeaDronesSee_SAHI.yaml"
    sahi_train = (output_root / "train" / "images").resolve()
    sahi_val   = (output_root / "val"   / "images").resolve()
    test_dir   = (base_dir / "data" / "test" / "images").resolve()

    content = f"""\
# SeaDronesSee — SAHI pre-processed dataset config for YOLOv7
# Generated automatically by prepare_sahi_dataset.py

train: {sahi_train.as_posix()}
val:   {sahi_val.as_posix()}
test:  {test_dir.as_posix()}

nc: 7
names: ['ignored', 'swimmer', 'boat', 'jetski', 'life_saving_appliances', 'buoy']
"""
    with open(yaml_path, "w") as f:
        f.write(content)
    print(f"\n  📄 YAML written to: {yaml_path}")
    return yaml_path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SAHI pre-processing for SeaDronesSee OD dataset")
    parser.add_argument("--slice",   type=int,   default=DEFAULT_SLICE_SIZE,
                        help="Patch size in pixels (default: 640)")
    parser.add_argument("--overlap", type=float, default=DEFAULT_OVERLAP,
                        help="Overlap ratio between patches, 0–1 (default: 0.1)")
    parser.add_argument("--minarea", type=float, default=DEFAULT_MIN_AREA,
                        help="Min bbox area ratio to keep (default: 0.1)")
    parser.add_argument("--sample",  type=float, default=DEFAULT_SAMPLE,
                        help="Fraction of patches to keep, 0–1 (default: 1.0 = all)")
    parser.add_argument("--splits",  nargs="+",  default=["train", "val"],
                        choices=["train", "val", "test"],
                        help="Which splits to process (default: train val)")
    args = parser.parse_args()

    # Resolve paths relative to this script's location (yolov7-main/)
    BASE_DIR    = Path(__file__).parent.resolve()
    DATA_DIR    = BASE_DIR / "data"
    ANN_DIR     = DATA_DIR / "annotations"
    OUTPUT_ROOT = DATA_DIR / "sahi"

    print("\n" + "="*60)
    print("  SeaDronesSee — SAHI Dataset Preparation")
    print("="*60)
    print(f"  Base dir    : {BASE_DIR}")
    print(f"  Output root : {OUTPUT_ROOT}")

    for split in args.splits:
        prepare_split(
            split_name   = split,
            coco_json    = ANN_DIR / f"instances_{split}.json",
            images_dir   = DATA_DIR / split / "images",
            output_root  = OUTPUT_ROOT,
            slice_size   = args.slice,
            overlap      = args.overlap,
            min_area     = args.minarea,
            sample_ratio = args.sample,
        )

    # Write the YAML config for YOLOv7
    yaml_path = write_yaml(OUTPUT_ROOT, BASE_DIR)

    print("\n" + "="*60)
    print("  ALL DONE.")
    print("="*60)
    print("\nNext steps:")
    print("  1. Run the verifier:  python verify_sahi_output.py")
    print("  2. Train YOLOv7:")
    print(f"     python train.py --workers 4 --device 0 --batch-size 8 --epochs 50 \\")
    print(f"       --img {args.slice} {args.slice} --data {yaml_path} \\")
    print(f"       --hyp data/hyp.scratch.custom.yaml \\")
    print(f"       --cfg cfg/training/yolov7_SeaDronesSee.yaml \\")
    print(f"       --name yolov7-SAHI --weights yolov7_training.pt")
