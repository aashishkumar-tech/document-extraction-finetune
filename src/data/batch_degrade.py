"""Batch-process every downloaded (PDF, JSON) pair into 4 degradation tiers.

For each source document:
  - Loads the PDF (first page) + ground truth JSON
  - Runs degrade() at clean/light/medium/heavy
  - Saves each tier's image under data/degraded/<level>/
  - Saves ONE shared ground-truth JSON under data/degraded/labels/
    (same label reused across all 4 tiers of that document - this is what
    lets you compare accuracy by degradation tier on identical content)
  - Appends a row to data/degraded/manifest.csv describing every generated
    (image, label, template_type, level) sample - this manifest is what
    prepare_dataset.py will read to build the train/val/test split.

Usage:
    python src/data/batch_degrade.py
    python src/data/batch_degrade.py --limit 20   # process only first 20 per folder (quick test)
"""
import argparse
import csv
import json
import shutil
from pathlib import Path

from src.data.load_statement import load_statement
from src.data.degrade import degrade, LEVELS

RAW_ROOT = Path("data/raw/train")
OUT_ROOT = Path("data/degraded")

FOLDERS = [
    "India_Bank_Statement_Scanned_Type1",
    "India_Bank_Statement_Scanned_Type2",
    "India_Bank_Statement_Digital_Type1",
    "India_Bank_Statement_Digital_Type2",
]


def find_pairs(folder: Path):
    """Yield (idx_str, pdf_path, json_path) for every complete pair in a folder."""
    if not folder.exists():
        return
    for pdf_path in sorted(folder.glob("*.pdf")):
        idx = pdf_path.stem
        json_path = folder / f"{idx}.json"
        if json_path.exists():
            yield idx, pdf_path, json_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                         help="Max number of documents to process per folder (for quick testing)")
    args = parser.parse_args()

    for level in LEVELS:
        (OUT_ROOT / level).mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "labels").mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    total_docs = 0
    total_images = 0
    failures = []

    for folder_name in FOLDERS:
        folder_path = RAW_ROOT / folder_name
        pairs = list(find_pairs(folder_path))
        if args.limit:
            pairs = pairs[: args.limit]

        print(f"\n{folder_name}: {len(pairs)} document(s) found")

        for idx, pdf_path, json_path in pairs:
            doc_id = f"{folder_name}_{idx}"

            try:
                image, ground_truth = load_statement(str(pdf_path), str(json_path))
            except Exception as e:
                print(f"  [FAILED] {doc_id}: could not load ({e})")
                failures.append((doc_id, str(e)))
                continue

            # Save the shared ground truth once per document
            label_out_path = OUT_ROOT / "labels" / f"{doc_id}.json"
            with open(label_out_path, "w") as f:
                json.dump(ground_truth, f, indent=2)

            for level in LEVELS:
                try:
                    degraded_image = degrade(image, level)
                except Exception as e:
                    print(f"  [FAILED] {doc_id} @ {level}: {e}")
                    failures.append((f"{doc_id}_{level}", str(e)))
                    continue

                image_out_path = OUT_ROOT / level / f"{doc_id}_{level}.jpg"
                degraded_image.save(image_out_path, quality=90)

                manifest_rows.append({
                    "doc_id": doc_id,
                    "template_type": folder_name,
                    "level": level,
                    "image_path": str(image_out_path).replace("\\", "/"),
                    "label_path": str(label_out_path).replace("\\", "/"),
                })
                total_images += 1

            total_docs += 1
            if total_docs % 10 == 0:
                print(f"  ...processed {total_docs} documents so far")

    # Write manifest CSV - this is the single source of truth for dataset splitting later
    manifest_path = OUT_ROOT / "manifest.csv"
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["doc_id", "template_type", "level", "image_path", "label_path"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"\n{'=' * 50}")
    print(f"Done. {total_docs} documents processed -> {total_images} degraded images.")
    print(f"Manifest written to: {manifest_path}")
    if failures:
        print(f"\n{len(failures)} failure(s):")
        for name, err in failures[:10]:
            print(f"  - {name}: {err}")
        if len(failures) > 10:
            print(f"  ...and {len(failures) - 10} more")


if __name__ == "__main__":
    main()
