"""Build Donut-ready training manifests from train.csv / val.csv / test.csv.

For each row (image_path, label_path), this produces a JSONL file with:
    {"image_path": "...", "target_sequence": "<s_extraction>...</s_extraction>"}

This is what the Colab training notebook will load - it keeps images as
file paths (not embedded/base64) since Donut training loads and resizes
images lazily per-batch, and embedding would bloat these files enormously
for no benefit.

Ground truth is scoped to header fields only (see donut_format.py docstring
for why - decode-length limit + fairness vs. the zero-shot baseline).

Usage:
    python -m src.data.prepare_donut_dataset
    python -m src.data.prepare_donut_dataset --splits-dir data/splits --out-dir data/donut_ready
"""
import argparse
import csv
import json
from pathlib import Path

from src.data.donut_format import encode_ground_truth_from_file, decode_donut_output, HEADER_FIELDS

DEFAULT_SPLITS_DIR = Path("data/splits")
DEFAULT_OUT_DIR = Path("data/donut_ready")
SPLIT_NAMES = ["train", "val", "test"]


def build_split(split_csv: Path, out_path: Path):
    """One row per IMAGE (not per document) - each degradation tier of a
    document is its own training example, all sharing the same target
    sequence (same header fields regardless of image quality)."""
    with open(split_csv, "r", newline="") as f:
        rows = list(csv.DictReader(f))

    records = []
    failures = []
    seen_labels_cache = {}  # avoid re-reading the same label file 4x (once per tier)

    for row in rows:
        label_path = row["label_path"]
        try:
            if label_path not in seen_labels_cache:
                seen_labels_cache[label_path] = encode_ground_truth_from_file(label_path)
            target_sequence = seen_labels_cache[label_path]
        except Exception as e:
            failures.append((row["doc_id"], row["level"], str(e)))
            continue

        records.append({
            "doc_id": row["doc_id"],
            "template_type": row["template_type"],
            "level": row["level"],
            "image_path": row["image_path"],
            "target_sequence": target_sequence,
        })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    return len(records), failures


def sanity_check(out_path: Path, n: int = 3):
    """Round-trip a few records back through decode_donut_output as a smoke
    test - if this ever fails, the tag format itself has a bug, not just
    this run's data."""
    with open(out_path, "r") as f:
        lines = [json.loads(l) for l in f.readlines()[:n]]

    print(f"\nSanity check ({min(n, len(lines))} sample(s) from {out_path.name}):")
    for r in lines:
        decoded = decode_donut_output(r["target_sequence"])
        non_null = {k: v for k, v in decoded.items() if v is not None}
        print(f"  {r['doc_id']} ({r['level']}): {len(non_null)}/{len(HEADER_FIELDS)} fields populated")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits-dir", type=Path, default=DEFAULT_SPLITS_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    print(f"Building Donut-ready manifests (header fields only: {HEADER_FIELDS})\n")

    for split_name in SPLIT_NAMES:
        split_csv = args.splits_dir / f"{split_name}.csv"
        if not split_csv.exists():
            print(f"  [SKIP] {split_csv} not found")
            continue

        out_path = args.out_dir / f"{split_name}.jsonl"
        n_records, failures = build_split(split_csv, out_path)

        print(f"{split_name}: {n_records} image-level examples -> {out_path}")
        if failures:
            print(f"  {len(failures)} failure(s):")
            for doc_id, level, err in failures[:5]:
                print(f"    - {doc_id} ({level}): {err}")

        if n_records > 0:
            sanity_check(out_path)

    print(f"\nDone. Donut-ready manifests written to: {args.out_dir}/")
    print("Each line: {\"image_path\": ..., \"target_sequence\": \"<s_extraction>...\"}")


if __name__ == "__main__":
    main()