"""Build a stratified 70/15/15 train/val/test split from manifest.csv.

Key rules (per Technical Design Doc SS2.3 and Test & Evaluation Plan SS4):
  1. Split at the DOCUMENT level, not the image-row level. Each document
     contributes 4 rows (one per degradation tier) - if we split rows
     independently, the same document's "clean" version could end up in
     train while its "heavy" version ends up in test, which leaks the
     document's content/layout across the split boundary.
  2. Stratify by template_type, so train/val/test each get proportional
     representation of every template - avoids validating only on the
     "easy" template by chance.
  3. One entire template type is held out from train/val and used ONLY in
     test, to measure generalization to a template the model never saw
     during training. This is added on top of the normal stratified test
     slice, so the test set ends up larger than a plain 15% would be -
     that's intentional and documented in the printed summary.

Usage:
    python -m src.data.prepare_dataset
    python -m src.data.prepare_dataset --holdout-template India_Bank_Statement_Digital_Type2
    python -m src.data.prepare_dataset --train 0.7 --val 0.15 --test 0.15 --seed 42
"""
import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path

MANIFEST_PATH = Path("data/degraded/manifest.csv")
SPLITS_DIR = Path("data/splits")


def load_manifest(path: Path):
    with open(path, "r", newline="") as f:
        return list(csv.DictReader(f))


def group_by_doc(rows):
    """doc_id -> {template_type, rows: [row, row, row, row]}"""
    docs = defaultdict(lambda: {"template_type": None, "rows": []})
    for row in rows:
        doc = docs[row["doc_id"]]
        doc["template_type"] = row["template_type"]
        doc["rows"].append(row)
    return docs


def stratified_doc_split(doc_ids_by_template: dict, train_frac: float, val_frac: float, seed: int):
    """Split doc_ids within each template proportionally into train/val/test."""
    train_ids, val_ids, test_ids = [], [], []
    rng = random.Random(seed)

    for template, doc_ids in doc_ids_by_template.items():
        ids = list(doc_ids)
        rng.shuffle(ids)
        n = len(ids)
        n_train = int(n * train_frac)
        n_val = int(n * val_frac)

        train_ids.extend(ids[:n_train])
        val_ids.extend(ids[n_train:n_train + n_val])
        test_ids.extend(ids[n_train + n_val:])

    return train_ids, val_ids, test_ids


def write_split(path: Path, doc_ids: list, docs: dict, fieldnames: list):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for doc_id in doc_ids:
            writer.writerows(docs[doc_id]["rows"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=float, default=0.70)
    parser.add_argument("--val", type=float, default=0.15)
    parser.add_argument("--test", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--holdout-template", type=str, default=None,
                         help="Template type to hold out entirely for generalization testing. "
                              "Defaults to the last template type alphabetically if not specified.")
    args = parser.parse_args()

    assert abs(args.train + args.val + args.test - 1.0) < 1e-6, "train+val+test must sum to 1.0"

    rows = load_manifest(MANIFEST_PATH)
    fieldnames = list(rows[0].keys())
    docs = group_by_doc(rows)

    all_templates = sorted(set(d["template_type"] for d in docs.values()))
    holdout_template = args.holdout_template or all_templates[-1]
    if holdout_template not in all_templates:
        raise ValueError(f"--holdout-template '{holdout_template}' not found. Available: {all_templates}")

    print(f"Templates found: {all_templates}")
    print(f"Held-out generalization template: {holdout_template}\n")

    # Separate held-out template's docs (go entirely to test) from the rest
    holdout_doc_ids = [doc_id for doc_id, d in docs.items() if d["template_type"] == holdout_template]
    remaining_doc_ids_by_template = defaultdict(list)
    for doc_id, d in docs.items():
        if d["template_type"] != holdout_template:
            remaining_doc_ids_by_template[d["template_type"]].append(doc_id)

    # Stratified split of the remaining (non-holdout) documents
    train_ids, val_ids, test_ids = stratified_doc_split(
        remaining_doc_ids_by_template, args.train, args.val, args.seed
    )

    # Held-out template's documents are added entirely to test
    test_ids = test_ids + holdout_doc_ids

    # Write out the three CSVs
    write_split(SPLITS_DIR / "train.csv", train_ids, docs, fieldnames)
    write_split(SPLITS_DIR / "val.csv", val_ids, docs, fieldnames)
    write_split(SPLITS_DIR / "test.csv", test_ids, docs, fieldnames)

    # Summary
    def count_by_template(doc_ids):
        c = defaultdict(int)
        for doc_id in doc_ids:
            c[docs[doc_id]["template_type"]] += 1
        return c

    print(f"{'Split':<8} {'#Docs':<8} {'#Images':<10} By template")
    for name, ids in [("train", train_ids), ("val", val_ids), ("test", test_ids)]:
        n_images = sum(len(docs[d]["rows"]) for d in ids)
        by_template = dict(count_by_template(ids))
        print(f"{name:<8} {len(ids):<8} {n_images:<10} {by_template}")

    print(f"\nNote: test set includes the full held-out template "
          f"('{holdout_template}', {len(holdout_doc_ids)} docs) PLUS a normal "
          f"stratified {args.test:.0%} slice of the other templates - "
          f"so it's intentionally larger than a plain {args.test:.0%} split.")
    print(f"\nSplits written to: {SPLITS_DIR}/")


if __name__ == "__main__":
    main()