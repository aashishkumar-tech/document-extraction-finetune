"""Download AgamiAI Indian Bank Statements directly into data/raw/,
with a clean, predictable folder structure (no HF cache indirection).

Usage:
    python src/data/download_all.py            # download N_PER_FOLDER samples per type
    python src/data/download_all.py --all       # download the ENTIRE dataset (~1.37 GB)
"""
import argparse
from typing import Optional
from huggingface_hub import snapshot_download

REPO = "AgamiAI/Indian-Bank-Statements"
FOLDERS = [
    "India_Bank_Statement_Scanned_Type1",
    "India_Bank_Statement_Scanned_Type2",
    "India_Bank_Statement_Digital_Type1",
    "India_Bank_Statement_Digital_Type2",
]

# How many numbered samples (0001, 0002, ...) to pull per folder when NOT using --all.
N_PER_FOLDER = 250


def build_allow_patterns(n_per_folder: Optional[int]):
    """None -> match everything under train/. Otherwise match only 00001..0000N per folder."""
    if n_per_folder is None:
        return [f"train/{folder}/*" for folder in FOLDERS]

    patterns = []
    for folder in FOLDERS:
        for i in range(1, n_per_folder + 1):
            idx = f"{i:05d}"
            patterns.append(f"train/{folder}/{idx}.pdf")
            patterns.append(f"train/{folder}/{idx}.json")
    return patterns


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true", help="Download the entire dataset (~1.37 GB)")
    parser.add_argument("--n", type=int, default=N_PER_FOLDER, help="Samples per folder (ignored with --all)")
    args = parser.parse_args()

    allow_patterns = build_allow_patterns(None if args.all else args.n)

    print(f"Downloading {'ALL files' if args.all else f'{args.n} samples per folder'} "
          f"from {REPO} into ./data/raw/ ...")

    local_path = snapshot_download(
        repo_id=REPO,
        repo_type="dataset",
        allow_patterns=allow_patterns,
        local_dir="data/raw",
    )

    print(f"Done. Files are under: {local_path}")
    print("Expected structure: data/raw/train/<FolderType>/0000N.pdf + .json")


if __name__ == "__main__":
    main()
