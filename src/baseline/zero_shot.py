"""Zero-shot large-VLM baseline via OpenAI GPT-4o.

This is the "secondary reference point" comparison from the Test &
Evaluation Plan - a general-purpose large VLM, given the schema in the
prompt, with NO fine-tuning. Run on the same test set as the fine-tuned
model for a fair comparison.

Every run is saved to its own timestamped folder under results/runs/,
containing:
  - zero_shot_results.jsonl   (raw predictions, same as before)
  - run_config.json           (exactly what was run: model, sampling mode,
                                seed, row count, template coverage, timestamp)
This makes every run traceable and comparable later - nothing gets
silently overwritten by the next run.

Setup:
    pip install openai
    Create a .env file in the project root with: OPENAI_API_KEY=sk-...

Usage:
    python -m src.baseline.zero_shot --per-template 5   # 5 docs per template (recommended)
    python -m src.baseline.zero_shot --n 10              # first 10 rows (not stratified)
    python -m src.baseline.zero_shot --all                # entire test set
    python -m src.baseline.zero_shot --per-template 5 --run-name sanity_check
"""
import argparse
import base64
import csv
import json
import os
import random
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()  # reads .env in the project root and sets OPENAI_API_KEY, etc.

SCHEMA_PROMPT = """Extract the following fields from this bank statement image as JSON:
account_number, ifsc_code, micr_code, bank_name, account_holder,
opening_balance, closing_balance, start_date, end_date, statement_date,
and a list of transactions with date, description, debit, credit, balance.

If a field is unreadable or not present, use null.
Return ONLY valid JSON. No markdown formatting, no explanation, no code fences."""

TEST_CSV = Path("data/splits/test.csv")
RUNS_ROOT = Path("results/runs")


def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def extract_zero_shot(client: OpenAI, image_path: str, model: str = "gpt-4o") -> str:
    image_b64 = encode_image(image_path)

    response = client.chat.completions.create(
        model=model,
        max_tokens=3000,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": SCHEMA_PROMPT},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
            ],
        }],
    )
    return response.choices[0].message.content


def load_test_rows(limit: int = None):
    with open(TEST_CSV, "r", newline="") as f:
        rows = list(csv.DictReader(f))
    if limit:
        rows = rows[:limit]
    return rows


def load_stratified_sample(docs_per_template: int, seed: int = 42):
    """Sample N *documents* per template type (all 4 degradation tiers included
    for each sampled document), instead of just taking the first N rows in
    file order - which would silently only cover whichever template happens
    to be listed first in test.csv."""
    with open(TEST_CSV, "r", newline="") as f:
        rows = list(csv.DictReader(f))

    rows_by_doc = defaultdict(list)
    template_by_doc = {}
    for row in rows:
        rows_by_doc[row["doc_id"]].append(row)
        template_by_doc[row["doc_id"]] = row["template_type"]

    doc_ids_by_template = defaultdict(list)
    for doc_id, template in template_by_doc.items():
        doc_ids_by_template[template].append(doc_id)

    rng = random.Random(seed)
    selected_rows = []
    for template, doc_ids in sorted(doc_ids_by_template.items()):
        doc_ids = sorted(doc_ids)
        rng.shuffle(doc_ids)
        chosen = doc_ids[:docs_per_template]
        for doc_id in chosen:
            selected_rows.extend(rows_by_doc[doc_id])

    return selected_rows


def make_run_id(sampling_mode: str, run_name: str = None) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if run_name:
        return f"{timestamp}_{run_name}"
    return f"{timestamp}_{sampling_mode}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=None,
                         help="Number of test samples to run, taken from the start of test.csv "
                              "(WARNING: not stratified by template - use --per-template instead "
                              "for balanced coverage across all template types)")
    parser.add_argument("--per-template", type=int, default=None,
                         help="Number of DOCUMENTS to sample per template type "
                              "(each document contributes all 4 degradation tiers). "
                              "Recommended over --n for representative results.")
    parser.add_argument("--all", action="store_true", help="Run on the entire test set")
    parser.add_argument("--model", type=str, default="gpt-4o")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-name", type=str, default=None,
                         help="Optional label appended to the run folder name, e.g. 'sanity_check'")
    args = parser.parse_args()

    if not (args.all or args.per_template or args.n):
        args.n = 10  # default fallback if nothing specified

    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY environment variable not set. "
            "Create a .env file with: OPENAI_API_KEY=sk-..."
        )

    client = OpenAI()

    # --- Determine sampling mode and load rows ---
    if args.all:
        sampling_mode = "all"
        rows = load_test_rows(limit=None)
        print(f"Running zero-shot extraction on the FULL test set ({len(rows)} images) using {args.model}...\n")
    elif args.per_template:
        sampling_mode = f"pertemplate{args.per_template}"
        rows = load_stratified_sample(args.per_template, seed=args.seed)
        by_template = defaultdict(int)
        for r in rows:
            by_template[r["template_type"]] += 1
        print(f"Running zero-shot extraction on {args.per_template} document(s) per template "
              f"({len(rows)} images total) using {args.model}...")
        print(f"Coverage: {dict(by_template)}\n")
    else:
        sampling_mode = f"first{args.n}"
        rows = load_test_rows(limit=args.n)
        templates_seen = set(r["template_type"] for r in rows)
        print(f"Running zero-shot extraction on {len(rows)} test images using {args.model}...")
        if len(templates_seen) == 1:
            print(f"WARNING: all {len(rows)} samples are from a single template "
                  f"({list(templates_seen)[0]}). Use --per-template for balanced coverage.\n")
        else:
            print()

    # --- Set up this run's dedicated folder ---
    run_id = make_run_id(sampling_mode, args.run_name)
    run_dir = RUNS_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    results_path = run_dir / "zero_shot_results.jsonl"
    config_path = run_dir / "run_config.json"

    # --- Run extraction ---
    results = []
    for i, row in enumerate(rows, 1):
        image_path = row["image_path"]
        label_path = row["label_path"]

        try:
            raw_output = extract_zero_shot(client, image_path, model=args.model)
        except Exception as e:
            print(f"  [{i}/{len(rows)}] FAILED: {row['doc_id']} ({row['level']}) - {e}")
            results.append({
                "doc_id": row["doc_id"],
                "template_type": row["template_type"],
                "level": row["level"],
                "image_path": image_path,
                "label_path": label_path,
                "prediction_raw": None,
                "error": str(e),
            })
            continue

        results.append({
            "doc_id": row["doc_id"],
            "template_type": row["template_type"],
            "level": row["level"],
            "image_path": image_path,
            "label_path": label_path,
            "prediction_raw": raw_output,
            "error": None,
        })

        print(f"  [{i}/{len(rows)}] done: {row['doc_id']} ({row['level']})")
        time.sleep(0.5)  # gentle rate-limit buffer

    with open(results_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    # --- Write run_config.json: exactly what was run, for future reference ---
    n_failed = sum(1 for r in results if r["error"] is not None)
    by_template_final = defaultdict(int)
    by_level_final = defaultdict(int)
    for r in results:
        by_template_final[r["template_type"]] += 1
        by_level_final[r["level"]] += 1

    run_config = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "model": args.model,
        "sampling_mode": sampling_mode,
        "seed": args.seed,
        "total_images": len(results),
        "succeeded": len(results) - n_failed,
        "failed": n_failed,
        "coverage_by_template": dict(by_template_final),
        "coverage_by_level": dict(by_level_final),
        "results_file": str(results_path).replace("\\", "/"),
    }
    with open(config_path, "w") as f:
        json.dump(run_config, f, indent=2)

    print(f"\nDone. {len(results) - n_failed}/{len(results)} succeeded.")
    print(f"Run ID: {run_id}")
    print(f"Results:    {results_path}")
    print(f"Run config: {config_path}")
    print(f"\nNext step: python -m src.eval.scorer --run-id {run_id}")


if __name__ == "__main__":
    main()