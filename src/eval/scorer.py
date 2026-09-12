"""Field-level precision/recall/F1 scorer.

Compares model predictions (raw JSON strings, e.g. from zero_shot.py) against
ground truth JSON labels. Per the Test & Evaluation Plan SS3:
  - Exact match (after normalization) for codes/numbers/dates
  - Fuzzy match (normalized edit distance) for names/free text
  - Aggregated precision/recall/F1 per field, per degradation tier

Also flags "hallucination" cases separately: predictions where most/all
header fields are present but wrong (as opposed to missing) - this is the
GPT-4o heavy-tier failure mode observed during baseline testing, and is
worth reporting as a distinct category, not lumped in with ordinary errors.

Usage:
    python -m src.eval.scorer --run-id 20260908_143012_pertemplate5
    python -m src.eval.scorer --input path/to/results.jsonl --output path/to/report.csv
"""
import argparse
import csv
import json
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path

try:
    import Levenshtein
    def _similarity(a: str, b: str) -> float:
        if not a and not b:
            return 1.0
        if not a or not b:
            return 0.0
        return Levenshtein.ratio(a, b)
except ImportError:
    # Fallback if python-Levenshtein isn't installed - slower, no extra dependency
    def _similarity(a: str, b: str) -> float:
        if not a and not b:
            return 1.0
        if not a or not b:
            return 0.0
        if a == b:
            return 1.0
        longer, shorter = (a, b) if len(a) >= len(b) else (b, a)
        if len(longer) == 0:
            return 1.0
        matches = sum(1 for c in shorter if c in longer)
        return matches / len(longer)

DEFAULT_RUNS_ROOT = Path("results/runs")
RUNS_INDEX_PATH = Path("results/runs_index.csv")

# Field classification, per Test & Evaluation Plan SS3
EXACT_FIELDS = ["account_number", "ifsc_code", "micr_code"]
NUMERIC_FIELDS = ["opening_balance", "closing_balance"]
DATE_FIELDS = ["start_date", "end_date", "statement_date"]
FUZZY_FIELDS = ["bank_name", "account_holder"]
FUZZY_THRESHOLD = 0.85

ALL_HEADER_FIELDS = EXACT_FIELDS + NUMERIC_FIELDS + DATE_FIELDS + FUZZY_FIELDS


def normalize_code(value) -> str:
    """For account numbers / IFSC / MICR: strip whitespace, uppercase, remove common OCR noise chars."""
    if value is None:
        return ""
    s = str(value).strip().upper()
    s = re.sub(r"[\s\-]", "", s)
    return s


def normalize_number(value) -> float:
    """For balances: strip currency symbols/labels, commas, whitespace; parse as float. Returns None if unparseable."""
    if value is None:
        return None
    s = str(value)
    # Find the actual numeric substring (digits, commas, one decimal point) - ignores
    # any currency symbols/prefixes like "Rs.", "INR", "$" that would otherwise corrupt parsing
    match = re.search(r"-?[\d,]+(?:\.\d+)?", s)
    if not match:
        return None
    cleaned = match.group(0).replace(",", "")
    try:
        return float(cleaned)
    except ValueError:
        return None


def normalize_date(value) -> str:
    """Normalize common date formats to YYYY-MM-DD for comparison. Returns '' if unparseable."""
    if value is None:
        return ""
    s = str(value).strip()
    formats_tried = [
        r"(\d{4})-(\d{2})-(\d{2})",                                    # 2024-01-01
        r"(\d{2})-(\d{2})-(\d{4})",                                    # 31-03-2024 (DD-MM-YYYY)
        r"(\d{2})/(\d{2})/(\d{4})",                                    # 31/03/2024
    ]
    m = re.match(formats_tried[0], s)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    m = re.match(formats_tried[1], s)
    if m:
        return f"{m.group(3)}-{m.group(2)}-{m.group(1)}"
    m = re.match(formats_tried[2], s)
    if m:
        return f"{m.group(3)}-{m.group(2)}-{m.group(1)}"

    # "01 Jan 2024" / "Jan 01 2024" style
    months = {"jan": "01", "feb": "02", "mar": "03", "apr": "04", "may": "05", "jun": "06",
              "jul": "07", "aug": "08", "sep": "09", "oct": "10", "nov": "11", "dec": "12"}
    m = re.match(r"(\d{1,2})\s+([A-Za-z]{3})\w*\s+(\d{4})", s)
    if m and m.group(2).lower()[:3] in months:
        return f"{m.group(3)}-{months[m.group(2).lower()[:3]]}-{int(m.group(1)):02d}"

    return s  # fall back to raw string comparison if format is unrecognized


def parse_prediction(raw: str):
    """Extract a JSON object from a raw model response, stripping markdown fences if present."""
    if raw is None:
        return None
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def field_match(field: str, gt_value, pred_value) -> bool:
    if field in EXACT_FIELDS:
        return normalize_code(gt_value) == normalize_code(pred_value) and normalize_code(gt_value) != ""
    if field in NUMERIC_FIELDS:
        gt_num, pred_num = normalize_number(gt_value), normalize_number(pred_value)
        if gt_num is None or pred_num is None:
            return False
        return abs(gt_num - pred_num) < 0.01
    if field in DATE_FIELDS:
        gt_norm, pred_norm = normalize_date(gt_value), normalize_date(pred_value)
        return gt_norm == pred_norm and gt_norm != ""
    if field in FUZZY_FIELDS:
        gt_str = "" if gt_value is None else str(gt_value).strip().lower()
        pred_str = "" if pred_value is None else str(pred_value).strip().lower()
        if gt_str == "":
            return False
        return _similarity(gt_str, pred_str) >= FUZZY_THRESHOLD
    return gt_value == pred_value


def is_present(value) -> bool:
    return value is not None and str(value).strip() != ""


def score_one_document(ground_truth: dict, prediction: dict):
    """Returns per-field TP/FP/FN counts for one document's header fields."""
    counts = {}
    for field in ALL_HEADER_FIELDS:
        gt_value = ground_truth.get(field)
        pred_value = prediction.get(field) if prediction else None

        gt_present = is_present(gt_value)
        pred_present = is_present(pred_value)

        if not gt_present and not pred_present:
            continue  # true negative, not counted

        matched = gt_present and pred_present and field_match(field, gt_value, pred_value)

        tp = 1 if matched else 0
        fp = 1 if (pred_present and not matched) else 0
        fn = 1 if (gt_present and not matched) else 0
        counts[field] = (tp, fp, fn)

    return counts


def classify_failure_type(counts: dict) -> str:
    """Distinguish 'hallucination' (fields present but wrong) from 'missing' (fields not extracted)."""
    total_wrong = sum(fp for _, (tp, fp, fn) in counts.items())
    total_missing = sum(1 for _, (tp, fp, fn) in counts.items() if tp == 0 and fp == 0 and fn == 1)
    total_correct = sum(tp for _, (tp, fp, fn) in counts.items())
    total_fields = len(counts)

    if total_fields == 0:
        return "no_data"
    if total_correct / total_fields >= 0.8:
        return "good"
    if total_wrong >= total_missing and total_wrong > 0:
        return "likely_hallucination"  # confidently wrong, not just silent/missing
    return "mostly_missing"


def append_to_runs_index(run_id: str, tier_metrics: dict, failure_counts: dict):
    """Append a one-line summary of this scoring run to results/runs_index.csv,
    so multiple runs (different sample sizes, different models, before/after
    fine-tuning) can be compared side by side without opening each report."""
    RUNS_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    file_exists = RUNS_INDEX_PATH.exists()

    fieldnames = ["run_id", "scored_at", "clean_f1", "light_f1", "medium_f1", "heavy_f1",
                  "n_good", "n_hallucination", "n_missing", "n_api_error", "n_unparseable"]

    row = {
        "run_id": run_id,
        "scored_at": datetime.now().isoformat(),
        "clean_f1": f"{tier_metrics.get('clean', 0.0):.3f}",
        "light_f1": f"{tier_metrics.get('light', 0.0):.3f}",
        "medium_f1": f"{tier_metrics.get('medium', 0.0):.3f}",
        "heavy_f1": f"{tier_metrics.get('heavy', 0.0):.3f}",
        "n_good": failure_counts.get("good", 0),
        "n_hallucination": failure_counts.get("likely_hallucination", 0),
        "n_missing": failure_counts.get("mostly_missing", 0),
        "n_api_error": failure_counts.get("api_error", 0),
        "n_unparseable": failure_counts.get("unparseable_json", 0),
    }

    with open(RUNS_INDEX_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", type=str, default=None,
                         help="Run folder name under results/runs/ (as printed by zero_shot.py). "
                              "Reads results/runs/<run-id>/zero_shot_results.jsonl and writes "
                              "the report into the same folder.")
    parser.add_argument("--input", type=Path, default=None,
                         help="Explicit path to a results .jsonl file (overrides --run-id)")
    parser.add_argument("--output", type=Path, default=None,
                         help="Explicit path for the output report (overrides --run-id)")
    args = parser.parse_args()

    if args.input:
        input_path = args.input
        output_path = args.output or input_path.parent / "field_level_report.csv"
        run_id = input_path.parent.name
    elif args.run_id:
        run_dir = DEFAULT_RUNS_ROOT / args.run_id
        input_path = run_dir / "zero_shot_results.jsonl"
        output_path = run_dir / "field_level_report.csv"
        run_id = args.run_id
    else:
        # Fall back to the most recently modified run folder, if any exist
        if not DEFAULT_RUNS_ROOT.exists() or not any(DEFAULT_RUNS_ROOT.iterdir()):
            raise RuntimeError(
                "No --run-id or --input given, and no runs found under results/runs/. "
                "Run src.baseline.zero_shot first, or pass --run-id explicitly."
            )
        latest_run = max(DEFAULT_RUNS_ROOT.iterdir(), key=lambda p: p.stat().st_mtime)
        input_path = latest_run / "zero_shot_results.jsonl"
        output_path = latest_run / "field_level_report.csv"
        run_id = latest_run.name
        print(f"No --run-id given, using most recent run: {run_id}\n")

    with open(input_path, "r") as f:
        results = [json.loads(line) for line in f if line.strip()]

    # aggregate: (template_type, level, field) -> [tp, fp, fn]
    agg = defaultdict(lambda: [0, 0, 0])
    per_doc_summary = []

    for r in results:
        if r.get("error"):
            per_doc_summary.append({**r, "failure_type": "api_error"})
            continue

        prediction = parse_prediction(r["prediction_raw"])
        with open(r["label_path"], "r") as f:
            ground_truth = json.load(f)

        if prediction is None:
            per_doc_summary.append({**r, "failure_type": "unparseable_json"})
            continue

        counts = score_one_document(ground_truth, prediction)
        failure_type = classify_failure_type(counts)
        per_doc_summary.append({**r, "failure_type": failure_type})

        for field, (tp, fp, fn) in counts.items():
            key = (r["template_type"], r["level"], field)
            agg[key][0] += tp
            agg[key][1] += fp
            agg[key][2] += fn

    # write field x tier report
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["template_type", "level", "field", "precision", "recall", "f1", "tp", "fp", "fn"])
        for (template_type, level, field), (tp, fp, fn) in sorted(agg.items()):
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            writer.writerow([template_type, level, field, f"{precision:.3f}", f"{recall:.3f}", f"{f1:.3f}", tp, fp, fn])

    # print summary by degradation tier (collapsed across fields) - the headline chart
    tier_agg = defaultdict(lambda: [0, 0, 0])
    for (template_type, level, field), (tp, fp, fn) in agg.items():
        tier_agg[level][0] += tp
        tier_agg[level][1] += fp
        tier_agg[level][2] += fn

    print(f"Run ID: {run_id}\n")
    print(f"{'Level':<10} {'Precision':<12} {'Recall':<10} {'F1':<10}")
    tier_f1 = {}
    for level in ["clean", "light", "medium", "heavy"]:
        if level not in tier_agg:
            continue
        tp, fp, fn = tier_agg[level]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        tier_f1[level] = f1
        print(f"{level:<10} {precision:<12.3f} {recall:<10.3f} {f1:<10.3f}")

    # failure type breakdown - surfaces the hallucination finding quantitatively
    failure_counts = defaultdict(int)
    for doc in per_doc_summary:
        failure_counts[doc["failure_type"]] += 1
    print("\nFailure type breakdown (per document):")
    for ftype, count in sorted(failure_counts.items(), key=lambda x: -x[1]):
        print(f"  {ftype}: {count}")

    append_to_runs_index(run_id, tier_f1, failure_counts)

    print(f"\nField x tier report written to: {output_path}")
    print(f"Cross-run summary appended to:  {RUNS_INDEX_PATH}")


if __name__ == "__main__":
    main()