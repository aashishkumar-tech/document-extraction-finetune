# Project Plan

**Project:** Fine-Tuned Vision-Language Model for Scanned Document Extraction
**Status:** Draft v1.0
**Last updated:** 2026-09-08

---

## 1. Milestones

| # | Milestone | Deliverable | Depends on |
|---|---|---|---|
| M1 | Data pipeline complete | Loader + degradation pipeline producing clean/light/medium/heavy tiers for the full AgamiAI sample set, with stratified 70/15/15 split | — |
| M2 | Dataset formatted for training | HuggingFace `Dataset` object with Donut-tagged targets, sanity-checked on a handful of samples | M1 |
| M3 | Zero-shot baseline run | Zero-shot large-VLM results on the test set, scored against ground truth | M1 |
| M4 | Donut fine-tuning complete | Trained LoRA adapter + checkpoint, training curves logged | M2 |
| M5 | Evaluation complete | Field × degradation-tier × model results table, generalization-set score | M3, M4 |
| M6 | Cost/latency analysis | Cost-per-document and latency-per-document table, both paths | M3, M4 |
| M7 | (Stretch) Qwen2-VL-7B comparison | Second fine-tuned model result, same evaluation harness | M5 (time/budget permitting) |
| M8 | Write-up & documentation finalized | Final report, updated docs, charts, resume bullet points drafted | M5, M6 |

## 2. Definition of done (per PRD §5)

Project is complete when all checkboxes in [01_PRD.md](01_PRD.md) §5 are
checked, and this doc set reflects what was actually built (not just
planned).

## 3. Budget

| Item | Estimated cost |
|---|---|
| Donut fine-tuning (free Colab T4) | $0 |
| Zero-shot baseline API calls (small test set) | ~$1–5, depending on provider/volume |
| Qwen2-VL-7B QLoRA on rented A100 (stretch, M7) | $10–20 typical, up to ~$30 with debugging overhead |
| **Total budget ceiling** | **$35** |

## 4. Sequencing notes

- M1–M3 have no GPU dependency and can be done entirely on local hardware
  (i5, no dedicated GPU) — no cost, no time pressure.
- M4 (Donut training) is the first step requiring GPU compute, but uses
  the free Colab tier — still $0 if session limits are managed.
- M7 (Qwen2-VL-7B) is explicitly a **stretch goal**, gated on M5 being
  done first — the project has a complete, presentable result without it.
- M8 should be updated incrementally throughout, not left to the end —
  screenshots/numbers are easiest to capture right after each milestone.

## 5. Explicitly deferred (future work, not part of this project's scope)

- Full transaction-ledger extraction (blocked by Donut's decode-length
  ceiling — see Technical Design Doc §4)
- Confidence-based routing / human-in-the-loop escalation for
  low-confidence fields
- Active-learning loop for prioritizing hard examples in future training
  rounds
- Applying this pipeline to real, employer-authorized production data
  under proper compliance review
