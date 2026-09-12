# Technical Design Document (TDD)

**Project:** Fine-Tuned Vision-Language Model for Scanned Document Extraction
**Status:** Draft v1.0
**Last updated:** 2026-09-08
**Related:** [01_PRD.md](01_PRD.md), [04_Test_and_Evaluation_Plan.md](04_Test_and_Evaluation_Plan.md)

---

## 1. High-Level Design (HLD)

### 1.1 Objective

Compare a fine-tuned small vision-language model against (a) a traditional
OCR+rules baseline and (b) a zero-shot large VLM, on structured field
extraction from progressively degraded scanned documents.

### 1.2 System components

| Component | Responsibility |
|---|---|
| Data layer | Source PDFs + JSON ground truth (AgamiAI bank statements) |
| Augmentation layer | Produces clean / light / medium / heavy degraded image tiers |
| Model layer (Path A) | Fine-tuned small VLM (Donut or Qwen2-VL-7B via QLoRA) |
| Model layer (Path B) | Zero-shot large VLM (Claude / GPT-4V) via API, no training |
| Training infrastructure | Free Colab T4 (Donut) or rented A100 (Qwen2-VL-7B QLoRA) |
| Evaluation layer | Field-level scoring engine, model JSON vs. ground truth JSON |
| Reporting layer | Accuracy tables/charts by field type and degradation tier; cost/latency table |

### 1.3 Data flow

```
Source PDFs + JSON labels
        |
Degradation pipeline
(clean / light / medium / heavy)
        |
   -------------------
   |                 |
Fine-tune         Zero-shot
small VLM         large VLM
   |                 |
   -------------------
        |
Evaluate & compare
(F1 by field, by degradation tier, by cost/latency)
```

### 1.4 Technology stack

- **Language/runtime:** Python 3.10+
- **ML libraries:** HuggingFace `transformers`, `peft` (LoRA/QLoRA), `datasets`
- **Image processing:** OpenCV, PIL, `pdf2image`
- **Degradation:** Custom OpenCV pipeline; Augraphy for realistic print/scan/fax effects
- **Baseline model access:** Claude / GPT-4V API
- **Scoring:** pandas, `python-Levenshtein` (fuzzy field matching)
- **Compute:** Google Colab (free tier) and/or RunPod / Lambda Labs (A100 rental)

---

## 2. Low-Level Design (LLD)

### 2.1 Data ingestion module

```
load_statement(pdf_path, json_path) -> (raw_image, ground_truth_dict)
```

- PDF → image via `pdf2image` / `pdftoppm`, fixed DPI (150–200)
- `ground_truth_dict` loaded directly from the AgamiAI JSON label file

### 2.2 Degradation module

```
degrade(image, level: Literal["clean","light","medium","heavy"]) -> degraded_image
```

Pipeline steps, applied in sequence with tier-dependent parameters:

1. Rotation/skew — `cv2.warpAffine`
2. Downscale → upscale (resolution loss) — `cv2.resize`
3. Gaussian blur — `cv2.GaussianBlur`
4. Speckle/salt-and-pepper noise injection
5. Contrast/brightness shift — `cv2.convertScaleAbs`
6. JPEG re-encode at reduced quality — `cv2.imencode`

The **same ground truth JSON** is reused across all four degradation tiers
of a given source document — this is what enables the "accuracy by
degradation tier" comparison chart later.

> Note: for a more realistic degradation profile (fax lines, scanner
> banding, ink bleed) beyond generic blur/noise, the Augraphy library
> (used by the ShabbyPages dataset) should be substituted in for step 3–6
> once available in the training environment. See §5 for tradeoffs.

### 2.3 Dataset preparation module

- Stratified split: **70% train / 15% validation / 15% test**, stratified
  by degradation tier and template type
- One template variant held out entirely from training and used only in
  the test set, to measure generalization to an unseen layout
- Ground truth JSON converted into the target model's expected training
  format:
  - Donut: tagged token sequence, e.g. `<s_account_number>...</s_account_number>`
  - Qwen2-VL: instruction-tuned chat format with image + JSON target
- Output: a `datasets.Dataset` object, ready for a HuggingFace `Trainer`

### 2.4 Model configuration

**Option A — Donut** (`naver-clova-ix/donut-base`, ~200M params)
- LoRA fine-tune (not full fine-tune)
- Fits comfortably on a free Colab T4 (16GB VRAM)
- Known constraint: fixed max decode length (~768–1024 tokens) — see §5

**Option B — Qwen2-VL-7B** (~7B params)
- QLoRA (4-bit NF4) fine-tune
- Requires rented GPU (A100 40GB recommended)
- LoRA config: rank 8–16, alpha 16–32, target modules = attention
  query/value projections

### 2.5 Training loop

- Optimizer: AdamW, learning rate ≈ 1e-4 (LoRA typically needs a higher LR
  than full fine-tuning)
- Loss: token-level cross-entropy
- Epochs: 3–5, with early stopping on validation loss
- Checkpointing: save only the best validation-loss checkpoint (storage-
  constrained environment)

### 2.6 Baseline module (zero-shot)

- Fixed schema-instruction prompt + raw document image sent to a large VLM
  (Claude / GPT-4V) via API
- Optional 1–2 few-shot image→JSON examples included in the prompt
- Run on the **same test set** as the fine-tuned model, for a fair,
  apples-to-apples comparison

### 2.7 Evaluation module

- Field matching strategy:
  - Exact match — account numbers, codes, dates
  - Fuzzy match (normalized string distance) — names, free-text fields
- Metrics: precision / recall / F1 per field, aggregated per degradation
  tier
- Output: a single results table — **field × degradation tier × model** —
  this is the core evidence artifact for the write-up

### 2.8 Reporting module

- Bar charts: fine-tuned model vs. zero-shot baseline, broken down by
  degradation tier
- Cost/latency table: per-document inference cost and time, fine-tuned
  small model vs. large-model API calls

---

## 3. Model options considered

| Model | Params | Fits on | Notes |
|---|---|---|---|
| Donut-base | ~200M | Free Colab T4 | OCR-free, purpose-built for image→JSON, fastest to iterate |
| Pix2Struct-base | ~282M | Free Colab T4 | Similar OCR-free approach, broader pretraining |
| LayoutLMv3-base | ~133M | Free Colab T4 | Needs OCR text + bounding boxes as input, not pure image→JSON |
| Qwen2-VL-2B | ~2B | Free Colab T4 (QLoRA, slow) | Strong document understanding, attemptable on free tier |
| **Qwen2-VL-7B (selected stretch option)** | ~7B | Rented A100 (QLoRA) | Strong open VLM, needs paid GPU rental |
| InternVL2-8B / LLaVA-1.6-7B | 7–13B | Rented A100/A10 | Alternatives at the same compute tier |

**Decision:** Donut-base is the primary model (low-risk, free-tier
feasible). Qwen2-VL-7B is a stretch goal if budget/time allow, run as a
second comparison point rather than a replacement.

## 4. Known constraints and design decisions

| Constraint | Decision |
|---|---|
| Donut's ~768–1024 token decode limit | Scope extraction to header fields (account number, balances, IFSC/MICR, dates). Full transaction-ledger extraction is documented as future work, not silently dropped. |
| No local GPU (i5, 8GB RAM, no dedicated GPU) | All data prep and scripting done locally; training executed on free Colab (Donut) or rented A100 (Qwen2-VL-7B). |
| No public dataset combines real domain + real degradation + labels | Use AgamiAI (real schema, real labels, clean scans) + custom/Augraphy degradation on top, explicitly documented as a synthetic stand-in (see [03_Data_and_Ethics.md](03_Data_and_Ethics.md)). |

## 5. Alternatives considered and rejected

- **Full fine-tuning (no LoRA):** rejected — infeasible on available
  compute budget and hardware; LoRA/QLoRA gives most of the benefit at a
  fraction of the memory/compute cost.
- **Training directly on real EOB/bank correspondence:** rejected —
  contains PHI/PII, not usable in a public project.
- **Building a full sprawling comparison matrix (5+ models/techniques):**
  rejected — dilutes the story and invites unfocused "why didn't you try
  X" critique. Scoped to one primary + one secondary comparison instead
  (see PRD §4, Non-goals).
