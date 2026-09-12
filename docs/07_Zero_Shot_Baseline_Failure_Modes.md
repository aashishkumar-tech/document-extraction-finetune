# Zero-Shot Baseline: Observed Failure Modes

**Model tested:** GPT-4o (OpenAI), zero-shot, no fine-tuning
**Sample size:** 20 documents × 4 degradation tiers = 80 images, stratified evenly across all 4 template types (5 documents per template)
**Run ID:** `20260912_201022_pertemplate5`
**Date:** 2026-09-12
**Purpose:** This document tracks specific, reproducible failure patterns found while validating the zero-shot baseline. It exists so that once the fine-tuned model (Donut) is evaluated, we can check directly whether each of these failure modes is fixed, unchanged, or replaced by a new one — rather than comparing only aggregate F1 scores.

---

## Headline numbers (n=20 per tier, confirmed at scale)

| Tier | Precision | Recall | F1 |
| --- | --- | --- | --- |
| clean | 0.895 | 0.850 | 0.872 |
| light | 0.600 | 0.570 | 0.585 |
| medium | 0.336 | 0.205 | 0.255 |
| heavy | 0.244 | 0.100 | 0.142 |

**Failure type breakdown (80 documents scored):** 29 good, 31 likely_hallucination, 19 mostly_missing, 1 unparseable/empty response.

---

## Summary table

| # | Failure mode | Tier(s) observed | Root cause | Fixable by fine-tuning? |
| --- | --- | --- | --- | --- |
| 1 | Minor transcription noise | light (Scanned templates) | Ordinary OCR-style character slips on long digit strings | Partially — depends on training data volume |
| 2 | Schema-field confusion (`statement_date` vs `end_date`) | **all tiers, all templates — 0/20, fully confirmed** | Model substitutes a plausible default instead of locating the actual printed field | **Yes, likely** — schema-specific pattern a fine-tuned model should learn directly |
| 3 | Full hallucination on illegible input | medium, heavy | Image unreadable; model fabricates a plausible-looking fake record instead of signaling uncertainty | Uncertain — may require a confidence/abstention mechanism, not just fine-tuning |
| 4 | **Digital-template images degrade faster than Scanned-template images at the same nominal tier** | **light, medium** | Likely: Digital-source PDFs start from crisp vector/high-contrast renders, so the same resolution-relative blur/downscale destroys proportionally more fine detail than on Scanned-source images | N/A — this is a property of the degradation pipeline + source image characteristics, not a model failure. Worth noting when interpreting "accuracy vs. tier" charts. |
| 5 | Empty/refused response (no error, no content) | heavy (1 case observed) | Model returned nothing for a fully illegible image, rather than hallucinating or erroring | N/A — arguably the *most* honest failure mode observed; worth investigating whether this is reproducible or a one-off |

---

## Failure mode 1: Minor transcription noise (light tier, Scanned templates)

At light tier, `Scanned_Type1` and `Scanned_Type2` mostly held up well (account_number, ifsc_code, micr_code around 0.8), with occasional single-field misses consistent with ordinary OCR-style noise. Not systemic.

---

## Failure mode 2: Schema-field confusion — `statement_date` vs `end_date` (CONFIRMED at scale)

**Result across the full 80-image run: 0/20 correct at every single tier, every single template.** This is now a fully confirmed, universal failure — not specific to one document or one template.

**Root cause, confirmed by direct inspection (see prior investigation):**

```
Ground truth end_date:       2024-03-31
Ground truth statement_date: 2025-11-22   (a different, unrelated date)
Predicted statement_date:    2024-03-31   (= end_date, copied)
```

GPT-4o substitutes `end_date` for `statement_date` regardless of image quality. This is a schema-understanding gap, completely independent of degradation — it fails even on perfectly legible clean-tier images, 20 times out of 20.

---

## Failure mode 3: Full hallucination on illegible (medium/heavy-tier) input

31 of 80 documents were classified as `likely_hallucination` — fields present in the output but wrong, rather than missing. This was originally spotted at heavy tier on one document (fabricated account number, IFSC code, transaction descriptions with a completely different narrative style and wrong year) and is now confirmed as a widespread pattern across the larger sample, concentrated in medium and heavy tiers.

**Operational risk:** these outputs are well-formatted and confident-looking, making them hard to distinguish from correct extractions without independent verification.

---

## Failure mode 4 (NEW): Digital-template images degrade faster than Scanned-template images

Comparing the **same nominal degradation tier** across templates reveals a real asymmetry:

| Field | Scanned_Type1 @ light | Digital_Type1 @ light |
| --- | --- | --- |
| account_number | 0.800 | **0.000** |
| ifsc_code | 1.000 | **0.000** |
| micr_code | 0.800 | **0.000** |
| closing_balance | 1.000 | **0.000** |

At the identical "light" degradation setting, Digital-template documents have already collapsed on several fields where Scanned-template documents are still mostly intact.

**Likely explanation:** Digital-type source images are likely cleaner/higher-contrast to begin with (vector-rendered text, thin fonts), so the same relative blur/downscale in `degrade.py` may destroy proportionally more legibility on these than on Scanned-type sources, which may have more inherent visual redundancy (thicker print, more forgiving spacing).

**Action item:** this is worth a follow-up investigation — either (a) accept this as a real property of the two document families and report degradation curves separately per template rather than pooling them, or (b) consider whether `degrade.py`'s parameters should be tier-calibrated per template type rather than using one global parameter set. Given time constraints, (a) — reporting separately — is the lower-risk choice for this project's scope.

---

## Failure mode 5 (NEW): Empty response, no error, no hallucination

One document (`India_Bank_Statement_Digital_Type2_00014`, heavy tier) produced `prediction_raw: null` with `error: null` — meaning the API call succeeded, but GPT-4o's response contained no usable content. This is distinct from both hallucination (fabricated content) and API failure (an exception was raised). It's arguably the "most honest" failure mode observed, though at n=1 it's too rare yet to characterize confidently — worth watching for in future runs.

---

## Open items before treating these as fully final

- [x] Confirm failure mode 1 (transcription noise) — confirmed, minor and tier-appropriate
- [x] Confirm failure mode 2 (`statement_date` confusion) persists across templates — **confirmed, 0/20 universal**
- [x] Confirm failure mode 3 (hallucination) is consistent across multiple documents — **confirmed, 31/80 documents**
- [ ] Investigate failure mode 4 (Digital vs. Scanned degradation asymmetry) further — decide whether to report tier curves per-template instead of pooled
- [ ] Determine if failure mode 5 (empty response) is reproducible or a one-off — watch for it in future runs
- [ ] Once Donut fine-tuning is complete, re-run this same failure-mode analysis and fill in the comparison table below

## Comparison table (to be completed after fine-tuning)

| Failure mode | GPT-4o zero-shot (n=80) | Donut fine-tuned | Notes |
| --- | --- | --- | --- |
| Minor transcription noise (light) | Present, minor | *TBD* | |
| `statement_date` schema confusion (all tiers) | **0/20, universal** | *TBD* | |
| Full hallucination (medium/heavy) | 31/80 documents | *TBD* | |
| Digital vs. Scanned degradation asymmetry | Confirmed | *TBD* | May not apply if Donut is trained per-template |
| Empty response | 1/80 (rare) | *TBD* | |
