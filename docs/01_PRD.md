# Product Requirements Document (PRD)

**Project:** Fine-Tuned Vision-Language Model for Scanned Document Extraction
**Doc owner:** Aashish
**Status:** Draft v1.0
**Last updated:** 2026-09-08

---

## 1. Background

At work, a production document-extraction pipeline (rule-based logic on top
of Azure Document Intelligence OCR output) is achieving roughly **10% field
accuracy** on scanned correspondence documents (EOB letters, claim forms,
bank correspondence). Root-cause investigation points to two compounding
failures:

1. Many source documents are blurry, low-resolution, or multi-generation
   scans/photocopies — OCR text quality is poor before any parsing happens.
2. Rule-based/regex parsing is brittle against the layout variation across
   document templates (narrative letters, coded field-dumps, standardized
   forms like UB-04).

This is a real, observed production problem. This project is a **personal
portfolio project** that investigates an alternative architecture to this
problem, using public, non-PHI data as a structural stand-in for the real
(restricted) documents.

## 2. Problem statement

> Can a small, fine-tuned vision-language model — which reads document
> images directly and skips a separate OCR step — recover extraction
> accuracy that a traditional OCR+rules pipeline loses to blur, scan noise,
> and layout variation, and can it do so more cheaply than calling a large
> general-purpose model per document?

## 3. Goals

- **G1:** Demonstrate a working fine-tuning pipeline (data → training →
  evaluation) for image-to-structured-JSON extraction.
- **G2:** Produce one headline, defensible comparison: fine-tuned small
  model vs. OCR+rules baseline, with a zero-shot large VLM as a secondary
  reference point.
- **G3:** Quantify accuracy degradation across clean/light/medium/heavy
  scan-quality tiers, broken down by field type.
- **G4:** Produce a cost/latency comparison to frame the result in
  practical, not just academic, terms.
- **G5:** Produce documentation and a write-up suitable for a resume/
  portfolio and technical interview discussion.

## 4. Non-goals (explicitly out of scope)

- Not a production-ready system — no deployment, no monitoring, no
  real-time serving requirement.
- Not using any real PHI/PII or employer data, in any form, at any stage.
- Not attempting full transaction-table extraction if the chosen model
  (Donut) has a decode-length ceiling that makes it unreliable — this is
  explicitly flagged as future work, not solved in v1.
- Not building an exhaustive comparison matrix across every possible model
  and technique — scope is deliberately one primary comparison plus one
  secondary reference point (see [04_Test_and_Evaluation_Plan.md](04_Test_and_Evaluation_Plan.md)).
- Not training on real bank/insurance customer documents at any point.

## 5. Success criteria / definition of done

The project is considered complete when all of the following exist:

- [ ] A reproducible dataset pipeline (source data → degraded tiers → train/val/test split)
- [ ] A trained/fine-tuned model checkpoint with logged training metrics
- [ ] A zero-shot baseline result on the same test set, same schema
- [ ] A field-level, degradation-tier-level evaluation report (table + charts)
- [ ] A cost/latency comparison table
- [ ] A written report (README/blog-style) covering problem → approach →
      results → limitations → future work
- [ ] All documentation in this doc set kept up to date with what was
      actually built (not just what was planned)

## 6. Users / audience

This is a portfolio artifact. The primary "users" are:

- **Hiring managers / interviewers** evaluating ML engineering judgment
- **Aashish**, as a reusable reference for applying the same technique to
  real (compliant) production data in a work context later

## 7. Key risks (see [05_Risk_Register.md](05_Risk_Register.md) for full detail)

- Domain mismatch: public data (bank statements) is not the same as the
  real target domain (EOB/insurance correspondence) — must be explicitly
  and honestly framed, not hidden.
- Hardware constraint: no local GPU; depends on free-tier or rented cloud
  compute, which bounds how much iteration is affordable.
- Donut's output length ceiling may block full-document extraction,
  requiring a scoping decision (header fields only) rather than a fix.

## 8. Open questions

- Donut vs. Qwen2-VL-7B as the primary model, or run both? (Recommendation:
  Donut first as the low-risk path; Qwen2-VL-7B as a stretch comparison if
  time/budget allow — see Technical Design Doc §5.)
- How much of the transaction-ledger extraction is worth attempting given
  Donut's decode-length limitation?
