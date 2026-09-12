# Data Sourcing, Licensing & Ethics

**Project:** Fine-Tuned Vision-Language Model for Scanned Document Extraction
**Status:** Draft v1.0
**Last updated:** 2026-09-08

---

## 1. Why this document exists

The real-world motivating problem (extracting data from bank/insurance
correspondence, EOB letters) involves documents that contain PHI and PII.
This document exists to make explicit, in writing, exactly what data is
and is not used in this project, and why that's safe.

## 2. Hard rule

**No real customer, patient, or employer document — redacted or
otherwise — is used at any stage of data collection, training,
evaluation, or publication for this project.**

This includes documents originally viewed for structural/layout reference
only (see §4). Anything used for layout inspiration was viewed and then
recreated synthetically; no original file or extracted content from it is
stored, retained, or included in the project repository.

## 3. Datasets used

| Dataset | Source | License | Role | PHI/PII risk |
|---|---|---|---|---|
| AgamiAI Indian Bank Statements | HuggingFace | Apache 2.0 | Primary training data — clean PDFs + JSON labels | None — fully synthetic |
| FUNSD | Public academic release | Research use | Real-scan calibration reference / generic eval | None — public academic dataset of non-sensitive forms |
| Augraphy / ShabbyPages | Open-source library / dataset | Open source | Degradation engine / distortion reference | None |
| PureDocBench (considered, optional) | HuggingFace | CC BY 4.0 | Optional generic robustness reference | None — explicitly synthetic, dataset card prohibits treating content as factual records of real people/institutions |

## 4. Domain-substitution disclosure

The real target domain is bank/insurance correspondence (EOB letters, UB-04
claim forms, narrative adjuster letters). Public, non-restricted datasets
matching this exact domain **do not exist**, because of the same privacy
constraints this project is designed to respect.

**Approach:** structurally similar public data (bank statements, with the
same category of extraction problem — dense tabular/field-based documents
subject to scan degradation) is used as a **prototype stand-in**. This
substitution is stated plainly in every relevant report/write-up produced
by this project — it is not something to gloss over.

Wording to reuse in the final report:

> "Real-world target documents (bank/insurance correspondence, EOBs)
> contain PHI/PII and cannot be used in a public project. This project
> uses publicly available, structurally similar documents (bank statement
> data with real scan-style degradation applied) as a prototype stand-in,
> explicitly noted as such."

## 5. Data retention and handling

- All working data (downloaded datasets, generated degraded images,
  intermediate files) stays local or in the researcher's own cloud
  storage/compute account — never uploaded to a third-party chat tool or
  shared publicly beyond what's needed for the portfolio write-up.
- No dataset containing real names, account numbers, or identifiers tied
  to real individuals is ever introduced into the pipeline.
- If any real work document is viewed for structural reference in the
  future, it will be treated the same way it was during planning: viewed
  for layout/structure only, never quoted, retained, stored, or used to
  auto-label synthetic data.

## 6. Licensing compliance checklist

- [x] AgamiAI dataset — Apache 2.0, permits training/derivative use, no
      attribution blocker identified
- [x] FUNSD — used within its research/academic terms, not redistributed
- [x] Augraphy — open-source (MIT), safe for use as a processing library
- [ ] Confirm exact FUNSD license terms before any public redistribution
      of derived FUNSD-based examples (if the project repo becomes public)

## 7. Ethical framing for the write-up

This section exists so the "why not just use real data" question has a
ready, honest answer in an interview:

- Real EOB/bank documents are HIPAA/financial-compliance-restricted —
  using them in a personal project, even redacted, is not appropriate
  without explicit employer authorization, which was not sought or given.
- The chosen substitution (synthetic-schema bank statements + custom
  degradation) is a standard technique in document-AI research precisely
  because production-realistic labeled data in regulated domains is
  usually inaccessible publicly.
- Results generalize as a **methodology validation**, not as a claim about
  performance on the real production documents — that distinction should
  be stated explicitly in the final report's limitations section.
