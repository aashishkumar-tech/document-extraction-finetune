# Risk Register

**Project:** Fine-Tuned Vision-Language Model for Scanned Document Extraction
**Status:** Draft v1.0
**Last updated:** 2026-09-08

---

| # | Risk | Likelihood | Impact | Mitigation | Owner |
|---|---|---|---|---|---|
| R1 | Domain mismatch (bank statements vs. real EOB/insurance docs) undermines credibility of results | High | Medium | Explicitly disclosed in every doc/report (see [03_Data_and_Ethics.md](03_Data_and_Ethics.md)); framed as methodology validation, not domain-specific performance claim | Aashish |
| R2 | Donut's ~768–1024 token decode limit truncates long transaction tables | High | Medium | Scope v1 to header-field extraction only; document table extraction as future work rather than forcing a broken fix | Aashish |
| R3 | No local GPU — dependent on free-tier Colab availability/session limits | Medium | Medium | Keep Donut as primary path (fits free tier comfortably); budget $10–30 for A100 rental as a funded fallback/stretch path | Aashish |
| R4 | Cloud GPU costs exceed budget due to debugging/re-runs | Medium | Low | Smoke-test on tiny subset (10–20 examples) before full runs; hard budget cap; stop instances immediately after each phase | Aashish |
| R5 | Synthetic degradation doesn't realistically match real-world scan artifacts, weakening the "recovers real-world accuracy" claim | Medium | Medium | Calibrate degradation visually against FUNSD (real scanned forms); consider Augraphy for more realistic printer/fax/scanner effects | Aashish |
| R6 | Project scope creeps into a sprawling multi-model comparison, diluting the story and inviting unfocused critique | Medium | Medium | Enforce PRD non-goals and the single-headline-comparison rule in [04_Test_and_Evaluation_Plan.md](04_Test_and_Evaluation_Plan.md); any extra experiment is explicitly labeled "bonus" | Aashish |
| R7 | Accidental inclusion of real PHI/PII (e.g., from work documents viewed for reference) into the training set or repo | Low | Critical | Hard rule in [03_Data_and_Ethics.md](03_Data_and_Ethics.md): no real document content is ever stored/retained/used, even redacted; enforced by only ever recreating structure synthetically | Aashish |
| R8 | Fine-tuned model underperforms zero-shot baseline, undermining the core hypothesis | Medium | Low | Treat as a valid, reportable finding (see Test Plan §7) rather than a project failure; write up the "why" as the interesting result | Aashish |
| R9 | HuggingFace dataset preview/schema quirks (already observed) cause silent data-loading errors during training | Low | Medium | Manually inspect a sample record (image + JSON) after loading, before committing to a full training run | Aashish |
| R10 | Public sharing of the project inadvertently implies real production accuracy numbers from the employer's actual pipeline | Low | High | Clearly label the ~10% OCR+rules figure as a "reported reference figure" from prior context, not a number this project reproduces or claims ownership of; avoid naming the employer in any public write-up | Aashish |

## Review cadence

This register should be revisited at the end of each project phase (data
prep, training, evaluation, write-up) — not just once at the start — since
new risks (e.g., a specific training bug, a specific dataset gap) tend to
surface only once work begins.
