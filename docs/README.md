# Fine-Tuned Vision-Language Model for Scanned Document Extraction

**Status:** Planning complete — pre-execution
**Owner:** Aashish
**Last updated:** 2026-09-08

This folder contains the full project documentation, organized the way a
production ML team would structure it. Read in this order:

| Doc | Purpose |
|---|---|
| [01_PRD.md](01_PRD.md) | Why this project exists, what problem it solves, what "done" looks like |
| [02_Technical_Design_Doc.md](02_Technical_Design_Doc.md) | Architecture, data flow, model choices, HLD + LLD |
| [03_Data_and_Ethics.md](03_Data_and_Ethics.md) | Where data comes from, why it's safe to use, licensing, PHI/PII handling |
| [04_Test_and_Evaluation_Plan.md](04_Test_and_Evaluation_Plan.md) | How success is measured, metrics, baselines, acceptance criteria |
| [05_Risk_Register.md](05_Risk_Register.md) | What could go wrong, likelihood/impact, mitigation |
| [06_Project_Plan.md](06_Project_Plan.md) | Milestones, timeline, budget, definition of done |

## One-line summary

Testing whether a small vision-language model, fine-tuned on domain-specific
scanned documents, can match or beat both a traditional OCR+rules pipeline
and a zero-shot large VLM — at a fraction of the cost — for structured field
extraction from degraded scanned documents.

## Quick links

- Primary dataset: [AgamiAI/Indian-Bank-Statements](https://huggingface.co/datasets/AgamiAI/Indian-Bank-Statements) (Apache 2.0)
- Calibration dataset: FUNSD (real scanned forms)
- Degradation engine: Augraphy / custom OpenCV pipeline
- Candidate models: Donut-base (~200M), Qwen2-VL-7B (QLoRA)
- Compute: Free Colab T4 (Donut) or rented A100 (Qwen2-VL-7B)
