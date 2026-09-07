# Fine-Tuned Vision-Language Model for Scanned Document Extraction

Testing whether a small fine-tuned VLM can beat an OCR+rules pipeline and
a zero-shot large VLM on structured extraction from degraded scanned
documents, at lower cost.

See `docs/` for full project documentation (PRD, Technical Design Doc,
Data & Ethics, Test Plan, Risk Register, Project Plan).

## Quickstart

```bash
pip install -r requirements.txt
python src/data/download.py
python src/data/load_statement.py   # smoke test on a few samples
```

## Status

Planning complete — execution in progress. See `docs/06_Project_Plan.md`
for milestones.
