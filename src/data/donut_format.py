"""Convert ground truth JSON <-> Donut's tagged token sequence format.

Donut doesn't generate JSON directly - it's trained to generate a flat
sequence of XML-style tags, e.g.:
    <s_account_number>456860100091</s_account_number><s_bank_name>...</s_bank_name>

Scope decision (per Technical Design Doc SS4 and confirmed by the zero-shot
baseline's own field scope): this project trains Donut on HEADER FIELDS
ONLY. The `transactions` list is excluded because:
  1. Donut has a ~768-1024 token decode limit; a full transaction ledger
     (100+ rows in some AgamiAI statements) would be truncated anyway.
  2. The zero-shot baseline itself was scored on header fields only, so
     keeping Donut's scope identical keeps the comparison fair and
     apples-to-apples.
Transaction-table extraction remains documented future work, not silently
dropped (see docs/02_Technical_Design_Doc.md SS4).

Usage as a library:
    from src.data.donut_format import encode_ground_truth, decode_donut_output
"""
import json
import re

# Same field list the scorer evaluates - keeping these identical is what
# makes the zero-shot baseline and the fine-tuned model directly comparable.
HEADER_FIELDS = [
    "account_number", "ifsc_code", "micr_code", "bank_name", "account_holder",
    "opening_balance", "closing_balance", "start_date", "end_date", "statement_date",
]

TASK_START_TOKEN = "<s_extraction>"
TASK_END_TOKEN = "</s_extraction>"


def _sanitize_value(value) -> str:
    """Escape characters that would break tag parsing if they appeared in a
    field value (e.g. a stray '<' or '>' in an OCR'd/synthetic field)."""
    if value is None:
        return ""
    s = str(value)
    s = s.replace("<", "&lt;").replace(">", "&gt;")
    return s


def _unsanitize_value(value: str) -> str:
    return value.replace("&lt;", "<").replace("&gt;", ">")


def encode_ground_truth(ground_truth: dict, fields: list = None) -> str:
    """dict -> Donut tagged sequence string, header fields only by default.

    Missing/null fields are encoded as empty tags (<s_field></s_field>),
    not omitted - this teaches the model to explicitly signal "not present"
    rather than never learning the tag exists at all.
    """
    fields = fields or HEADER_FIELDS
    parts = [TASK_START_TOKEN]
    for field in fields:
        value = _sanitize_value(ground_truth.get(field))
        parts.append(f"<s_{field}>{value}</s_{field}>")
    parts.append(TASK_END_TOKEN)
    return "".join(parts)


def decode_donut_output(sequence: str, fields: list = None) -> dict:
    """Donut tagged sequence string -> dict. Inverse of encode_ground_truth.

    Used both to sanity-check encoding round-trips, and later to parse the
    fine-tuned model's actual generated output during evaluation.
    Missing/malformed tags decode to None for that field, rather than
    raising - a partially-broken generation should still yield partial
    scoreable output.
    """
    fields = fields or HEADER_FIELDS
    result = {}
    for field in fields:
        pattern = rf"<s_{field}>(.*?)</s_{field}>"
        match = re.search(pattern, sequence, re.DOTALL)
        if match:
            value = match.group(1).strip()
            result[field] = _unsanitize_value(value) if value != "" else None
        else:
            result[field] = None
    return result


def encode_ground_truth_from_file(label_path: str, fields: list = None) -> str:
    with open(label_path, "r") as f:
        ground_truth = json.load(f)
    return encode_ground_truth(ground_truth, fields)