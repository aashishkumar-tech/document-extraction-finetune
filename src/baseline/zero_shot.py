"""Zero-shot large-VLM baseline via Claude API."""
import base64
import anthropic

SCHEMA_PROMPT = """Extract the following fields from this bank statement image as JSON:
account_number, ifsc_code, opening_balance, closing_balance,
and a list of transactions with date, description, debit, credit, balance.
If a field is unreadable, use null. Return ONLY valid JSON, no other text."""


def extract_zero_shot(image_path: str, model: str = "claude-sonnet-4-6") -> str:
    client = anthropic.Anthropic()
    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode("utf-8")

    response = client.messages.create(
        model=model,
        max_tokens=2000,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": image_data}},
                {"type": "text", "text": SCHEMA_PROMPT},
            ],
        }],
    )
    return response.content[0].text
