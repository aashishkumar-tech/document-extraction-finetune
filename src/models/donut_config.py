"""LoRA config for Donut fine-tuning."""
from peft import LoraConfig

DONUT_LORA_CONFIG = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["query", "value"],
    lora_dropout=0.05,
    bias="none",
)
