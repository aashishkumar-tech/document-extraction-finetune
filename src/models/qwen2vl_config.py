"""QLoRA config for Qwen2-VL-2B fine-tuning (stretch goal)."""
from peft import LoraConfig

QWEN2VL_QLORA_CONFIG = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
)
