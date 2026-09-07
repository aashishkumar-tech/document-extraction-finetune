"""Download a small batch of AgamiAI Indian-Bank-Statements samples for smoke testing."""
from huggingface_hub import hf_hub_download

REPO = "AgamiAI/Indian-Bank-Statements"
FOLDERS = [
    "India_Bank_Statement_Scanned_Type1",
    "India_Bank_Statement_Scanned_Type2",
    "India_Bank_Statement_Digital_Type1",
    "India_Bank_Statement_Digital_Type2",
]
N_SAMPLES = 5  # increase once smoke test passes

if __name__ == "__main__":
    for folder in FOLDERS:
        for i in range(1, N_SAMPLES + 1):
            idx = f"{i:05d}"
            for ext in ["pdf", "json"]:
                path = hf_hub_download(
                    repo_id=REPO,
                    repo_type="dataset",
                    filename=f"train/{folder}/{idx}.{ext}",
                    local_dir="data/raw",
                )
                print(f"Downloaded: {path}")
