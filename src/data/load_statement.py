"""load_statement(pdf_path, json_path) -> (raw_image, ground_truth_dict)"""
import json
from pdf2image import convert_from_path


def load_statement(pdf_path: str, json_path: str, dpi: int = 150):
    pages = convert_from_path(pdf_path, dpi=dpi)
    raw_image = pages[0]  # first page; extend if multi-page
    with open(json_path, "r") as f:
        ground_truth_dict = json.load(f)
    return raw_image, ground_truth_dict


if __name__ == "__main__":
    # smoke test placeholder — point at a real downloaded sample
    image, gt = load_statement(
        "data/raw/train/India_Bank_Statement_Scanned_Type1/00001.pdf",
        "data/raw/train/India_Bank_Statement_Scanned_Type1/00001.json",
    )
    print("Image size:", image.size)
    print("Ground truth keys:", list(gt.keys()))
