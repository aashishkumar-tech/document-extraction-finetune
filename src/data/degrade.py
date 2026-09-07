"""degrade(image, level) -> degraded_image  (clean/light/medium/heavy)"""
import cv2
import numpy as np
from PIL import Image

LEVELS = ["clean", "light", "medium", "heavy"]

PARAMS = {
    "clean":  {"blur": 0, "noise": 0.0,  "skew": 0,   "jpeg_q": 95, "scale": 1.0},
    "light":  {"blur": 1, "noise": 0.01, "skew": 1,   "jpeg_q": 70, "scale": 0.9},
    "medium": {"blur": 3, "noise": 0.03, "skew": 2,   "jpeg_q": 50, "scale": 0.7},
    "heavy":  {"blur": 5, "noise": 0.06, "skew": 4,   "jpeg_q": 30, "scale": 0.5},
}


def degrade(image: Image.Image, level: str) -> Image.Image:
    assert level in LEVELS, f"level must be one of {LEVELS}"
    p = PARAMS[level]
    img = np.array(image.convert("RGB"))

    if p["scale"] != 1.0:
        h, w = img.shape[:2]
        small = cv2.resize(img, (int(w * p["scale"]), int(h * p["scale"])))
        img = cv2.resize(small, (w, h))

    if p["blur"] > 0:
        k = p["blur"] * 2 + 1
        img = cv2.GaussianBlur(img, (k, k), 0)

    if p["noise"] > 0:
        noise = np.random.normal(0, p["noise"] * 255, img.shape).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    if p["skew"] > 0:
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), p["skew"], 1.0)
        img = cv2.warpAffine(img, M, (w, h), borderValue=(255, 255, 255))

    ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, p["jpeg_q"]])
    img = cv2.imdecode(enc, cv2.IMREAD_COLOR)

    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
