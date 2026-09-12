"""degrade(image, level) -> degraded_image  (clean/light/medium/heavy)

Degradation strength scales with image size (blur kernel, warp jitter),
so a 5000px scanned statement gets proportionally as rough as a small
test image - fixed pixel-count params silently under-degrade large scans.
"""
import cv2
import numpy as np
from PIL import Image

LEVELS = ["clean", "light", "medium", "heavy"]

# blur_frac / warp_frac are fractions of the image's shorter side.
# scale is the downscale-then-upscale factor (the biggest driver of
# real unrecoverable detail loss - this matters more than blur).
PARAMS = {
    "clean":  {"blur_frac": 0.000, "noise": 0.00, "skew": 0, "jpeg_q": 95, "scale": 1.00, "warp_frac": 0.000, "vignette": 0.00, "jpeg_passes": 1},
    "light":  {"blur_frac": 0.003, "noise": 0.04, "skew": 2, "jpeg_q": 45, "scale": 0.35, "warp_frac": 0.012, "vignette": 0.20, "jpeg_passes": 2},
    "medium": {"blur_frac": 0.007, "noise": 0.08, "skew": 4, "jpeg_q": 20, "scale": 0.15, "warp_frac": 0.025, "vignette": 0.35, "jpeg_passes": 3},
    "heavy":  {"blur_frac": 0.012, "noise": 0.14, "skew": 8, "jpeg_q": 8,  "scale": 0.07, "warp_frac": 0.045, "vignette": 0.55, "jpeg_passes": 4},
}


def _odd(n: int) -> int:
    n = max(1, int(n))
    return n if n % 2 == 1 else n + 1


def _apply_perspective_warp(img, frac):
    if frac <= 0:
        return img
    h, w = img.shape[:2]
    jitter = frac * min(h, w)
    src = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    dst = np.float32([
        [np.random.uniform(0, jitter), np.random.uniform(0, jitter)],
        [w - np.random.uniform(0, jitter), np.random.uniform(0, jitter)],
        [np.random.uniform(0, jitter), h - np.random.uniform(0, jitter)],
        [w - np.random.uniform(0, jitter), h - np.random.uniform(0, jitter)],
    ])
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, M, (w, h), borderValue=(255, 255, 255))


def _apply_vignette(img, strength):
    if strength <= 0:
        return img
    h, w = img.shape[:2]
    kernel_x = cv2.getGaussianKernel(w, w * (1.0 - strength * 0.5))
    kernel_y = cv2.getGaussianKernel(h, h * (1.0 - strength * 0.5))
    mask = kernel_y @ kernel_x.T
    mask = mask / mask.max()
    mask = np.clip(mask * strength + (1 - strength), 0, 1)
    img = img.astype(np.float32)
    for c in range(3):
        img[:, :, c] *= mask
    return np.clip(img, 0, 255).astype(np.uint8)


def _apply_salt_pepper(img, amount):
    if amount <= 0:
        return img
    out = img.copy()
    h, w = img.shape[:2]
    num_pixels = int(amount * h * w * 0.3)
    ys = np.random.randint(0, h, num_pixels)
    xs = np.random.randint(0, w, num_pixels)
    salt_mask = np.random.rand(num_pixels) > 0.5
    out[ys[salt_mask], xs[salt_mask]] = 255
    out[ys[~salt_mask], xs[~salt_mask]] = 0
    return out


def degrade(image: Image.Image, level: str) -> Image.Image:
    assert level in LEVELS, f"level must be one of {LEVELS}"
    p = PARAMS[level]
    img = np.array(image.convert("RGB"))
    h0, w0 = img.shape[:2]
    short_side = min(h0, w0)

    # 1. Resolution loss - the dominant degradation factor. Downscaling a
    #    5000px image to 15% of its size (heavy) and back genuinely destroys
    #    fine detail in a way blur alone can't fake.
    if p["scale"] != 1.0:
        small_w, small_h = max(1, int(w0 * p["scale"])), max(1, int(h0 * p["scale"]))
        small = cv2.resize(img, (small_w, small_h), interpolation=cv2.INTER_AREA)
        img = cv2.resize(small, (w0, h0), interpolation=cv2.INTER_LINEAR)

    # 2. Blur - kernel size relative to image resolution, not a fixed pixel count
    if p["blur_frac"] > 0:
        k = _odd(short_side * p["blur_frac"])
        img = cv2.GaussianBlur(img, (k, k), 0)

    # 3. Perspective warp - relative jitter, so it's visible at any resolution
    img = _apply_perspective_warp(img, p["warp_frac"])

    # 4. Rotation/skew
    if p["skew"] > 0:
        h, w = img.shape[:2]
        angle = np.random.uniform(-p["skew"], p["skew"])
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        img = cv2.warpAffine(img, M, (w, h), borderValue=(255, 255, 255))

    # 5. Vignette / uneven lighting
    img = _apply_vignette(img, p["vignette"])

    # 6. Gaussian noise
    if p["noise"] > 0:
        noise = np.random.normal(0, p["noise"] * 255, img.shape).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # 7. Salt-and-pepper speckle
    img = _apply_salt_pepper(img, p["noise"])

    # 8. Heavy JPEG re-compression, applied multiple times (compounding
    #    artifacts - this is what real multi-generation photocopies/faxes do)
    for _ in range(p.get("jpeg_passes", 1)):
        ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, p["jpeg_q"]])
        img = cv2.imdecode(enc, cv2.IMREAD_COLOR)

    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))