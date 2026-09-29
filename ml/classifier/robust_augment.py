"""Image corruptions for robustness training and for the robustness benchmark.

TRAIN_CORRUPTIONS are used as training augmentation (`train_yolo.py --robust-aug P`).
HELDOUT_CORRUPTIONS are NEVER used in training; `evaluate_robustness.py` reports them separately, so
"robust to corruptions it has seen" and "robust to corruptions it has not seen" are two numbers.

Every corruption takes (PIL image, severity in [0, 1], random.Random) and returns a PIL RGB image.
"""
from __future__ import annotations

import io
import random

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


def _rgb(im: Image.Image) -> Image.Image:
    return im if im.mode == "RGB" else im.convert("RGB")


# --------------------------------------------------------------------------- #
# Seen in training
# --------------------------------------------------------------------------- #
def gaussian_blur(im, s, rng):
    return im.filter(ImageFilter.GaussianBlur(radius=0.4 + 3.0 * s * max(im.size) / 320))


def gaussian_noise(im, s, rng):
    a = np.asarray(im, dtype=np.float32)
    sigma = 4 + 26 * s
    g = np.random.default_rng(rng.randrange(2**31)).normal(0, sigma, a.shape)
    return Image.fromarray(np.clip(a + g, 0, 255).astype(np.uint8))


def jpeg(im, s, rng):
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=int(round(55 - 47 * s)))       # 55 .. 8
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def low_resolution(im, s, rng):
    w, h = im.size
    f = 0.6 - 0.48 * s                                         # 0.6 .. 0.12 of the size
    small = im.resize((max(8, int(w * f)), max(8, int(h * f))), Image.BILINEAR)
    return small.resize((w, h), Image.BILINEAR)


def exposure(im, s, rng):
    """Too dark (most of the time) or too bright, plus lower contrast."""
    dark = rng.random() < 0.7
    b = (1 - 0.65 * s) if dark else (1 + 0.8 * s)
    im = ImageEnhance.Brightness(im).enhance(b)
    return ImageEnhance.Contrast(im).enhance(1 - 0.5 * s)


TRAIN_CORRUPTIONS = {"gaussian_blur": gaussian_blur, "gaussian_noise": gaussian_noise, "jpeg": jpeg,
                     "low_resolution": low_resolution, "exposure": exposure}


# --------------------------------------------------------------------------- #
# Held out: never used in training
# --------------------------------------------------------------------------- #
def motion_blur(im, s, rng):
    k = max(3, int((5 + 22 * s) * max(im.size) / 320)) | 1
    ker = np.zeros((k, k), np.float32)
    ker[k // 2, :] = 1.0 / k
    m = cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), rng.uniform(0, 180), 1.0)
    ker = cv2.warpAffine(ker, m, (k, k))
    ker /= max(ker.sum(), 1e-6)
    return Image.fromarray(cv2.filter2D(np.asarray(im), -1, ker))


def haze(im, s, rng):
    a = np.asarray(im, dtype=np.float32)
    fog = 200 + 40 * rng.random()
    t = 1 - 0.65 * s
    return Image.fromarray(np.clip(a * t + fog * (1 - t), 0, 255).astype(np.uint8))


def salt_pepper(im, s, rng):
    a = np.asarray(im).copy()
    g = np.random.default_rng(rng.randrange(2**31))
    m = g.random(a.shape[:2])
    p = 0.01 + 0.09 * s
    a[m < p / 2] = 0
    a[m > 1 - p / 2] = 255
    return Image.fromarray(a)


def colour_cast(im, s, rng):
    a = np.asarray(im, dtype=np.float32)
    gains = 1 + (np.array([rng.uniform(-1, 1) for _ in range(3)]) * 0.45 * s)
    return Image.fromarray(np.clip(a * gains, 0, 255).astype(np.uint8))


HELDOUT_CORRUPTIONS = {"motion_blur": motion_blur, "haze": haze, "salt_pepper": salt_pepper,
                       "colour_cast": colour_cast}


# --------------------------------------------------------------------------- #
# Geometric: partial views (benchmark only; training uses a wider RandomResizedCrop instead)
# --------------------------------------------------------------------------- #
def partial_crop(im, s, rng):
    """Keep a random window with (1 - 0.7*s) of the area: 100% .. 30%."""
    w, h = im.size
    area = 1 - 0.7 * s
    side = np.sqrt(area)
    cw, ch = max(8, int(w * side)), max(8, int(h * side))
    x, y = rng.randint(0, w - cw), rng.randint(0, h - ch)
    return im.crop((x, y, x + cw, y + ch))


def occlusion(im, s, rng, patch: Image.Image | None = None):
    """Cover (10% + 30%*s) of the image with a block: a patch of ANOTHER image if given (clutter /
    overlapping objects), else a flat grey block."""
    w, h = im.size
    frac = 0.10 + 0.30 * s
    pw, ph = int(w * np.sqrt(frac)), int(h * np.sqrt(frac))
    x, y = rng.randint(0, w - pw), rng.randint(0, h - ph)
    im = im.copy()
    block = patch.resize((pw, ph)) if patch is not None else Image.new("RGB", (pw, ph), (128, 128, 128))
    im.paste(block, (x, y))
    return im


GEOMETRIC = {"partial_crop": partial_crop, "occlusion": occlusion}


# --------------------------------------------------------------------------- #
# Training transform
# --------------------------------------------------------------------------- #
class RandomCorruption:
    """With probability p apply 1-2 random TRAIN_CORRUPTIONS at a random severity. Picklable (used in
    DataLoader workers)."""

    def __init__(self, p: float = 0.5, max_severity: float = 1.0):
        self.p, self.max_severity = p, max_severity
        self.names = sorted(TRAIN_CORRUPTIONS)

    def __call__(self, im: Image.Image) -> Image.Image:
        if random.random() >= self.p:
            return im
        im = _rgb(im)
        rng = random.Random(random.getrandbits(32))
        for name in rng.sample(self.names, 1 if rng.random() < 0.7 else 2):
            im = _rgb(TRAIN_CORRUPTIONS[name](im, rng.uniform(0.1, self.max_severity), rng))
        return im

    def __repr__(self):
        return f"RandomCorruption(p={self.p}, max_severity={self.max_severity}, {self.names})"


def install(p: float) -> None:
    """Patch ultralytics' classification training transforms so RandomCorruption runs right after the
    random crop/flip (before colour jitter / RandAugment / ToTensor). Validation transforms are not
    touched (they don't go through classify_augmentations)."""
    import torchvision.transforms as T
    from ultralytics.data import dataset as ds

    original = ds.classify_augmentations
    if getattr(original, "_mm_robust", False):
        return

    def patched(*args, **kwargs):
        comp = original(*args, **kwargs)
        tf = list(comp.transforms)
        i = next(k for k, t in enumerate(tf) if isinstance(t, T.ToTensor))
        j = next((k for k, t in enumerate(tf)
                  if isinstance(t, (T.ColorJitter, T.RandAugment, T.AugMix, T.AutoAugment))), i)
        tf.insert(min(i, j), RandomCorruption(p))
        return T.Compose(tf)

    patched._mm_robust = True
    ds.classify_augmentations = patched
