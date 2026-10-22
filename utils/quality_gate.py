from dataclasses import dataclass, field

import cv2
import numpy as np

from utils import config


@dataclass
class QualityResult:
    passed: bool
    reasons: list = field(default_factory=list)   
    metrics: dict = field(default_factory=dict)


def decode_image(file_bytes: bytes):
    arr = np.frombuffer(file_bytes, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def blur_score(bgr) -> float:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    scale = 512 / max(h, w)
    if scale < 1:
        gray = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def lighting_stats(bgr) -> dict:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return {
        "mean_brightness": float(gray.mean()),
        "dark_fraction": float((gray < 20).mean()),
        "bright_fraction": float((gray > 245).mean()),
    }


def skin_fraction(bgr) -> float:
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    mask = cv2.inRange(ycrcb, (0, 130, 77), (255, 175, 127))
    return float(mask.mean() / 255.0)


def check_quality(bgr) -> QualityResult:
    if bgr is None:
        return QualityResult(False, ["We couldn't read that file. Please upload a JPG or PNG."])

    h, w = bgr.shape[:2]
    blur = blur_score(bgr)
    light = lighting_stats(bgr)
    skin = skin_fraction(bgr)
    metrics = {"width": w, "height": h, "blur_score": round(blur, 1),
               "skin_fraction": round(skin, 3), **{k: round(v, 3) for k, v in light.items()}}

    reasons = []
    if min(h, w) < config.MIN_SIDE_PX:
        reasons.append(f"Image is too small ({w}x{h}). Use a photo at least {config.MIN_SIDE_PX}px on the short side.")
    if blur < config.BLUR_THRESHOLD:
        reasons.append("Image looks blurry. Hold the camera steady and tap to focus, then retake.")
    if light["mean_brightness"] < config.DARK_MEAN or light["dark_fraction"] > config.MAX_CLIPPED_DARK_FRAC:
        reasons.append("Image is too dark. Retake in bright, even, natural light.")
    if light["mean_brightness"] > config.BRIGHT_MEAN or light["bright_fraction"] > config.MAX_CLIPPED_BRIGHT_FRAC:
        reasons.append("Image is overexposed or has strong glare. Avoid flash and direct sunlight.")
    if skin < config.MIN_SKIN_FRACTION:
        reasons.append("We couldn't find much skin in this photo. Move closer to the area of concern.")

    return QualityResult(passed=not reasons, reasons=reasons, metrics=metrics)
