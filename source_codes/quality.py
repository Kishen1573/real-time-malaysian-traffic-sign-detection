# quality.py
import cv2
import numpy as np

def var_laplacian(gray: np.ndarray) -> float:
    """Sharpness proxy: variance of Laplacian (higher = sharper)."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

def glare_ratio(bgr: np.ndarray, v_hi: float = 0.95) -> float:
    """Fraction of pixels that are near-saturated in value channel (glare proxy)."""
    if bgr is None or bgr.size == 0:
        return 0.0
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2].astype(np.float32) / 255.0
    return float((v > v_hi).mean())

def _normalize_0_100(x: float, lo: float, hi: float) -> float:
    """Map x in [lo, hi] to [0, 100] with clipping."""
    if hi <= lo:
        return 0.0
    x = float(x)
    if np.isnan(x) or np.isinf(x):
        return 0.0
    x = max(lo, min(hi, x))
    return 100.0 * (x - lo) / (hi - lo)

def compute_visibility_metrics(bgr_patch: np.ndarray) -> dict:
    """
    Compute blur, contrast, glare, and a composite visibility score in [0..100].
    Low-visibility rule is consistent with the dashboard: vis < 40 OR blur < 100.
    """
    if bgr_patch is None or bgr_patch.size == 0:
        return {
            "blur_score": 0.0,
            "contrast_score": 0.0,
            "glare_ratio": 0.0,
            "visibility_score": 0.0,
            "low_visibility": True
        }

    gray = cv2.cvtColor(bgr_patch, cv2.COLOR_BGR2GRAY)

    # Raw metrics
    blur = var_laplacian(gray)                 # ~0..400+ depending on camera
    contrast = float(gray.std())               # ~0..60+ (stddev of intensity)
    glare = glare_ratio(bgr_patch, v_hi=0.95)  # 0..1

    # Softer normalization: start from 0 instead of hard floors.
    # (Old code used lo=50 for blur and lo=10 for contrast which zeroed tiny values.)
    sharp_n = _normalize_0_100(blur,     lo=0.0,  hi=400.0)
    contr_n = _normalize_0_100(contrast, lo=0.0,  hi=60.0)
    glare_n = _normalize_0_100(glare,    lo=0.0,  hi=0.30)  # higher glare = worse

    # Weighted visibility (still 0..100-ish)
    visibility = 0.5 * sharp_n + 0.4 * contr_n - 0.1 * glare_n
    visibility = max(0.0, min(100.0, visibility))

    low_vis = (visibility < 40.0) or (blur < 100.0)

    return {
        "blur_score": round(blur, 2),
        "contrast_score": round(contrast, 2),
        "glare_ratio": round(glare, 3),
        "visibility_score": round(visibility, 1),
        "low_visibility": bool(low_vis)
    }
