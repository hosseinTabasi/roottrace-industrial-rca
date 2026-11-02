from __future__ import annotations
import numpy as np


def topk_hit(ranking: np.ndarray, truth: set, k: int) -> float:
    return float(any(r in truth for r in ranking[:k]))


def reciprocal_rank(ranking: np.ndarray, truth: set) -> float:
    for pos, r in enumerate(ranking):
        if r in truth:
            return 1.0 / (pos + 1)
    return 0.0


def detection_delay(score: np.ndarray, thr: float, onset: int, consec: int = 3) -> float:
    above = score > thr
    for t in range(onset, len(score) - consec + 1):
        if above[t:t + consec].all():
            return float(t - onset)
    return float("nan")


def jaccard(a: set, b: set) -> float:
    return len(a & b) / max(len(a | b), 1)
