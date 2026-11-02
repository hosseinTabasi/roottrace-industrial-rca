"""Counterfactual intervention attribution + personalized PageRank root-cause ranking."""
from __future__ import annotations
import numpy as np
import torch
from .graph import NeuralGranger, make_lagged


@torch.no_grad()
def _predict(m: NeuralGranger, x: np.ndarray) -> np.ndarray:
    X, _ = make_lagged(x, m.p)
    return m(torch.from_numpy(X)).numpy()


def anomaly_score(m: NeuralGranger, x: np.ndarray, sigma: np.ndarray) -> float:
    X, Y = make_lagged(x, m.p)
    r = (Y - _predict(m, x)) / sigma
    return float((r ** 2).mean())


def counterfactual_attribution(m: NeuralGranger, x: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """delta_j = score(x) - score(x with sensor j replaced by its normal-predicted trajectory).

    x: window including p lags of context. The replaced trajectory is the
    model's one-step-ahead prediction, i.e. what a normal process would have produced.
    """
    p = m.p
    base = anomaly_score(m, x, sigma)
    pred = _predict(m, x)
    d = x.shape[1]
    delta = np.zeros(d)
    for j in range(d):
        xc = x.copy()
        xc[p:, j] = pred[:, j]
        delta[j] = base - anomaly_score(m, xc, sigma)
    return delta


def personalized_pagerank(A: np.ndarray, v: np.ndarray, restart: float = 0.3, iters: int = 100) -> np.ndarray:
    """Walk from effects to causes. A[i, j] = strength of j -> i.

    From node i the walker moves to parent j with prob A[i,j]/sum_j A[i,j];
    with prob `restart` it jumps back according to personalization v.
    """
    A = A.copy(); np.fill_diagonal(A, 0)
    rs = A.sum(1, keepdims=True)
    P = np.where(rs > 0, A / np.maximum(rs, 1e-12), 0.0)
    dangling = (rs[:, 0] == 0)
    v = np.clip(v, 0, None); v = v / v.sum() if v.sum() > 0 else np.ones_like(v) / len(v)
    pi = v.copy()
    for _ in range(iters):
        pi = restart * v + (1 - restart) * (pi @ P + pi[dangling].sum() * v)
    return pi


def rank(scores: np.ndarray) -> np.ndarray:
    return np.argsort(-scores)
