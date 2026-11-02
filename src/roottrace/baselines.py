"""Baselines: PCA T2/SPE contributions, linear Granger graph, detection statistics."""
from __future__ import annotations
import numpy as np
from sklearn.decomposition import PCA
from scipy import stats


class PCAMonitor:
    def fit(self, z: np.ndarray, var: float = 0.9) -> "PCAMonitor":
        self.pca = PCA(n_components=var, svd_solver="full").fit(z)
        self.P = self.pca.components_.T; self.lam = self.pca.explained_variance_
        return self

    def t2(self, z): t = z @ self.P; return ((t ** 2) / self.lam).sum(1)
    def spe(self, z): e = z - z @ self.P @ self.P.T; return (e ** 2).sum(1)

    def t2_contrib(self, z):
        t = z @ self.P  # (T,k)
        return np.abs(((t / self.lam) @ self.P.T) * z).mean(0)

    def spe_contrib(self, z):
        e = z - z @ self.P @ self.P.T
        return (e ** 2).mean(0)


def linear_granger_adjacency(series: list[np.ndarray], p: int, alpha: float = 0.05) -> np.ndarray:
    """Pairwise-conditional Granger via a full VAR(p) OLS; A[i,j] = 1-pvalue of F-test j->i (0 if p>alpha)."""
    from .graph import make_lagged
    Xs, Ys = zip(*[make_lagged(s, p) for s in series])
    X = np.concatenate(Xs); Y = np.concatenate(Ys)
    n, d, _ = X.shape
    Xf = np.concatenate([X.reshape(n, -1), np.ones((n, 1))], 1)
    A = np.zeros((d, d))
    lam = 1e-3 * np.eye(Xf.shape[1])
    G = np.linalg.inv(Xf.T @ Xf + lam)
    B = G @ Xf.T @ Y
    R = Y - Xf @ B
    rss_full = (R ** 2).sum(0)
    df2 = n - Xf.shape[1]
    for j in range(d):
        keep = np.ones(Xf.shape[1], bool); keep[j * p:(j + 1) * p] = False
        Xr = Xf[:, keep]
        Br = np.linalg.lstsq(Xr, Y, rcond=None)[0]
        rss_r = ((Y - Xr @ Br) ** 2).sum(0)
        F = ((rss_r - rss_full) / p) / (rss_full / df2)
        pv = 1 - stats.f.cdf(F, p, df2)
        A[:, j] = np.where(pv < alpha, 1 - pv, 0.0)
    return A
