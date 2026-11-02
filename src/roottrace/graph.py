"""Neural Granger sparse sensor-dependency graph (component-wise MLP, batched).

For each target i: x_i(t) = f_i(x(t-1..t-p)), first-layer weights W1[i,:,j,:]
grouped per input sensor j. Proximal group lasso drives whole groups to zero,
so A[i,j] = ||W1[i,:,j,:]||_F is a sparse Granger adjacency (j -> i).
"""
from __future__ import annotations
import numpy as np
import torch
torch.set_num_threads(3)
import torch.nn as nn


def make_lagged(x: np.ndarray, p: int):
    T = x.shape[0]
    X = np.stack([x[p - k - 1:T - k - 1] for k in range(p)], axis=-1)  # (T-p, d, p)
    return X.astype(np.float32), x[p:].astype(np.float32)


class NeuralGranger(nn.Module):
    def __init__(self, d: int, p: int, hidden: int = 8):
        super().__init__()
        self.d, self.p = d, p
        self.W1 = nn.Parameter(torch.randn(d, hidden, d, p) * 0.05)
        self.b1 = nn.Parameter(torch.zeros(d, hidden))
        self.W2 = nn.Parameter(torch.randn(d, hidden) * 0.1)
        self.b2 = nn.Parameter(torch.zeros(d))
        self.skip = nn.Parameter(torch.zeros(d))  # AR(1) self term

    def forward(self, X: torch.Tensor) -> torch.Tensor:  # X (B,d,p)
        h = torch.relu(torch.einsum("ihjk,bjk->bih", self.W1, X) + self.b1)
        return (h * self.W2).sum(-1) + self.b2 + self.skip * X[:, :, 0]

    @torch.no_grad()
    def prox(self, lam: float):
        n = self.W1.pow(2).sum(dim=(1, 3), keepdim=True).sqrt()
        self.W1.mul_(torch.clamp(1 - lam / (n + 1e-12), min=0))

    def adjacency(self) -> np.ndarray:
        """A[i, j]: strength of j -> i."""
        return self.W1.detach().pow(2).sum(dim=(1, 3)).sqrt().numpy()


def fit_graph(series: list[np.ndarray], p: int, lam: float, seed: int = 0,
              epochs: int = 150, lr: float = 3e-3) -> NeuralGranger:
    torch.manual_seed(seed); np.random.seed(seed)
    Xs, Ys = zip(*[make_lagged(s, p) for s in series])
    X = torch.from_numpy(np.concatenate(Xs)); Y = torch.from_numpy(np.concatenate(Ys))
    m = NeuralGranger(X.shape[1], p)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    n = len(X)
    for _ in range(epochs):
        perm = torch.randperm(n)
        for b in range(0, n, 256):
            idx = perm[b:b + 256]
            loss = ((m(X[idx]) - Y[idx]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step(); m.prox(lam * lr)
    return m


@torch.no_grad()
def residuals(m: NeuralGranger, x: np.ndarray) -> np.ndarray:
    X, Y = make_lagged(x, m.p)
    return Y - m(torch.from_numpy(X)).numpy()  # (T-p, d)
