"""Tennessee Eastman Process (Braatz d00..d21) loaders."""
from __future__ import annotations
import os
from pathlib import Path
import numpy as np

DEFAULT_DIR = os.environ.get("TEP_DIR", "/workspace/industrial-ts-data/TEP/braatz")
FAULT_ONSET = 160  # test files: fault introduced after sample 160

VAR_NAMES = [f"XMEAS({i})" for i in range(1, 42)] + [f"XMV({i})" for i in range(1, 12)]


def _read(path: Path) -> np.ndarray:
    return np.loadtxt(path)


def load_train(fault: int, data_dir: str = DEFAULT_DIR) -> np.ndarray:
    x = _read(Path(data_dir) / f"d{fault:02d}.dat")
    if x.shape[0] == 52 and x.shape[1] != 52:  # d00.dat is stored transposed
        x = x.T
    return x


def load_test(fault: int, data_dir: str = DEFAULT_DIR) -> np.ndarray:
    return _read(Path(data_dir) / f"d{fault:02d}_te.dat")


class Standardizer:
    """Fit on normal data only."""
    def fit(self, x: np.ndarray) -> "Standardizer":
        self.mu = x.mean(0); self.sd = x.std(0) + 1e-8; return self

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mu) / self.sd
