"""Choose group-lasso strength on NORMAL validation data only (no fault labels):
largest lambda whose validation forecast MSE is within 5% of the best."""
import sys; from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from roottrace import data, graph
n0 = data.load_train(0); n0te = data.load_test(0)
std = data.Standardizer().fit(np.vstack([n0, n0te[:480]]))
tr = [std(n0), std(n0te[:480])]; va = std(n0te[480:])
rows = []
for lam in [0.002, 0.2, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]:
    m = graph.fit_graph(tr, 5, lam, seed=0)
    A = m.adjacency(); np.fill_diagonal(A, 0)
    rows.append(dict(sparsity=lam, val_mse=float((graph.residuals(m, va) ** 2).mean()), nonzero_offdiag_edges=int((A > 1e-6).sum())))
    print(rows[-1], flush=True)
df = pd.DataFrame(rows); best = df.val_mse.min()
df["selected"] = df.sparsity == df[df.val_mse <= 1.05 * best].sparsity.max()
df.to_csv(ROOT / "experiments/results/e4_sparsity_selection.csv", index=False); print(df)
