import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np, torch
from roottrace import graph, rca, metrics, baselines
from roottrace.ground_truth import ROOT_CAUSES, KNOWN_EDGES


def test_make_lagged_alignment():
    x = np.arange(20, dtype=float).reshape(10, 2)
    X, Y = graph.make_lagged(x, 3)
    assert X.shape == (7, 2, 3) and Y.shape == (7, 2)
    assert X[0, 0, 0] == x[2, 0] and X[0, 0, 2] == x[0, 0] and Y[0, 0] == x[3, 0]


def test_prox_zeroes_groups():
    m = graph.NeuralGranger(4, 2)
    m.prox(1e6)
    assert np.allclose(m.adjacency(), 0)


def test_neural_granger_recovers_simple_chain():
    rng = np.random.default_rng(0); T = 1500; x = np.zeros((T, 3))
    for t in range(1, T):
        x[t, 0] = 0.5 * x[t - 1, 0] + rng.normal()
        x[t, 1] = 0.9 * x[t - 1, 0] + 0.1 * rng.normal()
        x[t, 2] = rng.normal()
    m = graph.fit_graph([x], 2, 0.5, epochs=60)
    A = m.adjacency()
    assert A[1, 0] > A[1, 2] and A[1, 0] > A[2, 0]


def test_pagerank_moves_mass_to_parent():
    A = np.zeros((3, 3)); A[1, 0] = 1; A[2, 1] = 1  # 0 -> 1 -> 2
    pi = rca.personalized_pagerank(A, np.array([0, 0, 1.0]), restart=0.2)
    assert pi[0] > 0.2 and abs(pi.sum() - 1) < 1e-6


def test_counterfactual_identifies_corrupted_sensor():
    rng = np.random.default_rng(1); x = rng.normal(size=(800, 3))
    m = graph.fit_graph([x], 2, 0.1, epochs=20)
    sig = graph.residuals(m, x).std(0)
    w = rng.normal(size=(40, 3)); w[2:, 1] += 8
    d = rca.counterfactual_attribution(m, w, sig)
    assert np.argmax(d) == 1


def test_metrics():
    r = np.array([3, 1, 2]); assert metrics.topk_hit(r, {1}, 1) == 0 and metrics.topk_hit(r, {1}, 2) == 1
    assert metrics.reciprocal_rank(r, {2}) == 1 / 3
    s = np.r_[np.zeros(10), np.ones(10)]; assert metrics.detection_delay(s, 0.5, 5) == 5
    assert metrics.jaccard({1, 2}, {2, 3}) == 1 / 3


def test_pca_contrib_shapes():
    z = np.random.default_rng(0).normal(size=(200, 5)); p = baselines.PCAMonitor().fit(z)
    assert p.t2_contrib(z).shape == (5,) and p.spe_contrib(z).shape == (5,)


def test_ground_truth_indices_valid():
    for f, s in ROOT_CAUSES.items(): assert all(0 <= v < 52 for v in s)
    assert all(0 <= a < 52 and 0 <= b < 52 for a, b in KNOWN_EDGES)
