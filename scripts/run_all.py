"""Run E1-E5 for RootTrace on TEP. Writes experiments/results/*.csv and figures/*.png."""
from __future__ import annotations
import json, sys, time, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, pandas as pd, yaml
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from roottrace import data, graph, rca, baselines, metrics
from roottrace.ground_truth import ROOT_CAUSES, KNOWN_EDGES

cfg = yaml.safe_load(open(ROOT / "configs/default.yaml"))
P, LAM, ALPHA, W, SEEDS = cfg["lags"], cfg["sparsity"], cfg["restart"], cfg["window"], cfg["seeds"]
RES = ROOT / "experiments/results"; FIG = ROOT / "figures"; RES.mkdir(parents=True, exist_ok=True); FIG.mkdir(exist_ok=True)
FAULTS = list(range(1, 22)); ONSET = data.FAULT_ONSET
t0 = time.time()

# ---------- data ----------
n0 = data.load_train(0); n0te = data.load_test(0)
std = data.Standardizer().fit(np.vstack([n0, n0te[:480]]))
train_series = [std(n0), std(n0te[:480])]
calib = std(n0te[480:])
test = {f: std(data.load_test(f)) for f in FAULTS}
trainf = {f: std(data.load_train(f)) for f in FAULTS}
pca = baselines.PCAMonitor().fit(np.vstack(train_series))

# ---------- graphs ----------
models = {}
for s in SEEDS:
    models[s] = graph.fit_graph(train_series, P, LAM, seed=s)
    print(f"graph seed {s} fitted {time.time()-t0:.0f}s", flush=True)
A_lin = baselines.linear_granger_adjacency(train_series, P)
sig = {s: graph.residuals(models[s], calib).std(0) + 1e-6 for s in SEEDS}

def res_score(m, sg, x, smooth=5):
    r = graph.residuals(m, x) / sg
    sc = (r ** 2).mean(1)
    sc = np.convolve(sc, np.ones(smooth) / smooth, mode="full")[:len(sc)]
    return np.concatenate([np.full(m.p, np.nan), sc])

thr = {s: np.nanquantile(res_score(models[s], sig[s], calib), 0.99) for s in SEEDS}

# ---------- E1 + E5: root cause ----------
rows = []
rng = np.random.default_rng(0)
for s in SEEDS:
    m, sg = models[s], sig[s]; A = m.adjacency()
    for f, truth in ROOT_CAUSES.items():
        x = test[f]; sc = res_score(m, sg, x)
        det = metrics.detection_delay(sc, thr[s], ONSET)
        start = ONSET + (0 if np.isnan(det) else int(det))
        for c in range(5):
            a = min(start + c * W, len(x) - W); b = a + W
            win = x[a - P:b]; zwin = x[a:b]
            r = np.abs(graph.residuals(m, win) / sg).mean(0)
            cf = rca.counterfactual_attribution(m, win, sg)
            scores = {
                "Random": rng.random(52),
                "PCA-T2 contribution": pca.t2_contrib(zwin),
                "PCA-SPE contribution": pca.spe_contrib(zwin),
                "Forecast-residual ranking": r,
                "Linear Granger + PageRank": rca.personalized_pagerank(A_lin, r, ALPHA),
                "RootTrace (full)": rca.personalized_pagerank(A, cf, ALPHA),
                "Ablation: no counterfactual": rca.personalized_pagerank(A, r, ALPHA),
                "Ablation: no propagation": cf,
                "Ablation: linear graph": rca.personalized_pagerank(A_lin, cf, ALPHA),
                **{f"Sensitivity: restart={a}": rca.personalized_pagerank(A, cf, a) for a in (0.1, 0.5, 0.8)},
            }
            for meth, v in scores.items():
                rk = rca.rank(v)
                rows.append(dict(seed=s, fault=f, case=c, method=meth,
                                 top1=metrics.topk_hit(rk, truth, 1), top3=metrics.topk_hit(rk, truth, 3),
                                 top5=metrics.topk_hit(rk, truth, 5), mrr=metrics.reciprocal_rank(rk, truth)))
e1 = pd.DataFrame(rows); e1.to_csv(RES / "e1_rootcause_cases.csv", index=False)
per_seed = e1.groupby(["method", "seed"])[["top1", "top3", "top5", "mrr"]].mean()
summ = per_seed.groupby("method").agg(["mean", "std"]).round(3)
summ.columns = [f"{a}_{b}" for a, b in summ.columns]; summ = summ.sort_values("mrr_mean", ascending=False)
summ.to_csv(RES / "e1_e5_rootcause_summary.csv")
perfault = e1.groupby(["method", "fault"])["top3"].mean().unstack().round(2)
perfault.to_csv(RES / "e1_top3_per_fault.csv")
print(summ, flush=True)

# ---------- E2: fault diagnosis with few labeled runs ----------
SEG = 20
def segments(x, start):
    return [x[i:i + SEG] for i in range(start, len(x) - SEG + 1, SEG)]

def emb_raw(seg): return np.concatenate([seg.mean(0), seg.std(0)])
def emb_pca(seg): t = seg @ pca.P; return np.concatenate([t.mean(0), t.std(0)])
def make_emb_graph(m, sg):
    def f(seg_ctx):
        r = graph.residuals(m, seg_ctx) / sg
        return np.concatenate([r.mean(0), r.std(0), seg_ctx[P:].mean(0)])
    return f

rows = []
for s in SEEDS:
    embs = {"Raw mean/std": (emb_raw, 0), "PCA scores": (emb_pca, 0), "Graph-residual embedding": (make_emb_graph(models[s], sig[s]), P)}
    for name, (fn, ctx) in embs.items():
        trX = {f: np.array([fn(trainf[f][i - ctx:i + SEG]) for i in range(max(ctx, 20), 480 - SEG + 1, SEG)]) for f in FAULTS}
        teX, teY = [], []
        for f in FAULTS:
            for i in range(ONSET + 40, 960 - SEG + 1, SEG):
                teX.append(fn(test[f][i - ctx:i + SEG])); teY.append(f)
        teX = np.array(teX); teY = np.array(teY)
        for k in [1, 5, 10]:
            for rep in range(5):
                r2 = np.random.default_rng(1000 * s + rep)
                Xs, Ys = [], []
                for f in FAULTS:
                    idx = r2.choice(len(trX[f]), k, replace=False); Xs.append(trX[f][idx]); Ys += [f] * k
                Xs = np.vstack(Xs); mu, sd = Xs.mean(0), Xs.std(0) + 1e-6
                clf = LogisticRegression(max_iter=2000, C=1.0).fit((Xs - mu) / sd, Ys)
                pred = clf.predict((teX - mu) / sd)
                easy = ~np.isin(teY, [3, 9, 15])
                rows.append(dict(seed=s, rep=rep, embedding=name, k=k, acc=(pred == teY).mean(), acc_excl_3_9_15=(pred[easy] == teY[easy]).mean()))
e2 = pd.DataFrame(rows); e2.to_csv(RES / "e2_diagnosis_runs.csv", index=False)
e2s = e2.groupby(["embedding", "k", "seed"])[["acc", "acc_excl_3_9_15"]].mean().groupby(["embedding", "k"]).agg(["mean", "std"]).round(3)
e2s.columns = [f"{a}_{b}" for a, b in e2s.columns]; e2s.to_csv(RES / "e2_diagnosis_summary.csv"); print(e2s, flush=True)

# ---------- E3: detection delay vs false alarm rate ----------
def pca_scores(x, kind):
    return pca.t2(x) if kind == "T2" else pca.spe(x)
rows = []
qs = [0.9, 0.95, 0.98, 0.99, 0.995, 0.999]
detectors = {"PCA-T2": lambda x: pca_scores(x, "T2"), "PCA-SPE": lambda x: pca_scores(x, "SPE")}
for s in SEEDS:
    detectors[f"Graph residual (seed {s})"] = (lambda m, sg: (lambda x: res_score(m, sg, x)))(models[s], sig[s])
for name, fn in detectors.items():
    cal = fn(calib)
    for q in qs:
        th = np.nanquantile(cal, q)
        far = np.nanmean(np.concatenate([fn(test[f])[:ONSET] > th for f in FAULTS]))
        delays = [metrics.detection_delay(np.nan_to_num(fn(test[f]), nan=-np.inf), th, ONSET) for f in FAULTS]
        dd = np.array(delays)
        rows.append(dict(detector=name, quantile=q, far=far, median_delay=np.nanmedian(dd), detected_faults=int(np.sum(~np.isnan(dd)))))
e3 = pd.DataFrame(rows)
e3["family"] = e3.detector.str.replace(r" \(seed \d\)", "", regex=True)
e3.to_csv(RES / "e3_delay_far_raw.csv", index=False)
e3s = e3.groupby(["family", "quantile"])[["far", "median_delay", "detected_faults"]].agg(["mean", "std"]).round(3)
e3s.columns = [f"{a}_{b}" for a, b in e3s.columns]; e3s.to_csv(RES / "e3_delay_far_summary.csv"); print(e3s, flush=True)

# ---------- E4: graph sanity ----------
K = 150
def topk_edges(A, k=K):
    A = A.copy(); np.fill_diagonal(A, 0); idx = np.argsort(-A.ravel())[:k]
    return {(int(i // 52), int(i % 52)) for i in idx}  # (target, source)
sets = {s: topk_edges(models[s].adjacency()) for s in SEEDS}
rows = []
for i in SEEDS:
    for j in SEEDS:
        if i < j: rows.append(dict(pair=f"{i}-{j}", jaccard_top150=metrics.jaccard(sets[i], sets[j])))
stab = pd.DataFrame(rows)
known = {(t, s_) for s_, t in KNOWN_EDGES} | {(s_, t) for s_, t in KNOWN_EDGES}
def known_recall(E): return np.mean([((t, s_) in E) or ((s_, t) in E) for s_, t in KNOWN_EDGES])
rec = [dict(graph=f"NeuralGranger seed {s}", known_pair_recall_top150=known_recall(sets[s]),
            nonzero_edges=int((models[s].adjacency() > 1e-6).sum() - (np.diag(models[s].adjacency()) > 1e-6).sum())) for s in SEEDS]
rec.append(dict(graph="Linear Granger (F-test)", known_pair_recall_top150=known_recall(topk_edges(A_lin)), nonzero_edges=int((A_lin > 0).sum() - (np.diag(A_lin) > 0).sum())))
# random expectation: fraction of ordered off-diagonal pairs hit when picking K edges, either direction
rec.append(dict(graph="Random top-150 (expected)", known_pair_recall_top150=1 - (1 - K / (52 * 51)) ** 2, nonzero_edges=K))
e4 = pd.DataFrame(rec); e4.to_csv(RES / "e4_graph_sanity.csv", index=False); stab.to_csv(RES / "e4_graph_stability.csv", index=False)
print(e4, stab, flush=True)

# ---------- figures ----------
main = ["Random", "PCA-T2 contribution", "PCA-SPE contribution", "Forecast-residual ranking", "Linear Granger + PageRank", "RootTrace (full)"]
fig, ax = plt.subplots(figsize=(8, 4)); sub = summ.loc[main]
xs = np.arange(len(main));
for o, k in enumerate(["top1", "top3", "top5", "mrr"]):
    ax.bar(xs + o * 0.2, sub[f"{k}_mean"], 0.2, yerr=sub[f"{k}_std"], label=k)
ax.set_xticks(xs + 0.3); ax.set_xticklabels(main, rotation=25, ha="right", fontsize=8); ax.legend(); ax.set_title("E1 root-cause accuracy on TEP (10 mapped faults x 5 windows x 3 seeds)")
fig.tight_layout(); fig.savefig(FIG / "e1_rootcause.png", dpi=130); plt.close(fig)
fig, ax = plt.subplots(figsize=(6, 4))
for name, g in e2s.reset_index().groupby("embedding"):
    ax.errorbar(g.k, g.acc_mean, yerr=g.acc_std, marker="o", label=name)
ax.set_xlabel("labeled runs per fault"); ax.set_ylabel("accuracy (21 faults)"); ax.legend(); ax.set_title("E2 few-label fault diagnosis")
fig.tight_layout(); fig.savefig(FIG / "e2_diagnosis.png", dpi=130); plt.close(fig)
fig, ax = plt.subplots(figsize=(6, 4))
for name, g in e3s.reset_index().groupby("family"):
    ax.plot(g.far_mean, g.median_delay_mean, marker="o", label=name)
ax.set_xlabel("false alarm rate (pre-onset samples)"); ax.set_ylabel("median detection delay (samples)"); ax.legend(); ax.set_title("E3 delay vs FAR")
fig.tight_layout(); fig.savefig(FIG / "e3_delay_far.png", dpi=130); plt.close(fig)
fig, ax = plt.subplots(figsize=(6, 5)); Ad = models[SEEDS[0]].adjacency().copy(); np.fill_diagonal(Ad, 0)
ax.imshow(Ad, cmap="viridis"); ax.set_xlabel("source sensor j"); ax.set_ylabel("target sensor i"); ax.set_title("E4 learned Granger graph (seed 0)")
fig.tight_layout(); fig.savefig(FIG / "e4_graph.png", dpi=130); plt.close(fig)
json.dump(dict(runtime_s=round(time.time() - t0, 1), config=cfg), open(RES / "run_meta.json", "w"), indent=1)
print("done", time.time() - t0)
