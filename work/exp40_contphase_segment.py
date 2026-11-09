"""Experiment 40: (A) continuous months-behind features E_cont - k (merges integer gap + phase share);
(B) dedicated near-current segment model, cross-fitted, replace/average vs global model."""
import os, sys, json, numpy as np, pandas as pd, lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from scipy.stats import binomtest
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from features import build_v12, local_k_features
DATA = sys.argv[1]
tr = pd.read_csv(f"{DATA}/Train_set.csv"); te = pd.read_csv(f"{DATA}/Test_set.csv")
X, y, Xte = build_v12(tr, te); Xr = X.reset_index(drop=True)
F, Fte = local_k_features(tr, te, Ws=(20, 40, 80), share=0.08)
k = X["k_int"].values


def cont_feats(F, k):
    d = pd.DataFrame(index=F.index)
    for W in (20, 40, 80):
        d[f"cb{W}"] = (F[f"lk{W}_rmax"].values - k) - (1 - F[f"lk{W}_phase"].values)       # continuous months behind current level
    d["cb_mean"] = d[["cb20", "cb40", "cb80"]].mean(axis=1)
    return d


C = cont_feats(F, k)
F40 = F[[c for c in F.columns if c.startswith("lk40_")]]
lp = json.load(open(os.path.join(HERE, "lgb_best.json")))["params"]
folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(X, y))


def cv(Xv, seed, rows=None, label=None):
    P = dict(objective="binary", verbose=-1, n_jobs=12, bagging_freq=1, seed=seed, **lp)
    oof = np.full(len(Xv), np.nan)
    for a, b in folds:
        if rows is not None: a, b = a[rows[a]], b[rows[b]]
        m = lgb.train(P, lgb.Dataset(Xv.iloc[a], y[a]), 4000, valid_sets=[lgb.Dataset(Xv.iloc[b], y[b])], callbacks=[lgb.early_stopping(150, verbose=False)])
        oof[b] = m.predict(Xv.iloc[b], num_iteration=m.best_iteration)
    return oof


def paired(pa, pb, name, mask=None):
    m = np.ones(len(y), bool) if mask is None else mask
    ca, cb = (pa[m] > 0.5) == y[m], (pb[m] > 0.5) == y[m]
    f, b = int((~ca & cb).sum()), int((ca & ~cb).sum()); pv = binomtest(f, f + b, 0.5).pvalue if f + b else 1.0
    print(f"  {name}: {ca.mean():.5f} -> {cb.mean():.5f} net={f-b:+d} p={pv:.4f}", flush=True); return f - b


base_X = pd.concat([Xr, F40], axis=1)
res = {}
for seed in [42, 7, 2024]:
    base = cv(base_X, seed)
    A = cv(pd.concat([Xr, F40, C], axis=1), seed)
    res[seed] = paired(base, A, f"A: +continuous months-behind, seed {seed}")
print("A sum of nets:", sum(res.values()))

# ---- B: dedicated near-current segment model (seed 42)
base = cv(base_X, 42)
seg = np.isin(F40["lk40_k_minus_rmax"].values, [-1.0, 0.0])
print(f"\nB: near-current segment rows {seg.sum()} ({seg.mean():.1%}); global-model accuracy there {((base[seg]>0.5)==y[seg]).mean():.5f}")
segp = cv(base_X, 42, rows=seg)
rep = base.copy(); rep[seg] = segp[seg]
avg = base.copy(); avg[seg] = 0.5 * base[seg] + 0.5 * segp[seg]
paired(base, rep, "B1: replace with segment model", None)
paired(base, avg, "B2: 50/50 average in segment", None)
print("done")
