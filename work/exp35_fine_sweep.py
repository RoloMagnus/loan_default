"""Experiment 35: fine sweep around W40 (window size and robust-max share). Replace W40 only if +20 rows on seed 42."""
import os, sys, json, numpy as np, pandas as pd, lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from scipy.stats import binomtest
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from features import build_v12, local_k_features
DATA = sys.argv[1]
tr = pd.read_csv(f"{DATA}/Train_set.csv"); te = pd.read_csv(f"{DATA}/Test_set.csv")
X, y, Xte = build_v12(tr, te); Xr = X.reset_index(drop=True)
lp = json.load(open(os.path.join(HERE, "lgb_best.json")))["params"]
folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(X, y))
def cv(Xv, seed):
    P = dict(objective="binary", verbose=-1, n_jobs=12, bagging_freq=1, seed=seed, **lp); oof = np.zeros(len(Xv))
    for a, b in folds:
        m = lgb.train(P, lgb.Dataset(Xv.iloc[a], y[a]), 4000, valid_sets=[lgb.Dataset(Xv.iloc[b], y[b])], callbacks=[lgb.early_stopping(150, verbose=False)])
        oof[b] = m.predict(Xv.iloc[b], num_iteration=m.best_iteration)
    return oof
def paired(pa, pb, name):
    ca, cb = (pa > 0.5) == y, (pb > 0.5) == y
    f, b = int((~ca & cb).sum()), int((ca & ~cb).sum()); pv = binomtest(f, f + b, 0.5).pvalue if f + b else 1.0
    print(f"  {name}: {ca.mean():.5f} -> {cb.mean():.5f} net={f-b:+d} p={pv:.4f}", flush=True); return f - b
F40, _ = local_k_features(tr, te, Ws=(40,), share=0.08); w40 = cv(pd.concat([Xr, F40], axis=1), 42)
print(f"W40 share .08 reference: {((w40>0.5)==y).mean():.5f}", flush=True)
res = {}
for tag, (W, sh) in {"W30": (30, .08), "W50": (50, .08), "W60": (60, .08), "W40_s04": (40, .04), "W40_s15": (40, .15), "W40_s25": (40, .25)}.items():
    Ft, _ = local_k_features(tr, te, Ws=(W,), share=sh)
    o = cv(pd.concat([Xr, Ft], axis=1), 42); print(f"{tag}:"); res[tag] = paired(w40, o, "vs W40 seed 42")
json.dump(res, open(os.path.join(HERE, "exp35_result.json"), "w")); print(res)
