"""Experiment 33: window-resolution sweep for the local phase features (seed 42 paired vs v12 baseline), then 3-seed check."""
import os, sys, json, time, numpy as np, pandas as pd, lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from scipy.stats import binomtest
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from features import build_v12, local_k_features
DATA = sys.argv[1]
tr = pd.read_csv(f"{DATA}/Train_set.csv"); te = pd.read_csv(f"{DATA}/Test_set.csv")
X, y, Xte = build_v12(tr, te)
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
    print(f"  {name}: {ca.mean():.5f} -> {cb.mean():.5f} fixed={f} broken={b} net={f-b:+d} p={pv:.4f}", flush=True); return f - b
variants = {"W10": (10,), "W20": (20,), "W40": (40,), "W10_20_40": (10, 20, 40), "W15_25_40": (15, 25, 40), "W20_40_80": (20, 40, 80)}
base = cv(X, 42); print(f"baseline seed 42: {((base>0.5)==y).mean():.5f}", flush=True)
res = {}; F = {}
for tag, Ws in variants.items():
    Ftr, Fte = local_k_features(tr, te, Ws=Ws); F[tag] = Ftr
    o = cv(pd.concat([X.reset_index(drop=True), Ftr], axis=1), 42); print(f"{tag} ({Ftr.shape[1]} cols):"); res[tag] = paired(base, o, "seed 42")
best = max(res, key=res.get); print("best:", best, res[best])
json.dump({"results": res, "best": best, "Ws": variants[best]}, open(os.path.join(HERE, "exp33_result.json"), "w"))
if res[best] > 0:
    Xv = pd.concat([X.reset_index(drop=True), F[best]], axis=1)
    for s in [7, 2024]: paired(cv(X, s), cv(Xv, s), f"{best} seed {s}")
