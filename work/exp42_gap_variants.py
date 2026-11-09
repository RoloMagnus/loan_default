"""Experiment 42: variants of the gap-normalised block vs the reference block (Ws=60,150)."""
import os, sys, json, numpy as np, pandas as pd, lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from scipy.stats import binomtest
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from features import build_v12, local_k_features, gap_norm_features
DATA = sys.argv[1]
tr = pd.read_csv(f"{DATA}/Train_set.csv"); te = pd.read_csv(f"{DATA}/Test_set.csv")
X, y, Xte = build_v12(tr, te); Xr = X.reset_index(drop=True)
F, Fte = local_k_features(tr, te, Ws=(40,), share=0.08)
A = pd.concat([Xr, F], axis=1)
G0, _ = gap_norm_features(tr, te, X, Xte, F, Fte, Ws=(60, 150))
G1, _ = gap_norm_features(tr, te, X, Xte, F, Fte, Ws=(60, 150), kgroup_m=10)
G2, _ = gap_norm_features(tr, te, X, Xte, F, Fte, Ws=(30, 60, 150, 300))
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
tot1 = tot2 = 0
for s in [42, 7]:
    ref = cv(pd.concat([A, G0], axis=1), s); print(f"seed {s}: reference block acc {((ref>.5)==y).mean():.5f}", flush=True)
    tot1 += paired(ref, cv(pd.concat([A, G1], axis=1), s), "V1 +same-k baseline vs ref")
    tot2 += paired(ref, cv(pd.concat([A, G2], axis=1), s), "V2 windows 30/60/150/300 vs ref")
print("V1 sum:", tot1, "| V2 sum:", tot2)
