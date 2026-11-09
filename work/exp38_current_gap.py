"""Experiment 38: interest received vs what a CURRENT loan of this vintage would have paid (sharp elapsed estimate from W=40 window)."""
import os, sys, json, numpy as np, pandas as pd, lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from scipy.stats import binomtest
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from features import build_v12, local_k_features
DATA = sys.argv[1]
tr = pd.read_csv(f"{DATA}/Train_set.csv"); te = pd.read_csv(f"{DATA}/Test_set.csv")
X, y, Xte = build_v12(tr, te); Xr = X.reset_index(drop=True)
F, Fte = local_k_features(tr, te, Ws=(40,), share=0.08)


def current_gap(Xd, raw, F):
    d = pd.DataFrame(index=range(len(raw)))
    r = raw["interest_rate"].values / 1200.0; L = raw["loan_amnt"].values.astype(float)
    term_m = raw["loan_term"].str.extract(r"(\d+)").astype(float)[0].values * 12
    emi = Xd["emi"].values; ir = raw["interest_receive"].values
    E = np.minimum(F["lk40_rmax"].values, term_m)
    bal = L * (1 + r) ** E - emi * ((1 + r) ** E - 1) / r
    sched = emi * E - (L - bal)
    d["cur_sched_int"] = sched
    d["cur_gap"] = ir - sched
    d["cur_ratio"] = ir / (sched + 1)
    d["missed40"] = F["lk40_rmax"].values - Xd["k_implied"].values
    d["impl_minus_E40"] = Xd["k_implied"].values - F["lk40_rmax"].values
    return d
G = current_gap(Xr, tr, F)
lp = json.load(open(os.path.join(HERE, "lgb_best.json")))["params"]
folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(X, y))
def cv(Xv, seed):
    P = dict(objective="binary", verbose=-1, n_jobs=6, bagging_freq=1, seed=seed, **lp); oof = np.zeros(len(Xv))
    for a, b in folds:
        m = lgb.train(P, lgb.Dataset(Xv.iloc[a], y[a]), 4000, valid_sets=[lgb.Dataset(Xv.iloc[b], y[b])], callbacks=[lgb.early_stopping(150, verbose=False)])
        oof[b] = m.predict(Xv.iloc[b], num_iteration=m.best_iteration)
    return oof
def paired(pa, pb, name):
    ca, cb = (pa > 0.5) == y, (pb > 0.5) == y
    f, b = int((~ca & cb).sum()), int((ca & ~cb).sum()); pv = binomtest(f, f + b, 0.5).pvalue if f + b else 1.0
    print(f"  {name}: {ca.mean():.5f} -> {cb.mean():.5f} net={f-b:+d} p={pv:.4f}", flush=True); return f - b
A = pd.concat([Xr, F], axis=1); B = pd.concat([Xr, F, G], axis=1)
tot = 0
for s in [42, 7, 2024]:
    tot += paired(cv(A, s), cv(B, s), f"W40 -> W40+current-gap, seed {s}")
print("sum of nets over 3 seeds:", tot)
