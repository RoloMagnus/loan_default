"""Experiment 14: error analysis of the champion (3-seed LGB on exact features) + ID-cohort holdout."""
import os, sys, json, numpy as np, pandas as pd, lightgbm as lgb
from sklearn.model_selection import GroupKFold
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from features import build
pd.set_option("display.width", 200)
DATA = sys.argv[1]
tr = pd.read_csv(f"{DATA}/Train_set.csv"); te = pd.read_csv(f"{DATA}/Test_set.csv")
X, y, Xte = build(tr, te)
oof = np.load(os.path.join(HERE, "oof_lgb_exact_3seed.npy")); pred = (oof > 0.5).astype(int); err = pred != y
d = tr[["ID", "loan_term", "loan_grade"]].copy(); d["err"] = err; d["y"] = y; d["FN"] = (y == 1) & (pred == 0); d["FP"] = (y == 0) & (pred == 1)
d["installments_missed"] = X["installments_missed"].values; d["months_behind"] = X["months_behind_int"].values; d["k_implied_minus_k"] = X["k_implied_minus_k"].values
d["conf"] = np.abs(oof - 0.5)
print("overall acc", 1 - err.mean(), " FN", d.FN.sum(), " FP", d.FP.sum())
g = lambda key: d.groupby(key).agg(n=("err", "size"), err_rate=("err", "mean"), default_rate=("y", "mean"), FN=("FN", "sum"), FP=("FP", "sum")).round(3)
print("\nby ID decile:\n", g(pd.qcut(d.ID, 10, labels=False)))
print("\nby installments_missed:\n", g(pd.cut(d.installments_missed, [-100, -1, 0, 1, 2, 3, 6, 12, 24, 100])))
print("\nby months_behind:\n", g(pd.cut(d.months_behind, [-100, -1, 0, 1, 2, 3, 6, 12, 24, 100])))
print("\nby k_implied_minus_k:\n", g(pd.cut(d.k_implied_minus_k, [-100, -3, -1, 0, 1, 3, 100])))
print("\nby loan_term:\n", g("loan_term")); print("\nby grade:\n", g("loan_grade"))
print("\nerrors by confidence band |p-0.5|:\n", g(pd.cut(d.conf, [0, .1, .2, .3, .4, .5])))
print("share of errors with |p-0.5|<0.2:", round(d.loc[d.err, "conf"].lt(0.2).mean(), 3))

# ID-cohort holdout (GroupKFold on ID decile) with champion params
best = json.load(open(os.path.join(HERE, "lgb_best.json")))["params"]
P = dict(objective="binary", verbose=-1, n_jobs=12, bagging_freq=1, seed=42, **best)
groups = pd.qcut(tr.ID, 5, labels=False).values; oofg = np.zeros(len(X))
for a, b in GroupKFold(5).split(X, y, groups):
    m = lgb.train(P, lgb.Dataset(X.iloc[a], y[a]), 1000)
    oofg[b] = m.predict(X.iloc[b])
print("\nID-cohort holdout (5 vintage groups, no early stopping, 1000 rounds): acc =", round(((oofg > 0.5) == y).mean(), 4))
print(pd.Series((oofg > 0.5) == y).groupby(groups).mean().round(3).rename("acc by held-out vintage group").to_string())
