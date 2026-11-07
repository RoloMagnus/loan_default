"""Experiment 16: long-inactivity segment — maturity split, prepayment indicators, segment threshold, cross-fitted residual model."""
import os, sys, json, numpy as np, pandas as pd, lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from scipy.stats import binomtest
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from features import build
pd.set_option("display.width", 220)
DATA = sys.argv[1]
tr = pd.read_csv(f"{DATA}/Train_set.csv"); te = pd.read_csv(f"{DATA}/Test_set.csv")
X, y, Xte = build(tr, te)
p = np.load(os.path.join(HERE, "oof_lgb_exact_3seed.npy")); pred = (p > 0.5).astype(int)
FP = (y == 0) & (pred == 1); FN = (y == 1) & (pred == 0); err = pred != y

term_m = X["loan_term"].values * 12
d = pd.DataFrame({"y": y, "p": p, "pred": pred, "FP": FP, "FN": FN, "err": err,
                  "missed": X["installments_missed"].values, "k": X["k_int"].values, "el": X["months_elapsed_int"].values,
                  "term_m": term_m, "grade": tr["loan_grade"].values})
d["matured"] = d["el"] >= d["term_m"]                      # contractual maturity already passed at snapshot
d["completed_sched"] = d["k"] >= d["term_m"] - 1           # paid (about) all scheduled installments
d["completion_ratio"] = d["k"] / d["term_m"]
r = tr["interest_rate"].values / 1200; L = tr["loan_amnt"].values.astype(float); emi = X["emi"].values
d["remaining_principal_ratio"] = (L * (1 + r) ** d.k - emi * ((1 + r) ** d.k - 1) / r) / L
d["int_dev_exact"] = X["int_dev_exact"].values; d["k_resid"] = X["k_implied_resid"].values

seg = d[d.missed > 12]
print(f"long-inactivity segment (>12 installments missed): n={len(seg)} default={seg.y.mean():.3f} err={seg.err.mean():.3f} FP={seg.FP.sum()} FN={seg.FN.sum()}")
print("\nFP split by contractual maturity:\n", seg.groupby("matured").agg(n=("y", "size"), default_rate=("y", "mean"), FP=("FP", "sum"), FN=("FN", "sum"), err=("err", "mean")).round(3))
print("\nsegment by completed_sched (paid all installments):\n", seg.groupby("completed_sched").agg(n=("y", "size"), default_rate=("y", "mean"), FP=("FP", "sum"), FN=("FN", "sum")).round(3))
print("\nsegment, not completed: default rate by completion ratio:\n",
      seg[~seg.completed_sched].groupby(pd.cut(seg[~seg.completed_sched].completion_ratio, [0, .1, .25, .5, .75, .9, 1.01])).agg(n=("y", "size"), default_rate=("y", "mean"), FP=("FP", "sum")).round(3))
print("\nsegment, not completed: default rate by remaining principal ratio:\n",
      seg[~seg.completed_sched].groupby(pd.cut(seg[~seg.completed_sched].remaining_principal_ratio, [-.01, .1, .3, .5, .7, .9, 1.01])).agg(n=("y", "size"), default_rate=("y", "mean"), FP=("FP", "sum")).round(3))
print("\nsegment, not completed: default rate by interest deviation from schedule:\n",
      seg[~seg.completed_sched].groupby(pd.cut(seg[~seg.completed_sched].int_dev_exact, [-2, -.2, -.05, -.01, .01, .05, .2, 100])).agg(n=("y", "size"), default_rate=("y", "mean"), FP=("FP", "sum")).round(3))
print("\nsegment default rate by grade:", seg.groupby("grade").y.mean().round(3).to_dict())

# ---- challenger A: segment-specific threshold, cross-fitted
folds = list(StratifiedKFold(5, shuffle=True, random_state=7).split(X, y))
mask = (d.missed > 12).values; newpred = pred.copy()
for a, b in folds:
    a_s = a[mask[a]]; b_s = b[mask[b]]
    ths = np.arange(0.3, 0.9, 0.01); accs = [((p[a_s] > t) == y[a_s]).mean() for t in ths]; t = ths[int(np.argmax(accs))]
    newpred[b_s] = (p[b_s] > t).astype(int)
def paired(newpred, name):
    fixed = (err & (newpred == y)).sum(); broken = (~err & (newpred != y)).sum()
    pv = binomtest(int(fixed), int(fixed + broken), 0.5).pvalue if fixed + broken else 1
    print(f"{name}: acc {1-err.mean():.5f} -> {(newpred == y).mean():.5f}  fixed={fixed} broken={broken} net={fixed-broken} McNemar p={pv:.4f}")
paired(newpred, "A. segment threshold")

# ---- challenger B: cross-fitted residual model on the segment (champion OOF prob as a feature)
Xs = X[mask].copy(); Xs["champ_p"] = p[mask]; ys = y[mask]; ps = np.zeros(len(Xs))
P = dict(objective="binary", verbose=-1, n_jobs=12, learning_rate=0.03, num_leaves=31, min_child_samples=100, feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, lambda_l2=5, seed=42)
for a, b in StratifiedKFold(5, shuffle=True, random_state=11).split(Xs, ys):
    m = lgb.train(P, lgb.Dataset(Xs.iloc[a], ys[a]), 3000, valid_sets=[lgb.Dataset(Xs.iloc[b], ys[b])], callbacks=[lgb.early_stopping(100, verbose=False)])
    ps[b] = m.predict(Xs.iloc[b], num_iteration=m.best_iteration)
newpred2 = pred.copy(); newpred2[mask] = (ps > 0.5).astype(int)
print(f"segment-only accuracy: champion {((p[mask]>0.5)==ys).mean():.4f} -> residual model {((ps>0.5)==ys).mean():.4f}")
paired(newpred2, "B. residual model on segment")
