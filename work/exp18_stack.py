"""Experiment 18: cross-fitted stacker on saved OOF probabilities vs weight-grid blend."""
import os, sys, itertools, numpy as np, pandas as pd, lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from scipy.stats import binomtest
from scipy.special import logit
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from features import build
DATA = sys.argv[1]
tr = pd.read_csv(f"{DATA}/Train_set.csv"); te = pd.read_csv(f"{DATA}/Test_set.csv")
X, y, Xte = build(tr, te)
P = {k: np.load(os.path.join(HERE, f"{k}.npy")) for k in ["oof_lgb_tuned", "oof_cat", "oof_xgb"]}
acc = lambda p, t=0.5: ((p > t) == y).mean()
w = (0.1, 0.7, 0.2)
blend = w[0] * P["oof_lgb_tuned"] + w[1] * P["oof_cat"] + w[2] * P["oof_xgb"]
print(f"current blend OOF acc: {acc(blend):.5f}")

def paired(pa, pb, name):
    ca, cb = (pa > 0.5) == y, (pb > 0.5) == y
    f, b = int((~ca & cb).sum()), int((ca & ~cb).sum())
    pv = binomtest(f, f + b, 0.5).pvalue if f + b else 1.0
    print(f"{name}: {ca.mean():.5f} -> {cb.mean():.5f} fixed={f} broken={b} net={f-b} McNemar p={pv:.4f}")

# meta-features: logits of the 3 probs + a few raw discriminators
Z = pd.DataFrame({"l_lgb": logit(np.clip(P["oof_lgb_tuned"], 1e-6, 1 - 1e-6)),
                  "l_cat": logit(np.clip(P["oof_cat"], 1e-6, 1 - 1e-6)),
                  "l_xgb": logit(np.clip(P["oof_xgb"], 1e-6, 1 - 1e-6))})
Zx = Z.copy()
for c in ["installments_missed", "months_behind_int", "int_dev_exact", "weeks_since_last_pay", "id_rank", "loan_term", "elapsed_vs_term"]:
    Zx[c] = X[c].values

folds = list(StratifiedKFold(5, shuffle=True, random_state=99).split(Z, y))  # fresh fold seed for the meta level
# A. logistic stacker on logits
sa = np.zeros(len(y))
for a, b in folds:
    m = LogisticRegression(max_iter=1000).fit(Z.iloc[a], y[a]); sa[b] = m.predict_proba(Z.iloc[b])[:, 1]
paired(blend, sa, "A. logistic on logits")
# B. small LGB stacker on logits only
pb_params = dict(objective="binary", learning_rate=0.05, num_leaves=15, min_child_samples=200, feature_fraction=0.9,
                 bagging_fraction=0.9, bagging_freq=1, lambda_l2=10, verbose=-1, n_jobs=12, seed=42)
sb = np.zeros(len(y))
for a, b in folds:
    m = lgb.train(pb_params, lgb.Dataset(Z.iloc[a], y[a]), 2000, valid_sets=[lgb.Dataset(Z.iloc[b], y[b])],
                  callbacks=[lgb.early_stopping(100, verbose=False)])
    sb[b] = m.predict(Z.iloc[b], num_iteration=m.best_iteration)
paired(blend, sb, "B. LGB stacker on logits")
# C. small LGB stacker on logits + raw discriminators
sc = np.zeros(len(y))
for a, b in folds:
    m = lgb.train(pb_params, lgb.Dataset(Zx.iloc[a], y[a]), 2000, valid_sets=[lgb.Dataset(Zx.iloc[b], y[b])],
                  callbacks=[lgb.early_stopping(100, verbose=False)])
    sc[b] = m.predict(Zx.iloc[b], num_iteration=m.best_iteration)
paired(blend, sc, "C. LGB stacker + raw features")
# D. finer weight grid + per-model threshold calibration via rank-average
best = (acc(blend), None)
for w0 in np.arange(0, 0.45, 0.05):
    for w1 in np.arange(0.3, 1.0 - w0 + 1e-9, 0.05):
        w2 = 1 - w0 - w1
        if w2 < -1e-9: continue
        a = acc(w0 * P["oof_lgb_tuned"] + w1 * P["oof_cat"] + max(w2, 0) * P["oof_xgb"])
        if a > best[0]: best = (a, (round(w0, 2), round(w1, 2), round(max(w2, 0), 2)))
print("finer grid best:", best)
