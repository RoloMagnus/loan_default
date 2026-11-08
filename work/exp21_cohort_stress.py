"""Experiment 21: ID-neighbourhood cohort-stress features (target-free, train+test pooled)."""
import os, sys, json, time, numpy as np, pandas as pd, lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from scipy.stats import binomtest
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from features import build, prep, vintage_table, add_vintage, add_schedule_exact, CAT_COLS
DATA = sys.argv[1]
tr = pd.read_csv(f"{DATA}/Train_set.csv"); te = pd.read_csv(f"{DATA}/Test_set.csv")
X, y, Xte = build(tr, te)
best = json.load(open(os.path.join(HERE, "lgb_best.json")))["params"]
P = dict(objective="binary", verbose=-1, n_jobs=12, bagging_freq=1, seed=42, **best)


def cohort_stress(tr, te, X, Xte, window=1001):
    """Rolling feature aggregates over ID order on train+test pooled (no target)."""
    all_ids = np.r_[tr["ID"].values, te["ID"].values]
    f = pd.DataFrame({"ID": all_ids,
                      "missed": np.r_[X["installments_missed"].values, Xte["installments_missed"].values],
                      "dev": np.r_[X["int_dev_exact"].values, Xte["int_dev_exact"].values],
                      "behind": np.r_[X["months_behind_int"].values, Xte["months_behind_int"].values]}).sort_values("ID").reset_index(drop=True)
    out = pd.DataFrame(index=f.index)
    out["nb_stress_rate"] = (f["missed"] > 2).rolling(window, center=True, min_periods=100).mean()
    out["nb_missed_mean"] = f["missed"].rolling(window, center=True, min_periods=100).mean()
    out["nb_dev_mean"] = f["dev"].clip(-1, 1).rolling(window, center=True, min_periods=100).mean()
    out["ID"] = f["ID"]
    return out.set_index("ID")


cs = cohort_stress(tr, te, X, Xte)
X2 = X.copy(); Xte2 = Xte.copy()
for c in ["nb_stress_rate", "nb_missed_mean", "nb_dev_mean"]:
    X2[c] = cs.loc[tr["ID"].values, c].values
    Xte2[c] = cs.loc[te["ID"].values, c].values
# relative-to-cohort versions
X2["missed_vs_cohort"] = X2["installments_missed"] - X2["nb_missed_mean"]
Xte2["missed_vs_cohort"] = Xte2["installments_missed"] - Xte2["nb_missed_mean"]

oof5 = np.load(os.path.join(HERE, "oof_lgb_tuned.npy"))
acc = lambda p: ((p > 0.5) == y).mean()
oof = np.zeros(len(X2)); t0 = time.time()
for a, b in StratifiedKFold(5, shuffle=True, random_state=42).split(X2, y):
    m = lgb.train(P, lgb.Dataset(X2.iloc[a], y[a]), 4000, valid_sets=[lgb.Dataset(X2.iloc[b], y[b])],
                  callbacks=[lgb.early_stopping(150, verbose=False)])
    oof[b] = m.predict(X2.iloc[b], num_iteration=m.best_iteration)
ca, cb = (oof5 > 0.5) == y, (oof > 0.5) == y
f, br = int((~ca & cb).sum()), int((ca & ~cb).sum())
pv = binomtest(f, f + br, 0.5).pvalue if f + br else 1
print(f"LGB base {acc(oof5):.5f} -> +cohort stress {acc(oof):.5f}  fixed={f} broken={br} net={f-br} p={pv:.4f}  [{time.time()-t0:.0f}s]")
