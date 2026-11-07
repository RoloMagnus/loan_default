"""Experiment 12: which exact-schedule features carry the gain? + refinements."""
import os, sys, json, time, numpy as np, pandas as pd, lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from features import build, report
DATA = sys.argv[1]
tr = pd.read_csv(f"{DATA}/Train_set.csv"); te = pd.read_csv(f"{DATA}/Test_set.csv")
X, y, Xte = build(tr, te)
best = json.load(open(os.path.join(HERE, "lgb_best.json")))["params"]
P = dict(objective="binary", verbose=-1, n_jobs=12, bagging_freq=1, seed=42, **best)
folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(X, y))


def cv(Xa, label):
    oof = np.zeros(len(Xa)); models = []
    for a, b in folds:
        m = lgb.train(P, lgb.Dataset(Xa.iloc[a], y[a]), 4000, valid_sets=[lgb.Dataset(Xa.iloc[b], y[b])], callbacks=[lgb.early_stopping(150, verbose=False)])
        oof[b] = m.predict(Xa.iloc[b], num_iteration=m.best_iteration); models.append(m)
    report(label, oof, y); return oof, models


t0 = time.time()
oof, models = cv(X, "full exact set")
imp = pd.Series(sum(m.feature_importance("gain") for m in models), index=X.columns).sort_values(ascending=False)
print(imp.head(15).round(0).to_string(), f"\n{time.time()-t0:.0f}s")
np.save(os.path.join(HERE, "oof_lgb_exact.npy"), oof)

new = ["k_int", "int_dev_exact", "int_gap_exact", "k_implied", "k_implied_resid", "k_implied_minus_k", "months_elapsed_int", "installments_missed", "months_behind_int"]
for drop in [["k_implied", "k_implied_resid", "k_implied_minus_k", "installments_missed"], ["int_dev_exact", "int_gap_exact"], ["months_elapsed_int", "months_behind_int"]]:
    cv(X.drop(columns=drop), f"drop {drop}")

# refinement: fractional implied k by linear interpolation between grid points
def frac_k(d, raw):
    r = raw["interest_rate"].values / 1200.0; L = raw["loan_amnt"].values.astype(float)
    term_m = raw["loan_term"].str.extract(r"(\d+)").astype(float)[0].values * 12
    emi = d["emi"].values; ir = raw["interest_receive"].values
    ks = np.arange(0, 61)
    cum = np.stack([emi * kk - (L - (L * (1 + r) ** kk - emi * ((1 + r) ** kk - 1) / r)) for kk in ks], axis=1)
    cum = np.where(ks[None, :] > term_m[:, None], np.nan, cum)
    out = np.empty(len(ir))
    for i in range(len(ir)):
        c = cum[i]; c = c[~np.isnan(c)]
        out[i] = np.interp(ir[i], c, np.arange(len(c)))
    return out
X2 = X.copy(); X2["k_implied_frac"] = frac_k(X, tr); X2["installments_missed_frac"] = X2["months_elapsed_int"] - X2["k_implied_frac"]
cv(X2, "+fractional implied k")
