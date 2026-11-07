"""Experiment 11: exact integer-month schedule + implied payments, seed variance, pseudo-labelling."""
import os, sys, json, time, numpy as np, pandas as pd, lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from features import build, report
DATA = sys.argv[1]
tr = pd.read_csv(f"{DATA}/Train_set.csv"); te = pd.read_csv(f"{DATA}/Test_set.csv")
X, y, Xte = build(tr, te)
best = json.load(open(os.path.join(HERE, "lgb_best.json")))["params"]
P = dict(objective="binary", verbose=-1, n_jobs=12, bagging_freq=1, **best)
folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(X, y))


def cv(Xa, ya, Xb_extra=None, yb_extra=None, seed=42, Xtest=None):
    p = dict(P, seed=seed); oof = np.zeros(len(Xa)); tp = np.zeros(len(Xtest)) if Xtest is not None else None
    for a, b in folds:
        Xf, yf = Xa.iloc[a], ya[a]
        if Xb_extra is not None:
            Xf = pd.concat([Xf, Xb_extra]); yf = np.r_[yf, yb_extra]
        m = lgb.train(p, lgb.Dataset(Xf, yf), 4000, valid_sets=[lgb.Dataset(Xa.iloc[b], ya[b])], callbacks=[lgb.early_stopping(150, verbose=False)])
        oof[b] = m.predict(Xa.iloc[b], num_iteration=m.best_iteration)
        if Xtest is not None: tp += m.predict(Xtest, num_iteration=m.best_iteration) / 5
    return oof, tp


def sharpen(d, raw):
    d = d.copy()
    r = raw["interest_rate"].values / 1200.0; L = raw["loan_amnt"].values
    term_m = raw["loan_term"].str.extract(r"(\d+)").astype(float)[0].values * 12
    emi = d["emi"].values
    k = np.round(raw["last_week_pay"].values * 12 / 52.0)                      # integer installments at last payment
    bal = L * (1 + r) ** k - emi * ((1 + r) ** k - 1) / r
    sched = emi * k - (L - bal)
    d["k_int"] = k
    d["int_dev_exact"] = (raw["interest_receive"].values - sched) / (sched + 1)
    d["int_gap_exact"] = raw["interest_receive"].values - sched
    # implied number of installments from interest actually received (invert cumulative interest, grid search)
    ks = np.arange(0, 61)
    cum = np.stack([emi * kk - (L - (L * (1 + r) ** kk - emi * ((1 + r) ** kk - 1) / r)) for kk in ks], axis=1)  # n x 61
    cum = np.where(ks[None, :] > term_m[:, None], np.inf, cum)
    k_impl = np.argmin(np.abs(cum - raw["interest_receive"].values[:, None]), axis=1).astype(float)
    d["k_implied"] = k_impl
    d["k_implied_minus_k"] = k_impl - k
    el = np.round(d["weeks_elapsed_est"].values * 12 / 52.0)
    d["months_elapsed_int"] = el
    d["installments_missed"] = el - k_impl
    d["months_behind_int"] = el - k
    return d


X1 = sharpen(X, tr); Xte1 = sharpen(Xte, te)
t0 = time.time()
o0, _ = cv(X, y); report("base tuned", o0, y)
o1, _ = cv(X1, y); report("+sharpen", o1, y); print(f"{time.time()-t0:.0f}s")
# seed variance on base
for s in [1, 2]:
    os_, _ = cv(X, y, seed=s); report(f"base seed {s}", os_, y)
# pseudo-labelling with confident test predictions (from v1 blend probabilities)
tp = 0.6 * np.load(os.path.join(HERE, "test_lgb_tuned.npy")) + 0.2 * np.load(os.path.join(HERE, "test_cat.npy")) + 0.2 * np.load(os.path.join(HERE, "test_xgb.npy"))
conf = (tp < 0.03) | (tp > 0.97); print("confident test rows:", conf.sum(), "of", len(tp))
o2, _ = cv(X, y, Xte[conf], (tp[conf] > 0.5).astype(int)); report("+pseudo-labels", o2, y)
np.save(os.path.join(HERE, "oof_exp11_sharpen.npy"), o1)
