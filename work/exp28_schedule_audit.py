"""Experiment 28: audit of the exact-schedule reconstruction — term-boundary capping and off-by-one.
Variants vs champion feature set, paired per seed; promote only if net>0 consistently across seeds."""
import os, sys, json, time, numpy as np, pandas as pd, lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from scipy.stats import binomtest
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from features import build
DATA = sys.argv[1]
tr = pd.read_csv(f"{DATA}/Train_set.csv"); te = pd.read_csv(f"{DATA}/Test_set.csv")
X, y, Xte = build(tr, te)
lp = json.load(open(os.path.join(HERE, "lgb_best.json")))["params"]
folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(X, y))
acc = lambda p: ((p > 0.5) == y).mean()

print("rows with k_int > contractual term (train):", int((X["k_int"] > X["loan_term"] * 12).sum()),
      "| test:", int((Xte["k_int"] > Xte["loan_term"] * 12).sum()))


def schedule_fix(d, raw, cap=True, offbyone=True):
    d = d.copy()
    r = raw["interest_rate"].values / 1200.0
    L = raw["loan_amnt"].values.astype(float)
    term_m = raw["loan_term"].str.extract(r"(\d+)").astype(float)[0].values * 12
    emi = d["emi"].values
    ir = raw["interest_receive"].values
    k_raw = np.round(raw["last_week_pay"].values * 12 / 52.0)
    def cum_int(kk):
        kk = np.minimum(kk, term_m) if cap else kk
        bal = L * (1 + r) ** kk - emi * ((1 + r) ** kk - 1) / r
        return emi * kk - (L - bal)
    if cap:
        k_eff = np.minimum(k_raw, term_m)
        d["k_overrun"] = k_raw - term_m
        sched = cum_int(k_eff)
        d["int_dev_exact"] = (ir - sched) / (sched + 1)
        d["int_gap_exact"] = ir - sched
        d["k_int"] = k_eff
    if offbyone:
        s_m1 = cum_int(np.maximum(k_raw - 1, 0)); s_p1 = cum_int(k_raw + 1)
        d["int_dev_km1"] = (ir - s_m1) / (s_m1 + 1)
        d["int_dev_kp1"] = (ir - s_p1) / (s_p1 + 1)
        # which hypothesis fits best: 0 = k-1, 1 = k, 2 = k+1
        devs = np.stack([np.abs(ir - s_m1), np.abs(ir - cum_int(k_raw)), np.abs(ir - s_p1)], axis=1)
        d["best_k_hypothesis"] = np.argmin(devs, axis=1).astype(float)
    return d


def cv(Xv, seed):
    P = dict(objective="binary", verbose=-1, n_jobs=12, bagging_freq=1, seed=seed, **lp)
    oof = np.zeros(len(Xv))
    for a, b in folds:
        m = lgb.train(P, lgb.Dataset(Xv.iloc[a], y[a]), 4000, valid_sets=[lgb.Dataset(Xv.iloc[b], y[b])],
                      callbacks=[lgb.early_stopping(150, verbose=False)])
        oof[b] = m.predict(Xv.iloc[b], num_iteration=m.best_iteration)
    return oof


def paired(pa, pb, name):
    ca, cb = (pa > 0.5) == y, (pb > 0.5) == y
    f, b = int((~ca & cb).sum()), int((ca & ~cb).sum())
    pv = binomtest(f, f + b, 0.5).pvalue if f + b else 1.0
    print(f"  {name}: {ca.mean():.5f} -> {cb.mean():.5f} net={f-b:+d} p={pv:.4f}", flush=True)
    return f - b


variants = {"A_cap": dict(cap=True, offbyone=False),
            "B_offbyone": dict(cap=False, offbyone=True),
            "C_both": dict(cap=True, offbyone=True)}
base42 = cv(X, 42); print(f"baseline seed 42: {acc(base42):.5f}", flush=True)
results = {}
for tag, kw in variants.items():
    Xv = schedule_fix(X, tr, **kw)
    o = cv(Xv, 42)
    print(f"{tag}:"); results[tag] = paired(base42, o, "seed 42")

best_tag = max(results, key=results.get)
if results[best_tag] > 0:
    print(f"\nbest variant {best_tag} — checking consistency across seeds 7, 2024:")
    Xv = schedule_fix(X, tr, **variants[best_tag])
    for s in [7, 2024]:
        bs = cv(X, s); os_ = cv(Xv, s)
        paired(bs, os_, f"seed {s}")
