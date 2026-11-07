"""Experiment 6: sharper schedule features (integer months, implied months from interest)."""
import sys, os, time, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from features import build, lgb_cv, report

DATA = sys.argv[1]
tr = pd.read_csv(f"{DATA}/Train_set.csv"); te = pd.read_csv(f"{DATA}/Test_set.csv")
X, y, Xte = build(tr, te)
print("distinct last_week_pay values:", sorted(tr.last_week_pay.dropna().unique())[:30])


def sharpen(d, raw):
    d = d.copy()
    W2M = 12 / 52.0
    term_m = d["loan_term"] * 12
    r = d["interest_rate"] / 1200.0
    emi = d["emi"]
    m_paid = np.round(raw["last_week_pay"].values * W2M)
    m_elapsed = np.round(d["weeks_elapsed_est"].values * W2M)
    d["months_paid_int"] = m_paid
    d["months_elapsed_int"] = m_elapsed
    d["months_behind"] = m_elapsed - m_paid
    d["months_remaining_at_last_pay"] = term_m - m_paid
    k = np.clip(m_paid, 0, None)
    bal_k = d["loan_amnt"] * (1 + r) ** k - emi * ((1 + r) ** k - 1) / r
    sched = emi * k - (d["loan_amnt"] - bal_k)
    d["int_ratio_int"] = d["interest_receive"] / (sched + 1)
    d["int_gap_int"] = d["interest_receive"] - sched
    # implied number of payments from interest received (invert cumulative interest by bisection)
    lo = np.zeros(len(d)); hi = np.full(len(d), 80.0)
    L, R_, E, I = d["loan_amnt"].values, r.values, emi.values, d["interest_receive"].values
    for _ in range(40):
        mid = (lo + hi) / 2
        bal = L * (1 + R_) ** mid - E * ((1 + R_) ** mid - 1) / R_
        cum_int = E * mid - (L - bal)
        go_up = cum_int < I
        lo = np.where(go_up, mid, lo); hi = np.where(go_up, hi, mid)
    d["months_implied_by_interest"] = (lo + hi) / 2
    d["implied_minus_paid"] = d["months_implied_by_interest"] - m_paid
    d["implied_minus_elapsed"] = d["months_implied_by_interest"] - m_elapsed
    # expected remaining balance at last payment
    d["balance_at_last_pay_ratio"] = bal_k / d["loan_amnt"]
    return d


Xs = sharpen(X, tr)
print(tr.groupby(Xs.months_behind.clip(-1, 12))["default"].agg(["mean", "size"]))
print(tr.groupby(pd.cut(Xs.int_ratio_int, [-1, 0.5, 0.8, 0.9, 0.95, 0.99, 1.01, 1.05, 1.1, 1.3, 2, 1e9]))["default"].agg(["mean", "size"]))
print(tr.groupby(pd.cut(Xs.implied_minus_paid, [-100, -3, -2, -1, -0.5, 0.5, 1, 2, 3, 100]))["default"].agg(["mean", "size"]))

t0 = time.time(); oof, it, models = lgb_cv(X, y); report("base", oof, y); print(it, f"{time.time()-t0:.0f}s")
t0 = time.time(); oof, it, models = lgb_cv(Xs, y); report("sharpened", oof, y); print(it, f"{time.time()-t0:.0f}s")
imp = pd.Series(models[-1].feature_importance("gain"), index=Xs.columns).sort_values(ascending=False)
print(imp.head(20).round(0).to_string())
err = (oof > 0.5) != y
print("error rate by months_behind:\n", pd.Series(err).groupby(Xs.months_behind.clip(-1, 12)).agg(["mean", "sum"]))
