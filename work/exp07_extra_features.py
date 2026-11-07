"""Experiment 7: extra features aimed at the stopped-payer segment + row-order check."""
import sys, os, time, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from features import build, lgb_cv, report
from sklearn.model_selection import StratifiedKFold
pd.set_option("display.width", 200)

DATA = sys.argv[1]
tr = pd.read_csv(f"{DATA}/Train_set.csv"); te = pd.read_csv(f"{DATA}/Test_set.csv")
print("default rate by train ROW-ORDER decile:", tr.groupby(pd.qcut(np.arange(len(tr)), 10))["default"].mean().round(3).values)
print("lag-1 autocorr of default in file order:", round(pd.Series(tr["default"]).autocorr(1), 4))

X, y, Xte = build(tr, te)
base, it, _ = lgb_cv(X, y); report("base", base, y)


def extra(d, raw):
    d = d.copy()
    r = raw["interest_rate"].values / 1200.0
    term_m = raw["loan_term"].str.extract(r"(\d+)").astype(float)[0].values * 12
    emi = d["emi"].values
    k_el = np.clip(d["weeks_elapsed_est"].values / 52 * 12, 0, term_m)
    bal = raw["loan_amnt"].values * (1 + r) ** k_el - emi * ((1 + r) ** k_el - 1) / r
    sched_int_elapsed = emi * k_el - (raw["loan_amnt"].values - bal)
    d["int_ratio_to_elapsed"] = raw["interest_receive"].values / (sched_int_elapsed + 1)
    d["int_gap_to_elapsed"] = raw["interest_receive"].values - sched_int_elapsed
    d["months_behind"] = np.round(d["weeks_since_last_pay"].values / 4.345)
    d["months_elapsed"] = np.round(d["weeks_elapsed_est"].values / 4.345)
    d["months_remaining_at_stop"] = term_m - d["months_paid"].values
    d["remaining_principal_at_stop"] = raw["loan_amnt"].values * (1 + r) ** d["months_paid"].values - emi * ((1 + r) ** d["months_paid"].values - 1) / r
    d["remaining_principal_to_income"] = d["remaining_principal_at_stop"] / (raw["annual_income"].values + 1)
    d["int_excess_over_sched"] = (d["int_ratio_to_sched"] > 1.02).astype(int)
    d["total_paid_est"] = emi * d["months_paid"].values
    d["total_paid_to_loan"] = d["total_paid_est"] / raw["loan_amnt"].values
    return d


X1 = extra(X, tr); o1, _, m1 = lgb_cv(X1, y); report("base+extra", o1, y)
imp = pd.Series(m1[-1].feature_importance("gain"), index=X1.columns).sort_values(ascending=False); print(imp.head(12).round(0).to_string())

# OOF target encoding of vintage neighbourhood (ID-sorted rolling mean of target, out-of-fold)
def vintage_te(X, raw, y, folds, window=500):
    te_col = np.full(len(X), np.nan)
    order = np.argsort(raw["ID"].values)
    for a, b in folds:
        s = pd.Series(np.nan, index=np.arange(len(X))); s.iloc[a] = y[a]
        s_sorted = s.iloc[order]
        roll = s_sorted.rolling(window, center=True, min_periods=50).mean().bfill().ffill()
        vals = pd.Series(roll.values, index=s_sorted.index)
        te_col[b] = vals.loc[b].values
    return te_col

folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(X, y))
X2 = X1.copy(); X2["vintage_default_rate"] = vintage_te(X1, tr, y, folds)
o2, _, _ = lgb_cv(X2, y); report("base+extra+vintageTE", o2, y)
np.save(os.path.join(HERE, "oof_exp07_extra.npy"), o1)
