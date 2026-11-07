"""Experiment 6: residual diagnostics on the vintage model + sharper vintage estimators."""
import sys, os, time, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from features import prep, add_vintage, lgb_cv, report, CAT_COLS
pd.set_option("display.width", 200)

DATA = sys.argv[1]
tr = pd.read_csv(f"{DATA}/Train_set.csv"); te = pd.read_csv(f"{DATA}/Test_set.csv")
y = tr["default"].values
oof = np.load(os.path.join(HERE, "oof_lgb_fe_id_vintage.npy"))
pred = (oof > 0.5).astype(int); err = pred != y
print("overall acc", 1 - err.mean())

allp = pd.concat([tr.drop(columns=["default"]), te]).sort_values("ID").reset_index(drop=True)


def vt(window, q, how="quantile"):
    s = allp["last_week_pay"]
    roll = (s.rolling(window, center=True, min_periods=50).max() if how == "max"
            else s.rolling(window, center=True, min_periods=50).quantile(q)).bfill().ffill()
    roll = np.maximum.accumulate(roll.values[::-1])[::-1]
    out = allp[["ID"]].copy(); out["weeks_elapsed_est"] = roll; out["id_rank"] = allp["ID"].rank(pct=True)
    return out.set_index("ID")


vf = vt(1500, 0.95)
X = add_vintage(prep(tr, True, True), tr, vf)
d = tr.copy(); d["wsl"] = X["weeks_since_last_pay"].values; d["we"] = X["weeks_elapsed_est"].values
d["err"] = err; d["pred"] = pred; d["p"] = oof
d["wsl_bin"] = pd.cut(d.wsl, [-1e9, 0, 4, 8, 13, 26, 52, 104, 1e9])
print("\nerror rate / default rate / size by weeks_since_last_pay:\n", d.groupby("wsl_bin")[["err", "default"]].mean().round(3).join(d.groupby("wsl_bin").size().rename("n")))
print("\nconfusion by wsl bin (FN=default missed, FP=false alarm):")
d["FN"] = (d.default == 1) & (d.pred == 0); d["FP"] = (d.default == 0) & (d.pred == 1)
print(d.groupby("wsl_bin")[["FN", "FP"]].sum())
print("\nby vintage decile:\n", d.groupby(pd.qcut(d.ID, 10))[["err", "default"]].mean().round(3))
print("\nby loan_term x wsl>8:\n", d.groupby(["loan_term", d.wsl > 8])[["err", "default"]].mean().round(3))
# among stopped payers (wsl>13), what separates default vs non-default?
s = d[d.wsl > 13].copy()
s["int_ratio"] = X.loc[s.index, "int_ratio_to_sched"]; s["pp"] = X.loc[s.index, "pay_progress"]
print("\nstopped payers (wsl>13), n=", len(s), " default rate", s.default.mean().round(3))
for c in ["loan_amnt", "interest_rate", "annual_income", "debt_to_income", "last_week_pay", "interest_receive", "int_ratio", "pp", "total_current_balance", "revolving_balance"]:
    print(f"  {c:22s} default median={s.loc[s.default==1,c].median():10.2f}  non-default median={s.loc[s.default==0,c].median():10.2f}")
print(s.groupby("loan_grade")["default"].mean().round(3).to_dict())
print(s.groupby("loan_term")["default"].mean().round(3).to_dict())
print(s.groupby(pd.cut(s.int_ratio, [-1, .8, .9, .95, 1, 1.05, 1.2, 1e9]))[["default"]].agg(["mean", "size"]).round(3))
print("\nerror by last_week_pay missing:", d.groupby(d.last_week_pay.isna())["err"].mean().round(4).to_dict())

# ---- sharper vintage estimators
for w, q, how in [(300, 0.98, "quantile"), (150, None, "max"), (400, None, "max"), (600, 0.99, "quantile")]:
    vf2 = vt(w, q, how)
    X2 = add_vintage(prep(tr, True, True), tr, vf2)
    oof2, it2, _ = lgb_cv(X2, y); report(f"{how} window={w} q={q}", oof2, y)
