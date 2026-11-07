"""Experiment 3: vintage-elapsed-time features derived from ID ordering (train+test, no target)."""
import sys, os, time, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import prep, lgb_cv, report

DATA = sys.argv[1]
tr = pd.read_csv(f"{DATA}/Train_set.csv"); te = pd.read_csv(f"{DATA}/Test_set.csv")
y = tr["default"].values


def vintage_features(tr, te, window=1500, q=0.95):
    """Estimate weeks elapsed since issue for each ID using a rolling high quantile of
    last_week_pay over ID order (train+test pooled; no target used)."""
    allp = pd.concat([tr.drop(columns=["default"]), te]).sort_values("ID").reset_index(drop=True)
    lwp = allp["last_week_pay"]
    roll = lwp.rolling(window, center=True, min_periods=200).quantile(q)
    roll = roll.bfill().ffill()
    # enforce monotone non-increasing in ID (older loans => more weeks elapsed)
    # each loan's elapsed time must be >= that of every newer loan (higher ID)
    roll = pd.Series(np.maximum.accumulate(roll.values[::-1])[::-1], index=allp.index)
    allp["weeks_elapsed_est"] = roll.values
    allp["id_rank"] = allp["ID"].rank(pct=True)
    return allp.set_index("ID")[["weeks_elapsed_est", "id_rank"]]


def add_vintage(d, raw, vf):
    v = vf.loc[raw["ID"].values]
    d = d.copy()
    d["weeks_elapsed_est"] = v["weeks_elapsed_est"].values
    d["id_rank"] = v["id_rank"].values
    d["weeks_since_last_pay"] = d["weeks_elapsed_est"] - raw["last_week_pay"].values
    d["pay_recency_ratio"] = raw["last_week_pay"].values / (d["weeks_elapsed_est"] + 1)
    term_weeks = raw["loan_term"].str.extract(r"(\d+)").astype(float)[0].values * 52
    d["term_overrun_weeks"] = raw["last_week_pay"].values - term_weeks
    d["elapsed_vs_term"] = d["weeks_elapsed_est"] / term_weeks
    d["loan_should_be_done"] = (d["weeks_elapsed_est"] > term_weeks).astype(int)
    return d


vf = vintage_features(tr, te)
print(vf.describe().T)
X = add_vintage(prep(tr, fe=True, use_id=True), tr, vf)
print("default rate by weeks_since_last_pay bins:\n",
      tr.groupby(pd.cut(X.weeks_since_last_pay, [-1e9, -10, 0, 4, 8, 13, 26, 52, 104, 1e9]))["default"].agg(["mean", "size"]))

t0 = time.time(); oof, it, models = lgb_cv(X, y); report("FE+ID+vintage", oof, y); print("iters", it, f"{time.time()-t0:.0f}s")
imp = pd.Series(models[-1].feature_importance("gain"), index=X.columns).sort_values(ascending=False)
print(imp.head(20).round(0).to_string())
np.save(os.path.join(os.path.dirname(os.path.abspath(__file__)), "oof_lgb_fe_id_vintage.npy"), oof)

# sensitivity to window/quantile
for w, q in [(800, 0.95), (3000, 0.95), (1500, 0.90), (1500, 0.99)]:
    vf2 = vintage_features(tr, te, w, q)
    X2 = add_vintage(prep(tr, fe=True, use_id=True), tr, vf2)
    oof2, it2, _ = lgb_cv(X2, y); report(f"window={w} q={q}", oof2, y)
