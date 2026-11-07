"""Experiment 2: error structure diagnostics + ID-as-vintage feature."""
import sys, os, time, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import prep, lgb_cv, report

DATA = sys.argv[1]
tr = pd.read_csv(f"{DATA}/Train_set.csv"); te = pd.read_csv(f"{DATA}/Test_set.csv")
y = tr["default"].values
pd.set_option("display.width", 200)

# ---- diagnostics
print("default rate by last_week_pay missing:\n", tr.groupby(tr.last_week_pay.isna())["default"].agg(["mean", "size"]))
print("default rate by total_current_balance missing:\n", tr.groupby(tr.total_current_balance.isna())["default"].agg(["mean", "size"]))
print("default rate by ID decile:\n", tr.groupby(pd.qcut(tr.ID, 10))["default"].agg(["mean", "size"]))
print("test ID deciles vs train quantiles:", np.quantile(te.ID, [0, .25, .5, .75, 1]), np.quantile(tr.ID, [0, .25, .5, .75, 1]))
X = prep(tr, fe=True)
print("default rate by int_ratio_to_sched bins:\n", tr.groupby(pd.cut(X.int_ratio_to_sched, [-1, 0.5, 0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.3, 2, 1e9]))["default"].agg(["mean", "size"]))
print("default rate by last_week_pay bins:\n", tr.groupby(pd.cut(tr.last_week_pay, [-1, 4, 13, 26, 52, 78, 104, 156, 208, 260, 300]))["default"].agg(["mean", "size"]))

# ---- CV: FE + ID
X2 = prep(tr, fe=True, use_id=True)
t0 = time.time(); oof2, it2, models = lgb_cv(X2, y); report("FE+ID", oof2, y); print("iters", it2, f"{time.time()-t0:.0f}s")
imp = pd.Series(models[-1].feature_importance("gain"), index=X2.columns).sort_values(ascending=False)
print(imp.head(15).round(0).to_string())
np.save(os.path.join(os.path.dirname(os.path.abspath(__file__)), "oof_lgb_fe_id.npy"), oof2)

# where are the errors?
err = ((oof2 > 0.5) != y)
print("error rate by ID decile:\n", pd.Series(err).groupby(pd.qcut(tr.ID, 10)).mean().round(3))
print("error rate by last_week_pay bin:\n", pd.Series(err).groupby(pd.cut(tr.last_week_pay, [-1, 4, 13, 26, 52, 78, 104, 156, 208, 260, 300])).mean().round(3))
print("error rate by true class:", pd.Series(err).groupby(y).mean().round(3).to_dict())
