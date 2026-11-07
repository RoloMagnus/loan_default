"""Experiment 9: first submission (blend of tuned LGB + CatBoost + XGBoost) + cheap leakage checks."""
import os, sys, json, time, numpy as np, pandas as pd, lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from features import build, report
DATA = sys.argv[1]
tr = pd.read_csv(f"{DATA}/Train_set.csv"); te = pd.read_csv(f"{DATA}/Test_set.csv")

# ---- leakage checks
key = ["loan_amnt", "loan_term", "interest_rate", "loan_subgrade", "annual_income", "debt_to_income", "revolving_balance", "total_acc", "interest_receive"]
m = te.merge(tr[key + ["default"]], on=key, how="inner")
print("test rows matching a train row on 9 exact feature values:", len(m), "of", len(te))
print("train rows with interest_receive==0:", (tr.interest_receive == 0).sum(), "default rate", tr.loc[tr.interest_receive == 0, "default"].mean().round(3))
print("train rows with last_week_pay==0:", (tr.last_week_pay == 0).sum(), "default rate", tr.loc[tr.last_week_pay == 0, "default"].mean().round(3))
print("train rows with last_week_pay NaN:", tr.last_week_pay.isna().sum(), "default rate", tr.loc[tr.last_week_pay.isna(), "default"].mean().round(3))

# ---- tuned LGB, 5-fold, test preds averaged
X, y, Xte = build(tr, te)
best = json.load(open(os.path.join(HERE, "lgb_best.json")))
params = dict(objective="binary", verbose=-1, n_jobs=12, seed=42, bagging_freq=1, **best["params"])
oof = np.zeros(len(X)); tp = np.zeros(len(Xte)); t0 = time.time()
for a, b in StratifiedKFold(5, shuffle=True, random_state=42).split(X, y):
    mdl = lgb.train(params, lgb.Dataset(X.iloc[a], y[a]), 4000, valid_sets=[lgb.Dataset(X.iloc[b], y[b])],
                    callbacks=[lgb.early_stopping(150, verbose=False)])
    oof[b] = mdl.predict(X.iloc[b], num_iteration=mdl.best_iteration); tp += mdl.predict(Xte, num_iteration=mdl.best_iteration) / 5
report("LGB tuned", oof, y); print(f"{time.time()-t0:.0f}s")
np.save(os.path.join(HERE, "test_lgb_tuned.npy"), tp)
cat = np.load(os.path.join(HERE, "test_cat.npy")); xgb = np.load(os.path.join(HERE, "test_xgb.npy"))
blend = 0.6 * tp + 0.2 * cat + 0.2 * xgb
oof_blend = 0.6 * oof + 0.2 * np.load(os.path.join(HERE, "oof_cat.npy")) + 0.2 * np.load(os.path.join(HERE, "oof_xgb.npy"))
report("blend", oof_blend, y)
sub = pd.DataFrame({"ID": te["ID"], "default": (blend > 0.5).astype(int)})
os.makedirs(os.path.join(DATA, "submissions"), exist_ok=True)
sub.to_csv(os.path.join(DATA, "submissions", "submission_v1_blend.csv"), index=False)
print("submission rows:", len(sub), "predicted default rate:", sub["default"].mean().round(4), "(train rate 0.2375)")
