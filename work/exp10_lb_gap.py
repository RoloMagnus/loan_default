"""Experiment 10: why is LB (0.9291) below CV (0.9321)? consistency + hidden-structure checks."""
import os, sys, numpy as np, pandas as pd, lightgbm as lgb
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from features import build
pd.set_option("display.width", 200)
DATA = sys.argv[1]
tr = pd.read_csv(f"{DATA}/Train_set.csv"); te = pd.read_csv(f"{DATA}/Test_set.csv"); ss = pd.read_csv(f"{DATA}/Sample_Submission.csv")
sub = pd.read_csv(f"{DATA}/submissions/submission_v1_blend.csv")
X, y, Xte = build(tr, te)

# (a) sample submission informativeness
print("sample_sub default rate:", ss["default"].mean().round(4), " agreement with our preds:", (ss["default"].values == sub["default"].values).mean().round(4))

# (b) prediction-rate consistency
oof = 0.6 * np.load(os.path.join(HERE, "oof_lgb_tuned.npy")) + 0.2 * np.load(os.path.join(HERE, "oof_cat.npy")) + 0.2 * np.load(os.path.join(HERE, "oof_xgb.npy"))
print("train actual rate", y.mean().round(4), " OOF predicted rate", (oof > 0.5).mean().round(4), " test predicted rate", sub["default"].mean().round(4))
dec = pd.qcut(tr["ID"], 10, labels=False, retbins=True)
bins = dec[1]; bins[0] = -np.inf; bins[-1] = np.inf
print("predicted rate by ID decile, train-OOF vs test:")
print(pd.DataFrame({"train_oof": pd.Series(oof > 0.5).groupby(pd.cut(tr["ID"], bins)).mean().values,
                    "train_actual": pd.Series(y).groupby(pd.cut(tr["ID"], bins)).mean().values,
                    "test_pred": sub["default"].groupby(pd.cut(te["ID"], bins)).mean().values}).round(3))
# feature distribution comparison
cmp = pd.DataFrame({"train_mean": X.select_dtypes("number").mean(), "test_mean": Xte.select_dtypes("number").mean()})
cmp["rel_diff"] = (cmp.test_mean - cmp.train_mean) / (cmp.train_mean.abs() + 1e-9)
print("largest train/test mean differences:\n", cmp.reindex(cmp.rel_diff.abs().sort_values(ascending=False).index).head(8).round(4))
for c in ["home_ownership", "state_code"]:
    print(c, "codes equal categories:", list(X[c].cat.categories) == list(Xte[c].cat.categories))

# adversarial validation
Z = pd.concat([X, Xte]); z = np.r_[np.zeros(len(X)), np.ones(len(Xte))]
p = cross_val_predict(lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, verbose=-1, n_jobs=12), Z, z, cv=3, method="predict_proba")[:, 1]
print("adversarial AUC (0.5 = train/test indistinguishable):", round(roc_auc_score(z, p), 4))

# (c) local label structure along ID
ids = tr[["ID"]].values.astype(float)
for k in [1, 5, 25, 200]:
    pk = cross_val_predict(KNeighborsClassifier(k), ids, y, cv=StratifiedKFold(5, shuffle=True, random_state=42), method="predict_proba")[:, 1]
    print(f"KNN on ID only, k={k}: OOF acc={((pk > 0.5) == y).mean():.4f}")

# (d) exact-schedule deviation
dev = (tr["interest_receive"] - X["interest_sched_to_date"]) / (X["interest_sched_to_date"] + 1)
print("rel. deviation from schedule, quantiles by class:\n", pd.DataFrame({"dev": dev, "y": y}).groupby("y")["dev"].quantile([.05, .25, .5, .75, .95]).unstack().round(3))
print("share |dev|<1% by class:", pd.Series(dev.abs() < 0.01).groupby(y).mean().round(3).to_dict())
