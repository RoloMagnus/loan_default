"""Submission v3: blend 0.1*LGB(retuned) + 0.7*CatBoost + 0.2*XGBoost on exact features (CV 0.9431)."""
import os, sys, json, time, numpy as np, pandas as pd, lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from features import build, report
DATA = sys.argv[1]
tr = pd.read_csv(f"{DATA}/Train_set.csv"); te = pd.read_csv(f"{DATA}/Test_set.csv")
X, y, Xte = build(tr, te)
best = json.load(open(os.path.join(HERE, "lgb_best.json")))["params"]
P = dict(objective="binary", verbose=-1, n_jobs=12, bagging_freq=1, seed=42, **best)
oof = np.zeros(len(X)); tp = np.zeros(len(Xte)); t0 = time.time()
for a, b in StratifiedKFold(5, shuffle=True, random_state=42).split(X, y):
    m = lgb.train(P, lgb.Dataset(X.iloc[a], y[a]), 4000, valid_sets=[lgb.Dataset(X.iloc[b], y[b])], callbacks=[lgb.early_stopping(150, verbose=False)])
    oof[b] = m.predict(X.iloc[b], num_iteration=m.best_iteration); tp += m.predict(Xte, num_iteration=m.best_iteration) / 5
print(f"retuned LGB refit {time.time()-t0:.0f}s"); report("LGB retuned", oof, y)
np.save(os.path.join(HERE, "test_lgb_retuned.npy"), tp)
cat_o = np.load(os.path.join(HERE, "oof_cat.npy")); xgb_o = np.load(os.path.join(HERE, "oof_xgb.npy"))
cat_t = np.load(os.path.join(HERE, "test_cat.npy")); xgb_t = np.load(os.path.join(HERE, "test_xgb.npy"))
W = (0.1, 0.7, 0.2)
report("blend OOF", W[0] * oof + W[1] * cat_o + W[2] * xgb_o, y)
blend_t = W[0] * tp + W[1] * cat_t + W[2] * xgb_t
sub = pd.DataFrame({"ID": te["ID"], "default": (blend_t > 0.5).astype(int)})
sub.to_csv(os.path.join(DATA, "submissions", "submission_v3_blend_exact.csv"), index=False)
v2 = pd.read_csv(os.path.join(DATA, "submissions", "submission_v2_lgb_exact.csv"))
print("rows", len(sub), "pred default rate", sub["default"].mean().round(4), "| differs from v2 on", (sub["default"].values != v2["default"].values).sum(), "rows")
