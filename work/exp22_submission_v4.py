"""Submission v4: 10-fold, multi-seed LGB(tuned) + CatBoost(tuned) + XGBoost; weight grid; paired test vs v3."""
import os, sys, json, time, numpy as np, pandas as pd, lightgbm as lgb
import xgboost as xgb_lib
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import StratifiedKFold
from scipy.stats import binomtest
import itertools
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from features import build, CAT_COLS
DATA = sys.argv[1]
tr = pd.read_csv(f"{DATA}/Train_set.csv"); te = pd.read_csv(f"{DATA}/Test_set.csv")
X, y, Xte = build(tr, te)
acc = lambda p: ((p > 0.5) == y).mean()
NF = 10

# ---------------- LGB (tuned), 3 seeds ----------------
lp = json.load(open(os.path.join(HERE, "lgb_best.json")))["params"]
t0 = time.time(); oof_l = np.zeros(len(X)); tst_l = np.zeros(len(Xte)); SL = [42, 7, 2024]
for s in SL:
    P = dict(objective="binary", verbose=-1, n_jobs=12, bagging_freq=1, seed=s, **lp)
    for a, b in StratifiedKFold(NF, shuffle=True, random_state=42).split(X, y):
        m = lgb.train(P, lgb.Dataset(X.iloc[a], y[a]), 4000, valid_sets=[lgb.Dataset(X.iloc[b], y[b])],
                      callbacks=[lgb.early_stopping(150, verbose=False)])
        oof_l[b] += m.predict(X.iloc[b], num_iteration=m.best_iteration) / len(SL)
        tst_l += m.predict(Xte, num_iteration=m.best_iteration) / (NF * len(SL))
print(f"LGB 10-fold x3 seeds: {acc(oof_l):.5f}  [{time.time()-t0:.0f}s]", flush=True)

# ---------------- XGBoost, 3 seeds ----------------
t0 = time.time(); oof_x = np.zeros(len(X)); tst_x = np.zeros(len(Xte))
dtest = xgb_lib.DMatrix(Xte, enable_categorical=True)
for s in SL:
    xp = dict(objective="binary:logistic", eval_metric="logloss", eta=0.03, max_depth=7, min_child_weight=5,
              subsample=0.8, colsample_bytree=0.7, reg_lambda=1.0, tree_method="hist", nthread=12, seed=s)
    for a, b in StratifiedKFold(NF, shuffle=True, random_state=42).split(X, y):
        dtr = xgb_lib.DMatrix(X.iloc[a], y[a], enable_categorical=True)
        dva = xgb_lib.DMatrix(X.iloc[b], y[b], enable_categorical=True)
        m = xgb_lib.train(xp, dtr, 5000, evals=[(dva, "va")], early_stopping_rounds=150, verbose_eval=False)
        oof_x[b] += m.predict(dva, iteration_range=(0, m.best_iteration + 1)) / len(SL)
        tst_x += m.predict(dtest, iteration_range=(0, m.best_iteration + 1)) / (NF * len(SL))
print(f"XGB 10-fold x3 seeds: {acc(oof_x):.5f}  [{time.time()-t0:.0f}s]", flush=True)

# ---------------- CatBoost (tuned), 2 seeds ----------------
cp = json.load(open(os.path.join(HERE, "cat_best.json")))["params"]
Xc = X.copy(); Xtc = Xte.copy()
for c in CAT_COLS: Xc[c] = Xc[c].astype(str); Xtc[c] = Xtc[c].astype(str)
t0 = time.time(); oof_c = np.zeros(len(X)); tst_c = np.zeros(len(Xte)); SC = [42, 7]
for s in SC:
    for a, b in StratifiedKFold(NF, shuffle=True, random_state=42).split(Xc, y):
        m = CatBoostClassifier(iterations=3000, eval_metric="Logloss", od_type="Iter", od_wait=100,
                               random_seed=s, verbose=False, thread_count=12, **cp)
        m.fit(Pool(Xc.iloc[a], y[a], cat_features=CAT_COLS), eval_set=Pool(Xc.iloc[b], y[b], cat_features=CAT_COLS), use_best_model=True)
        oof_c[b] += m.predict_proba(Xc.iloc[b])[:, 1] / len(SC)
        tst_c += m.predict_proba(Xtc)[:, 1] / (NF * len(SC))
print(f"Cat 10-fold x2 seeds: {acc(oof_c):.5f}  [{time.time()-t0:.0f}s]", flush=True)

for k, v in {"oof_l_v4": oof_l, "oof_x_v4": oof_x, "oof_c_v4": oof_c, "tst_l_v4": tst_l, "tst_x_v4": tst_x, "tst_c_v4": tst_c}.items():
    np.save(os.path.join(HERE, f"{k}.npy"), v)

# ---------------- blend + paired test vs v3 ----------------
best = (0, None)
for w in itertools.product(np.arange(0, 1.01, 0.05), repeat=2):
    w0, w1 = w; w2 = 1 - w0 - w1
    if w2 < -1e-9: continue
    a = acc(w0 * oof_l + w1 * oof_c + max(w2, 0) * oof_x)
    if a > best[0]: best = (a, (round(w0, 2), round(w1, 2), round(max(w2, 0), 2)))
print("v4 best weights (lgb,cat,xgb):", best, flush=True)
w = best[1]
oof_v4 = w[0] * oof_l + w[1] * oof_c + w[2] * oof_x
tst_v4 = w[0] * tst_l + w[1] * tst_c + w[2] * tst_x
oof_v3 = 0.1 * np.load(os.path.join(HERE, "oof_lgb_tuned.npy")) + 0.7 * np.load(os.path.join(HERE, "oof_cat.npy")) + 0.2 * np.load(os.path.join(HERE, "oof_xgb.npy"))
ca, cb = (oof_v3 > 0.5) == y, (oof_v4 > 0.5) == y
f, br = int((~ca & cb).sum()), int((ca & ~cb).sum())
pv = binomtest(f, f + br, 0.5).pvalue if f + br else 1
print(f"v3 {ca.mean():.5f} -> v4 {cb.mean():.5f}  fixed={f} broken={br} net={f-br} McNemar p={pv:.4f}")
sub = pd.DataFrame({"ID": te["ID"], "default": (tst_v4 > 0.5).astype(int)})
sub.to_csv(os.path.join(DATA, "submissions", "submission_v4_10fold_tunedcat.csv"), index=False)
v3 = pd.read_csv(os.path.join(DATA, "submissions", "submission_v3_blend_exact.csv"))
print("rows", len(sub), "pred rate", sub["default"].mean().round(4), "| differs from v3 on", int((sub["default"].values != v3["default"].values).sum()), "rows")
