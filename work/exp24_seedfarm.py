"""Experiment 24: seed farm — 3 extra CatBoost seeds + 2 extra LGB seeds at 10-fold; v5 candidate."""
import os, sys, json, time, numpy as np, pandas as pd, lightgbm as lgb
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import StratifiedKFold
from scipy.stats import binomtest
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from features import build, CAT_COLS
DATA = sys.argv[1]
tr = pd.read_csv(f"{DATA}/Train_set.csv"); te = pd.read_csv(f"{DATA}/Test_set.csv")
X, y, Xte = build(tr, te)
acc = lambda p: ((p > 0.5) == y).mean()
NF = 10
folds = list(StratifiedKFold(NF, shuffle=True, random_state=42).split(X, y))

# ---- extra LGB seeds
lp = json.load(open(os.path.join(HERE, "lgb_best.json")))["params"]
t0 = time.time(); oof_l2 = np.zeros(len(X)); tst_l2 = np.zeros(len(Xte)); SL = [11, 99]
for s in SL:
    P = dict(objective="binary", verbose=-1, n_jobs=12, bagging_freq=1, seed=s, **lp)
    for a, b in folds:
        m = lgb.train(P, lgb.Dataset(X.iloc[a], y[a]), 4000, valid_sets=[lgb.Dataset(X.iloc[b], y[b])],
                      callbacks=[lgb.early_stopping(150, verbose=False)])
        oof_l2[b] += m.predict(X.iloc[b], num_iteration=m.best_iteration) / len(SL)
        tst_l2 += m.predict(Xte, num_iteration=m.best_iteration) / (NF * len(SL))
print(f"LGB extra 2 seeds: {acc(oof_l2):.5f} [{time.time()-t0:.0f}s]", flush=True)

# ---- extra CatBoost seeds
cp = json.load(open(os.path.join(HERE, "cat_best.json")))["params"]
Xc = X.copy(); Xtc = Xte.copy()
for c in CAT_COLS: Xc[c] = Xc[c].astype(str); Xtc[c] = Xtc[c].astype(str)
t0 = time.time(); oof_c2 = np.zeros(len(X)); tst_c2 = np.zeros(len(Xte)); SC = [11, 2025, 777]
for s in SC:
    for a, b in folds:
        m = CatBoostClassifier(iterations=3000, eval_metric="Logloss", od_type="Iter", od_wait=100,
                               random_seed=s, verbose=False, thread_count=12, **cp)
        m.fit(Pool(Xc.iloc[a], y[a], cat_features=CAT_COLS), eval_set=Pool(Xc.iloc[b], y[b], cat_features=CAT_COLS), use_best_model=True)
        oof_c2[b] += m.predict_proba(Xc.iloc[b])[:, 1] / len(SC)
        tst_c2 += m.predict_proba(Xtc)[:, 1] / (NF * len(SC))
print(f"Cat extra 3 seeds: {acc(oof_c2):.5f} [{time.time()-t0:.0f}s]", flush=True)

# ---- combine with v4 arrays: 5-seed LGB, 5-seed Cat, 3-seed XGB
oof_l = (3 * np.load(os.path.join(HERE, "oof_l_v4.npy")) + 2 * oof_l2) / 5
tst_l = (3 * np.load(os.path.join(HERE, "tst_l_v4.npy")) + 2 * tst_l2) / 5
oof_c = (2 * np.load(os.path.join(HERE, "oof_c_v4.npy")) + 3 * oof_c2) / 5
tst_c = (2 * np.load(os.path.join(HERE, "tst_c_v4.npy")) + 3 * tst_c2) / 5
oof_x = np.load(os.path.join(HERE, "oof_x_v4.npy")); tst_x = np.load(os.path.join(HERE, "tst_x_v4.npy"))
print(f"LGB 5-seed: {acc(oof_l):.5f}  Cat 5-seed: {acc(oof_c):.5f}")
oof_v5 = 0.25 * oof_l + 0.70 * oof_c + 0.05 * oof_x
tst_v5 = 0.25 * tst_l + 0.70 * tst_c + 0.05 * tst_x
oof_v4 = 0.25 * np.load(os.path.join(HERE, "oof_l_v4.npy")) + 0.70 * np.load(os.path.join(HERE, "oof_c_v4.npy")) + 0.05 * oof_x
ca, cb = (oof_v4 > 0.5) == y, (oof_v5 > 0.5) == y
f, b = int((~ca & cb).sum()), int((ca & ~cb).sum())
pv = binomtest(f, f + b, 0.5).pvalue if f + b else 1.0
print(f"v4 {ca.mean():.5f} -> v5 seedfarm {cb.mean():.5f} fixed={f} broken={b} net={f-b} p={pv:.4f}")
sub = pd.DataFrame({"ID": te["ID"], "default": (tst_v5 > 0.5).astype(int)})
sub.to_csv(os.path.join(DATA, "submissions", "submission_v5_seedfarm.csv"), index=False)
v4s = pd.read_csv(os.path.join(DATA, "submissions", "submission_v4_10fold_tunedcat.csv"))
print("v5 rows differing from v4:", int((sub['default'].values != v4s['default'].values).sum()), "| pred rate", sub["default"].mean().round(4))
