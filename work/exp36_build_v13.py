"""Submission v13 build: v12 features + local payment-month phase features (W=40). Resumable, stage-wise.
Stages (each saved to work/v13_*.npy and skipped if present): lgb, cat42, xgb, cat7."""
import os, sys, json, time, numpy as np, pandas as pd, lightgbm as lgb
import xgboost as xgb_lib
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import StratifiedKFold
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from features import build_v12, local_k_features, CAT_COLS
DATA = sys.argv[1]
tr = pd.read_csv(f"{DATA}/Train_set.csv"); te = pd.read_csv(f"{DATA}/Test_set.csv")
X, y, Xte = build_v12(tr, te)
Ftr, Fte = local_k_features(tr, te, Ws=(40,), share=0.08)
X = pd.concat([X.reset_index(drop=True), Ftr], axis=1); Xte = pd.concat([Xte.reset_index(drop=True), Fte], axis=1)
acc = lambda p: ((p > 0.5) == y).mean()
NF = 10; folds = list(StratifiedKFold(NF, shuffle=True, random_state=42).split(X, y))
have = lambda n: os.path.exists(os.path.join(HERE, f"v13_{n}_oof.npy"))
save = lambda n, o, t: (np.save(os.path.join(HERE, f"v13_{n}_oof.npy"), o), np.save(os.path.join(HERE, f"v13_{n}_tst.npy"), t))
print("features:", X.shape, flush=True)


def stage_lgb():
    lp = json.load(open(os.path.join(HERE, "lgb_best.json")))["params"]
    oof = np.zeros(len(X)); tst = np.zeros(len(Xte)); SL = [42, 7, 2024]; t0 = time.time()
    for s in SL:
        P = dict(objective="binary", verbose=-1, n_jobs=12, bagging_freq=1, seed=s, **lp)
        for a, b in folds:
            m = lgb.train(P, lgb.Dataset(X.iloc[a], y[a]), 4000, valid_sets=[lgb.Dataset(X.iloc[b], y[b])], callbacks=[lgb.early_stopping(150, verbose=False)])
            oof[b] += m.predict(X.iloc[b], num_iteration=m.best_iteration) / len(SL); tst += m.predict(Xte, num_iteration=m.best_iteration) / (NF * len(SL))
    print(f"LGB x3: {acc(oof):.5f} [{time.time()-t0:.0f}s]", flush=True); save("lgb", oof, tst)


def stage_xgb():
    oof = np.zeros(len(X)); tst = np.zeros(len(Xte)); SL = [42, 7, 2024]; t0 = time.time()
    dtest = xgb_lib.DMatrix(Xte, enable_categorical=True)
    for s in SL:
        xp = dict(objective="binary:logistic", eval_metric="logloss", eta=0.03, max_depth=7, min_child_weight=5, subsample=0.8, colsample_bytree=0.7, reg_lambda=1.0, tree_method="hist", nthread=12, seed=s)
        for a, b in folds:
            dtr = xgb_lib.DMatrix(X.iloc[a], y[a], enable_categorical=True); dva = xgb_lib.DMatrix(X.iloc[b], y[b], enable_categorical=True)
            m = xgb_lib.train(xp, dtr, 5000, evals=[(dva, "va")], early_stopping_rounds=150, verbose_eval=False)
            oof[b] += m.predict(dva, iteration_range=(0, m.best_iteration + 1)) / len(SL); tst += m.predict(dtest, iteration_range=(0, m.best_iteration + 1)) / (NF * len(SL))
    print(f"XGB x3: {acc(oof):.5f} [{time.time()-t0:.0f}s]", flush=True); save("xgb", oof, tst)


def stage_cat(seed):
    cp = json.load(open(os.path.join(HERE, "cat_best.json")))["params"]
    Xc = X.copy(); Xtc = Xte.copy()
    for c in CAT_COLS: Xc[c] = Xc[c].astype(str); Xtc[c] = Xtc[c].astype(str)
    oof = np.zeros(len(X)); tst = np.zeros(len(Xte)); t0 = time.time()
    for i, (a, b) in enumerate(folds):
        m = CatBoostClassifier(iterations=3000, eval_metric="Logloss", od_type="Iter", od_wait=100, random_seed=seed, verbose=False, thread_count=12, **cp)
        m.fit(Pool(Xc.iloc[a], y[a], cat_features=CAT_COLS), eval_set=Pool(Xc.iloc[b], y[b], cat_features=CAT_COLS), use_best_model=True)
        oof[b] = m.predict_proba(Xc.iloc[b])[:, 1]; tst += m.predict_proba(Xtc)[:, 1] / NF
        print(f"  cat seed {seed} fold {i+1}/{NF} [{time.time()-t0:.0f}s]", flush=True)
    print(f"Cat seed {seed}: {acc(oof):.5f} [{time.time()-t0:.0f}s]", flush=True); save(f"cat{seed}", oof, tst)


for name, fn in [("lgb", stage_lgb), ("cat42", lambda: stage_cat(42)), ("xgb", stage_xgb), ("cat7", lambda: stage_cat(7))]:
    if have(name): print("skip", name, flush=True); continue
    fn()
print("ALL STAGES DONE", flush=True)
