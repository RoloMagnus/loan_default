"""Experiment 5: CatBoost and XGBoost 5-fold CV (same folds as exp04) -> OOF + test preds."""
import sys, os, time, warnings
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from features import build, CAT_COLS, report

DATA = sys.argv[1]
tr = pd.read_csv(f"{DATA}/Train_set.csv"); te = pd.read_csv(f"{DATA}/Test_set.csv")
X, y, Xte = build(tr, te)
folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(X, y))

# ---------- CatBoost (native categoricals)
from catboost import CatBoostClassifier, Pool
Xc = X.copy(); Xtec = Xte.copy()
for c in CAT_COLS:
    Xc[c] = Xc[c].astype(str); Xtec[c] = Xtec[c].astype(str)
oof = np.zeros(len(X)); tp = np.zeros(len(Xte)); iters = []; t0 = time.time()
for a, b in folds:
    m = CatBoostClassifier(iterations=5000, learning_rate=0.05, depth=7, l2_leaf_reg=3, random_seed=42,
                           eval_metric="Logloss", od_type="Iter", od_wait=200, verbose=False, thread_count=6)
    m.fit(Pool(Xc.iloc[a], y[a], cat_features=CAT_COLS), eval_set=Pool(Xc.iloc[b], y[b], cat_features=CAT_COLS), use_best_model=True)
    oof[b] = m.predict_proba(Xc.iloc[b])[:, 1]; tp += m.predict_proba(Xtec)[:, 1] / 5; iters.append(m.get_best_iteration())
report("CatBoost", oof, y); print("iters", iters, f"{time.time()-t0:.0f}s", flush=True)
np.save(os.path.join(HERE, "oof_cat.npy"), oof); np.save(os.path.join(HERE, "test_cat.npy"), tp)

# ---------- XGBoost (native categorical support)
import xgboost as xgb
oof = np.zeros(len(X)); tp = np.zeros(len(Xte)); iters = []; t0 = time.time()
params = dict(objective="binary:logistic", eval_metric="logloss", eta=0.03, max_depth=7, min_child_weight=5,
              subsample=0.8, colsample_bytree=0.7, reg_lambda=1.0, tree_method="hist", nthread=6, seed=42, max_cat_to_onehot=1)
dte = xgb.DMatrix(Xte, enable_categorical=True)
for a, b in folds:
    dtr = xgb.DMatrix(X.iloc[a], y[a], enable_categorical=True); dva = xgb.DMatrix(X.iloc[b], y[b], enable_categorical=True)
    m = xgb.train(params, dtr, 5000, evals=[(dva, "va")], early_stopping_rounds=200, verbose_eval=False)
    oof[b] = m.predict(dva, iteration_range=(0, m.best_iteration + 1)); tp += m.predict(dte, iteration_range=(0, m.best_iteration + 1)) / 5
    iters.append(m.best_iteration)
report("XGBoost", oof, y); print("iters", iters, f"{time.time()-t0:.0f}s", flush=True)
np.save(os.path.join(HERE, "oof_xgb.npy"), oof); np.save(os.path.join(HERE, "test_xgb.npy"), tp)
