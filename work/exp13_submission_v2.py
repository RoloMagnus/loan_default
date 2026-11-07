"""Submission v2: tuned LightGBM on the exact-schedule feature set (5-fold average, 3 seeds)."""
import os, sys, json, time, numpy as np, pandas as pd, lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from features import build, report
DATA = sys.argv[1]; SEEDS = [42, 1, 2]
tr = pd.read_csv(f"{DATA}/Train_set.csv"); te = pd.read_csv(f"{DATA}/Test_set.csv")
X, y, Xte = build(tr, te)
best = json.load(open(os.path.join(HERE, "lgb_best.json")))["params"]
oof = np.zeros(len(X)); tp = np.zeros(len(Xte)); t0 = time.time()
for s in SEEDS:
    P = dict(objective="binary", verbose=-1, n_jobs=12, bagging_freq=1, seed=s, **best)
    for a, b in StratifiedKFold(5, shuffle=True, random_state=42).split(X, y):
        m = lgb.train(P, lgb.Dataset(X.iloc[a], y[a]), 4000, valid_sets=[lgb.Dataset(X.iloc[b], y[b])], callbacks=[lgb.early_stopping(150, verbose=False)])
        oof[b] += m.predict(X.iloc[b], num_iteration=m.best_iteration) / len(SEEDS)
        tp += m.predict(Xte, num_iteration=m.best_iteration) / (5 * len(SEEDS))
    print("seed", s, "done", f"{time.time()-t0:.0f}s", flush=True)
report("LGB exact, 3-seed avg", oof, y)
np.save(os.path.join(HERE, "oof_lgb_exact_3seed.npy"), oof); np.save(os.path.join(HERE, "test_lgb_exact_3seed.npy"), tp)
sub = pd.DataFrame({"ID": te["ID"], "default": (tp > 0.5).astype(int)})
sub.to_csv(os.path.join(DATA, "submissions", "submission_v2_lgb_exact.csv"), index=False)
print("rows", len(sub), "predicted default rate", sub["default"].mean().round(4))
