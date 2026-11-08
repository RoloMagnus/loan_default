"""Experiment 20: does 10-fold CV beat 5-fold for the retuned LGB? (more training data per fold)"""
import os, sys, json, time, numpy as np, pandas as pd, lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from scipy.stats import binomtest
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from features import build
DATA = sys.argv[1]
tr = pd.read_csv(f"{DATA}/Train_set.csv"); te = pd.read_csv(f"{DATA}/Test_set.csv")
X, y, Xte = build(tr, te)
best = json.load(open(os.path.join(HERE, "lgb_best.json")))["params"]
P = dict(objective="binary", verbose=-1, n_jobs=6, bagging_freq=1, seed=42, **best)
oof5 = np.load(os.path.join(HERE, "oof_lgb_tuned.npy"))
acc = lambda p: ((p > 0.5) == y).mean()

t0 = time.time()
oof10 = np.zeros(len(X))
for a, b in StratifiedKFold(10, shuffle=True, random_state=42).split(X, y):
    m = lgb.train(P, lgb.Dataset(X.iloc[a], y[a]), 4000, valid_sets=[lgb.Dataset(X.iloc[b], y[b])],
                  callbacks=[lgb.early_stopping(150, verbose=False)])
    oof10[b] = m.predict(X.iloc[b], num_iteration=m.best_iteration)
ca, cb = (oof5 > 0.5) == y, (oof10 > 0.5) == y
f, br = int((~ca & cb).sum()), int((ca & ~cb).sum())
pv = binomtest(f, f + br, 0.5).pvalue if f + br else 1
print(f"LGB 5-fold {acc(oof5):.5f} -> 10-fold {acc(oof10):.5f}  fixed={f} broken={br} net={f-br} McNemar p={pv:.4f}  [{time.time()-t0:.0f}s]")
np.save(os.path.join(HERE, "oof_lgb_10fold.npy"), oof10)
