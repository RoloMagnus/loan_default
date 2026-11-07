"""Experiment 15: paired OOF comparison champion (3-seed LGB) vs refreshed blend; McNemar + paired bootstrap."""
import os, sys, itertools, numpy as np, pandas as pd
from scipy.stats import binomtest
HERE = os.path.dirname(os.path.abspath(__file__))
y = pd.read_csv(f"{sys.argv[1]}/Train_set.csv")["default"].values
champ0 = np.load(os.path.join(HERE, "oof_lgb_exact_3seed.npy"))
champ = np.load(os.path.join(HERE, "oof_lgb_tuned.npy"))  # retuned on exact features
cat = np.load(os.path.join(HERE, "oof_cat.npy")); xgb = np.load(os.path.join(HERE, "oof_xgb.npy"))
acc = lambda p, t=0.5: ((p > t) == y).mean()
print(f"LGB 3-seed old params: {acc(champ0):.5f}   LGB retuned: {acc(champ):.5f}   cat: {acc(cat):.5f}   xgb: {acc(xgb):.5f}")
print("corr:", np.corrcoef([champ, cat, xgb]).round(4))


def paired(a, b, name):
    ca = (a > 0.5) == y; cb = (b > 0.5) == y
    fixed = (~ca & cb).sum(); broken = (ca & ~cb).sum()
    p = binomtest(fixed, fixed + broken, 0.5).pvalue if fixed + broken else 1.0
    rng = np.random.default_rng(0); diffs = []
    for _ in range(2000):
        idx = rng.integers(0, len(y), len(y)); diffs.append(cb[idx].mean() - ca[idx].mean())
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    print(f"{name}: acc {ca.mean():.5f} -> {cb.mean():.5f}  fixed={fixed} broken={broken} net={fixed-broken}  McNemar p={p:.4f}  bootstrap 95% CI of gain=[{lo:+.5f},{hi:+.5f}]")


best = (acc(champ), (1, 0, 0))
for w in itertools.product(np.arange(0, 1.01, 0.1), repeat=3):
    if abs(sum(w) - 1) > 1e-9: continue
    a = acc(w[0] * champ + w[1] * cat + w[2] * xgb)
    if a > best[0]: best = (a, w)
print("best blend weights (lgb,cat,xgb):", best)
w = best[1]; blend = w[0] * champ + w[1] * cat + w[2] * xgb
paired(champ0, champ, "old champion -> retuned LGB")
paired(champ0, cat, "old champion -> CatBoost")
paired(champ, cat, "retuned LGB -> CatBoost")
paired(champ, blend, "retuned LGB -> best-weight blend")
paired(champ, (champ + cat + xgb) / 3, "retuned LGB -> equal blend")
np.save(os.path.join(HERE, "blend_weights.npy"), np.array(w))
