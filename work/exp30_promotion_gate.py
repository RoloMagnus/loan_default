"""Experiment 30: promotion gate for v12 — paired bootstrap CI + error attribution
(affected overrun rows vs rest, FP vs FN separately)."""
import os, sys, numpy as np, pandas as pd
from scipy.stats import binomtest
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from features import build
DATA = sys.argv[1]
tr = pd.read_csv(f"{DATA}/Train_set.csv"); te = pd.read_csv(f"{DATA}/Test_set.csv")
X, y, Xte = build(tr, te)
oof_new = np.load(os.path.join(HERE, "oof_v12.npy"))
oof_ref = 0.25 * np.load(os.path.join(HERE, "oof_l_v4.npy")) + 0.70 * np.load(os.path.join(HERE, "oof_c_v4.npy")) + 0.05 * np.load(os.path.join(HERE, "oof_x_v4.npy"))
ca, cb = (oof_ref > 0.5) == y, (oof_new > 0.5) == y
f, b = int((~ca & cb).sum()), int((ca & ~cb).sum())
pv = binomtest(f, f + b, 0.5).pvalue if f + b else 1.0
print(f"v4 blend {ca.mean():.5f} -> v12 blend {cb.mean():.5f}  fixed={f} broken={b} net={f-b:+d}  McNemar p={pv:.4f}")
rng = np.random.default_rng(0)
diffs = [cb[i].mean() - ca[i].mean() for i in (rng.integers(0, len(y), len(y)) for _ in range(2000))]
lo, hi = np.percentile(diffs, [2.5, 97.5])
print(f"paired bootstrap 95% CI of accuracy gain: [{lo:+.5f}, {hi:+.5f}]")

# attribution
term_m = X["loan_term"].values * 12
affected = (X["k_int"].values > term_m)
print(f"\naffected rows (k > contractual term): {affected.sum()}")
for name, m in [("affected", affected), ("rest", ~affected)]:
    pr, pn = (oof_ref[m] > 0.5).astype(int), (oof_new[m] > 0.5).astype(int)
    yy = y[m]
    fp_r, fn_r = int(((pr == 1) & (yy == 0)).sum()), int(((pr == 0) & (yy == 1)).sum())
    fp_n, fn_n = int(((pn == 1) & (yy == 0)).sum()), int(((pn == 0) & (yy == 1)).sum())
    print(f"{name:9s} n={m.sum():6d} | ref FP={fp_r} FN={fn_r} -> new FP={fp_n} FN={fn_n} | net corrected {fp_r+fn_r-fp_n-fn_n:+d}")
