"""Experiment 31: residual structure audit of v12.
 (a) Do OOF residuals of ID-adjacent train loans correlate? (unmodelled local/batch structure)
 (b) Error rate by estimated cohort month and by position inside the cohort (due-date phase hypothesis).
 (c) How exact is our elapsed-month estimate? Compare against a sharper local estimator."""
import os, sys, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from features import build_v12
pd.set_option("display.width", 220); pd.set_option("display.max_rows", 200)
DATA = sys.argv[1]
tr = pd.read_csv(f"{DATA}/Train_set.csv"); te = pd.read_csv(f"{DATA}/Test_set.csv")
X, y, Xte = build_v12(tr, te)
p = np.load(os.path.join(HERE, "oof_v12.npy")); r = y - p; err = (p > 0.5) != y
print(f"v12 blend OOF acc {1-err.mean():.5f}  errors {err.sum()}")

# ---------- (a) residual autocorrelation along ID
order = np.argsort(tr["ID"].values); rs = r[order]; es = err[order].astype(float)
def neigh_mean(v, w):
    k = np.ones(2 * w + 1); k[w] = 0
    num = np.convolve(v, k, mode="same"); den = np.convolve(np.ones_like(v), k, mode="same")
    return num / den
print("\n(a) correlation between a loan's residual and its ID-neighbours' mean residual")
for w in [2, 5, 10, 25, 100]:
    nm = neigh_mean(rs, w)
    c = np.corrcoef(rs, nm)[0, 1]
    # permutation baseline: shuffle residuals within blocks of 2000 consecutive IDs
    rng = np.random.default_rng(0); base = []
    for _ in range(20):
        sh = rs.copy()
        for s in range(0, len(sh), 2000): rng.shuffle(sh[s:s + 2000])
        base.append(np.corrcoef(sh, neigh_mean(sh, w))[0, 1])
    print(f"  window ±{w:3d}: corr={c:+.4f}   permutation baseline {np.mean(base):+.4f} ± {np.std(base):.4f}")

# ---------- (b) error by estimated cohort month
m_est = X["months_elapsed_int"].values
d = pd.DataFrame({"m": m_est, "err": err, "y": y, "k": X["k_int"].values, "behind": X["months_behind_int"].values, "ID": tr["ID"].values})
g = d.groupby("m").agg(n=("err", "size"), err_rate=("err", "mean"), default=("y", "mean")).round(3)
print("\n(b) error rate by estimated elapsed month (only months with n>=300):")
print(g[g.n >= 300].T.to_string())

# ---------- (c) sharpness of the vintage estimate: share of loans with behind<0 and with behind in {0,1} by cohort
print("\n(c) months_behind distribution for NON-defaulted loans (should be concentrated on 0/1 for current loans):")
print(d[d.y == 0].behind.clip(-3, 12).value_counts(normalize=True).sort_index().round(4).to_string())

# position inside cohort: rank of ID within its estimated month cohort (pooled train+test)
allm = pd.DataFrame({"ID": np.r_[tr["ID"].values, te["ID"].values], "m": np.r_[m_est, Xte["months_elapsed_int"].values]}).sort_values("ID")
allm["pos"] = allm.groupby("m")["ID"].rank(pct=True)
pos = allm.set_index("ID").loc[tr["ID"].values, "pos"].values
d["pos"] = pos
d["pos_bin"] = pd.cut(d.pos, [0, .2, .4, .6, .8, 1.0])
print("\nshare of NON-defaulted loans with behind==1 (vs 0), by position inside the cohort — a due-date phase would show a trend:")
nd = d[(d.y == 0) & d.behind.isin([0, 1])]
print(nd.groupby("pos_bin", observed=True).behind.mean().round(3).to_string())
print("\nerror rate (behind in {0,1}) by position inside cohort:")
print(d[d.behind.isin([0, 1])].groupby("pos_bin", observed=True).agg(n=("err", "size"), err=("err", "mean"), default=("y", "mean")).round(4).to_string())
