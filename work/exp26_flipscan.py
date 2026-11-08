"""Experiment 26: 'flip engine' — scan for rule-definable subgroups with OOF error rate > 50%,
discovered on half A of the training data and verified on half B (split-half control for selection bias)."""
import os, sys, numpy as np, pandas as pd
from scipy.stats import binomtest
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from features import build
pd.set_option("display.width", 220)
DATA = sys.argv[1]
tr = pd.read_csv(f"{DATA}/Train_set.csv"); te = pd.read_csv(f"{DATA}/Test_set.csv")
X, y, Xte = build(tr, te)
oof = 0.25 * np.load(os.path.join(HERE, "oof_l_v4.npy")) + 0.70 * np.load(os.path.join(HERE, "oof_c_v4.npy")) + 0.05 * np.load(os.path.join(HERE, "oof_x_v4.npy"))
pred = (oof > 0.5).astype(int); err = (pred != y).astype(int)
print(f"reference OOF acc {1 - err.mean():.5f}, total errors {err.sum()}")

# candidate binning features (coarse, pre-registered style)
cand = {}
for c, bins in {"installments_missed": [-100, 0, 1, 2, 3, 6, 12, 18, 24, 36, 100],
                "months_behind_int": [-100, 0, 1, 2, 3, 6, 12, 18, 24, 36, 100],
                "k_implied_minus_k": [-100, -3, -1, 0, 1, 3, 100],
                "int_dev_exact": [-2, -0.2, -0.05, -0.01, 0.01, 0.05, 0.2, 100],
                "elapsed_vs_term": [0, 0.25, 0.5, 0.75, 1.0, 1.25, 2, 10],
                "id_rank": [0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0],
                "loan_term": [2, 4, 6],
                "pay_recency_ratio": [-1, 0.25, 0.5, 0.75, 0.9, 1.1]}.items():
    cand[c] = pd.cut(X[c], bins)
cand["pred"] = pd.Series(pred, index=X.index).astype("category")
cand["conf_band"] = pd.cut(pd.Series(np.abs(oof - 0.5), index=X.index), [0, 0.05, 0.1, 0.2, 0.35, 0.5])

rng = np.random.default_rng(0)
half = rng.permutation(len(X)) < len(X) // 2  # half A = True

rows = []
# single and pairwise predicates
keys = list(cand)
for i, k1 in enumerate(keys):
    g1 = cand[k1]
    for v1 in g1.dropna().unique():
        m1 = np.asarray(g1 == v1)
        for k2 in keys[i + 1:]:
            g2 = cand[k2]
            for v2 in g2.dropna().unique():
                m = m1 & np.asarray(g2 == v2)
                a = m & half
                if a.sum() < 150: continue
                ea = err[a].mean()
                if ea <= 0.52: continue
                b = m & ~half
                if b.sum() < 100: continue
                eb = err[b].mean()
                pv = binomtest(int(err[b].sum()), int(b.sum()), 0.5, alternative="greater").pvalue
                rows.append((f"{k1}={v1} & {k2}={v2}", int(a.sum()), round(ea, 3), int(b.sum()), round(eb, 3), round(pv, 4)))
res = pd.DataFrame(rows, columns=["rule", "nA", "errA", "nB", "errB", "pB_gt_50%"]).sort_values("pB_gt_50%")
print(f"\ncandidate rules with half-A error>52% (n={len(res)}):")
print(res.head(15).to_string(index=False) if len(res) else "  none found")
sig = res[(res.errB > 0.5) & (res["pB_gt_50%"] < 0.01)]
print(f"\nrules surviving half-B verification (errB>50%, p<0.01): {len(sig)}")
if len(sig): print(sig.to_string(index=False))
