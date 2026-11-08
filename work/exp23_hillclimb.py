"""Experiment 23: Caruana hill-climb ensemble selection over the saved model library."""
import os, sys, numpy as np, pandas as pd
from scipy.stats import binomtest
HERE = os.path.dirname(os.path.abspath(__file__))
y = pd.read_csv(f"{sys.argv[1]}/Train_set.csv")["default"].values
# components: (oof_file, test_file)
LIB = {"lgb5": ("oof_lgb_tuned", "test_lgb_retuned"),
       "cat5": ("oof_cat", "test_cat"),
       "xgb5": ("oof_xgb", "test_xgb"),
       "lgb5_old3seed": ("oof_lgb_exact_3seed", "test_lgb_exact_3seed"),
       "lgb10": ("oof_l_v4", "tst_l_v4"),
       "xgb10": ("oof_x_v4", "tst_x_v4"),
       "cat10t": ("oof_c_v4", "tst_c_v4")}
O = {k: np.load(os.path.join(HERE, f"{v[0]}.npy")) for k, v in LIB.items()}
T = {k: np.load(os.path.join(HERE, f"{v[1]}.npy")) for k, v in LIB.items()}
acc = lambda p: ((p > 0.5) == y).mean()
for k in LIB: print(f"{k:14s} OOF acc {acc(O[k]):.5f}")

# v4 reference blend
v4 = 0.25 * O["lgb10"] + 0.70 * O["cat10t"] + 0.05 * O["xgb10"]
print(f"\nv4 blend: {acc(v4):.5f}")

# hill-climb with replacement (Caruana), 60 steps, started from the best single
rng = np.random.default_rng(0)
keys = list(LIB)
def hill_climb(seed_key, steps=60):
    sel = [seed_key]; cur = O[seed_key].copy()
    for _ in range(steps):
        best_k, best_a = None, acc(cur / len(sel) if False else cur / len(sel))
        best_a = acc(cur / len(sel))
        for k in keys:
            a = acc((cur + O[k]) / (len(sel) + 1))
            if a > best_a: best_a, best_k = a, k
        if best_k is None: break
        sel.append(best_k); cur += O[best_k]
    return sel, cur / len(sel)

sel, oof_hc = hill_climb(max(keys, key=lambda k: acc(O[k])))
from collections import Counter
w = Counter(sel); total = sum(w.values())
print("hill-climb weights:", {k: round(v / total, 3) for k, v in w.items()}, f"OOF {acc(oof_hc):.5f}")

def paired(pa, pb, name):
    ca, cb = (pa > 0.5) == y, (pb > 0.5) == y
    f, b = int((~ca & cb).sum()), int((ca & ~cb).sum())
    pv = binomtest(f, f + b, 0.5).pvalue if f + b else 1.0
    print(f"{name}: {ca.mean():.5f} -> {cb.mean():.5f} fixed={f} broken={b} net={f-b} p={pv:.4f}")
paired(v4, oof_hc, "v4 -> hill-climb")
tst_hc = sum(T[k] for k in sel) / len(sel)
np.save(os.path.join(HERE, "oof_hc.npy"), oof_hc); np.save(os.path.join(HERE, "tst_hc.npy"), tst_hc)
te = pd.read_csv(f"{sys.argv[1]}/Test_set.csv")
sub = pd.DataFrame({"ID": te["ID"], "default": (tst_hc > 0.5).astype(int)})
sub.to_csv(os.path.join(sys.argv[1], "submissions", "submission_v5_hillclimb.csv"), index=False)
v4s = pd.read_csv(os.path.join(sys.argv[1], "submissions", "submission_v4_10fold_tunedcat.csv"))
print("v5 rows differing from v4:", int((sub['default'].values != v4s['default'].values).sum()), "| pred rate", sub["default"].mean().round(4))
