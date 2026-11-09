"""Plan B: robust fixed-weight blend of the frozen v14 stage outputs (weights NOT tuned on OOF). Does not touch champion/."""
import os, sys, numpy as np, pandas as pd
from scipy.stats import binomtest
HERE = os.path.dirname(os.path.abspath(__file__)); DATA = sys.argv[1]
tr = pd.read_csv(f"{DATA}/Train_set.csv"); te = pd.read_csv(f"{DATA}/Test_set.csv"); y = tr["default"].values
ld = lambda n: (np.load(os.path.join(HERE, f"v14_{n}_oof.npy")), np.load(os.path.join(HERE, f"v14_{n}_tst.npy")))
acc = lambda p: ((p > 0.5) == y).mean()
(lo, lt), (c4o, c4t), (c7o, c7t), (xo, xt) = ld("lgb"), ld("cat42"), ld("cat7"), ld("xgb")
co, ct = (c4o + c7o) / 2, (c4t + c7t) / 2
W = (0.4, 0.4, 0.2)   # fixed in advance: LGB / Cat(2 seeds) / XGB
oof = W[0] * lo + W[1] * co + W[2] * xo; tst = W[0] * lt + W[1] * ct + W[2] * xt
champ_oof = np.load(os.path.join(HERE, "oof_v14b.npy")); champ_tst = np.load(os.path.join(HERE, "tst_v14b.npy"))
print(f"plan B OOF {acc(oof):.5f} | champion OOF {acc(champ_oof):.5f}")
ca, cb = (champ_oof > 0.5) == y, (oof > 0.5) == y
f, b = int((~ca & cb).sum()), int((ca & ~cb).sum())
print(f"plan B vs champion: fixed={f} broken={b} net={f-b:+d} McNemar p={binomtest(f, f+b, 0.5).pvalue:.3f}")
pred = (tst > 0.5).astype(int)
SIG = ["loan_amnt", "interest_rate", "loan_term", "interest_receive", "last_week_pay"]
key = lambda d: d[SIG].astype(str).agg("|".join, axis=1)
ktr, kte = key(tr), key(te); g = tr.groupby(ktr)["default"].agg(["size", "sum"])
nt = kte.map(g["size"]).fillna(0).values; st = kte.map(g["sum"]).fillna(0).values
rule = (nt >= 2) & ((st == 0) | (st == nt)); sp = pred.copy(); sp[rule] = (st == nt).astype(int)[rule]
out = pd.DataFrame({"ID": te["ID"], "default": sp}); out.to_csv(os.path.join(DATA, "submissions", "submission_v14c_planB_sig.csv"), index=False)
champ = pd.read_csv(os.path.join(DATA, "champion", "CHAMPION_submission_v14b_sig.csv"))["default"].values
a14 = pd.read_csv(os.path.join(DATA, "submissions", "submission_v14a_sig.csv"))["default"].values
ok = len(out) == 39933 and (out.ID.values == te.ID.values).all()
print(f"plan B written | format {'OK' if ok else 'PROBLEM'} | pred rate {sp.mean():.4f} | differs from champion on {int((sp != champ).sum())} rows, from v14a on {int((sp != a14).sum())} rows")
