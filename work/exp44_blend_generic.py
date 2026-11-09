"""Submission v13: blend stage outputs, promotion gate vs v12, calibration (only if EV>=1 row), write CSV."""
import os, sys, itertools, numpy as np, pandas as pd
from scipy.stats import binomtest
from sklearn.isotonic import IsotonicRegression
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
DATA = sys.argv[1]; P_TRUE = 9590; TAG = sys.argv[2] if len(sys.argv) > 2 else "v13_phase"; PREFIX = sys.argv[3] if len(sys.argv) > 3 else "v13"; REF = sys.argv[4] if len(sys.argv) > 4 else "oof_v12.npy"
tr = pd.read_csv(f"{DATA}/Train_set.csv"); te = pd.read_csv(f"{DATA}/Test_set.csv"); y = tr["default"].values
ld = lambda n: (np.load(os.path.join(HERE, f"{PREFIX}_{n}_oof.npy")), np.load(os.path.join(HERE, f"{PREFIX}_{n}_tst.npy")))
acc = lambda p: ((p > 0.5) == y).mean()
have = lambda n: os.path.exists(os.path.join(HERE, f"{PREFIX}_{n}_oof.npy"))

M = {}
M["lgb"] = ld("lgb")
cats = [ld(n) for n in ("cat42", "cat7") if have(n)]
M["cat"] = (np.mean([c[0] for c in cats], axis=0), np.mean([c[1] for c in cats], axis=0)); print(f"cat seeds available: {len(cats)}")
if have("xgb"): M["xgb"] = ld("xgb")
for k, (o, t) in M.items(): print(f"{k}: OOF {acc(o):.5f}")

keys = list(M); best = (0, None)
for w in itertools.product(np.arange(0, 1.0001, 0.05), repeat=len(keys) - 1):
    last = 1 - sum(w)
    if last < -1e-9: continue
    ws = list(w) + [max(last, 0)]
    a = acc(sum(wi * M[k][0] for wi, k in zip(ws, keys)))
    if a > best[0]: best = (a, ws)
print("best weights", dict(zip(keys, np.round(best[1], 2))), f"OOF {best[0]:.5f}")
oof = sum(wi * M[k][0] for wi, k in zip(best[1], keys)); tst = sum(wi * M[k][1] for wi, k in zip(best[1], keys))

# ---- promotion gate vs v12 blend
ref = np.load(os.path.join(HERE, REF))
ca, cb = (ref > 0.5) == y, (oof > 0.5) == y
f, b = int((~ca & cb).sum()), int((ca & ~cb).sum()); pv = binomtest(f, f + b, 0.5).pvalue if f + b else 1.0
rng = np.random.default_rng(0)
diffs = [cb[i].mean() - ca[i].mean() for i in (rng.integers(0, len(y), len(y)) for _ in range(2000))]
lo, hi = np.percentile(diffs, [2.5, 97.5])
print(f"\nGATE: v12 blend {ca.mean():.5f} -> v13 blend {cb.mean():.5f} fixed={f} broken={b} net={f-b:+d} McNemar p={pv:.4f} | bootstrap 95% CI [{lo:+.5f},{hi:+.5f}]")
pr, pn = (ref > 0.5).astype(int), (oof > 0.5).astype(int)
print(f"errors by type: v12 FP={int(((pr==1)&(y==0)).sum())} FN={int(((pr==0)&(y==1)).sum())} -> v13 FP={int(((pn==1)&(y==0)).sum())} FN={int(((pn==0)&(y==1)).sum())}")

# ---- calibration (same procedure as v10/v12; applied only if expected gain >= 1 row)
iso = IsotonicRegression(out_of_bounds="clip").fit(oof, y)
qhat = iso.predict(tst); delta = (qhat.sum() - P_TRUE) / len(tst); q_corr = np.clip(qhat - delta, 0, 1)
pred = (tst > 0.5).astype(int)
gain = np.where(pred == 1, 1 - 2 * q_corr, 2 * q_corr - 1)
order = np.argsort(-gain); cum = np.cumsum(gain[order])
k_opt = int(np.argmax(cum)) + 1 if cum.max() > 0 else 0
print(f"calibration: E[P]={qhat.sum():.0f} vs measured {P_TRUE} | optimal flips {k_opt} | EV {cum.max():.2f} rows")
newpred = pred.copy()
if k_opt and cum.max() >= 1.0:
    newpred[order[:k_opt]] = 1 - newpred[order[:k_opt]]; print("flips APPLIED")
else:
    print("flips SKIPPED (EV < 1 row)")
sub = pd.DataFrame({"ID": te["ID"], "default": newpred})
sub.to_csv(os.path.join(DATA, "submissions", f"submission_{TAG}.csv"), index=False)
v12 = pd.read_csv(os.path.join(DATA, "submissions", "submission_v13b_sig.csv"))["default"].values
print(f"v13 written | pred_pos {int(newpred.sum())} | differs from v12 on {int((newpred != v12).sum())} rows")
np.save(os.path.join(HERE, f"oof_{TAG}.npy"), oof); np.save(os.path.join(HERE, f"tst_{TAG}.npy"), tst)


# ---------------- repayment-signature override (exp39): unanimous labels from >=2 matching train loans ----------------
SIG = ["loan_amnt", "interest_rate", "loan_term", "interest_receive", "last_week_pay"]
key = lambda d: d[SIG].astype(str).agg("|".join, axis=1)
ktr, kte = key(tr), key(te)
g = tr.groupby(ktr)["default"].agg(["size", "sum"])
n_all = ktr.map(g["size"]).values; s_all = ktr.map(g["sum"]).values
n_o = n_all - 1; s_o = s_all - y
rule_tr = (n_o >= 2) & ((s_o == 0) | (s_o == n_o)); lab_tr = (s_o == n_o).astype(int)
pc = (oof > 0.5).astype(int)
fixed = int(((pc != y) & (lab_tr == y) & rule_tr).sum()); broken = int(((pc == y) & (lab_tr != y) & rule_tr).sum())
pv = binomtest(fixed, fixed + broken, 0.5).pvalue if fixed + broken else 1.0
print(f"\nSIGNATURE OVERRIDE (OOF, leave-one-out): rule rows {int(rule_tr.sum())} | fixes {fixed} breaks {broken} net {fixed-broken:+d} p={pv:.4f}")
nt = kte.map(g["size"]).fillna(0).values; st = kte.map(g["sum"]).fillna(0).values
rule_te = (nt >= 2) & ((st == 0) | (st == nt)); lab_te = (st == nt).astype(int)
sig_pred = newpred.copy(); sig_pred[rule_te] = lab_te[rule_te]
print(f"test: rule rows {int(rule_te.sum())} | overrides that change the submission: {int((sig_pred != newpred).sum())}")
pd.DataFrame({"ID": te["ID"], "default": sig_pred}).to_csv(os.path.join(DATA, "submissions", f"submission_{TAG}_sig.csv"), index=False)
print(f"written submission_{TAG}_sig.csv | pred_pos {int(sig_pred.sum())}")
