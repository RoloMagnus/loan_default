"""Experiment 39: repayment-signature lookup vs champion — paired, leave-one-out (own implementation).
Signature = (loan_amnt, interest_rate, loan_term, interest_receive, last_week_pay)."""
import os, sys, numpy as np, pandas as pd
from scipy.stats import binomtest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from scipy.special import logit
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = sys.argv[1]
pd.set_option("display.width", 200)
tr = pd.read_csv(f"{DATA}/Train_set.csv"); te = pd.read_csv(f"{DATA}/Test_set.csv"); y = tr["default"].values
SIG = ["loan_amnt", "interest_rate", "loan_term", "interest_receive", "last_week_pay"]
key = lambda d: d[SIG].astype(str).agg("|".join, axis=1)
ktr, kte = key(tr), key(te)
g = tr.groupby(ktr)["default"].agg(["size", "sum"])
n_all = ktr.map(g["size"]).values; s_all = ktr.map(g["sum"]).values
n_o = n_all - 1; s_o = s_all - y                       # leave-one-out counts for train rows
matched = n_o >= 1
look = np.where(matched, (s_o > (n_o - s_o)).astype(int), -1)  # ties -> 0
unan = matched & ((s_o == 0) | (s_o == n_o))
print(f"train rows with >=1 other loan sharing the signature: {matched.sum()} ({matched.mean():.1%}); unanimous: {unan.sum()}")
print(f"test rows matched to train: {kte.map(g['size']).notna().sum()} ({kte.map(g['size']).notna().mean():.1%})")

for name, f in [("v12 blend", "oof_v12.npy"), ("v13a interim blend", "oof_v13a_interim.npy")]:
    p = np.load(os.path.join(HERE, f)); pc = (p > 0.5).astype(int)
    print(f"\n==================== champion = {name} (OOF acc {(pc == y).mean():.5f}) ====================")
    m = matched
    print(f"matched rows: lookup acc {(look[m] == y[m]).mean():.5f} | champion acc on the same rows {(pc[m] == y[m]).mean():.5f} | majority-class acc {max(y[m].mean(), 1 - y[m].mean()):.5f}")
    rows = []
    bands = [("n_o=1", (n_o == 1)), ("n_o=2", (n_o == 2)), ("n_o=3-4", (n_o >= 3) & (n_o <= 4)), ("n_o>=5", n_o >= 5)]
    for tag, bm in bands:
        for un in [True, False]:
            mm = m & bm & (unan == un)
            if mm.sum() == 0: continue
            fixed = int(((pc != y) & (look == y) & mm).sum()); broken = int(((pc == y) & (look != y) & mm).sum())
            rows.append((tag, "unanimous" if un else "mixed", int(mm.sum()), round((pc[mm] == y[mm]).mean(), 4), round((look[mm] == y[mm]).mean(), 4), fixed, broken, fixed - broken))
    print(pd.DataFrame(rows, columns=["support", "labels", "n", "champ_acc", "lookup_acc", "lookup_fixes", "lookup_breaks", "net_if_override"]).to_string(index=False))
    # candidate override rules
    print("\noverride rules (apply lookup label where rule holds):")
    for rname, rule in [("R1 unanimous & n_o>=2", unan & (n_o >= 2)), ("R2 unanimous & n_o>=3", unan & (n_o >= 3)),
                        ("R3 unanimous & n_o>=1", unan & (n_o >= 1)), ("R4 any match", m)]:
        fixed = int(((pc != y) & (look == y) & rule).sum()); broken = int(((pc == y) & (look != y) & rule).sum())
        pv = binomtest(fixed, fixed + broken, 0.5).pvalue if fixed + broken else 1.0
        print(f"  {rname}: rows {int(rule.sum())} | disagree {fixed + broken} | fixes {fixed} breaks {broken} net {fixed - broken:+d} (p={pv:.3f}) | test-scale net ≈ {(fixed - broken) * len(te) / len(tr):+.1f}")
    # cross-fitted stacker: does lookup info add to the champion probability?
    eps = 1e-4
    lr_feat = np.c_[logit(np.clip(p, eps, 1 - eps)), np.where(matched, (s_o + 0.5) / (n_o + 1.0), 0.5), np.log1p(n_o), matched.astype(float)]
    lr_feat[:, 1] = logit(np.clip(lr_feat[:, 1], eps, 1 - eps))
    oof_lr = np.zeros(len(y))
    for a, b in StratifiedKFold(5, shuffle=True, random_state=1).split(lr_feat, y):
        oof_lr[b] = LogisticRegression(max_iter=1000, C=10).fit(lr_feat[a], y[a]).predict_proba(lr_feat[b])[:, 1]
    fixed = int(((pc != y) & ((oof_lr > 0.5) == y)).sum()); broken = int(((pc == y) & ((oof_lr > 0.5) != y)).sum())
    print(f"  cross-fitted logistic stacker [champion logit, lookup logit, support]: acc {((oof_lr>0.5)==y).mean():.5f} (champion {(pc==y).mean():.5f}) net {fixed-broken:+d}")

# test-side view vs the v13a interim submission
sub = pd.read_csv(f"{DATA}/submissions/submission_v13a_interim.csv")["default"].values
nt = kte.map(g["size"]).fillna(0).values; st = kte.map(g["sum"]).fillna(0).values
mt = nt >= 1; lt = np.where(mt, (st > (nt - st)).astype(int), -1); ut = mt & ((st == 0) | (st == nt))
dis = mt & (lt != sub)
print(f"\nTEST: matched {mt.sum()}; lookup-vs-v13a disagreements {dis.sum()} (v13a=1,lookup=0: {int((dis & (sub == 1)).sum())}; v13a=0,lookup=1: {int((dis & (sub == 0)).sum())})")
for rname, rule in [("unanimous & n>=2", ut & (nt >= 2)), ("unanimous & n>=3", ut & (nt >= 3))]:
    print(f"  {rname}: rows {int(rule.sum())}, disagreements with v13a {int((rule & (lt != sub)).sum())}")
