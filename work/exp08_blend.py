"""Experiment 8: blend OOF predictions of tuned LGB, CatBoost, XGBoost; choose weights + threshold."""
import os, sys, itertools, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
y = pd.read_csv(f"{sys.argv[1]}/Train_set.csv")["default"].values
P = {k: np.load(os.path.join(HERE, f"oof_{k}.npy")) for k in ["lgb_tuned", "cat", "xgb"]}
acc = lambda p, t=0.5: ((p > t) == y).mean()
for k, p in P.items():
    print(f"{k:10s} acc@0.5={acc(p):.5f}  logloss={-(y*np.log(p+1e-9)+(1-y)*np.log(1-p+1e-9)).mean():.4f}")
print("pairwise corr:\n", pd.DataFrame(P).corr().round(4))
best = (0, None)
for w in itertools.product(np.arange(0, 1.01, 0.1), repeat=3):
    if abs(sum(w) - 1) > 1e-9: continue
    p = w[0] * P["lgb_tuned"] + w[1] * P["cat"] + w[2] * P["xgb"]
    a = acc(p)
    if a > best[0]: best = (a, w)
print("best weights (lgb,cat,xgb):", best)
p = sum(w * P[k] for w, k in zip(best[1], P))
ths = np.arange(0.40, 0.60, 0.005); accs = [acc(p, t) for t in ths]
print("threshold sweep:", {round(t, 3): round(a, 5) for t, a in zip(ths, accs) if a >= max(accs) - 0.0003})
print("equal-weight avg acc@0.5:", round(acc((P["lgb_tuned"] + P["cat"] + P["xgb"]) / 3), 5))
# rank average
r = sum(pd.Series(P[k]).rank().values / len(y) for k in P) / 3
print("rank-avg best acc:", round(max(((r > t) == y).mean() for t in np.arange(0.6, 0.9, 0.005)), 5))
