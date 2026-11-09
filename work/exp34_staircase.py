"""Experiment 34: monotone staircase estimate of the current-loan payment month along ID + distance to nearest step."""
import os, sys, json, time, numpy as np, pandas as pd, lightgbm as lgb
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import StratifiedKFold
from scipy.stats import binomtest
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from features import build_v12, local_k_features
DATA = sys.argv[1]
tr = pd.read_csv(f"{DATA}/Train_set.csv"); te = pd.read_csv(f"{DATA}/Test_set.csv")
X, y, Xte = build_v12(tr, te)


def staircase_features(tr, te, W=40):
    ids = np.r_[tr["ID"].values, te["ID"].values]; n = len(ids)
    Ftr, Fte = local_k_features(tr, te, Ws=(W,))
    rmax = np.r_[Ftr[f"lk{W}_rmax"].values, Fte[f"lk{W}_rmax"].values]
    k = np.round(np.r_[tr["last_week_pay"].values, te["last_week_pay"].values] * 12 / 52.0)
    order = np.argsort(ids, kind="stable")
    rm_s = pd.Series(rmax[order]).ffill().bfill().values  # rows without payment history inherit neighbours' level
    iso = IsotonicRegression(increasing=False).fit(np.arange(n), rm_s)
    stair = np.round(iso.predict(np.arange(n)))
    chg = np.r_[True, stair[1:] != stair[:-1]]
    pos = np.arange(n)
    last_chg = np.maximum.accumulate(np.where(chg, pos, 0))
    nxt = np.empty(n, dtype=int); nxt[-1] = n
    next_c = n
    for i in range(n - 1, -1, -1):
        nxt[i] = next_c
        if i > 0 and chg[i]: next_c = i
    # robust: next change strictly after i
    idx_chg = np.where(chg)[0]
    nxt = np.empty(n, dtype=int)
    j = np.searchsorted(idx_chg, pos, side="right")
    nxt = np.where(j < len(idx_chg), idx_chg[np.minimum(j, len(idx_chg) - 1)], n)
    out = pd.DataFrame({"st_E": stair, "st_dist_prev": pos - last_chg, "st_dist_next": nxt - pos})
    ks = k[order]
    out["st_behind"] = stair - ks
    out["st_dist_min"] = np.minimum(out.st_dist_prev, out.st_dist_next)
    inv = np.empty(n, dtype=int); inv[order] = np.arange(n)
    out = out.iloc[inv].reset_index(drop=True)
    return out.iloc[:len(tr)].reset_index(drop=True), out.iloc[len(tr):].reset_index(drop=True)


St, Ste = staircase_features(tr, te)
print("staircase steps:", int((St.st_E.diff().fillna(0) != 0).sum()), "| E range", St.st_E.min(), St.st_E.max())
nd = y == 0
print("non-defaulted: share st_behind==0:", round((St.loc[nd, "st_behind"] == 0).mean(), 4), " ==1:", round((St.loc[nd, "st_behind"] == 1).mean(), 4))
print("default rate by st_behind:"); print(pd.Series(y).groupby(St.st_behind.clip(-3, 8).values).agg(["mean", "size"]).round(3).to_string())

lp = json.load(open(os.path.join(HERE, "lgb_best.json")))["params"]
folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(X, y))
def cv(Xv, seed):
    P = dict(objective="binary", verbose=-1, n_jobs=12, bagging_freq=1, seed=seed, **lp); oof = np.zeros(len(Xv))
    for a, b in folds:
        m = lgb.train(P, lgb.Dataset(Xv.iloc[a], y[a]), 4000, valid_sets=[lgb.Dataset(Xv.iloc[b], y[b])], callbacks=[lgb.early_stopping(150, verbose=False)])
        oof[b] = m.predict(Xv.iloc[b], num_iteration=m.best_iteration)
    return oof
def paired(pa, pb, name):
    ca, cb = (pa > 0.5) == y, (pb > 0.5) == y
    f, b = int((~ca & cb).sum()), int((ca & ~cb).sum()); pv = binomtest(f, f + b, 0.5).pvalue if f + b else 1.0
    print(f"  {name}: {ca.mean():.5f} -> {cb.mean():.5f} fixed={f} broken={b} net={f-b:+d} p={pv:.4f}", flush=True); return f - b
W40, W40te = local_k_features(tr, te, Ws=(40,))
Xr = X.reset_index(drop=True)
base = cv(Xr, 42); w40 = cv(pd.concat([Xr, W40], axis=1), 42)
print(f"baseline {((base>0.5)==y).mean():.5f} | W40 {((w40>0.5)==y).mean():.5f}", flush=True)
s1 = cv(pd.concat([Xr, St], axis=1), 42); print("S1 stair only:"); r1 = paired(base, s1, "vs base")
s2 = cv(pd.concat([Xr, W40, St], axis=1), 42); print("S2 W40 + stair:"); r2 = paired(base, s2, "vs base"); r2b = paired(w40, s2, "vs W40")
best = "S2" if r2 >= r1 else "S1"
json.dump({"S1_vs_base": r1, "S2_vs_base": r2, "S2_vs_W40": r2b}, open(os.path.join(HERE, "exp34_result.json"), "w"))
if max(r1, r2) > 62 or r2b > 0:
    Xv = pd.concat([Xr, W40, St], axis=1) if best == "S2" else pd.concat([Xr, St], axis=1)
    for s in [7, 2024]:
        b_s = cv(Xr, s); paired(b_s, cv(Xv, s), f"{best} vs base seed {s}")
        if best == "S2": paired(cv(pd.concat([Xr, W40], axis=1), s), cv(Xv, s), f"{best} vs W40 seed {s}")
