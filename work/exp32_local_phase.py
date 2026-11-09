"""Experiment 32: local payment-month profile features (due-date phase hypothesis).
Pooled train+test (target-free) ID-ordered windows over k = installments at last payment.
Features: local modal k, robust local max (elapsed-month proxy at phase resolution), phase share,
rank of my k inside the window, k minus local mode/max."""
import os, sys, json, time, numpy as np, pandas as pd, lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from scipy.stats import binomtest
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from features import build_v12

DATA = sys.argv[1]
tr = pd.read_csv(f"{DATA}/Train_set.csv"); te = pd.read_csv(f"{DATA}/Test_set.csv")
X, y, Xte = build_v12(tr, te)


def local_k_features(tr, te, Ws=(40, 120, 360), share=0.08):
    ids = np.r_[tr["ID"].values, te["ID"].values]
    lwp = np.r_[tr["last_week_pay"].values, te["last_week_pay"].values]
    k = np.round(lwp * 12 / 52.0)
    order = np.argsort(ids, kind="stable"); n = len(ids)
    ks = k[order]; valid = ~np.isnan(ks)
    K = int(np.nanmax(ks)) + 2
    oh = np.zeros((n, K), dtype=np.float32)
    vi = np.where(valid)[0]; oh[vi, ks[vi].astype(int)] = 1
    cs = np.vstack([np.zeros((1, K), dtype=np.float64), np.cumsum(oh, axis=0, dtype=np.float64)])
    rows = np.arange(n); kk = np.where(valid, ks, 0).astype(int)
    feats = {}
    for W in Ws:
        lo = np.clip(rows - W, 0, n); hi = np.clip(rows + W + 1, 0, n)
        C = (cs[hi] - cs[lo]).astype(np.float32); tot = np.maximum(C.sum(1), 1)
        mode = C.argmax(1)
        mask = C >= share * tot[:, None]
        rmax = K - 1 - np.argmax(mask[:, ::-1], axis=1)
        c_eq = C[rows, kk]; Cc = np.cumsum(C, axis=1); n_le = Cc[rows, kk]; n_lt = n_le - c_eq
        c_r = C[rows, rmax]; c_r1 = C[rows, np.maximum(rmax - 1, 0)]
        f = {f"lk{W}_mode": mode.astype(float), f"lk{W}_rmax": rmax.astype(float),
             f"lk{W}_k_minus_mode": ks - mode, f"lk{W}_k_minus_rmax": ks - rmax,
             f"lk{W}_frac_lt": n_lt / tot, f"lk{W}_frac_eq": c_eq / tot, f"lk{W}_mode_share": C.max(1) / tot,
             f"lk{W}_phase": c_r / np.maximum(c_r + c_r1, 1)}
        for name, v in f.items():
            v = np.asarray(v, dtype=float); v[~valid] = np.nan; feats[name] = v
    F = pd.DataFrame(feats)
    inv = np.empty(n, dtype=int); inv[order] = np.arange(n)
    F = F.iloc[inv].reset_index(drop=True)
    return F.iloc[:len(tr)].reset_index(drop=True), F.iloc[len(tr):].reset_index(drop=True)


t0 = time.time()
Ftr, Fte = local_k_features(tr, te)
print(f"local features built: {Ftr.shape[1]} cols [{time.time()-t0:.0f}s]")
nd = (y == 0) & Ftr["lk120_k_minus_rmax"].notna().values
print("non-defaulted loans: share with k == local robust max (W=120):", round((Ftr.loc[nd, "lk120_k_minus_rmax"] == 0).mean(), 4),
      "| k == max-1:", round((Ftr.loc[nd, "lk120_k_minus_rmax"] == -1).mean(), 4),
      "(global-estimate version: behind==0 0.479, behind==1 0.396)")
print("default rate by k - local robust max (W=120):")
print(pd.Series(y).groupby(Ftr["lk120_k_minus_rmax"].clip(-6, 1).values).agg(["mean", "size"]).round(3).to_string())

lp = json.load(open(os.path.join(HERE, "lgb_best.json")))["params"]
folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(X, y))


def cv(Xv, seed):
    P = dict(objective="binary", verbose=-1, n_jobs=12, bagging_freq=1, seed=seed, **lp)
    oof = np.zeros(len(Xv))
    for a, b in folds:
        m = lgb.train(P, lgb.Dataset(Xv.iloc[a], y[a]), 4000, valid_sets=[lgb.Dataset(Xv.iloc[b], y[b])],
                      callbacks=[lgb.early_stopping(150, verbose=False)])
        oof[b] = m.predict(Xv.iloc[b], num_iteration=m.best_iteration)
    return oof


def paired(pa, pb, name):
    ca, cb = (pa > 0.5) == y, (pb > 0.5) == y
    f, b = int((~ca & cb).sum()), int((ca & ~cb).sum())
    pv = binomtest(f, f + b, 0.5).pvalue if f + b else 1.0
    print(f"  {name}: {ca.mean():.5f} -> {cb.mean():.5f} fixed={f} broken={b} net={f-b:+d} p={pv:.4f}", flush=True)
    return f - b


variants = {"V1_all": list(Ftr.columns),
            "V2_W120": [c for c in Ftr.columns if c.startswith("lk120_")],
            "V3_W40": [c for c in Ftr.columns if c.startswith("lk40_")]}
t0 = time.time(); base = cv(X, 42); print(f"\nbaseline v12 features seed 42: {((base>0.5)==y).mean():.5f} [{time.time()-t0:.0f}s]", flush=True)
res = {}
for tag, cols in variants.items():
    Xv = pd.concat([X.reset_index(drop=True), Ftr[cols]], axis=1)
    o = cv(Xv, 42); print(f"{tag} ({len(cols)} cols):"); res[tag] = paired(base, o, "seed 42")
best = max(res, key=res.get)
json.dump({"results": res, "best": best}, open(os.path.join(HERE, "exp32_result.json"), "w"))
if res[best] > 0:
    print(f"\n{best} leads; consistency check on seeds 7, 2024:")
    Xv = pd.concat([X.reset_index(drop=True), Ftr[variants[best]]], axis=1)
    for s in [7, 2024]:
        paired(cv(X, s), cv(Xv, s), f"seed {s}")
