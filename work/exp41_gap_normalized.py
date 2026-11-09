"""Experiment 41: cohort-normalised interest deviation. gap_m = (interest_received - scheduled_at_k) / monthly interest;
baseline = median gap_m among current-level neighbours (k == local robust max) in ID-ordered pooled windows (label-free)."""
import os, sys, json, numpy as np, pandas as pd, lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from scipy.stats import binomtest
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from features import build_v12, local_k_features
DATA = sys.argv[1]
tr = pd.read_csv(f"{DATA}/Train_set.csv"); te = pd.read_csv(f"{DATA}/Test_set.csv")
X, y, Xte = build_v12(tr, te); Xr = X.reset_index(drop=True); Xter = Xte.reset_index(drop=True)
F, Fte = local_k_features(tr, te, Ws=(40,), share=0.08)


def gap_features(raw, Xd, Fd):
    r = raw["interest_rate"].values / 1200; L = raw["loan_amnt"].values.astype(float)
    out = pd.DataFrame(index=range(len(raw)))
    out["gap_m"] = Xd["int_gap_exact"].values / (L * r)
    out["gap_m_km1"] = (Xd["int_dev_km1"].values * (0)) + np.nan  # placeholder (km1 gap not stored as dollars)
    out["cur_level"] = (Fd["lk40_k_minus_rmax"].values == 0)
    return out.drop(columns=["gap_m_km1"])


Gtr = gap_features(tr, Xr, F); Gte = gap_features(te, Xter, Fte)
ids = np.r_[tr["ID"].values, te["ID"].values]; order = np.argsort(ids, kind="stable"); n = len(ids)
gap_all = np.r_[Gtr.gap_m.values, Gte.gap_m.values][order]; cur_all = np.r_[Gtr.cur_level.values, Gte.cur_level.values][order]
base_series = pd.Series(np.where(cur_all, gap_all, np.nan))
feats = {}
inv = np.empty(n, dtype=int); inv[order] = np.arange(n)
for W in (60, 150):
    med = base_series.rolling(2 * W + 1, center=True, min_periods=15).median().values
    q25 = base_series.rolling(2 * W + 1, center=True, min_periods=15).quantile(0.25).values
    q75 = base_series.rolling(2 * W + 1, center=True, min_periods=15).quantile(0.75).values
    feats[f"gapbase{W}"] = med[inv]; feats[f"gapiqr{W}"] = (q75 - q25)[inv]
feat_all = pd.DataFrame(feats)
G = pd.concat([Gtr[["gap_m"]], feat_all.iloc[:len(tr)].reset_index(drop=True)], axis=1)
for W in (60, 150):
    G[f"gap_rel{W}"] = G["gap_m"] - G[f"gapbase{W}"]
    G[f"gap_z{W}"] = G[f"gap_rel{W}"] / (G[f"gapiqr{W}"] + 0.02)
print("new features:", list(G.columns))
cur = F["lk40_k_minus_rmax"].values == 0
print("current-level loans: |gap_rel60|<0.03 share:", round((G.loc[cur, "gap_rel60"].abs() < 0.03).mean(), 3),
      "| default rate by gap_rel60 bucket:")
b = pd.cut(G.loc[cur, "gap_rel60"], [-100, -0.5, -0.1, -0.03, 0.03, 0.1, 0.5, 100])
print(pd.Series(y[cur]).groupby(b.values, observed=True).agg(["size", "mean"]).round(3).to_string())

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
    print(f"  {name}: {ca.mean():.5f} -> {cb.mean():.5f} net={f-b:+d} p={pv:.4f}", flush=True); return f - b
A = pd.concat([Xr, F[[c for c in F.columns if c.startswith("lk40_")]]], axis=1)
tot = 0
for s in [42, 7, 2024]:
    tot += paired(cv(A, s), cv(pd.concat([A, G], axis=1), s), f"+gap-normalised features, seed {s}")
print("sum of nets over 3 seeds:", tot)
