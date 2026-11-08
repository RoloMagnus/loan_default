"""Experiment 25: diversity candidates (LGB dart, LGB extra-trees, sklearn HGB) + re-blend vs v5."""
import os, sys, json, time, numpy as np, pandas as pd, lightgbm as lgb
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from scipy.stats import binomtest
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from features import build, CAT_COLS
DATA = sys.argv[1]
tr = pd.read_csv(f"{DATA}/Train_set.csv"); te = pd.read_csv(f"{DATA}/Test_set.csv")
X, y, Xte = build(tr, te)
acc = lambda p: ((p > 0.5) == y).mean()
folds = list(StratifiedKFold(10, shuffle=True, random_state=42).split(X, y))
lp = json.load(open(os.path.join(HERE, "lgb_best.json")))["params"]

def run_lgb(params, rounds, es, tag):
    oof = np.zeros(len(X)); tst = np.zeros(len(Xte)); t0 = time.time()
    for a, b in folds:
        cb = [lgb.early_stopping(es, verbose=False)] if es else []
        vs = [lgb.Dataset(X.iloc[b], y[b])] if es else None
        m = lgb.train(params, lgb.Dataset(X.iloc[a], y[a]), rounds, valid_sets=vs, callbacks=cb)
        ni = m.best_iteration if es else rounds
        oof[b] = m.predict(X.iloc[b], num_iteration=ni); tst += m.predict(Xte, num_iteration=ni) / len(folds)
    print(f"{tag}: OOF {acc(oof):.5f} [{time.time()-t0:.0f}s]", flush=True)
    return oof, tst

# 1. dart
dart = dict(objective="binary", boosting="dart", drop_rate=0.1, skip_drop=0.5, verbose=-1, n_jobs=12,
            bagging_freq=1, seed=42, **{k: v for k, v in lp.items()})
oof_d, tst_d = run_lgb(dart, 600, None, "LGB dart")
# 2. extra-trees mode
xt = dict(objective="binary", extra_trees=True, verbose=-1, n_jobs=12, bagging_freq=1, seed=42, **lp)
oof_e, tst_e = run_lgb(xt, 4000, 150, "LGB extra-trees")
# 3. sklearn HGB (ordinal-encode cats)
Xh = X.copy(); Xth = Xte.copy()
for c in CAT_COLS:
    Xh[c] = Xh[c].cat.codes; Xth[c] = Xth[c].cat.codes
oof_h = np.zeros(len(X)); tst_h = np.zeros(len(Xte)); t0 = time.time()
for a, b in folds:
    m = HistGradientBoostingClassifier(max_iter=1200, learning_rate=0.05, max_leaf_nodes=63, early_stopping=True,
                                       n_iter_no_change=50, validation_fraction=0.1,
                                       categorical_features=[Xh.columns.get_loc(c) for c in CAT_COLS], random_state=42)
    m.fit(Xh.iloc[a], y[a])
    oof_h[b] = m.predict_proba(Xh.iloc[b])[:, 1]; tst_h += m.predict_proba(Xth)[:, 1] / len(folds)
print(f"HGB: OOF {acc(oof_h):.5f} [{time.time()-t0:.0f}s]", flush=True)

for k, v in {"oof_dart": oof_d, "tst_dart": tst_d, "oof_xt": oof_e, "tst_xt": tst_e, "oof_hgb": oof_h, "tst_hgb": tst_h}.items():
    np.save(os.path.join(HERE, f"{k}.npy"), v)

# ---- v5 components
oof_l = (3 * np.load(os.path.join(HERE, "oof_l_v4.npy")) + 2 * np.load(os.path.join(HERE, "oof_dummy.npy")) ) if False else None
oof_l5 = (3 * np.load(os.path.join(HERE, "oof_l_v4.npy"))) / 3  # placeholder replaced below
# rebuild v5 blend arrays exactly as exp24 did
oof_l5 = None
import importlib
# simpler: recompute from saved parts
oof_l_v4 = np.load(os.path.join(HERE, "oof_l_v4.npy")); tst_l_v4 = np.load(os.path.join(HERE, "tst_l_v4.npy"))
oof_c_v4 = np.load(os.path.join(HERE, "oof_c_v4.npy")); tst_c_v4 = np.load(os.path.join(HERE, "tst_c_v4.npy"))
oof_x = np.load(os.path.join(HERE, "oof_x_v4.npy")); tst_x = np.load(os.path.join(HERE, "tst_x_v4.npy"))
# exp24 extras were not saved separately; reconstruct v5 blend from the submission is impossible for OOF.
# Use v4-component blend as the reference (OOF 0.94401) — conservative, since v5 OOF was 0.94389.
ref_oof = 0.25 * oof_l_v4 + 0.70 * oof_c_v4 + 0.05 * oof_x
ref_tst = 0.25 * tst_l_v4 + 0.70 * tst_c_v4 + 0.05 * tst_x
print(f"reference blend OOF: {acc(ref_oof):.5f}")

def paired(pa, pb, name):
    ca, cb = (pa > 0.5) == y, (pb > 0.5) == y
    f, b = int((~ca & cb).sum()), int((ca & ~cb).sum())
    pv = binomtest(f, f + b, 0.5).pvalue if f + b else 1.0
    print(f"{name}: {ca.mean():.5f} -> {cb.mean():.5f} fixed={f} broken={b} net={f-b} p={pv:.4f}")
    return f - b

best = (acc(ref_oof), None, None)
for tag, (o, t) in {"dart": (oof_d, tst_d), "xt": (oof_e, tst_e), "hgb": (oof_h, tst_h)}.items():
    print(f"corr with ref: {np.corrcoef(ref_oof, o)[0,1]:.4f} ({tag})")
    for w in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]:
        a = acc((1 - w) * ref_oof + w * o)
        if a > best[0]: best = (a, tag, w)
print("best add-on:", best)
if best[1]:
    o, t = {"dart": (oof_d, tst_d), "xt": (oof_e, tst_e), "hgb": (oof_h, tst_h)}[best[1]]
    new_oof = (1 - best[2]) * ref_oof + best[2] * o
    net = paired(ref_oof, new_oof, f"ref -> +{best[1]}@{best[2]}")
    if net > 0:
        new_tst = (1 - best[2]) * ref_tst + best[2] * t
        sub = pd.DataFrame({"ID": te["ID"], "default": (new_tst > 0.5).astype(int)})
        sub.to_csv(os.path.join(DATA, "submissions", "submission_v6_diversity.csv"), index=False)
        v5s = pd.read_csv(os.path.join(DATA, "submissions", "submission_v5_seedfarm.csv"))
        print("v6 written | differs from v5 on", int((sub['default'].values != v5s['default'].values).sum()), "rows")
