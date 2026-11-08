"""Submission v12: full champion pipeline rebuilt on the off-by-one-corrected schedule features,
plus the (already-measured) probe calibration flips. The one final, OOF-validated challenger."""
import os, sys, json, time, numpy as np, pandas as pd, lightgbm as lgb
import xgboost as xgb_lib
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import StratifiedKFold
from sklearn.isotonic import IsotonicRegression
from scipy.stats import binomtest
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from features import build, CAT_COLS


def schedule_fix(d, raw, cap=False, offbyone=True):
    """Off-by-one deviation features (audited in exp28; cap variant was harmful and is off)."""
    d = d.copy()
    r = raw["interest_rate"].values / 1200.0
    L = raw["loan_amnt"].values.astype(float)
    term_m = raw["loan_term"].str.extract(r"(\d+)").astype(float)[0].values * 12
    emi = d["emi"].values
    ir = raw["interest_receive"].values
    k_raw = np.round(raw["last_week_pay"].values * 12 / 52.0)
    def cum_int(kk):
        bal = L * (1 + r) ** kk - emi * ((1 + r) ** kk - 1) / r
        return emi * kk - (L - bal)
    s_m1 = cum_int(np.maximum(k_raw - 1, 0)); s_p1 = cum_int(k_raw + 1)
    d["int_dev_km1"] = (ir - s_m1) / (s_m1 + 1)
    d["int_dev_kp1"] = (ir - s_p1) / (s_p1 + 1)
    devs = np.stack([np.abs(ir - s_m1), np.abs(ir - cum_int(k_raw)), np.abs(ir - s_p1)], axis=1)
    d["best_k_hypothesis"] = np.argmin(devs, axis=1).astype(float)
    return d

DATA = "." if len(sys.argv) < 2 else sys.argv[1]
tr = pd.read_csv(f"{DATA}/Train_set.csv"); te = pd.read_csv(f"{DATA}/Test_set.csv")
Xb, y, Xteb = build(tr, te)
X = schedule_fix(Xb, tr, cap=False, offbyone=True)
Xte = schedule_fix(Xteb, te, cap=False, offbyone=True)
acc = lambda p: ((p > 0.5) == y).mean()
NF = 10; folds = list(StratifiedKFold(NF, shuffle=True, random_state=42).split(X, y))
P_TRUE = 9590

lp = json.load(open(os.path.join(HERE, "lgb_best.json")))["params"]
t0 = time.time(); oof_l = np.zeros(len(X)); tst_l = np.zeros(len(Xte)); SL = [42, 7, 2024]
for s in SL:
    P = dict(objective="binary", verbose=-1, n_jobs=12, bagging_freq=1, seed=s, **lp)
    for a, b in folds:
        m = lgb.train(P, lgb.Dataset(X.iloc[a], y[a]), 4000, valid_sets=[lgb.Dataset(X.iloc[b], y[b])],
                      callbacks=[lgb.early_stopping(150, verbose=False)])
        oof_l[b] += m.predict(X.iloc[b], num_iteration=m.best_iteration) / len(SL)
        tst_l += m.predict(Xte, num_iteration=m.best_iteration) / (NF * len(SL))
print(f"LGB: {acc(oof_l):.5f} [{time.time()-t0:.0f}s]", flush=True)

t0 = time.time(); oof_x = np.zeros(len(X)); tst_x = np.zeros(len(Xte))
dtest = xgb_lib.DMatrix(Xte, enable_categorical=True)
for s in SL:
    xp = dict(objective="binary:logistic", eval_metric="logloss", eta=0.03, max_depth=7, min_child_weight=5,
              subsample=0.8, colsample_bytree=0.7, reg_lambda=1.0, tree_method="hist", nthread=12, seed=s)
    for a, b in folds:
        dtr = xgb_lib.DMatrix(X.iloc[a], y[a], enable_categorical=True)
        dva = xgb_lib.DMatrix(X.iloc[b], y[b], enable_categorical=True)
        m = xgb_lib.train(xp, dtr, 5000, evals=[(dva, "va")], early_stopping_rounds=150, verbose_eval=False)
        oof_x[b] += m.predict(dva, iteration_range=(0, m.best_iteration + 1)) / len(SL)
        tst_x += m.predict(dtest, iteration_range=(0, m.best_iteration + 1)) / (NF * len(SL))
print(f"XGB: {acc(oof_x):.5f} [{time.time()-t0:.0f}s]", flush=True)

cp = json.load(open(os.path.join(HERE, "cat_best.json")))["params"]
Xc = X.copy(); Xtc = Xte.copy()
for c in CAT_COLS: Xc[c] = Xc[c].astype(str); Xtc[c] = Xtc[c].astype(str)
t0 = time.time(); oof_c = np.zeros(len(X)); tst_c = np.zeros(len(Xte)); SC = [42, 7]
for s in SC:
    for a, b in folds:
        m = CatBoostClassifier(iterations=3000, eval_metric="Logloss", od_type="Iter", od_wait=100,
                               random_seed=s, verbose=False, thread_count=12, **cp)
        m.fit(Pool(Xc.iloc[a], y[a], cat_features=CAT_COLS), eval_set=Pool(Xc.iloc[b], y[b], cat_features=CAT_COLS), use_best_model=True)
        oof_c[b] += m.predict_proba(Xc.iloc[b])[:, 1] / len(SC)
        tst_c += m.predict_proba(Xtc)[:, 1] / (NF * len(SC))
print(f"Cat: {acc(oof_c):.5f} [{time.time()-t0:.0f}s]", flush=True)

oof = 0.25 * oof_l + 0.70 * oof_c + 0.05 * oof_x
tst = 0.25 * tst_l + 0.70 * tst_c + 0.05 * tst_x
oof_v4 = 0.25 * np.load(os.path.join(HERE, "oof_l_v4.npy")) + 0.70 * np.load(os.path.join(HERE, "oof_c_v4.npy")) + 0.05 * np.load(os.path.join(HERE, "oof_x_v4.npy"))
ca, cb = (oof_v4 > 0.5) == y, (oof > 0.5) == y
f, b = int((~ca & cb).sum()), int((ca & ~cb).sum())
print(f"v4 blend {ca.mean():.5f} -> v12 blend {cb.mean():.5f} net={f-b:+d} McNemar p={binomtest(f, f+b, 0.5).pvalue:.4f}")

# identical calibration correction with the already-measured P_TRUE
iso = IsotonicRegression(out_of_bounds="clip").fit(oof, y)
qhat = iso.predict(tst)
delta = (qhat.sum() - P_TRUE) / len(tst)
q_corr = np.clip(qhat - delta, 0, 1)
pred = (tst > 0.5).astype(int)
gain = np.where(pred == 1, 1 - 2 * q_corr, 2 * q_corr - 1)
order = np.argsort(-gain); cum = np.cumsum(gain[order])
k_opt = int(np.argmax(cum)) + 1 if cum.max() > 0 else 0
print(f"calibration: E[P]={qhat.sum():.0f} vs {P_TRUE}, delta={delta:+.5f}, flips={k_opt}, EV={cum.max():.1f} rows")
newpred = pred.copy()
if k_opt: newpred[order[:k_opt]] = 1 - newpred[order[:k_opt]]
sub = pd.DataFrame({"ID": te["ID"], "default": newpred})
sub.to_csv(os.path.join(DATA, "submissions", "submission_v12_offbyone.csv"), index=False)
v10 = pd.read_csv(os.path.join(DATA, "submissions", "submission_v10_calibrated.csv"))["default"].values
print("v12 written | pred_pos", int(newpred.sum()), "| differs from v10 on", int((newpred != v10).sum()), "rows")
np.save(os.path.join(HERE, "oof_v12.npy"), oof); np.save(os.path.join(HERE, "tst_v12.npy"), tst)
