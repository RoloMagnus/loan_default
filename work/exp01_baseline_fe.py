"""Experiment 1: raw features vs engineered features, LightGBM 5-fold CV accuracy."""
import sys, time, warnings
import numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
warnings.filterwarnings("ignore")

DATA = sys.argv[1]
tr = pd.read_csv(f"{DATA}/Train_set.csv")
te = pd.read_csv(f"{DATA}/Test_set.csv")
y = tr["default"].values

def prep(df, fe=True):
    d = df.copy()
    d["loan_term"] = d["loan_term"].str.extract(r"(\d+)").astype(float)
    d["job_experience"] = d["job_experience"].map({"<5 Years": 0, "6-10 years": 1, "10+ years": 2})
    d["loan_grade"] = d["loan_grade"].map({g: i for i, g in enumerate("ABCDEFG")})
    d["loan_subgrade"] = d["loan_subgrade"].map(lambda s: "ABCDEFG".index(s[0]) * 5 + int(s[1]) - 1)
    for c in ["home_ownership", "income_verification_status", "loan_purpose", "state_code", "application_type"]:
        d[c] = d[c].astype("category")
    if fe:
        term_m = d["loan_term"] * 12
        r = d["interest_rate"] / 1200.0
        emi = d["loan_amnt"] * r * (1 + r) ** term_m / ((1 + r) ** term_m - 1)
        d["emi"] = emi
        d["total_interest_sched"] = emi * term_m - d["loan_amnt"]
        d["weeks_in_term"] = d["loan_term"] * 52
        d["pay_progress"] = d["last_week_pay"] / d["weeks_in_term"]
        months_paid = d["last_week_pay"] / 52.0 * 12.0
        d["months_paid"] = months_paid
        # scheduled cumulative interest after k payments (amortisation closed form)
        k = months_paid.clip(lower=0)
        bal_after_k = d["loan_amnt"] * (1 + r) ** k - emi * ((1 + r) ** k - 1) / r
        principal_paid = d["loan_amnt"] - bal_after_k
        d["interest_sched_to_date"] = emi * k - principal_paid
        d["int_ratio_to_sched"] = d["interest_receive"] / (d["interest_sched_to_date"] + 1)
        d["int_recv_per_loan"] = d["interest_receive"] / d["loan_amnt"]
        d["int_recv_per_week"] = d["interest_receive"] / (d["last_week_pay"] + 1)
        d["int_recv_per_rate"] = d["interest_receive"] / (d["loan_amnt"] * d["interest_rate"] / 100 + 1)
        d["emi_to_income"] = emi * 12 / (d["annual_income"] + 1)
        d["loan_to_income"] = d["loan_amnt"] / (d["annual_income"] + 1)
        d["revol_util"] = d["revolving_balance"] / (d["total_revolving_limit"] + 1)
        d["bal_to_income"] = d["total_current_balance"] / (d["annual_income"] + 1)
        d["revol_to_income"] = d["revolving_balance"] / (d["annual_income"] + 1)
        d["lwp_missing"] = d["last_week_pay"].isna().astype(int)
        d["tcb_missing"] = d["total_current_balance"].isna().astype(int)
    return d.drop(columns=["ID", "default"], errors="ignore")

params = dict(objective="binary", learning_rate=0.03, num_leaves=63, min_child_samples=40,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
              verbose=-1, n_jobs=12, seed=42)

for fe in [False, True]:
    X = prep(tr, fe)
    skf = StratifiedKFold(5, shuffle=True, random_state=42)
    oof = np.zeros(len(X)); t0 = time.time(); iters = []
    for f, (a, b) in enumerate(skf.split(X, y)):
        m = lgb.train(params, lgb.Dataset(X.iloc[a], y[a]), 5000,
                      valid_sets=[lgb.Dataset(X.iloc[b], y[b])],
                      callbacks=[lgb.early_stopping(200, verbose=False)])
        oof[b] = m.predict(X.iloc[b], num_iteration=m.best_iteration); iters.append(m.best_iteration)
    acc = ((oof > 0.5) == y).mean()
    # best threshold
    ths = np.arange(0.3, 0.7, 0.01); accs = [((oof > t) == y).mean() for t in ths]
    print(f"FE={fe}: acc@0.5={acc:.5f}  best_thr={ths[int(np.argmax(accs))]:.2f} acc={max(accs):.5f}  iters={iters}  {time.time()-t0:.0f}s", flush=True)
    if fe:
        imp = pd.Series(m.feature_importance("gain"), index=X.columns).sort_values(ascending=False)
        print(imp.head(25).round(0).to_string())
