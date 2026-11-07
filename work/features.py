"""Shared feature engineering for the NBFC loan-default experiments."""
import numpy as np, pandas as pd

CAT_COLS = ["home_ownership", "income_verification_status", "loan_purpose", "state_code", "application_type"]


def prep(df, fe=True, use_id=False):
    d = df.copy()
    d["loan_term"] = d["loan_term"].str.extract(r"(\d+)").astype(float)
    d["job_experience"] = d["job_experience"].map({"<5 Years": 0, "6-10 years": 1, "10+ years": 2})
    d["loan_grade"] = d["loan_grade"].map({g: i for i, g in enumerate("ABCDEFG")})
    d["loan_subgrade"] = d["loan_subgrade"].map(lambda s: "ABCDEFG".index(s[0]) * 5 + int(s[1]) - 1)
    for c in CAT_COLS:
        d[c] = d[c].astype("category")
    if fe:
        term_m = d["loan_term"] * 12
        r = d["interest_rate"] / 1200.0
        emi = d["loan_amnt"] * r * (1 + r) ** term_m / ((1 + r) ** term_m - 1)
        d["emi"] = emi
        d["total_interest_sched"] = emi * term_m - d["loan_amnt"]
        d["pay_progress"] = d["last_week_pay"] / (d["loan_term"] * 52)
        k = (d["last_week_pay"] / 52.0 * 12.0).clip(lower=0)
        d["months_paid"] = k
        bal_after_k = d["loan_amnt"] * (1 + r) ** k - emi * ((1 + r) ** k - 1) / r
        d["interest_sched_to_date"] = emi * k - (d["loan_amnt"] - bal_after_k)
        d["int_ratio_to_sched"] = d["interest_receive"] / (d["interest_sched_to_date"] + 1)
        d["int_gap_to_sched"] = d["interest_receive"] - d["interest_sched_to_date"]
        d["int_recv_per_loan"] = d["interest_receive"] / d["loan_amnt"]
        d["int_recv_per_week"] = d["interest_receive"] / (d["last_week_pay"] + 1)
        d["int_recv_per_rate"] = d["interest_receive"] / (d["loan_amnt"] * d["interest_rate"] / 100 + 1)
        d["int_recv_share_total"] = d["interest_receive"] / (d["total_interest_sched"] + 1)
        d["emi_to_income"] = emi * 12 / (d["annual_income"] + 1)
        d["loan_to_income"] = d["loan_amnt"] / (d["annual_income"] + 1)
        d["revol_util"] = d["revolving_balance"] / (d["total_revolving_limit"] + 1)
        d["bal_to_income"] = d["total_current_balance"] / (d["annual_income"] + 1)
        d["revol_to_income"] = d["revolving_balance"] / (d["annual_income"] + 1)
        d["lwp_missing"] = d["last_week_pay"].isna().astype(int)
        d["tcb_missing"] = d["total_current_balance"].isna().astype(int)
    if use_id:
        d["loan_id_vintage"] = d["ID"]
    return d.drop(columns=["ID", "default"], errors="ignore")


def vintage_table(tr, te, window=1500, q=0.95):
    """Estimate weeks elapsed since issue for each ID via a rolling high quantile of
    last_week_pay over ID order (train+test pooled, target never used)."""
    allp = pd.concat([tr.drop(columns=["default"], errors="ignore"), te]).sort_values("ID").reset_index(drop=True)
    roll = allp["last_week_pay"].rolling(window, center=True, min_periods=200).quantile(q).bfill().ffill()
    roll = np.maximum.accumulate(roll.values[::-1])[::-1]  # non-increasing in ID
    allp["weeks_elapsed_est"] = roll
    allp["id_rank"] = allp["ID"].rank(pct=True)
    return allp.set_index("ID")[["weeks_elapsed_est", "id_rank"]]


def add_vintage(d, raw, vf):
    v = vf.loc[raw["ID"].values]
    d = d.copy()
    d["weeks_elapsed_est"] = v["weeks_elapsed_est"].values
    d["id_rank"] = v["id_rank"].values
    d["weeks_since_last_pay"] = d["weeks_elapsed_est"] - raw["last_week_pay"].values
    d["pay_recency_ratio"] = raw["last_week_pay"].values / (d["weeks_elapsed_est"] + 1)
    term_weeks = raw["loan_term"].str.extract(r"(\d+)").astype(float)[0].values * 52
    d["term_overrun_weeks"] = raw["last_week_pay"].values - term_weeks
    d["elapsed_vs_term"] = d["weeks_elapsed_est"] / term_weeks
    d["loan_should_be_done"] = (d["weeks_elapsed_est"] > term_weeks).astype(int)
    return d


def add_schedule_exact(d, raw, max_k=60):
    """Exact integer-installment amortisation features.
    last_week_pay takes discrete monthly values (4, 9, 13, 17, ...), so the number of
    installments at the last payment is k = round(weeks * 12 / 52). Comparing the interest
    actually received with the cumulative scheduled interest after k installments, and
    inverting the schedule to get the number of installments implied by the interest,
    exposes missed / irregular payments."""
    d = d.copy()
    r = raw["interest_rate"].values / 1200.0; L = raw["loan_amnt"].values.astype(float)
    term_m = raw["loan_term"].str.extract(r"(\d+)").astype(float)[0].values * 12
    emi = d["emi"].values; ir = raw["interest_receive"].values
    k = np.round(raw["last_week_pay"].values * 12 / 52.0)
    def cum_int(kk):
        bal = L * (1 + r) ** kk - emi * ((1 + r) ** kk - 1) / r
        return emi * kk - (L - bal)
    sched = cum_int(k)
    d["k_int"] = k
    d["int_dev_exact"] = (ir - sched) / (sched + 1)
    d["int_gap_exact"] = ir - sched
    ks = np.arange(0, max_k + 1)
    cum = np.stack([cum_int(kk) for kk in ks], axis=1)
    cum = np.where(ks[None, :] > term_m[:, None], np.inf, cum)
    idx = np.argmin(np.abs(cum - ir[:, None]), axis=1)
    d["k_implied"] = idx.astype(float)
    d["k_implied_resid"] = ir - cum[np.arange(len(ir)), idx]
    d["k_implied_minus_k"] = d["k_implied"] - k
    el = np.round(d["weeks_elapsed_est"].values * 12 / 52.0)
    d["months_elapsed_int"] = el
    d["installments_missed"] = el - d["k_implied"]
    d["months_behind_int"] = el - k
    return d


def build(tr, te, exact=True):
    """Full pipeline -> X_train, y, X_test (category dtypes aligned)."""
    vf = vintage_table(tr, te)
    Xtr = add_vintage(prep(tr, True, True), tr, vf)
    Xte = add_vintage(prep(te, True, True), te, vf)
    if exact:
        Xtr = add_schedule_exact(Xtr, tr); Xte = add_schedule_exact(Xte, te)
    for c in CAT_COLS:
        cats = pd.api.types.union_categoricals([Xtr[c], Xte[c]]).categories
        Xtr[c] = pd.Categorical(Xtr[c], categories=cats); Xte[c] = pd.Categorical(Xte[c], categories=cats)
    return Xtr, tr["default"].values, Xte[Xtr.columns]


LGB_PARAMS = dict(objective="binary", learning_rate=0.03, num_leaves=63, min_child_samples=40,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                  verbose=-1, n_jobs=12, seed=42)


def lgb_cv(X, y, params=LGB_PARAMS, n_splits=5, seed=42, rounds=5000, es=200):
    import lightgbm as lgb
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits, shuffle=True, random_state=seed)
    oof = np.zeros(len(X)); iters = []; models = []
    for a, b in skf.split(X, y):
        m = lgb.train(params, lgb.Dataset(X.iloc[a], y[a]), rounds,
                      valid_sets=[lgb.Dataset(X.iloc[b], y[b])],
                      callbacks=[lgb.early_stopping(es, verbose=False)])
        oof[b] = m.predict(X.iloc[b], num_iteration=m.best_iteration)
        iters.append(m.best_iteration); models.append(m)
    return oof, iters, models


def report(name, oof, y):
    acc = ((oof > 0.5) == y).mean()
    ths = np.arange(0.3, 0.7, 0.01); accs = [((oof > t) == y).mean() for t in ths]
    print(f"{name}: acc@0.5={acc:.5f}  best_thr={ths[int(np.argmax(accs))]:.2f} acc={max(accs):.5f}", flush=True)
    return acc
