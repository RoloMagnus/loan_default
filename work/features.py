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


def add_offbyone(d, raw):
    """Off-by-one deviation features (v12): received interest vs schedule at k-1 and k+1 installments,
    plus which of k-1 / k / k+1 fits best."""
    d = d.copy()
    r = raw["interest_rate"].values / 1200.0
    L = raw["loan_amnt"].values.astype(float)
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


def build_v12(tr, te):
    """Champion v12 feature set: build() + off-by-one deviations."""
    X, y, Xte = build(tr, te)
    return add_offbyone(X, tr), y, add_offbyone(Xte, te)


def local_k_features(tr, te, Ws=(15, 25, 40), share=0.08):
    """Local payment-month profile (due-date phase) features.
    Pooled train+test (target-free) ID-ordered windows over k = installments at last payment.
    Returns (train_frame, test_frame)."""
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


def gap_norm_features(tr, te, X, Xte, F, Fte, Ws=(60, 150), kgroup_m=None):
    """Cohort-normalised interest deviation (label-free, pooled train+test, ID-ordered).
    gap_m = (interest received - scheduled interest at k) / monthly interest. Baseline = rolling median of gap_m among
    current-level loans (k == local robust max, W=40 window feature) over ID windows; optional second baseline = median over
    loans with the SAME k among ID-neighbours within that k group (kgroup_m = half-window in group positions)."""
    raw = pd.concat([tr.drop(columns=["default"], errors="ignore"), te], ignore_index=True)
    Xa = pd.concat([X.reset_index(drop=True), Xte.reset_index(drop=True)], ignore_index=True)
    Fa = pd.concat([F.reset_index(drop=True), Fte.reset_index(drop=True)], ignore_index=True)
    r = raw["interest_rate"].values / 1200; L = raw["loan_amnt"].values.astype(float)
    gap_m = Xa["int_gap_exact"].values / (L * r)
    cur = Fa["lk40_k_minus_rmax"].values == 0
    ids = raw["ID"].values; order = np.argsort(ids, kind="stable"); n = len(ids)
    inv = np.empty(n, dtype=int); inv[order] = np.arange(n)
    base_series = pd.Series(np.where(cur, gap_m, np.nan)[order])
    out = pd.DataFrame({"gap_m": gap_m})
    for W in Ws:
        roll = base_series.rolling(2 * W + 1, center=True, min_periods=15)
        med = roll.median().values[inv]; iqr = (roll.quantile(0.75).values - roll.quantile(0.25).values)[inv]
        out[f"gapbase{W}"] = med; out[f"gapiqr{W}"] = iqr
        out[f"gap_rel{W}"] = gap_m - med; out[f"gap_z{W}"] = (gap_m - med) / (iqr + 0.02)
    if kgroup_m:
        k = Xa["k_int"].values
        d = pd.DataFrame({"k": k, "id": ids, "gap": gap_m, "pos": np.arange(n)}).dropna(subset=["k"]).sort_values(["k", "id"])
        w = 2 * kgroup_m + 1
        d["kmed"] = d.groupby("k")["gap"].transform(lambda s: s.rolling(w, center=True, min_periods=5).median())
        d["kiqr"] = d.groupby("k")["gap"].transform(lambda s: s.rolling(w, center=True, min_periods=5).quantile(0.75) - s.rolling(w, center=True, min_periods=5).quantile(0.25))
        kmed = pd.Series(np.nan, index=range(n)); kiqr = pd.Series(np.nan, index=range(n))
        kmed.loc[d["pos"].values] = d["kmed"].values; kiqr.loc[d["pos"].values] = d["kiqr"].values
        out["gapbase_k"] = kmed.values; out["gap_rel_k"] = gap_m - kmed.values; out["gap_z_k"] = (gap_m - kmed.values) / (kiqr.values + 0.02)
    return out.iloc[:len(tr)].reset_index(drop=True), out.iloc[len(tr):].reset_index(drop=True)
