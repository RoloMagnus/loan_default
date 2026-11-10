# Winning a Loan-Default Hackathon One Falsifiable Hypothesis at a Time — a Deep Dive into the Methodology

*A practical case study in data detective work, exact amortisation reconstruction, pre-registered promotion gates, paired statistical testing, and an honest answer to the sharpest comment on my last article.*

The model was, again, only half the problem. The other half was deciding — with evidence, before every single upload — whether a candidate deserved to replace the frozen champion. After my Restaurant Turnover article, a reader left the comment every author hopes for and fears: *"the part everyone skips is whether the agent workflow improved out-of-sample performance or just made leaderboard overfitting faster… without ablations against one careful human and a fixed compute budget, 'agentic orchestration' may just be expensive parallel prompting with better branding."* This article is my answer: the same competition circuit, a deliberately different workflow, and — this time — the ablation ledger, the statistical gates, and the receipts published in full.

## 1. The Problem

The task, organised by Great Learning: predict whether a client of an Indian NBFC (non-banking financial company) will **default on a loan**. 93,174 labeled training loans, 39,933 unlabeled test loans, 22 predictors spanning loan terms (amount, term, interest rate, grade/subgrade), borrower attributes (income, employment, home ownership, state), credit-bureau counts (delinquencies, public records, revolving balances), and two deceptively quiet columns describing the loan *after* origination: `interest_receive` (cumulative interest collected) and `last_week_pay` (how long the borrower had been paying). The metric was plain **accuracy**; 23.75% of training loans defaulted, so "nobody defaults" already scores 76.25% and the contest would be decided in the third decimal — the final margin was about 20 loans out of 39,933.

The objective, as before, was not to fit the most complex model. It was to build a defensible system: audit the data, understand the *servicing mechanics* that generated it, engineer features that reconstruct those mechanics, and promote a challenger only when a paired statistical test on identical validation folds said the improvement was real.

## 2. Technology Stack

| Layer | Tools |
|---|---|
| Language / runtime | Python 3.12 |
| Core data stack | pandas 2.2, NumPy |
| Modeling | LightGBM, CatBoost, XGBoost (+ scikit-learn DT/RF/GB/HGB and a Keras ANN as graded-notebook baselines) |
| Hyperparameter search | Optuna (TPE; resumable SQLite studies) |
| Statistics | SciPy (McNemar exact test, binomial tests), paired bootstrap |
| Visualization | matplotlib, seaborn |
| Persistence & reproducibility | 46 numbered experiment scripts, OOF prediction arrays on disk, SHA-256-checksummed frozen champion, 31+ git commits |
| IDE / version control | Visual Studio Code, GitHub |
| AI workflow | **One Claude Code conversation** (no multi-agent harness this time), relayed across three Claude models — Fable 5.1 → Fable 5 → Sonnet 5.5 — plus **ChatGPT as an unmetered outside reviewer** |

## 3. From Volleyball Team to Singles Match — the Workflow Experiment

Last time I orchestrated multiple agents through the DontPanic harness. This time I deliberately benched the team: one continuous AI conversation as the only engineering partner, me at every decision line, and a second LLM (a ChatGPT thread) used the way a team uses an external referee — it read the raw CSVs independently, challenged my leakage framing, demanded stricter promotion criteria, and twice proposed experiments (one of which, repayment-signature matching, ended up changing 7 test predictions).

### 3.1 What survived from the volleyball playbook

The harness disappeared; its disciplines did not. They became standing rules of a single conversation:

- **Serve** — I framed each question, the constraint, and the current champion before an experiment started.
- **Receive/Set/Attack collapsed into one player** — the same conversation explored, implemented, and attacked its own results; the referee LLM supplied independent challenge.
- **Return to human** — every submission, every "stop here", every gamble was my explicit call.

### 3.2 The promotion gate (the DEV → UAT → PROD of this project)

The single most important methodological object in this project was a rule written down *before* the first submission and never relaxed:

> **A challenger replaces the frozen champion only if, on identical cross-validation folds, a paired comparison of out-of-fold predictions shows it fixes more rows than it breaks — tested with McNemar's exact test and a paired bootstrap CI — and, for feature changes, the gain replicates with the same sign across 3 independent seeds.**

DEV was cheap scripting on 5-fold CV. UAT was the gate: identical folds, paired counts of fixed-vs-broken rows, seed replication. PROD was the frozen champion — an immutable CSV with a SHA-256 manifest, never edited, only ever *replaced* by a gate-passing successor. When my machine crashed mid-campaign (it did, several times), everything resumed from artifacts, because the system of record was never the chat.

## 4. Data Audit — Trust Nothing Until Verified

Column-by-column audit before any model:

- Train 93,174 × 23, test 39,933 × 22; **zero duplicate rows, zero duplicate IDs, zero train/test ID overlap**; the sample-submission row order matched the test file exactly.
- Missingness was *structured*, not random: `total_current_balance` and `total_revolving_limit` missing **together** on 7.9% of rows; `job_experience` 5.0%; `last_week_pay` 2.1%. The audit flagged these for the EDA question "is missingness itself a signal?" (it was — dramatically, see 5.4).
- `last_week_pay`, nominally "weeks", took only the values 4, 9, 13, 17, 22, 26… — **whole months expressed in weeks**. This one observation, made during the audit, later unlocked the entire feature-engineering programme.
- `annual_income` max ₹9.5M against a ₹64k median, and a `total_revolving_limit` of 9,999,999 — sentinel values noted; tree models handle them, so no clipping was imposed.
- A nine-attribute exact-match test confirmed **no test loan duplicates any training loan** — ruling out the cheapest possible exploit and, later, motivating the more subtle signature analysis of Section 9.

## 5. Exploratory Data Analysis

### 5.1 The target and the trap

23.75% default. Accuracy as a metric meant both classes mattered; threshold 0.5 was audited repeatedly on out-of-fold probabilities and the optimum never moved beyond noise (0.48–0.54 across seeds, ≤0.03 pp).

### 5.2 The paradox that cracked the dataset: grade A defaults more than grade E

The first bivariate pass produced nonsense: **grade A loans defaulted at 27.4% versus 18.6% for grade E**; verified incomes defaulted more than unverified; low debt-to-income looked *riskier* than high. Risk grades cannot be that wrong — so something correlated with grade had to be confounding everything.

That something was time. **Default rate fell monotonically from 78% in the oldest ID decile to 0.9% in the newest** — IDs are issued sequentially, so the ID column is a loan-age clock in disguise. Older vintages have had years of exposure in which to default, and they happened to contain more A/B loans. Restricted to the newest half of vintages, the grade ordering snapped back to normal (A 5.0% → G 8.2%). Every subsequent analysis conditioned on vintage; every raw correlation was re-checked against this confounder (debt-to-income's "protective" effect evaporated the same way).

### 5.3 Dating the undated: estimating loan age without a date column

There is no date field anywhere. But the *upper envelope* of `last_week_pay` — the longest any loan in an ID neighbourhood has been paying — decreases almost perfectly monotonically in ID. A rolling 95th percentile over ID order (train and test pooled; the target never touches this computation), forced monotone, yields an **estimated weeks-since-issue for every loan**. That estimate turned "this borrower paid for 65 weeks" into the far more useful "this borrower paid for 65 of an available 120 weeks — and then stopped."

### 5.4 Missingness as signal

- `total_current_balance` missing → **79.7% default** (an old reporting cohort).
- `last_week_pay` missing → **0.0% default** (brand-new loans, no payment due yet).

Imputing these away would have destroyed information; instead, explicit missingness flags were kept and the gradient-boosting models were allowed to route NaNs natively.

### 5.5 What the correlation matrix could not see

No raw feature correlated with the target beyond 0.27. The signal was *relational* — it lived in the comparison between what a borrower **did** pay and what the amortisation schedule says they **should have** paid by now. That diagnosis set the entire feature agenda of Section 7.

## 6. A Snapshot Problem in Disguise — Framing Before Fitting

Before engineering anything, one framing question had to be settled, because it decides what "leakage" even means here: *when is the prediction supposed to happen?* With post-origination payment columns in the feature set, this competition is **snapshot-status classification** — "is this loan, as observed at the data-extraction date, in default?" — which is neither origination-time credit scoring ("should we approve this applicant?") nor horizon-based servicing prediction ("will this current loan default within 12 months?").

The hackathon's FAQ explicitly rewards accuracy, so the payment columns are fair game — but the distinction was documented in the graded notebook and quantified: a model restricted to origination-time features reaches **0.824** against the 0.763 majority baseline, while the full snapshot feature set reaches **0.946**. Nearly all predictive power is post-origination repayment behaviour. Section 14 returns to what that means for generalisation claims.

## 7. Feature Engineering — Reconstructing the Loan Servicing System

This is where the competition was won, in four increasingly surgical passes. The ablation ladder below is the article in one picture.

**[FIGURE F6_feature_ladder.png]**

### 7.1 Pass one: financial ratios and the EMI (0.878 → 0.907)

Standard affordability engineering: the amortised monthly installment (EMI) from amount, rate and term via the closed-form annuity formula; EMI-to-income; loan-to-income; revolving utilisation; balance ratios; missingness flags.

### 7.2 Pass two: the clock (0.907 → 0.931)

The vintage features from 5.3: estimated weeks since issue, ID rank, **weeks since last payment** (elapsed minus `last_week_pay`), pay-progress fraction, elapsed-versus-term ratio. "How long ago did the payments stop?" is most of the default signal — but only once you can estimate "how long ago".

### 7.3 Pass three: exact schedule inversion (0.931 → 0.941) — the decisive step

Because `last_week_pay` is whole months in weeks, the number of installments at the last payment is exactly `k = round(weeks × 12/52)`. The closed-form amortisation then gives the *scheduled cumulative interest after k installments*. Two constructions followed:

- **Deviation features**: received interest minus scheduled interest at k, as a gap and a ratio. For most performing loans this deviation sits within 1% of zero — the schedule really is exact.
- **Schedule inversion**: scan k = 0…60 and find the installment count whose scheduled cumulative interest best matches the interest actually received — the **installments implied by money**, independent of the timing column. Its residual, and above all **installments missed = months elapsed − installments implied**, became the strongest features in the model. Default rate climbs from 2–3% at ≤1 missed installment to 61% at 2–3 missed to 77% at 12+.

### 7.4 Pass four: the audits that kept paying (0.941 → 0.946)

Three later refinements each came from *auditing the reconstruction itself* rather than adding new data:

- **Off-by-one deviations** (v12): the week→month rounding means received interest sometimes fits k−1 or k+1 better than k. Evaluating the schedule at all three and adding a "best-fitting k" indicator gained +29 OOF rows, replicating with the same sign across 3 seeds. The mirror experiment — capping k at contractual maturity, which "should" be a correctness fix — **lost 44 rows**: the overshoot beyond maturity was itself signal (lingering late payers), a reminder that the gate, not intuition, decides.
- **Due-date phase** (v13): a residual audit revealed that whether a *current* borrower looks "0 months behind" or "1 month behind" depends on where their due date falls inside the issue-month cohort — 13% of current loans look one-behind in the earliest cohort positions versus 87% in the latest. A smooth time estimate blurs current and late. The fix: label-free **local windows over neighbouring IDs** (±40 loans, train+test pooled), extracting each loan's distance from its own cohort's modal and maximal installment count. Replicated +62/+98/+37 rows across seeds; the gain was almost entirely false-negative reduction.
- **Cohort-normalised interest gaps** (v14): the same normalisation applied to the interest-deviation features — your deviation relative to your 40 nearest contemporaries, not to a global schedule. Replicated +106/+84/+187 rows across seeds.

**[FIGURE F5_phase_discovery.png]**

The common thread: every one of these was a **falsifiable hypothesis about loan mechanics**, written as a feature, and executed through the same gate. None of them is "try a bigger model".

## 8. External Information as an Enrichment Layer — One Deliberate Probe

The Restaurant Turnover project tested external geographic data as enrichment. Here the only external information available was the leaderboard itself, and it was used **once, openly, as a measurement instrument**: a single all-zeros submission, whose score is by definition 1 − (true test default rate). Result: the test set contains exactly 9,590 defaulters (24.015%).

That one number converted threshold calibration from guesswork into arithmetic. An isotonic calibration fit on out-of-fold predictions expected 9,666 positives on test — only 76 rows from truth, a strong external validation of the model's calibration — and the residual gap implied that flipping the 90 most marginal positive predictions had an expected value of **+3.8 rows**. Realised: **+6**. A second calibration pass on the final champion predicted +0.26 rows and was therefore *not applied* — the same arithmetic saying "don't".

Disclosure, then and now: this uses competition feedback, so the public score is not a fully independent evaluation of the selected model. It was one submission out of eighteen, its mechanism is published, and the final champion's predictions do not depend on it (its calibration flips were skipped as sub-threshold).

## 9. Models — What Was Tried and Why

| Model | Why it was included | Outcome |
|---|---|---|
| Decision Tree, Random Forest, sklearn GB, ANN | Graded-notebook baselines; sanity floor | 1.5–3 pp behind the GBMs |
| **LightGBM** | Fast iteration engine for all 46 experiments; native NaN/categorical handling | Core blend member (every feature ablation ran on it) |
| **CatBoost** | Strongest single model on this data; ordered target statistics for categoricals | Highest-weight blend member (0.40–0.70) |
| **XGBoost** | Third independent boosting implementation for residual diversity | Minor blend member (0.05–0.20) |
| LightGBM dart / extra-trees, sklearn HistGB | Deliberate diversity candidates for the endgame | Correlation >0.99 with the blend; zero added rows — rejected |
| Logistic / LightGBM stackers (3 variants) | Learned combination vs weight grid | All failed the paired gate — rejected |
| Repayment-signature lookup (referee's idea) | 8,348 test loans share a 5-attribute repayment signature with training loans; unanimous signatures (≥2 matches) are 99.8% accurate held-out | Kept as a 7-row override on the final champion |

The deeper finding: on a tabular problem where the features already encode the domain mechanics, **three boosting implementations converge to nearly the same function** (pairwise OOF correlation 0.992–0.997). Ensembling bought a real but modest ~0.1 pp; architecture diversity bought nothing. The leverage was never in the model zoo.

## 10. Hyperparameter Tuning

Optuna TPE with resumable SQLite studies, always scored by out-of-fold accuracy on fixed folds: 65 trials for LightGBM across two feature generations (learning rate, leaves, depth, child samples, feature/bagging fractions, L1/L2, min-gain, categorical smoothing), 12 deliberately-budgeted trials for CatBoost (0.94246 → 0.94269 — the search confirmed the defaults were nearly right), XGBoost left at experiment-validated settings given its 0.05–0.2 blend weight. Tuning totalled roughly +0.3 pp of the 6.8 pp journey — real, bounded, and an order of magnitude less than the feature work. Final-generation training used **10-fold CV with 2–3 seeds per model**, averaging both out-of-fold and test predictions (10-fold's larger training share was worth ~+0.04 pp over 5-fold, adopted for the final sprint).

## 11. Model Ensembling Technique — and Its Statistical Gate

Out-of-fold probabilities were stored for every candidate, so every ensembling decision was evaluated on identical rows:

- **Weight grids** over the three models' OOF predictions (0.05 steps), evaluated by OOF accuracy — but *promoted* only by the paired test. The grid-best blend beat the best single model by net +85 rows, McNemar p = 0.007, bootstrap 95% CI strictly positive → promoted. The final champion blend: LightGBM 0.50 / CatBoost 0.40 / XGBoost 0.10.
- **Caruana hill-climb** over a library of seven saved model runs: converged back to the same two models and lost 5 net rows → rejected.
- **Stacking** (logistic on logits, LightGBM on logits, LightGBM on logits + raw features): net −39 to +8 rows, p ≥ 0.13 → all rejected.
- **Seed farms**: averaging 5 seeds per model shifted the mean less than +0.02 pp but shrank the submission-to-submission luck spread — variance reduction, not skill.

## 12. What Improved — and What Did Not

The project improved because weak ideas were rejected as aggressively as strong ideas were promoted; the rejections *are* the ablation study the commenter asked for.

**[TABLE T4_experiment_verdicts.png]**

Three rejected ideas deserve a paragraph each, because each *felt* certain to work:

- **The "flip engine."** With errors concentrated in long-inactive loans, the obvious move is a second-stage model (or a rule) that flips predictions in the hard segment. Flipping only profits where the model is wrong **more than 50% of the time**, so I scanned every pairwise rule over payment features, confidence bands and predicted class — roughly a thousand candidate subgroups, discovered on one half of the data and verified on the other. **Zero subgroups survived.** A boosted model with these features has already internalised every rule that coarse; the worst honest pocket was ~45% wrong, and flipping it loses rows.
- **The maturity cap.** Capping installment counts at contractual term is the "obviously correct" accounting fix — and it cost 44 rows, because payment weeks beyond maturity encode lingering distressed payers. Correctness of representation and usefulness of representation are different claims; only the gate can tell them apart.
- **Segment-specific correction for the irreducible core.** Loans that stopped paying 12+ installments ago are ~21% errors and about half of all remaining mistakes — because a loan **paid off early** and a loan **charged off** become observationally identical once payments stop, and no field in this data separates them. A cross-fitted residual model and segment thresholds both failed the gate. Knowing *where the data runs out* is a result, not a failure: it went straight into the business recommendations.

## 13. Performance Across Champions

**[FIGURE F1_leaderboard_progression.png]**

**[TABLE T2_submission_ledger.png]**

Eighteen submissions over three days, three times overtaken, three times retaken. The sequence tells the method's story compactly: every durable jump on that chart is a named, replicated feature hypothesis (v2 schedule inversion, v12 off-by-one, v13 phase, v14 cohort gaps); every flat stretch is the gate rejecting something; the three deliberate "equal-skill redraws" (v6–v8, blends identical in validated skill but differing on coin-flip rows) all landed below the champion — a cheap, honest lesson in what luck does and doesn't buy. Final: **0.94383, first of 33, ~20 rows ahead of second place**.

## 14. Competition Performance vs. Generalization

This was the heart of the reader's critique last time, so this time it gets the strongest figure in the article:

**[FIGURE F2_cv_vs_leaderboard.png]**

Across all nine champions, spanning three days and five feature generations, **the public leaderboard tracked pre-submission cross-validation with a constant offset of ≈0.25 pp, never deviating by more than 0.03 pp**. That stability is the measurable signature that the gains were generalisation rather than leaderboard fitting: a workflow that was tuning to the leaderboard would show the gap *widening* with every submission. (An adversarial-validation check — a classifier trying to distinguish train from test rows — scored AUC 0.50, confirming the two samples are exchangeable, which is why the offset is both small and stable.)

The same honesty cuts the other way, with three documented limits: the all-zeros probe and its +6-row calibration *did* use competition feedback (Section 8); a vintage-cohort holdout — train on four loan-age groups, score the fifth — drops to 0.9166 overall and 0.75 on the oldest vintage, so the model **cannot extrapolate to unseen loan ages** and would need vintage-matched retraining in any deployment; and the origination-only model's 0.824 (Section 6) is the honest ceiling for *approval-time* prediction with these attributes. Competition-valid, deployment-caveated — the same two-numbers discipline as last time, now with the offset chart to prove the first number was earned.

## 15. What It Cost — a Short Aside on Metered AI

Because the commenter also asked about compute budgets, this project's AI cost was *measured*, from the session log, per API call: **about $186 at list-price equivalent** over 67 hours (364 calls, 85 human prompts). Two findings worth a data scientist's minute: the spend was **70% cache writes and only 8% actual model output** — the cost of a long-context AI workflow is re-reading state after interruptions, not reasoning (18% of calls, the cold starts after pauses and crashes, caused 96% of cache-write tokens); and the August hackathon's cost *cannot* be compared because flat-rate subscription seats never produced a log. The honest contrast is not "agents vs single model" but **metered visibility vs flat-rate opacity** — this one felt more expensive because, for the first time, the meter was visible. Full method, caveats, and a pre-registration checklist for a fair future comparison live in the repo (`article/METHODOLOGY_NOTE.md`).

**[TABLE T3_measured_cost.png]** *(optional: F3_cost_anatomy.png / T1_two_hackathons.png as a sidebar)*

## 16. Main Assumptions

- Train and test were assumed exchangeable; verified empirically (adversarial AUC 0.50, probe-measured test default rate within 0.3 pp of train).
- `last_week_pay` was interpreted as weeks-from-issue at the last payment, in whole-month steps; validated by the near-exact match between schedule-implied and received interest on performing loans.
- The ID column was treated as a monotone issue-date proxy; validated by the monotone `last_week_pay` envelope and the vintage-conditioned grade ordering.
- All label-dependent statistics (encodings, calibration, gates) were computed strictly out-of-fold; the pooled train+test computations (vintage envelope, cohort windows) use feature columns only, never the target.
- The snapshot framing of Section 6 bounds every claim: this model classifies observed status at extraction time; it is not an origination scorer and was never promoted as one.

## 17. Business and Modeling Recommendations

- **Audit identifiers before features: your ID column may be a clock.** Sequential IDs leak time; conditioning on it reversed every naive risk correlation in this data. Any portfolio analysis at an NBFC must control for loan age before comparing segments.
- **Reconstruct the mechanism, don't approximate it.** The amortisation formula is public mathematics; inverting it turned two opaque columns into "installments missed", worth more than every raw attribute combined. The earliest actionable alarm sits at 2–3 missed installments (3% → 61% default) — a concrete trigger for servicing outreach.
- **Treat structured missingness as a feature and record what's absent from the schema.** A payoff-versus-charge-off status field — one column — would eliminate this model's largest remaining error source.
- **Pre-register the promotion gate and let it reject.** Paired tests on identical folds, replicated across seeds, kept 18 submissions honest where 622 once ran free; the constant CV-to-leaderboard offset is the audit trail.
- **Use the leaderboard (or any scored oracle) as an instrument, sparingly and openly.** One designed probe yielded one clean parameter (test prevalence) worth exactly its predicted +4–6 rows; repeated score-chasing redraws yielded nothing, measurably.
- **Keep the system of record outside the chat, whatever the AI workflow.** Numbered scripts, on-disk OOF arrays and a checksummed champion survived three machine crashes and two model handoffs; conversational memory did not need to.

## 18. Conclusion and Lessons Learned

The strongest lesson from this project is a sharpening of the last one. Then, I argued that disciplined validation beats model complexity; now I can quantify it: of 6.8 percentage points gained, **~6.0 came from reconstructing the loan-servicing mechanics in features, ~0.5 from models/tuning/ensembling, and ~0.3 from measurement-driven calibration** — and the fourteen-idea ledger behind those numbers rejected more than it promoted. The volleyball team became a singles match and the discipline survived the translation, because the discipline was never really about the agents: frozen champions, paired gates, seed replication, artifacts over memory.

To the commenter who asked whether the workflow improves out-of-sample performance or just accelerates leaderboard overfitting: here is the ablation ledger, here is the constant 0.25 pp CV-to-leaderboard offset across every champion, here is the one disclosed probe, and here is the meter. Rapid search and reliable systems are not opposites — provided the search is fast and the *promotion* is slow.

---

*The full repository — 46 experiment scripts, the executed solution notebook, the frozen checksummed champion, and the cost-measurement scripts — is public: github.com/RoloMagnus/loan_default. Final result: 1st of 33, accuracy 0.94383, November 9, 2026.*
