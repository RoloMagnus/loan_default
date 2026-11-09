# This Time I Benched the Volleyball Team: Winning a Loan-Default Hackathon with One AI Conversation — and a Visible Meter

*A follow-up case study: one Claude Code session instead of a multi-agent harness, 18 submissions instead of 622, a measured $186 of tokens — and an answer to the sharpest comment on the last article.*

---

> **Suggested subtitle:** What a reader's critique about "expensive parallel prompting with better branding" taught me, and what the receipts say.

*(Hero image suggestion: reuse the volleyball visual language — one player alone at the net this time, or a relay-baton image for the three-model handoff.)*

## 0. The comment that started this

After the Restaurant Turnover article, one reader left the comment every author hopes for and fears:

> "the part everyone skips is whether the agent workflow improved out-of-sample performance or just made leaderboard overfitting faster. hackathons reward rapid search, not reliable systems. without ablations against one careful human and a fixed compute budget, 'agentic orchestration' may just be expensive parallel prompting with better branding"

They were partly right, and my own article said so: the final LB01–LB05 phase of that hackathon was disclosed as competition-specific calibration, and I reported the clean cross-validated RMSE (₹20.1M) separately from the leaderboard number (₹2.89M) for exactly that reason.

So when the next Great Learning hackathon opened — **NBFC Loan Default**, 93,174 training loans, 39,933 test loans, metric: plain accuracy — I changed the experiment. No DontPanic harness. No agent volleyball. One Claude Code conversation as the only engineering partner, ChatGPT as an outside reviewer, and a promotion rule I committed to before the first upload: **nothing goes to the leaderboard unless it beats the frozen champion on paired out-of-fold predictions.**

This article is the result: another first place — and, for the first time, receipts.

**[TABLE T1_two_hackathons.png]**

## 1. The problem

Predict whether a client defaults on a loan. 22 predictors: loan terms, grades, income, credit-bureau counts, and — crucially — two columns that describe the loan *after* origination: `interest_receive` (cumulative interest collected) and `last_week_pay` (how long the borrower had been paying). 23.75% of training loans defaulted, so "predict nobody defaults" already scores 76.25%. The winner would be decided in the third decimal.

## 2. The discoveries that actually moved the score

Everything that mattered came from reading the data as a *loan servicing system* rather than a feature matrix.

- **The ID column is a clock.** Default rate falls from 78% in the oldest ID decile to 0.9% in the newest — IDs are issued sequentially, so ID is loan age in disguise. Older loans have simply had more time to default.
- **`last_week_pay` is whole months expressed in weeks** (4, 9, 13, 17, 22, 26…). That means you can reconstruct the exact amortisation schedule, invert it, and compute *how many installments the interest actually received corresponds to*. The gap between installments-implied-by-interest and months-elapsed — **installments missed** — turned out to be the single most powerful feature family (CV accuracy 0.9321 → 0.9412).
- **Loans have a due-date phase.** Whether a current borrower looks "0 months behind" or "1 month behind" depends on where their due date falls inside the issue-month cohort — a smooth time estimate blurs the two. Replacing the global clock with *local cohort windows over neighbouring IDs* (label-free, computed on train+test pooled) fixed roughly 900 false negatives and powered the final sprint from 0.9417 to 0.9438.

**[FIGURE F5_phase_discovery.png]**

An honest framing note that also went into the graded notebook: with post-origination payment columns, this task is **snapshot-status classification** ("is this loan, as observed today, in default?"), not origination-time credit scoring. An origination-features-only model manages 0.824 against the 0.763 majority baseline; the payment-behaviour reconstruction supplies essentially all the rest. The model also cannot extrapolate to unseen vintages (a cohort-holdout check drops to 0.75 on the oldest slice). Competition-valid, deployment-caveated.

## 3. The campaign: 18 submissions, three overtakes, one all-zeros upload

**[FIGURE F1_leaderboard_progression.png]**

Three moments deserve explanation:

- **The redraws (v6–v8).** Once rivals were within ~10 test rows, I knowingly bought lottery tickets: equal-skill blends whose predictions differed only on coin-flip rows. All three landed below my best. Lesson paid for and logged.
- **The probe (v9).** One submission of all zeros. Its score is, by definition, 1 minus the true test default rate — a single measurement that turned threshold calibration from guesswork into arithmetic. The calibrated flip of 90 marginal predictions had an expected value of +3.8 rows; it realised +6. (Disclosed then, disclosed now: this uses leaderboard feedback, so the public score is not a fully independent evaluation of the final model.)
- **The endgame.** Overtaken twice on the final day (by Wonder, then by Jay Solo Army's +41-row jump), the answer both times was a new *falsifiable feature hypothesis*, not a bigger search: off-by-one installment deviations, then the due-date phase features. Final margin: about 20 rows.

## 4. The part the commenter asked about: did validation predict reality?

This is the figure I wish I'd had for the first article.

**[FIGURE F2_cv_vs_leaderboard.png]**

Across all nine champions submitted over three days, the public leaderboard tracked out-of-fold cross-validation with a **constant ≈0.25 pp offset, never deviating more than 0.03 pp**. Every claimed gain generalised. That is what distinguishes "rapid search with gates" from "leaderboard overfitting made faster": the gate was a paired McNemar test plus bootstrap CI on identical folds, applied before every upload — and it rejected more ideas than it promoted.

**[TABLE T4_experiment_verdicts.png]**

Eight of fourteen major ideas died at that gate, including several that *felt* certain to work (model diversity; a "flip engine" for the hard segment — a split-half scan found literally zero subgroups where the model was wrong more often than right). The graveyard is the ablation study.

**[TABLE T2_submission_ledger.png]**

## 5. The money: what a "single-LLM" hackathon actually costs

In August, running a multi-agent harness on subscription seats, compute felt free — 622 submissions' worth of iteration at zero marginal cost per call. This time, running one metered conversation, I ran out of credits twice and felt every recharge. Was it *actually* more expensive? Here's what I can and cannot say.

What I can say, because Claude Code logs every API call's token counts:

**[TABLE T3_measured_cost.png]**

**[FIGURE F3_cost_anatomy.png]**

$186 at list-price equivalent. And the anatomy is the real finding: **the money didn't go to thinking — it went to re-reading.** Output tokens, including all extended reasoning, were 8% of cost. Seventy percent was cache writes: every time the session paused longer than the five-minute cache window (72 times), crashed, or restarted, the next call re-ingested a ~247k-token context from scratch. Eighteen percent of calls — the cold starts — caused 96% of cache-write spend.

**[FIGURE F4_cold_starts.png]**

What I cannot say: that this was more expensive than the harness. The August cost was never logged — it was dissolved into Copilot and Codex flat-rate seats. So the honest comparison is not "agentic vs single-LLM cost" but **metered visibility vs flat-rate opacity**. The single conversation felt more expensive because, for the first time, I could see the meter. The commenter asked for a fixed compute budget; I can't reconstruct one for August, but this article publishes the full meter for November, and the measurement script ships with the repo.

Two more disclosures for symmetry. "Single LLM" deserves quotes: the one conversation was relayed across three Claude models (Fable 5.1 → Fable 5 → Sonnet 5.5 — and Sonnet 5.5, the cheapest of the three, built the features that won the endgame). And a ChatGPT chat served as an unmetered outside reviewer, challenging leakage framing, demanding stricter promotion gates, and contributing one idea (repayment-signature matching) that ended up changing 7 test predictions.

## 6. What transferred from the volleyball playbook — and what I'd keep from each

The harness is gone; its *disciplines* survived as habits of one conversation:

- **The frozen champion** — every challenger compared against an immutable, checksummed best.
- **Promotion gates** — paired tests before any upload, replicated across seeds for feature changes.
- **Artifacts over chat memory** — 46 numbered experiment scripts, 31 commits, OOF arrays on disk. When the machine restarted mid-campaign (it did, several times), the state survived because the state was never the conversation.
- **The human at the decision line** — every upload, every "stop here", every lottery ticket was a human call.

What the single conversation added: coherence. One context holding three days of experiment history meant the off-by-one audit on day three could reference a rejection from day one without re-explanation. What it cost: those 247k tokens of context had to be re-read on every cold start — coherence, it turns out, is exactly what you pay for.

## 7. Lessons

1. **Answer critics with instrumentation, not adjectives.** The CV-vs-LB offset chart is one scatter plot, and it settles the overfitting question better than any paragraph.
2. **Domain reconstruction beats model search.** Six promoted ideas: five were loan-mechanics features, one was a measurement (the probe). Zero were "try a bigger model".
3. **The graveyard is the methodology.** Pre-registered gates that reject eight of fourteen ideas are what keep the leaderboard honest.
4. **In metered workflows, manage context like money.** Cost ≈ context size × cold starts. Shorter sessions, deliberate pauses, and compaction are budget levers, not conveniences.
5. **Measure before you claim.** I almost wrote "this time was more expensive." The log said: *this time was visible.* Those are different findings, and the second one is more useful.

---

*The full repo — 46 experiment scripts, the executed solution notebook, the frozen champion with checksums, and the cost-measurement script — is public: github.com/RoloMagnus/loan_default. Final result: 1st of 33, accuracy 0.94383, Nov 9 2026.*
