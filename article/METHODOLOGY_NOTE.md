# Can I honestly say "this time was more expensive"? — Measurement note

*Prepared for the Medium article. Everything here is reproducible from `article/scripts/measure_session_cost.py` and the session log.*

## 1. Yes, there is a way to measure it — and it has now been done for this hackathon

Claude Code writes a local session log (`~/.claude/projects/<project>/<session>.jsonl`) that records,
for every API call, the model and the exact token counts: uncached input, output (with thinking tokens),
cache reads, and cache writes. Multiplying those counts by published per-token prices gives a
**list-price-equivalent cost** for the whole project:

| | |
|---|---|
| Total, Nov 6 → Nov 9 | **$186** (list-price equivalent) |
| By model | Fable 5.1 $30 → Fable 5 $124 → Sonnet 5.5 $32 |
| By component | cache writes $131 (70%) · cache reads $41 (22%) · output $14 (8%) |
| Volume | 364 API calls, 85 human prompts, ~426k output tokens, 69M cache-read tokens |

Two caveats keep this honest:
- It is a **list-price equivalent**, not a billing statement. Credits and bundles price differently.
- It **excludes the ChatGPT reviewer chat** ("Hackathon Deadline Reminder"), which was heavily used
  and is not metered anywhere I can read.

## 2. No, the comparison with the Restaurant Turnover hackathon cannot be made rigorous — retroactively

The August workflow ran through Copilot and Codex **subscription seats** plus the DontPanic harness.
No per-token logs survive (I checked: the local Codex folder holds two unrelated sessions). So the honest
statement is asymmetric:

> *"This hackathon cost a measured $186 in list-price tokens. The previous one cost an unmeasured amount
> hidden inside flat-rate subscriptions — it **felt** cheap because the marginal cost of one more agent
> call was zero at the point of use."*

That is still a publishable and interesting claim — but it is a claim about **cost visibility and pricing
models** (metered API vs flat-rate seats), not a measured claim that single-LLM > multi-agent in cost.

## 3. What the measurement actually shows (more interesting than "it was expensive")

- **The money did not go to thinking. It went to re-reading.** 70% of cost was cache writes; 18% of calls
  (the first call after a pause, crash, or restart) caused **96%** of cache-write tokens, ≈ $123. Median
  context per call was 247k tokens. The cost of this workflow is `context size × number of cold starts`,
  not reasoning volume.
- **Three models, one baton.** The "single LLM" framing needs a footnote: the conversation was relayed
  across Claude Fable 5.1 → Fable 5 → Sonnet 5.5 (and Sonnet 5.5, the cheapest, built the features that
  won). Plus ChatGPT as an unmetered outside reviewer whose pasted notes (~13 long advisory messages)
  shaped the experiment queue and contributed the repayment-signature idea.
- **622 → 18 submissions is the real workflow story.** The August workflow searched fast and finished
  with disclosed leaderboard-specific calibration. The November workflow promoted only paired-test
  winners, and the CV-to-leaderboard offset stayed at ~0.25 pp (max deviation 0.03 pp) across all nine
  champions — the measurable signature that gains were generalisation, not leaderboard fitting.

## 4. Recommended framing for the article (fair to both workflows)

1. Lead with the commenter's critique; concede what was true of hackathon 1's final phase (the previous
   article itself disclosed LB01–LB05 as competition-specific).
2. Present hackathon 2 as the deliberate opposite: fewer submissions, pre-registered gates, constant
   CV→LB offset (Figure F2 is the proof).
3. On cost: give the measured $186 with its anatomy, state plainly that the August cost was not logged,
   and frame the contrast as *metered visibility vs flat-rate opacity* — "it felt more expensive because
   for the first time I could see the meter."
4. Credit all four models (three Claude + ChatGPT). Disclose the probe/calibration submissions explicitly.
5. Avoid: "agentic was cheaper/more expensive than single-LLM" as a general law; n=1 vs n=1, different
   problems, metrics, and model generations.

## 5. If you want a rigorous comparison next time (pre-registration checklist)

- Fix a compute budget in tokens or dollars, logged per call, before starting.
- Keep one log per workflow arm (harness vs single conversation), same model generation on both arms.
- Register the promotion gate (paired test, threshold) before the first submission.
- Track: submissions, experiments run, CV→LB offset stability, wall-clock, human prompts.
- Then — and only then — "X was cheaper per leaderboard point" is a measurement, not a feeling.
