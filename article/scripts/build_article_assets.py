"""Builds every figure and table-image for the Medium article (article/figures, article/tables).
All numbers are from measured sources: the leaderboard screenshots/receipts quoted in the session,
OOF accuracies printed by the experiment scripts, and article/data/*.json|csv from measure_session_cost.py."""
import os, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ART = os.path.dirname(HERE)
FIG = os.path.join(ART, "figures"); TAB = os.path.join(ART, "tables"); DATA = os.path.join(ART, "data")
os.makedirs(FIG, exist_ok=True); os.makedirs(TAB, exist_ok=True)

# dataviz palette (light mode)
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6"
SURFACE, INK, INK2 = "#fcfcfb", "#0b0b0b", "#52514e"
plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                     "text.color": INK, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.grid": True, "grid.color": "#e8e7e3", "grid.linewidth": 0.8,
                     "axes.spines.top": False, "axes.spines.right": False, "font.size": 10.5,
                     "axes.titlesize": 12.5, "axes.titleweight": "bold", "figure.dpi": 200})
SAVE = dict(bbox_inches="tight", pad_inches=0.25)

# ---------------- F1: leaderboard progression ----------------
subs = [  # (champion label, LB accuracy, note)
    ("v1", 0.9291063, "3-model blend"), ("v2", 0.9386973, "exact amortisation\nschedule features"),
    ("v3", 0.9406506, "retuned blend"), ("v4", 0.9412266, "10-fold + tuned CatBoost"),
    ("v5", 0.9412766, "5-seed farm"), ("v6", 0.9409261, "redraw"), ("v7", 0.9412516, "redraw"),
    ("v8", 0.9409761, "redraw"), ("v10", 0.9413768, "probe-calibrated\nthreshold (+6 rows)"),
    ("v11", 0.9413267, "redraw"), ("v12", 0.9416773, "off-by-one\nschedule features"),
    ("v13b", 0.9425538, "due-date phase\nfeatures"), ("v14a", 0.9435555, "+ cohort\ninterest gaps"),
    ("v14b", 0.9438309, "2nd CatBoost seed"),
]
rivals = [(1.6, 0.9383968, "Jay Solo Army 0.93840", (0, -13)), (5.6, 0.9415521, "Wonder 0.94155", (0, 5)),
          (8.6, 0.9423534, "Jay Solo Army 0.94235", (0, 5)), (9.6, 0.9431548, "Arun 0.94315", (0, 5))]
fig, ax = plt.subplots(figsize=(11, 5.8))
x = np.arange(len(subs)); y = [s[1] for s in subs]
best = np.maximum.accumulate(y)
ax.plot(x, best, color=BLUE, lw=2, zorder=3, label="RolusMagnus — best score so far")
ax.scatter(x, y, s=46, color=BLUE, zorder=4)
low = [i for i in range(len(y)) if y[i] < best[i] - 1e-9]
ax.scatter([x[i] for i in low], [y[i] for i in low], s=46, facecolor=SURFACE, edgecolor=BLUE, zorder=5,
           label="equal-skill redraws (kept the best score)")
for xr, yr, lab, off_xy in rivals:
    ax.hlines(yr, xr, len(subs) - 0.6, color=ORANGE, lw=1.6, ls=(0, (4, 3)), zorder=2)
    ax.annotate(lab, (xr + 0.05, yr), textcoords="offset points", xytext=off_xy, color=ORANGE, fontsize=8.7)
notes = {"v1": ("3-model blend", 14), "v2": ("exact amortisation\nschedule features", -34), "v3": ("retuned blend", -24),
         "v10": ("probe-calibrated\nthreshold (+6 rows)", -40), "v12": ("off-by-one\nschedule features", -56),
         "v13b": ("due-date phase\nfeatures", -62), "v14b": ("+ cohort gaps,\n2nd CatBoost seed", -58)}
for i, (lab, yy, note) in enumerate(subs):
    if lab in notes:
        txt, dy = notes[lab]
        ax.annotate(txt, (i, yy), textcoords="offset points", xytext=(0, dy), ha="center", fontsize=8,
                    color=INK2, arrowprops=dict(arrowstyle="-", color="#c9c8c3", lw=.8))
ax.set_xticks(x, [s[0] for s in subs]); ax.set_ylim(0.9272, 0.9462)
ax.set_xlabel("scored submissions, in order (v9 all-zeros probe omitted — accuracy 0.7598 by design)")
ax.set_ylabel("public leaderboard accuracy")
ax.set_title("Three days, 18 submissions: every jump is a named feature discovery, every flat stretch a rejected experiment")
ax.legend(loc="lower right", frameon=False, fontsize=9)
fig.savefig(os.path.join(FIG, "F1_leaderboard_progression.png"), **SAVE); plt.close(fig)

# ---------------- F2: CV vs LB (the answer to the overfitting critique) ----------------
pairs = [  # (label, OOF accuracy, LB accuracy)
    ("v2", 0.94124, 0.9386973), ("v3", 0.94308, 0.9406506), ("v4", 0.94401, 0.9412266),
    ("v5", 0.94389, 0.9412766), ("v10", 0.94401, 0.9413768), ("v12", 0.94415, 0.9416773),
    ("v13b", 0.94496, 0.9425538), ("v14a", 0.94618, 0.9435555), ("v14b", 0.94612, 0.9438309)]
fig, ax = plt.subplots(figsize=(7.4, 6.6))
ox = [p[1] for p in pairs]; oy = [p[2] for p in pairs]
lims = [0.9365, 0.9478]
ax.plot(lims, lims, color=INK2, lw=1, ls=":", label="y = x (no generalisation gap)")
off = np.mean([a - b for _, a, b in pairs])
ax.plot(lims, [v - off for v in lims], color=AQUA, lw=1.6, ls="--",
        label=f"y = x − {off*100:.2f} pp (fitted constant offset)")
ax.scatter(ox, oy, s=55, color=BLUE, zorder=5)
lab_off = {"v4": (7, -10), "v5": (-30, -2), "v10": (7, 3), "v14b": (-42, -4), "v14a": (7, -9)}
for lab, a, b in pairs:
    ax.annotate(lab, (a, b), textcoords="offset points", xytext=lab_off.get(lab, (7, -3)), fontsize=8.6, color=INK2)
ax.set_xlim(lims); ax.set_ylim(lims[0] - 0.003, lims[1] - 0.0015)
ax.set_xlabel("out-of-fold CV accuracy (5–10 fold, computed before submitting)")
ax.set_ylabel("public leaderboard accuracy")
resid = [abs((a - b) - off) for _, a, b in pairs]
ax.set_title("Validation honesty check: CV predicted every leaderboard score\n"
             f"offset ≈ {off*100:.2f} pp, max deviation {max(resid)*100:.2f} pp across 9 champions")
ax.legend(loc="upper left", frameon=False, fontsize=9)
fig.savefig(os.path.join(FIG, "F2_cv_vs_leaderboard.png"), **SAVE); plt.close(fig)

# ---------------- F3: where the money went ----------------
m = json.load(open(os.path.join(DATA, "session_effort_metrics.json")))
comp = m["cost_components_usd"]
fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.4), gridspec_kw={"wspace": 0.42})
ax = axes[0]
keys = ["cache_write", "cache_read", "output", "input_uncached"]
labels = ["cache writes\n(re-reading context\nafter every pause)", "cache reads", "output tokens\n(incl. thinking)", "uncached input"]
vals = [comp[k] for k in keys]
bars = ax.barh(labels[::-1], vals[::-1], color=[BLUE, AQUA, ORANGE, YELLOW][::-1], height=0.62)
for b, v in zip(bars, vals[::-1]):
    ax.annotate(f"${v:,.0f}", (b.get_width(), b.get_y() + b.get_height() / 2), xytext=(6, 0),
                textcoords="offset points", va="center", fontsize=10)
ax.set_title(f"Where the ${m['total_cost_usd_list']:,.0f} went (API list price)", loc="left")
ax.set_xlabel("USD, list-price equivalent"); ax.grid(axis="y", visible=False)
ax = axes[1]
bm = m["cost_by_model_usd"]
order = ["claude-fable-5-1", "claude-fable-5", "claude-sonnet-5-5"]
names = ["Fable 5.1\nNov 6–7 night\n(setup → v2)", "Fable 5\nNov 7–8\n(v3 → v12, the grind)", "Sonnet 5.5\nNov 8–9 night\n(phase features,\nv13–v14 endgame)"]
bars = ax.bar(names, [bm[o] for o in order], color=[VIOLET, BLUE, AQUA], width=0.6)
for b, o in zip(bars, order):
    ax.annotate(f"${bm[o]:,.0f}", (b.get_x() + b.get_width() / 2, b.get_height()), xytext=(0, 4),
                textcoords="offset points", ha="center", fontsize=10.5)
ax.set_title("Cost by model — three relay runners", loc="left")
ax.set_ylabel("USD, list-price equivalent"); ax.grid(axis="x", visible=False)
fig.savefig(os.path.join(FIG, "F3_cost_anatomy.png"), **SAVE); plt.close(fig)

# ---------------- F4: the cold-start mechanism ----------------
fig, ax = plt.subplots(figsize=(8.6, 4.2))
cats = [f"{m['cold_start_calls']} cold-start calls\n(first call after a pause /\ncrash / compaction)",
        f"{m['api_calls'] - m['cold_start_calls']} warm calls\n(cache still alive)"]
shares = [m["cold_start_share_of_cache_write_tokens"], 1 - m["cold_start_share_of_cache_write_tokens"]]
bars = ax.bar(cats, [s * 100 for s in shares], color=[ORANGE, BLUE], width=0.45)
for b, s in zip(bars, shares):
    ax.annotate(f"{s:.0%} of all cache-write tokens", (b.get_x() + b.get_width() / 2, b.get_height()),
                xytext=(0, 5), textcoords="offset points", ha="center", fontsize=11)
ax.set_ylabel("% of cache-write tokens (≈70% of total cost)")
ax.set_title(f"The cost driver nobody talks about: {m['cold_start_share_of_calls']:.0%} of calls caused "
             f"{m['cold_start_share_of_cache_write_tokens']:.0%} of cache writes ≈ ${m['cold_start_cost_usd_list']:,.0f}\n"
             f"median context per call: {m['context_tokens_per_call']['median']/1000:.0f}k tokens; "
             f"{m['gaps_over_5min']} pauses outlived the 5-minute cache")
ax.set_ylim(0, 108); ax.grid(axis="x", visible=False)
fig.savefig(os.path.join(FIG, "F4_cold_starts.png"), **SAVE); plt.close(fig)

# ---------------- F5: the decisive domain discovery (phase ladder) ----------------
fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.6), gridspec_kw={"wspace": 0.34})
ax = axes[0]
pos = ["0–20%", "20–40%", "40–60%", "60–80%", "80–100%"]
share = [0.132, 0.227, 0.419, 0.667, 0.865]
ax.bar(pos, [s * 100 for s in share], color=BLUE, width=0.6)
for i, s in enumerate(share): ax.annotate(f"{s:.0%}", (i, s * 100), xytext=(0, 4), textcoords="offset points", ha="center")
ax.set_xlabel("position inside issue-month cohort (by ID)")
ax.set_ylabel('% of CURRENT loans that look\n"1 month behind" to a smooth estimate')
ax.set_title("The due-date phase effect a smooth time\nestimate cannot see (train, non-defaulted loans)")
ax.grid(axis="x", visible=False)
ax = axes[1]
ks = ["≤ −6", "−5", "−4", "−3", "−2", "−1", "0", "+1"]
dr = [0.732, 0.694, 0.679, 0.671, 0.434, 0.081, 0.056, 0.362]
cols = [ORANGE if v > 0.4 else BLUE for v in dr]
ax.bar(ks, [v * 100 for v in dr], color=cols, width=0.62)
for i, v in enumerate(dr): ax.annotate(f"{v:.0%}", (i, v * 100), xytext=(0, 4), textcoords="offset points", ha="center", fontsize=9)
ax.set_xlabel("installments paid − local cohort installment count (±40 neighbours)")
ax.set_ylabel("default rate")
ax.set_title("After the fix: distance from your own cohort's\npayment count is a near-perfect risk ladder")
ax.grid(axis="x", visible=False)
fig.savefig(os.path.join(FIG, "F5_phase_discovery.png"), **SAVE); plt.close(fig)

# ---------------- table-image helper (manual renderer: wraps text, no clipping) ----------------
import textwrap
def table_image(fname, title, cols, rows, col_w, fontsize=9.3, row_h=0.34, align=None, highlight=None, note=None):
    # col_w are relative widths in inches; wrap text to column width
    CHAR_W = fontsize * 0.0072  # approx inches per character
    def wrap(txt, w_in):
        out = []
        for part in str(txt).split(chr(10)):
            out += textwrap.wrap(part, max(int(w_in / CHAR_W), 8)) or [""]
        return out
    wrapped = [[wrap(c, col_w[j] - 0.18) for j, c in enumerate(r)] for r in rows]
    hdr = [wrap(c, col_w[j] - 0.18) for j, c in enumerate(cols)]
    line_h = fontsize * 0.0155 + 0.055
    heights = [max(len(c) for c in hdr) * line_h + 0.12] + [max(len(c) for c in r) * line_h + 0.12 for r in wrapped]
    note_h = (note.count(chr(10)) + 1) * 0.16 + 0.15 if note else 0
    W = sum(col_w) + 0.3; H = 0.52 + sum(heights) + note_h + 0.12
    fig = plt.figure(figsize=(W, H)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
    ax.text(0.15, H - 0.18, title, fontsize=12.5, fontweight="bold", va="top")
    y = H - 0.52
    xs = [0.15]
    for w in col_w: xs.append(xs[-1] + w)
    for i, (h, cells) in enumerate(zip(heights, [hdr] + wrapped)):
        y0 = y - h
        if i == 0: bg = "#e8effb"
        elif highlight and (i - 1) in highlight: bg = "#e9f2ff"
        elif i % 2 == 0: bg = "#f4f3f0"
        else: bg = SURFACE
        ax.add_patch(plt.Rectangle((xs[0], y0), sum(col_w), h, facecolor=bg, edgecolor="#e0dfda", lw=0.7))
        for j, lines in enumerate(cells):
            ha = "right" if (align and j < len(align) and align[j] == "r" and i > 0) else "left"
            tx = xs[j + 1] - 0.12 if ha == "right" else xs[j] + 0.12
            ax.text(tx, y0 + h / 2, chr(10).join(lines), fontsize=fontsize, va="center", ha=ha,
                    fontweight="bold" if i == 0 else "normal", color=INK, linespacing=1.25)
        y = y0
    if note:
        ax.text(0.15, y - 0.14, note, fontsize=8.2, color=INK2, va="top")
    fig.savefig(os.path.join(TAB, fname), **SAVE); plt.close(fig)

# ---------------- T1: the two hackathons ----------------
table_image("T1_two_hackathons.png",
    "Two hackathons, two workflows — what is comparable and what is not",
    ["", "Restaurant Turnover (Aug 2026)", "NBFC Loan Default (Nov 2026)"],
    [["Result", "1st place (RMSE metric)", "1st place — 0.94383 accuracy"],
     ["Workflow", "DontPanic multi-agent harness:\nCopilot (Claude Sonnet) + OpenAI Codex\n'volleyball' roles, human decision gates", "One Claude Code conversation\n(Fable 5.1 → Fable 5 → Sonnet 5.5)\n+ ChatGPT as outside reviewer"],
     ["Submissions", "622", "18 (incl. 1 deliberate probe)"],
     ["Experiments", "~91 experiment artifacts, 71-feature pipeline", "46 numbered experiment scripts, 31 commits"],
     ["Human prompts", "not logged", "85 prompts over ~67 hours"],
     ["Final-phase strategy", "LB01–LB05 leaderboard-specific calibration\n(disclosed as competition-specific,\n₹12.1M → ₹2.89M public RMSE)", "OOF-gated promotion (McNemar + bootstrap);\nCV-to-LB offset stayed at ~0.25 pp\nacross all 9 champions"],
     ["Measured compute cost", "NOT measured (subscription seats;\nno per-token logs survive)", "$186 API list-price equivalent,\nmeasured from session logs"],
     ["Felt like", "cheap, fast, parallel", "slower, deliberate, visibly expensive"]],
    [1.25, 3.1, 3.1], fontsize=8.8, row_h=0.62,
    note="Fair-comparison caveats: different problems (regression vs classification), different metrics, different model generations,\n"
         "n = 1 each, and the first hackathon's cost was never logged — so 'more expensive' is a measured number on one side and a recollection on the other.")

# ---------------- T2: submission ledger ----------------
table_image("T2_submission_ledger.png",
    "Submission ledger — every upload, what changed, and whether validation predicted it",
    ["#", "Champion", "What changed", "OOF acc", "LB acc"],
    [["1", "v1", "LGB+Cat+XGB blend, engineered ratios + vintage features", "0.9321", "0.92911"],
     ["2", "v2", "Exact amortisation schedule: installments implied by interest received", "0.9412", "0.93870"],
     ["3", "v3", "Retuned LGB + CatBoost; weighted blend (McNemar-gated)", "0.9431", "0.94065"],
     ["4", "v4", "10-fold CV, multi-seed, Optuna-tuned CatBoost", "0.9440", "0.94123"],
     ["5", "v5", "5-seed farm (variance reduction)", "0.9439", "0.94128"],
     ["6–8", "v6–v8", "Equal-skill decorrelated redraws (all kept below best)", "≈0.9439", "0.94093–0.94125"],
     ["9", "v9", "All-zeros probe → measures true test default rate = 0.24015", "—", "0.75985 (by design)"],
     ["10", "v10", "Probe-calibrated threshold: flip 90 marginal positives (EV +3.8 rows)", "0.9440", "0.94138 (+6 rows)"],
     ["11", "v11", "Same flips on v5 base (EV ≈ +1 row)", "0.9439", "0.94133"],
     ["12", "v12", "Off-by-one schedule deviations (k−1 / k+1), 3-seed replicated", "0.9442", "0.94168"],
     ["13", "v13b", "Due-date phase features (local cohort windows)", "0.9450", "0.94255"],
     ["14", "v14a", "+ cohort-normalised interest-gap features", "0.9462", "0.94356"],
     ["15", "v14b", "Second CatBoost seed (equal-skill redraw)", "0.9461", "0.94383  🏆"]],
    [0.55, 0.85, 4.6, 0.9, 1.5], fontsize=8.8, row_h=0.36, highlight=[12],
    note="18 platform submissions in total; a few interim variants (e.g. v13a) are folded into their champion row. OOF = out-of-fold cross-validated accuracy, computed before each upload.")

# ---------------- T3: measured cost ----------------
t = json.load(open(os.path.join(DATA, "session_effort_metrics.json")))
bm = t["cost_by_model_usd"]
table_image("T3_measured_cost.png",
    "What one 'single-LLM' hackathon actually consumed (measured from the Claude Code session log)",
    ["Metric", "Value"],
    [["Wall-clock span (first → last API call)", "66.9 hours (Nov 6 night → Nov 9)"],
     ["Human prompts", "85"],
     ["API calls (assistant turns)", f"{t['api_calls']}"],
     ["Tool invocations (scripts, git, files)", f"{t['tool_calls']}"],
     ["Output tokens (incl. 197k thinking)", f"{t['tokens']['output']:,}"],
     ["Cache-read tokens", f"{t['tokens']['cache_read']:,} (≈69 full re-reads of a 1M-token context)"],
     ["Cache-write tokens", f"{t['tokens']['cache_write']:,}"],
     ["Median context per call", f"{t['context_tokens_per_call']['median']:,} tokens"],
     ["Cost — Fable 5.1 (setup → v2)", f"${bm['claude-fable-5-1']:.0f}"],
     ["Cost — Fable 5 (v3 → v12)", f"${bm['claude-fable-5']:.0f}"],
     ["Cost — Sonnet 5.5 (v13 → v14b)", f"${bm['claude-sonnet-5-5']:.0f}"],
     ["TOTAL (API list-price equivalent)", f"${t['total_cost_usd_list']:.0f}"]],
    [3.3, 3.6], fontsize=9.2, row_h=0.35, highlight=[11],
    note="List-price equivalent computed per call from logged token counts × published per-token prices (incl. 1.25× cache-write premium).\n"
         "Billed amounts can differ (credits/bundles). Excludes the ChatGPT reviewer chat, which was not metered.")

# ---------------- T4: experiment verdicts ----------------
table_image("T4_experiment_verdicts.png",
    "The graveyard matters: every idea met the same gate (paired McNemar on identical folds)",
    ["Idea", "Verdict", "Evidence"],
    [["Exact amortisation schedule reconstruction", "PROMOTED", "+0.9 pp CV; biggest single gain"],
     ["ID-as-vintage + elapsed-time estimate", "PROMOTED", "+2.2 pp over no-ID baseline"],
     ["Off-by-one installment deviations", "PROMOTED", "+29 rows, same sign on 3 seeds"],
     ["Due-date phase (local cohort windows)", "PROMOTED", "+62/+98/+37 rows on 3 seeds"],
     ["Cohort-normalised interest gaps", "PROMOTED", "+106/+84/+187 rows on 3 seeds"],
     ["All-zeros probe → threshold calibration", "PROMOTED", "EV +3.8 rows; realised +6"],
     ["Capping installments at contractual maturity", "REJECTED", "−44 rows: the overshoot was signal"],
     ["Stacking (logistic / LGB meta-learners, 3 variants)", "REJECTED", "net −29 to +8, p ≥ 0.13"],
     ["Pseudo-labelling confident test rows", "REJECTED", "−0.08 pp"],
     ["Cohort-stress neighbourhood features", "REJECTED", "net +3 rows, p = 0.95"],
     ["Segment threshold / residual corrector for old loans", "REJECTED", "net +20 / −6, not significant"],
     ["'Flip engine' scan (any subgroup >50% wrong?)", "REJECTED", "0 subgroups survived split-half test"],
     ["Model diversity (dart / extra-trees / sklearn HGB)", "REJECTED", "corr > 0.99 with blend; +0 rows"],
     ["Hill-climb ensemble over 7 saved models", "REJECTED", "re-found the same blend, −5 rows"]],
    [3.4, 1.0, 2.6], fontsize=8.9, row_h=0.345,
    note="6 promotions, 8 rejections. Rejections are what keep the CV-to-leaderboard offset constant — they are the ablations.")

print("figures:", sorted(os.listdir(FIG)))
print("tables:", sorted(os.listdir(TAB)))

# ---------------- F6: feature-ablation ladder ----------------
ladder = [  # (label, CV accuracy, kind)  kind: f=feature family, m=model/ensemble work, c=calibration
    ("raw columns,\nLightGBM", 0.8777, "m"),
    ("+ ratios\n& EMI", 0.9070, "f"),
    ("+ ID as\nloan-age clock", 0.9294, "f"),
    ("+ weeks since\nlast payment", 0.9305, "f"),
    ("+ installments\nmissed (exact\nschedule)", 0.9412, "f"),
    ("+ tuning, 10-fold\n3-model blend", 0.9440, "m"),
    ("+ off-by-one\ndeviations", 0.9442, "f"),
    ("+ due-date\nphase", 0.9450, "f"),
    ("+ cohort\ninterest gaps", 0.9462, "f"),
]
fig, ax = plt.subplots(figsize=(12.2, 5.4))
xx = np.arange(len(ladder)); yy = [l[1] for l in ladder]
cols = {"f": BLUE, "m": YELLOW, "c": AQUA}
ax.bar(xx, yy, color=[cols[l[2]] for l in ladder], width=0.62)
for i, v in enumerate(yy):
    d = f"+{(yy[i]-yy[i-1])*100:.2f}" if i else ""
    ax.annotate(f"{v:.4f}\n{d}", (i, v), xytext=(0, 5), textcoords="offset points", ha="center", fontsize=9)
ax.set_ylim(0.87, 0.952)
ax.set_xticks(xx, [l[0] for l in ladder], fontsize=8.4)
ax.tick_params(axis="x", pad=4)
ax.set_ylabel("5–10 fold out-of-fold accuracy")
import matplotlib.patches as mpatches
ax.legend(handles=[mpatches.Patch(color=BLUE, label="domain feature family (loan mechanics)"),
                   mpatches.Patch(color=YELLOW, label="model & ensemble work")],
          loc="upper left", frameon=False, fontsize=9.5)
ax.set_title("The ablation ladder: 6.8 pp of accuracy, and 6.0 pp of it came from reconstructing loan mechanics — not from bigger models")
ax.grid(axis="x", visible=False)
fig.savefig(os.path.join(FIG, "F6_feature_ladder.png"), **SAVE); plt.close(fig)
print("F6 written")
