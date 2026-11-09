"""Measure what this project actually consumed, from the Claude Code session log (per-call usage records).
Outputs (article/data): api_calls.csv (content-free, one row per API call), token_usage_by_model.csv,
token_usage_by_day_model.csv, session_effort_metrics.json.
Costs are API LIST-PRICE equivalents (USD per million tokens, Claude API reference cached 2026-10-06).
They are NOT necessarily what was billed to the account (credits, subscriptions and discounts differ)."""
import json, sys, os, statistics as st, datetime as dt
import pandas as pd
LOG, OUT = sys.argv[1], sys.argv[2]
UTC_OFFSET_H = -6   # local time (CST) for day bucketing
PRICE = {"claude-fable-5-1": (10, 50, 0.25), "claude-fable-5": (10, 50, 1.00), "claude-sonnet-5-5": (2, 10, 0.20)}  # input, output, cache read
calls, user_prompts = {}, 0
first = last = None
for line in open(LOG, encoding="utf8"):
    try: d = json.loads(line)
    except Exception: continue
    ts = d.get("timestamp")
    if ts:
        t = dt.datetime.fromisoformat(ts.replace("Z", "+00:00")); first = first or t; last = t
    m = d.get("message") if isinstance(d.get("message"), dict) else None
    if d.get("type") == "user" and m and not d.get("isMeta"):
        c = m.get("content")
        if isinstance(c, str) or (isinstance(c, list) and any(isinstance(b, dict) and b.get("type") == "text" for b in c)): user_prompts += 1
    if d.get("type") == "assistant" and m and m.get("usage") and m.get("model") not in (None, "<synthetic>"):
        mid = m.get("id") or d.get("uuid"); u = m["usage"]
        if mid not in calls or u.get("output_tokens", 0) >= calls[mid]["u"].get("output_tokens", 0):
            n_tool = sum(1 for b in (m.get("content") or []) if isinstance(b, dict) and b.get("type") == "tool_use")
            calls[mid] = {"u": u, "model": m["model"], "ts": ts, "tools": n_tool}
rows = []
for r in calls.values():
    u = r["u"]; cc = u.get("cache_creation") or {}
    w5 = cc.get("ephemeral_5m_input_tokens", 0) if cc else u.get("cache_creation_input_tokens", 0); w1 = cc.get("ephemeral_1h_input_tokens", 0) if cc else 0
    t = dt.datetime.fromisoformat(r["ts"].replace("Z", "+00:00")) + dt.timedelta(hours=UTC_OFFSET_H)
    rows.append(dict(ts_local=t.strftime("%Y-%m-%d %H:%M:%S"), day=t.strftime("%Y-%m-%d"), model=r["model"], input_tokens=u.get("input_tokens", 0),
                     output_tokens=u.get("output_tokens", 0), thinking_tokens=(u.get("output_tokens_details") or {}).get("thinking_tokens", 0),
                     cache_read=u.get("cache_read_input_tokens", 0), cache_write_5m=w5, cache_write_1h=w1, tool_calls=r["tools"]))
df = pd.DataFrame(rows).sort_values("ts_local").reset_index(drop=True)
df["context_tokens"] = df.input_tokens + df.cache_read + df.cache_write_5m + df.cache_write_1h
df["cold_start"] = (df.cache_read < 0.1 * df.context_tokens) & (df.context_tokens > 20000)
def parts(r):
    pi, po, pr = PRICE.get(r.model, (10, 50, 0.25))
    return pd.Series(dict(c_input=r.input_tokens * pi / 1e6, c_output=r.output_tokens * po / 1e6, c_read=r.cache_read * pr / 1e6,
                          c_write=(r.cache_write_5m * 1.25 + r.cache_write_1h * 2.0) * pi / 1e6))
df = pd.concat([df, df.apply(parts, axis=1)], axis=1); df["cost_usd_list"] = df[["c_input", "c_output", "c_read", "c_write"]].sum(axis=1)
df.round(5).to_csv(os.path.join(OUT, "api_calls.csv"), index=False)
g = df.groupby("model").agg(api_calls=("cost_usd_list", "size"), first_call=("ts_local", "min"), last_call=("ts_local", "max"), output_tokens=("output_tokens", "sum"),
                            thinking_tokens=("thinking_tokens", "sum"), cache_read_tokens=("cache_read", "sum"), cache_write_tokens=("cache_write_5m", "sum"),
                            cost_usd_list=("cost_usd_list", "sum")).round(2)
g.to_csv(os.path.join(OUT, "token_usage_by_model.csv"))
df.groupby(["day", "model"]).agg(api_calls=("cost_usd_list", "size"), cost_usd_list=("cost_usd_list", "sum")).round(2).to_csv(os.path.join(OUT, "token_usage_by_day_model.csv"))
ts = pd.to_datetime(df.ts_local); gaps = ts.diff().dt.total_seconds() / 60
cold = df[df.cold_start]
metrics = dict(
    as_of_local=str(dt.datetime.now().strftime("%Y-%m-%d %H:%M")), api_calls=len(df), user_prompts=user_prompts, tool_calls=int(df.tool_calls.sum()),
    span_hours_first_to_last_call=round((last - first).total_seconds() / 3600, 1), total_cost_usd_list=round(df.cost_usd_list.sum(), 2),
    cost_components_usd={"input_uncached": round(df.c_input.sum(), 2), "output": round(df.c_output.sum(), 2), "cache_read": round(df.c_read.sum(), 2), "cache_write": round(df.c_write.sum(), 2)},
    tokens=dict(output=int(df.output_tokens.sum()), thinking=int(df.thinking_tokens.sum()), cache_read=int(df.cache_read.sum()), cache_write=int(df.cache_write_5m.sum() + df.cache_write_1h.sum())),
    context_tokens_per_call=dict(median=int(df.context_tokens.median()), mean=int(df.context_tokens.mean()), max=int(df.context_tokens.max())),
    cold_start_calls=int(len(cold)), cold_start_share_of_calls=round(len(cold) / len(df), 3),
    cold_start_share_of_cache_write_tokens=round(cold.cache_write_5m.sum() / max(df.cache_write_5m.sum(), 1), 3),
    cold_start_cost_usd_list=round(cold.c_write.sum(), 2), gaps_over_5min=int((gaps > 5).sum()), gaps_over_30min=int((gaps > 30).sum()),
    cost_by_model_usd=g.cost_usd_list.to_dict())
json.dump(metrics, open(os.path.join(OUT, "session_effort_metrics.json"), "w"), indent=1)
print(g.to_string()); print(json.dumps(metrics, indent=1))
