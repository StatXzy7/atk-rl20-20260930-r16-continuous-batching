"""把 SimulationResult 渲染成人类可读的事件时间线（供核对/审阅）。"""

from __future__ import annotations

from .engine import SimulationResult


def render_timeline(res: SimulationResult) -> str:
    lines = [
        f"# 事件时间线（策略={res.policy}）",
        "# 警告：以下均为模拟器*预测*时间线，非真实硬件 trace。",
        f"# 配置: {res.cfg}",
        "",
        "== Step 时间线 ==",
    ]
    for st in res.steps:
        acts = []
        for a in st.actions:
            if a["kind"] == "prefill":
                tag = "重算" if a.get("replay") else "prefill"
                extra = (" (fresh%d+replay%d)" % (a["fresh"], a["replay"])
                         if a.get("replay") else "")
                acts.append(
                    f"{a['rid']}:{tag}{a['tokens']}{extra}"
                    f"[页{a['pages']}]")
            else:
                acts.append(
                    f"{a['rid']}:decode->#{a['generated']}[页{a['pages']}]"
                    f"{'*完成*' if a.get('env_finished') else ''}")
        lines.append(
            f"step{st.index:>3} t={st.start:>7.2f}->{st.end:>7.2f} "
            f"dur={st.duration:>6.2f} P={st.prefill_tokens} "
            f"D={st.decode_tokens} 页:{st.used_pages_before}"
            f"->{st.used_pages_after}->(释放后){st.used_pages_final} | "
            + "; ".join(acts))
    lines.append("")
    lines.append("== 离散事件 ==")
    for e in res.events:
        if e["event"] in ("frag_snapshot", "prefill", "decode"):
            continue
        lines.append(str(e))
    lines.append("")
    lines.append("== 每请求指标 ==")
    for m in res.requests:
        lines.append(
            f"{m.rid} [{m.status}] TTFT={m.ttft} "
            f"完成延迟={m.completion_latency} ITL={m.itls} "
            f"抢占={m.preemptions} 重算token={m.recompute_tokens} "
            f"decode={m.decode_tokens} 备注: {m.itl_note}")
    if res.deadlock:
        lines.append("")
        lines.append(f"== 死锁/活锁（t={res.makespan}）: {res.deadlock}")
    return "\n".join(lines)
