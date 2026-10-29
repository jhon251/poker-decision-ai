#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P30-9-GJ：真实会话覆盖复测（**口径 = 最终决策的锚定来源**）。

上一版按"查表命中"分类，看不到新增的**现场真解**升级 ⇒ 低估改进 ✗。本版按最终决策归类：
  * `incomplete_board`：公共牌不全（第 ② 项会阻断；此处如实标注）
  * `chart`：翻前图表锚定
  * `live_solve`：两人池 turn/river **现场真解**（本目标新增能力 ✓）
  * `anchored_library`：锚定在库内解（精确命中 ✓）
  * `approx`：仅近似先验/启发式（未锚定 ✓）
另记录每类延迟（实时预算 1200ms 参照）。

输出：`docs/coverage_real_sessions.json`（覆盖写）
"""
from __future__ import annotations

import asyncio
import collections
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = pathlib.Path(__file__).resolve().parents[1]
LOG = ROOT / "dist" / "logs"
NEED = {"preflop": 0, "flop": 3, "turn": 4, "river": 5}


def decisions() -> list[dict]:
    seen: dict[str, dict] = {}
    for f in sorted(LOG.glob("events-*.jsonl")):
        for line in f.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.strip():
                continue
            try:
                d = json.loads(line)
            except Exception:                                    # noqa: BLE001
                continue
            if d.get("event") != "agent_invocation" or d.get("agent") != "vision":
                continue
            vo = ((d.get("input") or {}).get("request") or {}).get("vision_observation")
            if not vo or not vo.get("street"):
                continue
            seen[str(d.get("request_id") or len(seen))] = vo
    return list(seen.values())


def classify(report_text: str, obs: dict) -> str:
    street = str(obs.get("street") or "preflop")
    got = len([c for c in (obs.get("board") or []) if c])
    if got < NEED.get(street, 0):
        return "incomplete_board"
    if "mode=live_solve" in report_text or "现场真解" in report_text:
        return "live_solve"
    if "求解器锚定" in report_text:
        return "anchored_library"
    if "图表" in report_text and "锚定" in report_text:
        return "chart"
    if "图表基线" in report_text or "preflop_chart" in report_text:
        return "chart"
    return "approx"


async def main() -> None:
    from poker_decision_ai import cli as cli_mod
    from poker_decision_ai.realtime import RealtimeRunner

    obs_list = decisions()
    rt = cli_mod.build_runtime()
    rr = RealtimeRunner(rt)
    rows = []
    for vo in obs_list:
        o = dict(vo)
        o.setdefault("source", "log_replay")
        o.setdefault("confidence", 0.95)
        try:
            rep = await rr.decide({"kind": "cash_hand", "hero_id": "Hero",
                                    "vision_observation": o, "session_id": "cov"})
            dd = rep.to_dict() if hasattr(rep, "to_dict") else dict(rep)
            txt = json.dumps(dd, ensure_ascii=False)
            fd = dd.get("final_decision") or {}
            action = str(fd.get("action") or "")
            ms = dd.get("latency_ms") or dd.get("elapsed_ms")
        except Exception as e:                                    # noqa: BLE001
            txt, action, ms = f"ERROR:{type(e).__name__}", "", None
        n_in = len([s for s in (vo.get("seats") or []) if s.get("in_hand")])
        rows.append({"street": vo.get("street"), "n_in_hand": n_in,
                      "board": "".join(vo.get("board") or []),
                      "kind": classify(txt, vo), "action": action, "ms": ms})

    kinds = collections.Counter(r["kind"] for r in rows)
    print(f"真实决策 {len(rows)} | 归类: {dict(kinds)}")
    post = [r for r in rows if r["street"] in ("flop", "turn", "river")]
    anchored = sum(1 for r in post
                    if r["kind"] in ("chart", "anchored_library", "live_solve"))
    print(f"\n翻后 {len(post)}：真解资源占比 {anchored}/{len(post)} = "
          f"{anchored/max(len(post),1)*100:.0f}%（图表/库精确/现场真解）")
    for st_ in ("flop", "turn", "river"):
        sub = [r for r in post if r["street"] == st_]
        if sub:
            c = collections.Counter(r["kind"] for r in sub)
            a = sum(v for k, v in c.items() if k in ("chart", "anchored_library", "live_solve"))
            print(f"  {st_}: {len(sub)} | 真解 {a}/{len(sub)} | {dict(c)}")
    for label, sel in (("两人池", lambda r: r["n_in_hand"] == 2),
                        ("多人≥3", lambda r: r["n_in_hand"] >= 3)):
        sub = [r for r in post if sel(r)]
        if sub:
            c = collections.Counter(r["kind"] for r in sub)
            a = sum(v for k, v in c.items() if k in ("chart", "anchored_library", "live_solve"))
            print(f"  {label}: {len(sub)} | 真解 {a}/{len(sub)} = {a/len(sub)*100:.0f}% | {dict(c)}")
    lat = [r["ms"] for r in rows if isinstance(r["ms"], (int, float))]
    if lat:
        lat.sort()
        print(f"\n延迟中位 {lat[len(lat)//2]:.0f} ms | p90 {lat[int(len(lat)*0.9)]:.0f} ms")

    (ROOT / "docs" / "coverage_real_sessions.json").write_text(
        json.dumps({"n": len(rows), "kinds": dict(kinds), "rows": rows},
                    ensure_ascii=False, indent=1), encoding="utf-8")
    print("已写 docs/coverage_real_sessions.json")


if __name__ == "__main__":
    asyncio.run(main())
