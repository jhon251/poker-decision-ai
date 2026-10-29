#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P30-9-EL：**真实帧均衡知情率**探针（产品路径 CashGameRunner.analyze）。

对每张真实帧报告：
  * 街/动作/尺寸；求解器来源（图表/spot/6-max/mock）与是否近似；
  * 查表命中模式（HU spot / 6-max 库）；
  * 判定类别：图表锚定 / 求解器锚定(spot) / 6-max 锚定 / 求解器知情偏离 / 启发式。
汇总"**均衡知情率**"= 由真实解（图表 GTO / HU spot / 6-max 库）参与决策的比例。
"""
from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from poker_decision_ai.agents.solver.agent import SolverAgent          # noqa: E402
from poker_decision_ai.core.providers import spot_table_6max as st6    # noqa: E402
from poker_decision_ai.vision2_platform.png_decode import decode_png   # noqa: E402
from cash_game_ui.runner import CashGameRunner                         # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
FRAMES = [
    (ROOT / "dist/logs/frames/20261010-112328-DECISION_UNAVAILABLE.png", {"sb": 0.01, "bb": 0.02}),
    (ROOT / "dist/logs/frames/20261010-113101-DECISION_UNAVAILABLE.png", {"sb": 0.01, "bb": 0.02}),
    (ROOT / "dist/logs/frames/20261010-114807-DECISION_UNAVAILABLE.png", {"sb": 0.01, "bb": 0.02}),
    (ROOT / "dist/logs/frames/20261010-114822-DECISION_UNAVAILABLE.png", {"sb": 0.01, "bb": 0.02}),
    (ROOT / "data/vision2/coinpoker_evidence/user_2250_preflop_q6s.png", {"sb": 0.02, "bb": 0.05}),
    (ROOT / "data/vision2/coinpoker_evidence/user_2300_preflop_kts.png", {"sb": 0.05, "bb": 0.15}),
    (ROOT / "data/vision2/coinpoker_evidence/8bf82f134651db0add626be0304a2ff1.png",
     {"sb": 1000.0, "bb": 2000.0}),
]

SIX: list[dict] = []
_orig6 = st6.SpotTable6max.lookup


def _lk6(self, q):
    h = _orig6(self, q)
    SIX.append({"actor": q.get("actor"), "node": q.get("node"),
                 "pot": q.get("pot_bb"), "ratio": q.get("bet_ratio"),
                 "hit": (h or {}).get("model"), "approx": (h or {}).get("is_approximate")})
    return h


st6.SpotTable6max.lookup = _lk6

SOL: list[dict] = []
_orig_run = SolverAgent.run


async def _run(self, inp, ctx):
    env = await _orig_run(self, inp, ctx)
    o = getattr(env, "output", env)
    o = o.to_dict() if hasattr(o, "to_dict") else o
    SOL.append({"solver_name": o.get("solver_name"), "source": o.get("source"),
                 "approx": o.get("is_approximate"),
                 "spot": ((o.get("source_detail") or {}).get("spot_id")),
                 "warn": [w.get("code") for w in (o.get("warnings") or [])]})
    return env


SolverAgent.run = _run

# P30-9-HQ：**帧回放**不是真实对局 ⇒ 证据事件须标来源 ✓
#   （此前未标 ⇒ 回放产生的事件被记成 `runtime` ✗ ⇒ "真实恢复率"分母被污染 ✗）
import os as _os
_os.environ.setdefault("DSH_EVIDENCE_SOURCE", "frames_replay")

run = CashGameRunner()
rows = []
for path, blinds in FRAMES:
    if not path.exists():
        continue
    SIX.clear()
    SOL.clear()
    w, h, rgba = decode_png(path.read_bytes())
    res = run.analyze(rgba, w, h, allow_recapture=False,
                       blinds_override={**blinds, "source": "probe"})
    d = res.to_dict() if hasattr(res, "to_dict") else {}
    reason = str(res.reasoning or "")
    sol = SOL[-1] if SOL else {}
    six = [s for s in SIX if s.get("hit")]
    cls = ("图表锚定" if ("图表基线" in reason or "图表锚定" in reason)
            else "求解器锚定" if "求解器锚定" in reason
            else "6-max 锚定" if ("spot_6max" in reason or (six and "6max" in str(six[-1]["hit"])))
            else "启发式" if "启发式" in reason else "未标注")
    informed = bool(sol and str(sol.get("source") or "") in
                     ("preflop_chart", "postflop_spot") and not sol.get("approx")) or bool(six)
    rows.append({"frame": path.name[:30], "street": d.get("street") or res.status,
                  "action": d.get("decision_action"), "solver_name": sol.get("solver_name"),
                  "source": sol.get("source"), "approx": sol.get("approx"),
                  "six": six, "cls": cls, "informed": informed, "reason": reason[:160]})
    print(f"{rows[-1]['frame']:<31} {str(rows[-1]['street']):<7} 动作={str(d.get('decision_action')):<5} "
          f"solver={str(sol.get('source')):<16} approx={str(sol.get('approx')):<5} → {cls}")
    if six:
        print(f"      6-max: {six[-1]['hit']} approx={six[-1]['approx']} actor={six[-1]['actor']}")
    if sol.get("warn"):
        print(f"      warn: {sol['warn']}")

n = len(rows)
inf = sum(1 for r in rows if r["informed"])
print(f"\n均衡知情率（真实解参与）: {inf}/{n} = {inf/max(n,1)*100:.0f}%")
for k in ("图表锚定", "求解器锚定", "6-max 锚定", "启发式", "未标注"):
    c = sum(1 for r in rows if r["cls"] == k)
    if c:
        print(f"  {k}: {c}/{n} = {c/max(n,1)*100:.0f}%")
(ROOT / "docs" / "coverage_real_frames.json").write_text(
    json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
print("已写 docs/coverage_real_frames.json")
