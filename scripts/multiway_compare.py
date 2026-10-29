#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""对照实验：修前（逐对手平均）vs 修后（多人联合胜率）在同一批局面上的建议差异。

**只在本进程内临时替换 `hero_equity`/`outcome`**（不改任何文件、不改冻结层），
用于让用户看清修正的影响面后再决定是否落地。
"""
from __future__ import annotations

import asyncio
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:                                              # noqa: BLE001
    pass

from poker_decision_ai import cli as cli_mod                    # noqa: E402
from poker_decision_ai.multiway_equity import multiway_equity   # noqa: E402
from poker_decision_ai.agents.equity import agent as eq_agent   # noqa: E402

#: 本进程内的临时"修后"覆盖（None = 不覆盖 ⇒ 即修前行为）
OVERRIDE: dict = {"value": None, "win": None, "tie": None}
_orig_handle = eq_agent.EquityAgent.handle


async def _patched_handle(self, inp, ctx):
    out = await _orig_handle(self, inp, ctx)
    if OVERRIDE["value"] is not None:
        try:
            out.hero_equity = float(OVERRIDE["value"])
            if OVERRIDE["win"] is not None:
                out.outcome = {"win": float(OVERRIDE["win"]),
                               "tie": float(OVERRIDE["tie"]),
                               "lose": round(1.0 - float(OVERRIDE["win"])
                                              - float(OVERRIDE["tie"]), 6),
                               "samples": int(OVERRIDE.get("samples") or 0),
                               "seed": 0, "method": "multiway_joint_mc"}
        except Exception:                                      # noqa: BLE001
            pass
    return out


eq_agent.EquityAgent.handle = _patched_handle

HANDS = [("72o（最弱）", ["7c", "2h"]), ("Q8o（你那次）", ["Qs", "8d"]),
         ("99（中强）", ["9c", "9d"]), ("AKo（强）", ["As", "Kd"])]
OPPONENTS = (1, 3, 5)


def obs(hole, n_opp, to_call):
    """n_opp = 除 hero 外**参与**的对手数（其余座位判为已弃牌）。"""
    seats = []
    for i in range(1, 7):
        in_hand = (i == 5) or (i <= n_opp)
        seats.append({"seat": i, "name": f"seat{i}", "stack": 100.0,
                       "in_hand": in_hand})
    return {"source": "v3f_snapshot", "street": "preflop", "hero_seat": 5,
            "button_seat": 3, "hole_cards": hole, "board": [], "seats": seats,
            "pot": 0.4, "blinds": {"sb": 0.05, "bb": 0.15}, "ante": 0.0,
            "to_call": to_call, "confidence": 0.96,
            "extra": {"v3f_hero_position": "BB"}}


async def decide(runtime, o):
    req = {"kind": "cash_hand", "hero_id": "Hero", "vision_observation": o}
    ctx = runtime.new_context(cli_mod.new_id("cmp"), "cli", allow_llm=False)
    rep = await runtime.runner.run(req, ctx)
    d = rep.to_dict()
    fd = d.get("final_decision") or {}
    eq = None
    for ev in (fd.get("key_evidence") or []):
        s = str(ev)
        if "hero_equity=" in s:
            try:
                eq = float(s.split("hero_equity=")[1].split()[0].rstrip(",;"))
            except (ValueError, IndexError):
                eq = None
            break
    return fd.get("action"), eq, fd.get("reasoning") or ""


async def main():
    runtime = cli_mod.build_runtime()
    print("局面：翻前，hero 在 BB，面对 0.35 的加注（to_call=0.35）")
    print(f"{'手牌':<16}{'对手':>4} | {'引擎胜率':>8} {'多人估计':>8} | "
          f"{'修前动作':<10}{'修后动作':<10}")
    print("-" * 74)
    for label, hole in HANDS:
        for n_opp in OPPONENTS:
            o = obs(hole, n_opp, 0.35)
            OVERRIDE.update({"value": None, "win": None, "tie": None})
            a_before, eq_before, _ = await decide(runtime, o)
            est = multiway_equity(hole, [], n_opp, samples=1500)
            OVERRIDE.update({"value": est.get("equity"), "win": est.get("win"),
                              "tie": est.get("tie"),
                              "samples": est.get("samples")})
            a_after, eq_after, _ = await decide(runtime, o)
            OVERRIDE.update({"value": None, "win": None, "tie": None})
            print(f"{label:<16}{n_opp:>4} | {str(eq_before):>8} "
                  f"{str(est.get('equity')):>8} | {str(a_before):<10}"
                  f"{str(a_after):<10}")


asyncio.run(main())
