#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""F1 正确修法对照：引擎(逐对手平均) vs 随机手牌联合 vs **范围化联合**。

* 挂钩 `EquityAgent.handle` 读取**引擎自己的对手范围**（同一批 ranges）；
* 逐局面输出：引擎胜率 / 随机手牌联合 / 范围化联合 / 修前动作 / 修后动作（范围化）。
"""
from __future__ import annotations

import asyncio
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:                                              # noqa: BLE001
    pass

from poker_decision_ai import cli as cli_mod                    # noqa: E402
from poker_decision_ai import joint_equity as JE                # noqa: E402
from poker_decision_ai.multiway_equity import multiway_equity    # noqa: E402

JE.install_joint_hook()
POT = 10.0


def obs(hole, board, n_opp, frac, street="flop"):
    seats = [{"seat": i, "name": f"seat{i}", "stack": 100.0,
               "in_hand": i == 5 or i <= n_opp} for i in range(1, 7)]
    to_call = round(POT * frac, 4)
    return {"source": "v3f_snapshot", "street": street, "hero_seat": 5,
            "button_seat": 3, "hole_cards": list(hole), "board": list(board),
            "seats": seats, "pot": POT, "blinds": {"sb": 0.05, "bb": 0.15},
            "ante": 0.0, "to_call": to_call, "confidence": 0.96,
            "extra": {"v3f_hero_position": "BB", "v3f_bets": {"S1": to_call}}}


async def decide(runtime, o):
    req = {"kind": "cash_hand", "hero_id": "Hero", "vision_observation": o}
    ctx = runtime.new_context(cli_mod.new_id("jt"), "cli", allow_llm=False)
    rep = await runtime.runner.run(req, ctx)
    fd = rep.to_dict().get("final_decision") or {}
    eq = None
    for ev in (fd.get("key_evidence") or []):
        s = str(ev)
        if "hero_equity=" in s:
            try:
                eq = float(s.split("hero_equity=")[1].split()[0].rstrip(",;"))
            except (ValueError, IndexError):
                eq = None
            break
    return fd.get("action"), eq


async def main():
    runtime = cli_mod.build_runtime()
    rows = []
    for lab, hole in (("AA", ["As", "Ah"]), ("AKo", ["As", "Kd"]),
                       ("99", ["9c", "9d"]), ("Q8o", ["Qs", "8d"])):
        for n in (1, 3, 5):
            o = obs(hole, ["Kd", "7c", "2h"], n, 0.33)
            JE.USE_JOINT_AS_HERO_EQUITY["on"] = False
            JE.LAST_JOINT.clear()
            a_off, eq_off = await decide(runtime, o)
            j = dict(JE.LAST_JOINT)
            rnd = multiway_equity(hole, ["Kd", "7c", "2h"], n, samples=600)
            JE.USE_JOINT_AS_HERO_EQUITY["on"] = True
            a_on, eq_on = await decide(runtime, o)
            JE.USE_JOINT_AS_HERO_EQUITY["on"] = False
            rows.append({"hole": lab, "n_opp": n, "bet_frac": 0.33,
                          "eq_engine": eq_off, "eq_random_joint": rnd.get("equity"),
                          "eq_range_joint": j.get("equity"),
                          "joint_basis": j.get("basis"),
                          "action_off": a_off, "action_on": a_on,
                          "flipped": a_off != a_on})
    pathlib.Path(".tmp/f1_range_table.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    hdr = (f"{'手牌':<5}{'对手':>4}{'引擎(逐对手平均)':>16}{'随机手牌联合':>13}"
           f"{'范围化联合':>11}{'修前':>7}{'修后':>7}{'翻转':>6}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        f = lambda v: (f"{v:.3f}" if isinstance(v, (int, float)) else "  -  ")
        print(f"{r['hole']:<5}{r['n_opp']:>4}{f(r['eq_engine']):>16}"
              f"{f(r['eq_random_joint']):>13}{f(r['eq_range_joint']):>11}"
              f"{str(r['action_off']):>7}{str(r['action_on']):>7}"
              f"{('是' if r['flipped'] else ''):>6}")
    print("\n已存: .tmp/f1_range_table.json")


asyncio.run(main())
