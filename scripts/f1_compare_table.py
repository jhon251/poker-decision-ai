#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""F1 对照表：**修前（逐对手平均）vs 修后（多人联合胜率）** 的建议差异。

只在本进程内临时替换 `hero_equity`/`outcome`（**不改任何文件**），用于让你看清影响面。
输出：手牌 × 对手数 × 下注大小 ⇒ 胜率(修前/多人估计) + 动作(修前/修后) + 是否翻转。
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
from poker_decision_ai.multiway_equity import multiway_equity   # noqa: E402
from poker_decision_ai.agents.equity import agent as eq_agent   # noqa: E402

OVERRIDE: dict = {"value": None, "win": None, "tie": None, "samples": 0}
_orig_handle = eq_agent.EquityAgent.handle


async def _patched(self, inp, ctx):
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


eq_agent.EquityAgent.handle = _patched

POT = 10.0
BB = 0.15


def obs(hole, board, n_opp, to_call, street="flop"):
    seats = [{"seat": i, "name": f"seat{i}", "stack": 100.0,
               "in_hand": i == 5 or i <= n_opp} for i in range(1, 7)]
    bets = {"S1": to_call} if to_call else {}
    return {"source": "v3f_snapshot", "street": street, "hero_seat": 5,
            "button_seat": 3, "hole_cards": list(hole), "board": list(board),
            "seats": seats, "pot": POT, "blinds": {"sb": 0.05, "bb": BB},
            "ante": 0.0, "to_call": to_call, "confidence": 0.96,
            "extra": {"v3f_hero_position": "BB", "v3f_bets": bets}}


async def decide(runtime, o):
    req = {"kind": "cash_hand", "hero_id": "Hero", "vision_observation": o}
    ctx = runtime.new_context(cli_mod.new_id("cmp"), "cli", allow_llm=False)
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


async def one(runtime, hole, board, n_opp, frac, street="flop"):
    to_call = round(POT * frac, 4)
    o = obs(hole, board, n_opp, to_call, street)
    OVERRIDE.update({"value": None, "win": None, "tie": None})
    a_off, eq_off = await decide(runtime, o)
    est = multiway_equity(hole, board, n_opp, samples=1200)
    OVERRIDE.update({"value": est.get("equity"), "win": est.get("win"),
                      "tie": est.get("tie"), "samples": est.get("samples")})
    a_on, eq_on = await decide(runtime, o)
    OVERRIDE.update({"value": None, "win": None, "tie": None})
    return {"hole": "".join(hole), "board": "".join(board[:3]), "n_opp": n_opp,
            "bet_frac": frac, "eq_engine": eq_off, "eq_multiway": est.get("equity"),
            "action_off": a_off, "action_on": a_on, "flipped": a_off != a_on,
            "equity_used_on": eq_on}


async def main():
    runtime = cli_mod.build_runtime()
    rows = []
    # 主表：干燥牌面（未成对），各手牌 × 人数 × 1/3 底池
    for lab, hole in (("AA", ["As", "Ah"]), ("KK", ["Ks", "Kh"]),
                       ("AKo", ["As", "Kd"]), ("99", ["9c", "9d"]),
                       ("Q8o", ["Qs", "8d"]), ("72o", ["7c", "2h"])):
        for n in (1, 3, 5):
            rows.append(await one(runtime, hole, ["Kd", "7c", "2h"], n, 0.33))
    # 用户局面：公对 + Q-high，两种下注
    for frac in (0.33, 0.75):
        rows.append(await one(runtime, ["Qs", "8d"], ["Kd", "7c", "7h"], 1, frac))
        rows.append(await one(runtime, ["Qs", "8d"], ["Kd", "7c", "7h"], 3, frac))

    out = pathlib.Path(".tmp/f1_compare_table.json")
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")

    hdr = (f"{'手牌':<6}{'牌面':<8}{'对手':>4}{'下注':>6}"
           f"{'引擎胜率':>10}{'多人估计':>10}{'修前':>8}{'修后':>8}{'翻转':>6}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['hole']:<6}{r['board']:<8}{r['n_opp']:>4}"
              f"{r['bet_frac']:>6.2f}"
              f"{(r['eq_engine'] if r['eq_engine'] is not None else float('nan')):>10.3f}"
              f"{(r['eq_multiway'] if r['eq_multiway'] is not None else float('nan')):>10.3f}"
              f"{str(r['action_off']):>8}{str(r['action_on']):>8}"
              f"{('是' if r['flipped'] else ''):>6}")
    print("\n已存:", out)


asyncio.run(main())
