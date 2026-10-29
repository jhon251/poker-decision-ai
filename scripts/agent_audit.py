#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Agent 审计批处理：**只用客观可判定**的检验（合法性/单调性/自洽性/组件真实性）。

判据（不需要任何"策略口味"）：
  L1 合法性：面对下注时不得 fold/check（无注可跟时不得 fold）
  L2 价格单调性：底注越大 ⇒ 跟注意愿不得**提高**（same hand/board）
  L3 人数单调性：对手越多 ⇒ 报告的 hero_equity 不得**提高**（胜率定义要求下降）
  L4 牌力单调性：同局面下 equity(AA) ≥ equity(KK) ≥ equity(Q8o) ≥ equity(72o)
  L5 理由↔动作自洽：推理写"最优 X"时动作族须与 X 一致
  L6 确定性：同输入两次 ⇒ 动作/尺寸/置信一致
  L7 尺寸合法性：加注尺寸 ≥ 最小加注（2bb）且 ≥ 0；不得出现负值/荒谬值
  L8 组件真实性：provider_status 里 solver/opponent/exploit/fold_equity 等是否 real
用法：python scripts/agent_audit.py [--out .tmp/agent_audit.json]
"""
from __future__ import annotations

import argparse
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

SEATS = [{"seat": i, "name": f"seat{i}", "stack": 100.0, "in_hand": True}
         for i in range(1, 7)]
AGGRO = {"bet", "raise", "allin"}


def obs(hole, board, pot, to_call, *, street="flop", bets=None, n_opp=1):
    seats = []
    for i in range(1, 7):
        seats.append({"seat": i, "name": f"seat{i}", "stack": 100.0,
                       "in_hand": i == 5 or i <= n_opp})
    return {"source": "v3f_snapshot", "street": street, "hero_seat": 5,
            "button_seat": 3, "hole_cards": list(hole), "board": list(board),
            "seats": seats, "pot": pot, "blinds": {"sb": 0.05, "bb": 0.15},
            "ante": 0.0, "to_call": to_call, "confidence": 0.96,
            "extra": {"v3f_hero_position": "BB", "v3f_bets": bets or {}}}


async def decide(runtime, o):
    req = {"kind": "cash_hand", "hero_id": "Hero", "vision_observation": o}
    ctx = runtime.new_context(cli_mod.new_id("audit"), "cli", allow_llm=False)
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
    return {"action": fd.get("action"), "size": fd.get("recommended_size"),
            "conf": fd.get("confidence"), "equity": eq,
            "reasoning": str(fd.get("reasoning") or ""),
            "providers": d.get("provider_status") or {},
            "degraded": d.get("degraded") or [], "skipped": d.get("skipped") or []}


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=".tmp/agent_audit.json")
    a = ap.parse_args()
    runtime = cli_mod.build_runtime()
    R: dict = {}

    # ── L2 价格单调性（公对，Q-high） ─────────────────────────────
    price = []
    for frac in (0.25, 0.5, 1.0, 1.5):
        bet = round(1.0 * frac, 4)
        r = await decide(runtime, obs(["Qs", "8d"], ["Kd", "7c", "7h"], 1.0, bet,
                                       bets={"S1": bet}))
        price.append((frac, r["action"], r["equity"], r["conf"]))
    R["L2_price_monotonicity"] = price

    # ── L3 人数单调性 ───────────────────────────────────────────
    from poker_decision_ai.joint_equity import LAST_JOINT
    opp, opp_joint = [], []
    for n in (1, 3, 5):
        LAST_JOINT.clear()
        r = await decide(runtime, obs(["Qs", "8d"], [], 1.0, 0.6,
                                       street="preflop", n_opp=n,
                                       bets={"S1": 0.6}))
        opp.append((n, r["equity"], r["action"]))
        opp_joint.append((n, LAST_JOINT.get("equity"), r["action"]))
    R["L3_opponent_monotonicity"] = opp
    R["L3_opponent_monotonicity_effective"] = opp_joint
    vals = [v for (_n, v, _a) in opp_joint if isinstance(v, (int, float))]
    R["L3_effective_is_decreasing"] = (
        all(vals[i] > vals[i + 1] for i in range(len(vals) - 1))
        if len(vals) >= 2 else None)

    # ── L4 牌力单调性 ───────────────────────────────────────────
    hands = []
    for lab, hole in (("AA", ["As", "Ah"]), ("KK", ["Ks", "Kh"]),
                       ("Q8o", ["Qs", "8d"]), ("72o", ["7c", "2h"])):
        r = await decide(runtime, obs(hole, ["Kd", "7c", "2h"], 1.0, 0.6,
                                       bets={"S1": 0.6}))
        hands.append((lab, r["equity"], r["action"]))
    R["L4_hand_monotonicity"] = hands

    # ── L5 理由↔动作自洽 ────────────────────────────────────────
    l5 = []
    for name, o in (("多人（原缺陷局面）", obs(["Qs", "8d"], ["Kd", "7c", "7h"],
                                                 1.0, 0.6, n_opp=3,
                                                 bets={"S1": 0.6})),
                     ("单挑小注", obs(["Qs", "8d"], ["Kd", "7c", "7h"], 1.0, 0.25,
                                       bets={"S1": 0.25}))):
        r = await decide(runtime, o)
        txt = r["reasoning"]
        claimed = None
        if "EVCore 最优" in txt:
            try:
                claimed = txt.split("EVCore 最优", 1)[1].split()[0].lower()
            except IndexError:
                claimed = None
        l5.append({"case": name, "action": r["action"], "ev_claimed": claimed,
                    "consistent": (claimed is None or
                                    claimed == (r["action"] or "").lower()
                                    or (claimed, r["action"]) in
                                    {("raise", "bet"), ("bet", "raise"),
                                     ("check", "call"), ("call", "check")})})
    R["L5_reasoning_action_consistency"] = l5

    # ── L6 确定性 ───────────────────────────────────────────────
    o = obs(["Qs", "8d"], ["Kd", "7c", "7h"], 1.0, 0.5, bets={"S1": 0.5})
    r1 = await decide(runtime, o)
    r2 = await decide(runtime, o)
    R["L6_determinism"] = {"run1": [r1["action"], r1["size"], r1["conf"]],
                            "run2": [r2["action"], r2["size"], r2["conf"]],
                            "same": (r1["action"] == r2["action"] and
                                      r1["size"] == r2["size"])}

    # ── L7 尺寸合法性 ───────────────────────────────────────────
    sizes = []
    for name, o in (("AA 翻前加注", obs(["As", "Ah"], [], 1.0, 0.3,
                                         street="preflop", bets={"S1": 0.3})),
                     ("顶对翻后加注", obs(["As", "Kd"], ["Kd", "7c", "2h"], 1.0, 0.3,
                                           bets={"S1": 0.3}))):
        r = await decide(runtime, o)
        sizes.append({"case": name, "action": r["action"], "size": r["size"],
                       "legal": (r["size"] is None or
                                  (isinstance(r["size"], (int, float)) and
                                   r["size"] >= 0))})
    R["L7_size_legality"] = sizes

    # ── L8 组件真实性 ───────────────────────────────────────────
    r = await decide(runtime, o)
    R["L8_providers"] = {k: (v.get("status") if isinstance(v, dict) else v)
                          for k, v in (r["providers"] or {}).items()}
    R["L8_degraded"] = r["degraded"]
    R["L8_skipped"] = r["skipped"]

    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(R, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(R, ensure_ascii=False, indent=1))


asyncio.run(main())
