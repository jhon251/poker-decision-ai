"""P30-9-BD：**范围化联合胜率**——F1 的正确修法。

问题（审计 F1，已证）：`agents/equity/agent.py` 的 `hero_equity` 是
「逐个对手单挑胜率的平均」，而多人底池必须**同时击败所有对手**。
实测 Q8o/5 人：引擎 0.45 vs 真值 ≈0.17（随机手牌）。

为什么不能直接用"随机手牌多人估计"替换（已证）：随机手牌**强于**对手的
**下注范围** ⇒ 单挑里会偏高（实测 Q8o 单挑 3/4 底池由 fold 变 call）。

本模块：用**引擎自己的对手范围**做**联合**模拟（同一批 ranges，联合采样、互不撞牌），
从而既修好多人聚合、又不改变单挑所依据的范围强度。
"""
from __future__ import annotations

import random
from typing import Any

JOINT_VERSION = "p30-9-bd-joint-range-equity-v1"
DEFAULT_JOINT_SAMPLES = 900

#: 最近一次联合估计（供诚实标注引用；不参与决策，除非显式开启）
LAST_JOINT: dict = {}
#: 是否把**范围化联合胜率**写入 hero_equity。
#: P30-9-BF（用户授权）：**默认开启**——它是 F1（逐对手平均）的正确修法。
#: 可用环境变量 `JOINT_EQUITY_MODE=off` 回退到引擎原行为。
import os as _os

_DEFAULT_ON = str(_os.environ.get("JOINT_EQUITY_MODE", "on")).lower() != "off"
USE_JOINT_AS_HERO_EQUITY: dict = {"on": _DEFAULT_ON}
_HOOKED = False


def _eval7(cards) -> int:
    from .domain.poker.evaluator import evaluate7
    return evaluate7(cards)


def joint_equity_range_aware(hero_hole: list[str], board: list[str],
                             villain_dists: list[dict | None], *,
                             samples: int = DEFAULT_JOINT_SAMPLES,
                             seed: int = 42) -> dict:
    """hero 对 N 个对手（各自**范围**）的联合胜率（同时击败所有人）。

    `villain_dists[i]` = Range Agent 的 `distribution`（{cat: weight}）；
    None/空 ⇒ 该对手退化为随机手牌（并如实标注）。
    """
    from .agents.equity.agent import _range_from_distribution
    from .domain.poker.cards import Card, parse_cards
    from .domain.poker.ranges import parse_range, sample_concrete_pair
    from .domain.poker.evaluator import evaluate7

    if len(hero_hole) != 2 or not villain_dists:
        return {"ok": False, "reason": "bad_input"}
    hero_cards = parse_cards([str(c) for c in hero_hole])
    board_cards = parse_cards([str(c) for c in board])
    ranges = []
    n_random = 0
    for d in villain_dists:
        if isinstance(d, dict) and d:
            r = _range_from_distribution(d)
            if not r.is_empty():
                ranges.append(r)
                continue
        ranges.append(parse_range("random:1.0", name="random"))
        n_random += 1
    rng = random.Random(seed)
    win = tie = used = 0
    n = max(1, int(samples))
    for _ in range(n):
        dead = list(hero_cards) + list(board_cards)
        opps = []
        ok = True
        for r in ranges:
            combo = r.sample_non_colliding(rng, dead)
            if combo is None:
                ok = False
                break
            pair = sample_concrete_pair(rng, combo[0], combo[1], combo[2], dead)
            if not pair:
                ok = False
                break
            opps.append(pair)
            dead = dead + pair
        if not ok:
            continue
        need = 5 - len(board_cards)
        if need < 0:
            return {"ok": False, "reason": "board_too_long"}
        dead_codes = {c.code for c in dead}
        pool = [c for c in range(52) if c not in dead_codes]
        rng.shuffle(pool)
        full_board = list(board_cards) + [Card(c) for c in pool[:need]]
        mine = evaluate7(hero_cards + full_board)
        best = max(evaluate7(o + full_board) for o in opps)
        used += 1
        if mine > best:
            win += 1
        elif mine == best:
            tie += 1
    if used == 0:
        return {"ok": False, "reason": "no_valid_sample"}
    eq = (win + tie / 2) / used
    return {"ok": True, "equity": round(eq, 4), "samples": used,
            "n_villains": len(ranges), "win": round(win / used, 6),
            "tie": round(tie / used, 6),
            "basis": ("ranges" if n_random == 0 else
                       ("ranges+random" if n_random < len(ranges) else "random")),
            "n_random_villains": n_random, "version": JOINT_VERSION,
            "note": "范围化联合胜率（同时击败所有对手；使用引擎自身范围）"}


def _extract_ranges(inp: dict) -> list[dict | None]:
    state = (inp or {}).get("state") or {}
    range_in = (inp or {}).get("range") or {}
    vmap = range_in.get("villains") or {}
    players = state.get("players") or []
    out: list[dict | None] = []
    for p in players:
        if not isinstance(p, dict) or p.get("is_hero"):
            continue
        if p.get("status") not in ("active", "allin"):
            continue
        pid = p.get("id") or p.get("player_id") or p.get("seat")
        vinfo = vmap.get(pid) or {}
        out.append(vinfo.get("distribution") if isinstance(vinfo, dict) else None)
    return out


def install_joint_hook() -> bool:
    """幂等挂钩：每次权益计算后，用**同一批范围**算一次联合胜率。"""
    global _HOOKED
    if _HOOKED:
        return True
    try:
        from .agents.equity.agent import EquityAgent
    except Exception:                                          # noqa: BLE001
        return False
    orig = EquityAgent.handle

    async def _patched(self, inp, ctx):
        out = await orig(self, inp, ctx)
        try:
            state = (inp or {}).get("state") or {}
            hero_p = next((p for p in (state.get("players") or [])
                            if isinstance(p, dict) and p.get("is_hero")), None)
            hole = [str(c) for c in ((hero_p or {}).get("hole_cards") or [])]
            board = [str(c) for c in (state.get("board") or [])]
            dists = _extract_ranges(inp)
            if len(hole) == 2 and dists:
                j = joint_equity_range_aware(hole, board, dists)
                if j.get("ok"):
                    LAST_JOINT.clear()
                    LAST_JOINT.update(j)
                    if USE_JOINT_AS_HERO_EQUITY.get("on"):
                        out.hero_equity = float(j["equity"])
                        out.outcome = {"win": float(j["win"]), "tie": float(j["tie"]),
                                        "lose": round(1.0 - float(j["win"])
                                                       - float(j["tie"]), 6),
                                        "samples": int(j["samples"]), "seed": 0,
                                        "method": "joint_range_mc"}
        except Exception:                                      # noqa: BLE001
            pass
        return out

    EquityAgent.handle = _patched
    _HOOKED = True
    return True
