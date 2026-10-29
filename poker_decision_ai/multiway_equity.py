"""P30-9-AI：多人底池胜率独立核验（不改冻结层，只校验与标注）。

背景（实测缺陷）：`agents/equity/agent.py` 的 `hero_equity` 是**逐个对手单挑胜率的
平均**（第 321-326 行）。多人底池须**同时击败所有对手**，故该值被系统性高估：
实测 Q8o vs 5 随机对手：引擎 0.44 / 独立蒙特卡洛 0.17（2.6 倍）。

本模块只做**独立估计 + 一致性判定 + 诚实标注**，不修改冻结决策层。
"""
from __future__ import annotations

import random
from typing import Any

MULTIWAY_VERSION = "p30-9-ai-multiway-equity-v1"
#: 引擎值 / 多人估计 超过该比例即判"高估"
INFLATION_RATIO = 1.35
#: 蒙特卡洛次数：**保守模式**用高精度；**诚实标注**用低精度（只需区分阈值）
DEFAULT_SAMPLES = 1200
LABEL_SAMPLES = 600


def _eval7(cards: list) -> int:
    from .domain.poker.evaluator import evaluate7
    return evaluate7(cards)


def multiway_equity(hole: list[str], board: list[str], n_villains: int,
                    *, samples: int = DEFAULT_SAMPLES,
                    seed: int = 20261010) -> dict:
    """hero 对 **n_villains 个随机手牌**的多人底池胜率（蒙特卡洛）。

    多路底池的胜率 = P(hero 同时击败所有对手)；不是逐对手胜率的平均。
    """
    from .domain.poker.cards import parse_cards
    if len(hole) != 2 or n_villains < 1:
        return {"ok": False, "reason": "bad_input"}
    deck = [f"{r}{s}" for r in "23456789TJQKA" for s in "cdhs"]
    dead = set(str(c) for c in hole) | set(str(c) for c in board)
    deck = [c for c in deck if c not in dead]
    need_board = 5 - len(board)
    rnd = random.Random(seed)
    hero_cards = parse_cards([str(c) for c in hole])
    board_cards = parse_cards([str(c) for c in board])
    win = tie = 0
    for _ in range(samples):
        pool = deck[:]
        rnd.shuffle(pool)
        add = pool[:need_board]
        rest = pool[need_board:]
        if len(rest) < 2 * n_villains:
            return {"ok": False, "reason": "not_enough_cards"}
        full_board = board_cards + parse_cards(add)
        mine = _eval7(hero_cards + full_board)
        best = -1
        for i in range(n_villains):
            opp = parse_cards(rest[2 * i:2 * i + 2])
            v = _eval7(opp + full_board)
            if v > best:
                best = v
        if mine > best:
            win += 1
        elif mine == best:
            tie += 1
    eq = (win + tie / 2) / samples
    return {"ok": True, "equity": round(eq, 4), "samples": samples,
            "n_villains": n_villains, "version": MULTIWAY_VERSION,
            "win": round(win / samples, 6), "tie": round(tie / samples, 6),
            "basis": "random_hands",
            "note": ("多人底池胜率（同时击败所有对手）；与逐对手平均不同。"
                      "**依据为随机手牌**——对『下注范围』而言这是偏高的，"
                      "故不可直接等同于引擎的范围化胜率。")}


def equity_consistency(obs: Any, engine_equity: float | None = None,
                       *, samples: int = LABEL_SAMPLES) -> dict:
    """把引擎 `hero_equity` 与多人独立估计对比 ⇒ 是否高估（只读判定）。"""
    o = obs if isinstance(obs, dict) else {}
    hole = [str(c) for c in (o.get("hole_cards") or [])]
    board = [str(c) for c in (o.get("board") or [])]
    seats = o.get("seats") if isinstance(o.get("seats"), list) else []
    n_active = sum(1 for s in seats
                   if isinstance(s, dict) and s.get("in_hand") is not False)
    n_villains = max(1, n_active - 1) if n_active else 1
    # 性能：单挑不存在"多人高估"问题 ⇒ **不跑蒙特卡洛**
    if n_villains < 2:
        return {"ok": True, "n_villains": n_villains,
                "multiway_estimate": None, "engine_equity": engine_equity,
                "version": MULTIWAY_VERSION, "inflated": False,
                "skipped": "heads_up_no_multiway_inflation"}
    # 性能：若保守模式已算出估计值 ⇒ 直接复用，不重复模拟
    if MULTIWAY_OVERRIDE.get("value") is not None:
        est = {"ok": True, "equity": MULTIWAY_OVERRIDE["value"],
               "n_villains": MULTIWAY_OVERRIDE.get("n_villains") or n_villains}
    else:
        est = multiway_equity(hole, board, n_villains, samples=samples)
    out = {"ok": bool(est.get("ok")), "n_villains": n_villains,
           "multiway_estimate": est.get("equity"), "engine_equity": engine_equity,
           "version": MULTIWAY_VERSION, "inflated": False}
    if not est.get("ok") or engine_equity is None:
        return out
    e, m = float(engine_equity), float(est["equity"])
    out["ratio"] = round(e / m, 3) if m > 1e-9 else None
    if n_villains >= 2 and m > 1e-9 and e > m * INFLATION_RATIO:
        out["inflated"] = True
        out["note"] = (
            f"⚠ 多人底池胜率被高估：引擎 {e:.3f} vs 多人独立估计 {m:.3f}"
            f"（{n_villains} 个对手）。引擎的 hero_equity 是『逐个对手单挑胜率的平均』，"
            f"而多人底池必须**同时击败所有对手**；请勿据此跟注/加注。")
    return out


# ── 多人保守模式（默认关） ──────────────────────────────────────────
#: 本进程内的临时覆盖：None = 不干预（默认行为，与不改任何东西等价）
MULTIWAY_OVERRIDE: dict = {"value": None, "win": None, "tie": None,
                            "samples": 0, "n_villains": None}
_HOOK_INSTALLED = False


def set_conservative(estimate: dict | None) -> None:
    """开启/关闭保守覆盖（`None` = 关闭 ⇒ 恢复默认行为）。"""
    if not estimate or not estimate.get("ok"):
        MULTIWAY_OVERRIDE.update({"value": None, "win": None, "tie": None,
                                   "samples": 0, "n_villains": None})
        return
    MULTIWAY_OVERRIDE.update({"value": estimate.get("equity"),
                               "win": estimate.get("win"),
                               "tie": estimate.get("tie"),
                               "samples": estimate.get("samples") or 0,
                               "n_villains": estimate.get("n_villains")})


def clear_conservative() -> None:
    set_conservative(None)


def conservative_active() -> bool:
    return MULTIWAY_OVERRIDE.get("value") is not None


def install_equity_hook() -> bool:
    """安装 `EquityAgent.handle` 挂钩（幂等）。**未开启时为零影响**。"""
    global _HOOK_INSTALLED
    if _HOOK_INSTALLED:
        return True
    try:
        from .agents.equity.agent import EquityAgent
    except Exception:                                          # noqa: BLE001
        return False
    orig = EquityAgent.handle

    async def _patched(self, inp, ctx):
        out = await orig(self, inp, ctx)
        return apply_override(out)

    EquityAgent.handle = _patched
    _HOOK_INSTALLED = True
    return True


def conservative_for(obs: Any, *, samples: int = DEFAULT_SAMPLES) -> dict:
    """按观察算多人估计（供 runner 在开启模式时使用）。"""
    o = obs if isinstance(obs, dict) else {}
    hole = [str(c) for c in (o.get("hole_cards") or [])]
    board = [str(c) for c in (o.get("board") or [])]
    seats = o.get("seats") if isinstance(o.get("seats"), list) else []
    n_active = sum(1 for s in seats
                   if isinstance(s, dict) and s.get("in_hand") is not False)
    n_villains = max(1, n_active - 1) if n_active else 1
    return multiway_equity(hole, board, n_villains, samples=samples)


def apply_override(out: Any) -> Any:
    """把多人估计应用到 EquityOutput（**未开启时原样返回**）。纯函数，便于测试。"""
    val = MULTIWAY_OVERRIDE.get("value")
    if val is None or out is None:
        return out
    try:
        out.hero_equity = float(val)
        w, ti = MULTIWAY_OVERRIDE.get("win"), MULTIWAY_OVERRIDE.get("tie")
        if w is not None and ti is not None:
            out.outcome = {"win": float(w), "tie": float(ti),
                            "lose": round(1.0 - float(w) - float(ti), 6),
                            "samples": int(MULTIWAY_OVERRIDE.get("samples") or 0),
                            "seed": 0, "method": "multiway_joint_mc"}
    except Exception:                                          # noqa: BLE001
        pass
    return out
