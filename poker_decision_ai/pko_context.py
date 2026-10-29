"""P67 — PKO/KO/Mystery Deep Strategy Context（bounty 因素层；确定性；mode 门控）。

只使用 state 既有真实上下文（players[].bounty、tournament.bounty_*、
stack、payouts）与冻结的 domain/bounty.py 公式；缺失 → UNKNOWN，
Mystery 未揭示金额不得伪造。

因素（全部经 P63 Strategy Fusion 独立来源接入，source=pko_context）：
  effective_bounty  有效赏金（bb）：KO=全额；PKO=growth_rate×victim；
                    Mystery 仅已揭示金额（未揭示 → UNKNOWN，不伪造）
  cover_relation    hero 对最大赏金对手的覆盖关系（covered/covered_by/
                    neither——淘汰可能性的前提）
  bounty_pressure   赏金筹码相对有效栈强度（复用冻结 bounty_pressure）
  icm_x_bounty      报告性：ICM k± 与 bounty utility 的交互已在 P44
                    objective 表达（淘汰型候选）→ 显式不调整（防双计）
  late_stage_bounty 剩余人数少 + 赏金/栈高 → bounty hunting 倾斜
                    （声明式；仅 bounty 上下文齐备时）

双计守卫：bounty chip EV 已由 P44 knockout utility（淘汰型候选）表达；
本层不重复计算 knockout EV——倾斜只来自"赏金压力/晚段猎赏"这一对手
行为侧因素。mode 门控：仅 resolved_mode ∈ {pko, ko, mystery}。
"""
from __future__ import annotations

from .domain.poker.bounty import (
    BountyInfo,
    bounty_pressure,
)
from .strategy_fusion import make_entry

MODULE = "pko_context"
LATE_STAGE_MAX_PLAYERS = 9
LATE_STAGE_PRESSURE = 0.25          # bounty_pressure 阈值（声明式）
CONFIDENCE = 0.6                    # 单因素声明置信（与 P66 同族）


def _bounty_chips_bb(p: dict, bb: float, growth_rate: float,
                     btype: str) -> float | None:
    """玩家 bounty（chips）→ bb 口径；PKO 按 growth_rate 折算有效赏金。"""
    b = p.get("bounty")
    if b is None:
        return None
    try:
        chips = float(b)
    except (TypeError, ValueError):
        return None
    if chips <= 0 or bb <= 0:
        return None
    if btype == "pko":
        chips = chips * growth_rate      # 有效赏金 = growth_rate 份额
    return chips / bb


def resolve_pko_context(state: dict) -> dict:
    """state → PKO 因素块（确定性；UNKNOWN-first；不伪造 Mystery 金额）。"""
    from .domain.poker.models import positions_for
    players = [p for p in (state.get("players") or [])
               if isinstance(p, dict)]
    n = len(players)
    bb = state.get("bb")
    info = BountyInfo.from_state(state)
    btype = info.bounty_type or ""
    growth = info.growth_rate

    factors: dict[str, dict] = {}

    # ---- effective_bounty（逐 active 玩家的有效赏金 bb） ----
    if not bb or bb <= 0:
        factors["effective_bounty"] = {
            "status": "UNKNOWN", "reason": "bb missing/invalid"}
    else:
        vals = []
        for p in players:
            if p.get("is_hero") or p.get("status") not in ("active", "allin"):
                continue
            cb = _bounty_chips_bb(p, float(bb), growth, btype)
            if cb is not None:
                vals.append({"player_id": str(p.get("player_id") or ""),
                             "bounty_bb": round(cb, 4)})
        if info.bounty_type == "mystery" and not vals:
            # Mystery 且无已揭示金额 → UNKNOWN（不伪造揭示值）
            factors["effective_bounty"] = {
                "status": "UNKNOWN",
                "reason": "mystery bounty unrevealed — amounts not "
                          "fabricated"}
        elif not vals:
            factors["effective_bounty"] = {
                "status": "UNKNOWN",
                "reason": "no active opponent bounty present"}
        else:
            factors["effective_bounty"] = {
                "status": "usable", "per_opponent": vals,
                "max_bounty_bb": round(max(v["bounty_bb"] for v in vals), 4),
                "bounty_type": info.bounty_type or "unknown",
                "note": ("mystery: revealed amounts only" if btype ==
                         "mystery" else "")}

    # ---- cover_relation（hero vs 最大赏金对手） ----
    hero = next((p for p in players if p.get("is_hero")), None)
    hero_stack = (hero or {}).get("stack")
    max_opp = None
    for p in players:
        if p.get("is_hero") or p.get("status") not in ("active", "allin"):
            continue
        b = p.get("bounty")
        if b is None:
            continue
        if max_opp is None or float(b) > float(max_opp.get("bounty") or 0):
            max_opp = p
    if hero_stack is None or max_opp is None:
        factors["cover_relation"] = {
            "status": "UNKNOWN", "reason": "hero stack or bounty opponent "
                                           "missing"}
    else:
        hs, os_ = float(hero_stack), float(max_opp.get("stack") or 0.0)
        rel = ("covered" if hs >= os_ else
               "covered_by" if hs < os_ else "neither")
        factors["cover_relation"] = {
            "status": "usable", "relation": rel,
            "max_bounty_opponent": str(max_opp.get("player_id") or ""),
            "hero_stack": round(hs, 4),
            "opponent_stack": round(os_, 4),
            "note": ("knockout probability > 0 only when hero covers "
                     "the opponent (bounty.py contract)")}

    # ---- bounty_pressure（复用冻结公式；最大赏金 bb × 有效栈） ----
    eff_bb = None
    if bb and hero_stack:
        eff_bb = float(hero_stack) / float(bb)
    max_bb = factors.get("effective_bounty", {}).get("max_bounty_bb")
    if eff_bb and max_bb:
        pressure = bounty_pressure(max_bb, eff_bb)
        factors["bounty_pressure"] = {
            "status": "usable", "pressure": round(pressure, 6),
            "provenance": "frozen bounty_pressure(bounty.py)"}
    else:
        factors["bounty_pressure"] = {
            "status": "UNKNOWN",
            "reason": "effective bounty or stack missing"}

    # ---- icm_x_bounty：报告性（防双计——P44 已表达） ----
    factors["icm_x_bounty"] = {
        "status": "context_only",
        "expressed_by": ("P44 knockout utility on elimination-type "
                         "candidates (bounty_util_bb)"),
        "reason": "no separate adjustment (anti-double-count)"}

    # ---- late_stage_bounty_pressure ----
    remaining = (state.get("tournament") or {}).get("players_remaining") \
        if isinstance(state.get("tournament"), dict) else None
    late = (remaining is not None and int(remaining)
            <= LATE_STAGE_MAX_PLAYERS
            and factors.get("bounty_pressure", {}).get("status")
            == "usable"
            and float(factors["bounty_pressure"].get("pressure") or 0)
            >= LATE_STAGE_PRESSURE)
    factors["late_stage_bounty_pressure"] = {
        "status": ("usable" if late else "UNKNOWN"),
        "reason": ("" if late else
                   "requires late stage (≤9 remaining) AND bounty "
                   "pressure ≥ " + str(LATE_STAGE_PRESSURE)),
        "players_remaining": remaining,
        "threshold_players": LATE_STAGE_MAX_PLAYERS,
        "threshold_pressure": LATE_STAGE_PRESSURE}
    return {"module": MODULE, "bounty_type": info.bounty_type or "unknown",
            "growth_rate": growth, "is_bounty": info.is_bounty,
            "bounty_mode": info.bounty_type if info.bounty_type in
            ("pko", "ko", "mystery") else "unknown",
            "factors": factors,
            "positions_contract": str(positions_for) is not None}


def pko_context_to_entries(context: dict) -> list[dict]:
    """因素块 → P63 ledger entries（source=pko_context；仅 usable 且
    上下文齐备的因素；Mystery 未揭示不产生金额倾斜）。"""
    factors = (context or {}).get("factors") or {}
    entries: list[dict] = []

    eff = factors.get("effective_bounty") or {}
    bp = factors.get("bounty_pressure") or {}
    cover = factors.get("cover_relation") or {}
    late = factors.get("late_stage_bounty_pressure") or {}

    # bounty hunting 倾斜需要：有效赏金 usable + 压力 usable
    if not (eff.get("status") == "usable" and bp.get("status") == "usable"):
        return entries
    pressure = float(bp.get("pressure") or 0.0)
    max_bb = float(eff.get("max_bounty_bb") or 0.0)

    # 淘汰激励：赏金压力高 → bet/raise +（猎赏），幅度 ∝ pressure
    hunt = pressure * CONFIDENCE
    entries.append(make_entry(MODULE, "bet", hunt, CONFIDENCE,
                              ["effective_bounty", "bounty_pressure"],
                              provenance=(f"bounty pressure "
                                          f"{pressure:.3f} "
                                          f"(max {max_bb:.2f}bb)")))
    entries.append(make_entry(MODULE, "raise", hunt, CONFIDENCE,
                              ["effective_bounty", "bounty_pressure"],
                              provenance=(f"bounty pressure "
                                          f"{pressure:.3f} "
                                          f"(max {max_bb:.2f}bb)")))
    # covered_by（hero 被大赏金对手覆盖）→ 对手猎赏动机 → 保守 −
    if cover.get("status") == "usable" and \
            cover.get("relation") == "covered_by":
        entries.append(make_entry(MODULE, "bet", -0.15, CONFIDENCE,
                                  ["cover_relation"],
                                  provenance="hero covered_by bounty "
                                             "leader — cautious"))
    # 晚段猎赏：剩余少 + 赏金压力 → call/悬牌追淘汰的既有 A9 之外
    # 的额外倾斜（声明式；仅 late usable）
    if late.get("status") == "usable":
        entries.append(make_entry(MODULE, "bet", 0.1, CONFIDENCE,
                                  ["late_stage_bounty_pressure"],
                                  provenance="late-stage bounty hunting"))
    return entries


def mode_application_gate(resolved_mode: str) -> dict:
    """mode 门控：仅 pko/ko/mystery 应用；cash/mtt 显式拒绝。"""
    if resolved_mode in ("pko", "ko", "mystery"):
        return {"applied": True, "reason": ""}
    return {"applied": False,
            "reason": (f"pko_context requires bounty mode (pko/ko/"
                       f"mystery), got {resolved_mode!r} "
                       "(mode switch isolation)")}
