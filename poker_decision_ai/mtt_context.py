"""P66 — MTT Deep Strategy Context（街机上下文因素层；确定性；mode 门控）。

只使用 state 既有真实上下文（state.tournament + players 栈/盲注），
缺失 → 该因素 UNKNOWN/fallback；绝不伪造 tournament context。

因素（全部经 P63 Strategy Fusion 作为独立来源接入，cap=FUSION_CAP_BB）：
  stack_pressure   M-ratio（栈 / 每圈成本）驱动的短码攻守倾斜：
                   M ≤ SHORT_STACK_M → bet/raise +、call −（推/弃域）；
                   M ≥ BIG_STACK_M → 轻微 value bet +（声明式）。
  bubble_pressure  players_remaining − itm_players ∈ [0, NEAR_BUBBLE_GAP]
                   且 hero 非 short → 对手 ICM 压力 → bet/raise +
                   （hero 自身 k± 不表达对手风险厌恶——非双计）。
  final_table      报告性上下文（≤9 人）：其风险厌恶已由 hero k± 表达 →
                   显式不调整（防与 ICM 双计）。
  payout_sensitivity 报告性：已由 chip_ev+icm objective 表达（P44 k±）→
                   显式不调整（防与 ICM 双计）。

mode 门控：仅 resolved_mode == "mtt" 应用；CASH/PKO 携带本上下文 →
not_applied + reason（不改变其行为）。

UNKNOWN-first：bb/sb 缺失 → stack_pressure UNKNOWN；itm_players 或
players_remaining 缺失 → bubble UNKNOWN。aggression/exploit 类倾斜不
在本层（P62 层职责）。
"""
from __future__ import annotations

SHORT_STACK_M = 8.0
BIG_STACK_M = 20.0
NEAR_BUBBLE_GAP = 2
MGG = "mtt_context"
FINAL_TABLE_SIZE = 9


def resolve_mtt_context(state: dict) -> dict:
    """state.tournament + 栈/盲注 → MTT 因素块（确定性；UNKNOWN-first）。"""
    t = state.get("tournament") if isinstance(state.get("tournament"),
                                              dict) else {}
    bb = state.get("bb")
    sb = state.get("sb")
    ante = state.get("ante") or 0.0
    hero = next((p for p in (state.get("players") or [])
                 if isinstance(p, dict) and p.get("is_hero")), None)
    hero_stack = hero.get("stack") if hero else None
    players = [p for p in (state.get("players") or []) if isinstance(p, dict)]
    n_players = len(players)
    per_round = (float(sb or 0) + float(bb or 0)
                 + float(ante) * max(n_players, 1))

    factors: dict[str, dict] = {}

    # ---- stack_pressure（M-ratio） ----
    if bb and hero_stack:
        m_ratio = float(hero_stack) / per_round if per_round > 0 else None
        if m_ratio is None or m_ratio <= 0:
            factors["stack_pressure"] = {
                "status": "UNKNOWN", "reason": "non-positive per-round cost"}
        else:
            zone = ("short" if m_ratio <= SHORT_STACK_M
                    else "big" if m_ratio >= BIG_STACK_M else "medium")
            factors["stack_pressure"] = {
                "status": "usable", "m_ratio": round(m_ratio, 4),
                "zone": zone,
                "thresholds": {"short": SHORT_STACK_M, "big": BIG_STACK_M}}
    else:
        factors["stack_pressure"] = {
            "status": "UNKNOWN",
            "reason": "missing bb/sb/hero stack"}

    # ---- bubble_pressure（对手 ICM 压力；hero k± 不表达） ----
    itm = t.get("itm_players")
    remaining = t.get("players_remaining")
    if itm and remaining:
        gap = int(remaining) - int(itm)
        if 0 <= gap <= NEAR_BUBBLE_GAP:
            hero_zone = (factors.get("stack_pressure") or {}).get("zone")
            if hero_zone == "short":
                factors["bubble_pressure"] = {
                    "status": "UNKNOWN",
                    "reason": "hero short at bubble — aggression tilt not "
                              "evidenced (Icm k± dominates; no fake shove)"}
            else:
                factors["bubble_pressure"] = {
                    "status": "usable", "gap": gap,
                    "mechanism": "opponent_icm_pressure"}
        else:
            factors["bubble_pressure"] = {
                "status": "UNKNOWN",
                "reason": f"not near bubble (gap={gap})"}
    else:
        factors["bubble_pressure"] = {
            "status": "UNKNOWN",
            "reason": "itm_players/players_remaining missing"}

    # ---- 报告性上下文（显式不调整——防 ICM 双计） ----
    stage = str(t.get("stage") or "")
    if not stage and remaining:
        stage = f"players_remaining={int(remaining)}"
    factors["final_table"] = {
        "status": "context_only",
        "final_table": bool(remaining and int(remaining)
                            <= FINAL_TABLE_SIZE),
        "reason": "risk aversion expressed by hero ICM k± — no separate "
                  "adjustment (anti-double-count)"}
    factors["payout_sensitivity"] = {
        "status": "context_only",
        "expressed_by": "chip_ev+icm objective (P44 k± scaling)",
        "reason": "no separate adjustment (anti-double-count with ICM)"}
    return {"module": MGG, "tournament_stage": stage or "unknown",
            "avg_stack": t.get("avg_stack"),
            "players_remaining": remaining,
            "factors": factors}


def mtt_context_to_entries(context: dict) -> list[dict]:
    """因素块 → P63 ledger entries（source=mtt_context；仅 usable 因素）。"""
    from .strategy_fusion import make_entry
    factors = (context or {}).get("factors") or {}
    entries = []
    conf = 0.6                      # 单因素声明置信（contract）
    sp = factors.get("stack_pressure") or {}
    if sp.get("status") == "usable" and sp.get("zone") == "short":
        # 短码：推/弃域 → 激进 +、被动 call −（幅度按 zone 内位置）
        m = float(sp.get("m_ratio") or 0.0)
        depth = max(0.0, min(1.0, (SHORT_STACK_M - m) / SHORT_STACK_M))
        entries.append(make_entry(MGG, "bet", 0.15 + 0.25 * depth, conf,
                                  ["mtt_stack_pressure"],
                                  provenance=f"M={m:.2f} short → "
                                             "shove/fold orientation"))
        entries.append(make_entry(MGG, "raise", 0.15 + 0.25 * depth, conf,
                                  ["mtt_stack_pressure"],
                                  provenance=f"M={m:.2f} short → "
                                             "shove/fold orientation"))
        entries.append(make_entry(MGG, "call", -0.1 - 0.1 * depth, conf,
                                  ["mtt_stack_pressure"],
                                  provenance=f"M={m:.2f} short → "
                                             "passive call dominated"))
    elif sp.get("status") == "usable" and sp.get("zone") == "big":
        entries.append(make_entry(MGG, "bet", 0.1, conf,
                                  ["mtt_stack_pressure"],
                                  provenance=f"M={sp.get('m_ratio')} big → "
                                             "value betting lane"))
    bp = factors.get("bubble_pressure") or {}
    if bp.get("status") == "usable" and bp.get("mechanism") == \
            "opponent_icm_pressure":
        gap = int(bp.get("gap") or 0)
        boost = 0.2 if gap <= 1 else 0.1
        entries.append(make_entry(MGG, "bet", boost, conf,
                                  ["mtt_bubble_pressure"],
                                  provenance=f"opponent ICM pressure "
                                             f"(gap={gap})"))
        entries.append(make_entry(MGG, "raise", boost, conf,
                                  ["mtt_bubble_pressure"],
                                  provenance=f"opponent ICM pressure "
                                             f"(gap={gap})"))
    return entries


def mode_application_gate(resolved_mode: str) -> dict:
    """mode 门控：仅 mtt 应用；其他模式显式 not_applied。"""
    if resolved_mode == "mtt":
        return {"applied": True, "reason": ""}
    if resolved_mode == "cash":
        return {"applied": False,
                "reason": "cash mode: MTT-specific factors forbidden"}
    return {"applied": False,
            "reason": f"mode {resolved_mode}: MTT-specific factors "
                      "not applicable"}
