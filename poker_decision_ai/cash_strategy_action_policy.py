"""V3H-4 — CASH Action Policy（候选动作 + provenance；**不做最终决策**）。

定位
----
在**既有合法动作/尺寸树**之上，产出"候选动作 + 可核验的模型内依据"。
本层是 V3H-3 路线审计中 `L2 policy_candidate` 的最小离线实现。

**本层不选动作**：不打最终决策、不接生产 Judge、不写回任何 state。
输出的是 candidates 列表与它们的 provenance；最终选择仍属未来（未授权）阶段。

数据接口（审计要求：生产 import 图零新增）
------------------------------------------
本层**不在模块顶层 import 任何 V3G 层**（V3G-11 边界测试按 AST 扫描包内全部
模块，只豁免层自身文件）。因此：

* **合法动作/尺寸树由调用方通过 `action_tree=` 传入**
  （由 `cash_action_tree.build_action_tree` 产出）；
* 未传入 → `present=False, reason="action_tree_required"`，
  **不做隐式导入、不猜、不默认**。

这样既满足"复用而非重实现"（不重写任何合法性/尺寸规则），
又保证 `realtime.py` / Judge 与其余生产模块的 import 图**零新增**。

其余复用
--------
* **手牌强度**：`domain.poker.equity.EquityCalculator.hand_vs_range`（显式 seed）；
* **模式门禁**：`cash_integrity_context`（V3G-2 适配层，已授权旁路点）。

硬约束
------
1. **只读 + 确定性**：不修改输入；**不 import `random`**、不生成未来牌、
   无 IO、无 wall-clock；同输入 + 同 seed → **逐字节同输出**。
2. **不默认任何东西**：不默认动作、下注尺寸、位置、座位、stack、table size、庄位。
3. **UNKNOWN ≠ EMPTY**：信息不足 → `unknown`；**绝不**用 0 值或默认动作填充；
   `unknown` 非空时不产出 `preferred`。
4. **不声称 GTO / equilibrium**：全部 `evidence_grade="MODEL_INTERNAL"`。
5. **CASH-only，fail closed**：非 CASH → `present=False`。
"""
from __future__ import annotations

from typing import Any

from .domain.poker.equity import EquityCalculator

CASH_STRATEGY_ACTION_POLICY_VERSION = "v3h4-cash-strategy-action-policy-v1"

EVIDENCE_GRADE = "MODEL_INTERNAL"

CLAIM_BOUNDARY: dict[str, bool] = {
    "is_gto": False,
    "is_equilibrium": False,
    "is_solver_output": False,
    "is_profit_evidence": False,
    "is_calibrated": False,
    "selects_action": False,
    "wired_to_production": False,
}

PROVENANCE_KEYS: tuple[str, ...] = (
    "legality_source", "sizing_source", "equity_source", "range_source",
    "state_integrity", "mode",
)

DEFAULT_EQUITY_SEED = 42
DEFAULT_EQUITY_ITERATIONS = 20000


def _is_cash(state: Any) -> bool:
    if not isinstance(state, dict):
        return False
    gt = state.get("game_type")
    return isinstance(gt, str) and gt.strip().lower() == "cash"


def _stub(reason: str) -> dict:
    return {
        "present": False,
        "version": CASH_STRATEGY_ACTION_POLICY_VERSION,
        "reason": reason,
        "evidence_grade": "NONE",
        "unknown": [reason],
        "candidates": [],
        "preferred": None,
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }


def _hero_hole_cards(state: dict) -> tuple[list[str], str | None]:
    hero_id = state.get("hero_id")
    players = state.get("players")
    if not isinstance(players, list):
        return [], "players_missing"
    for p in players:
        if not isinstance(p, dict):
            continue
        is_hero = bool(p.get("is_hero")) or (
            hero_id is not None and p.get("player_id") == hero_id)
        if not is_hero:
            continue
        hc = p.get("hole_cards")
        if isinstance(hc, list) and len(hc) == 2 and all(
                isinstance(c, str) and c.strip() for c in hc):
            return [str(c) for c in hc], None
        return [], "hero_hole_cards_missing"
    return [], "hero_not_found"


def _board(state: dict) -> tuple[list[str], str | None]:
    b = state.get("board")
    if not isinstance(b, list):
        return [], "board_missing"
    cards = [str(c) for c in b if isinstance(c, str) and c.strip()]
    if len(cards) != len(b):
        return [], "board_malformed"
    return cards, None


def _hand_strength(state: dict, *, seed: int = DEFAULT_EQUITY_SEED,
                   iterations: int = DEFAULT_EQUITY_ITERATIONS) -> dict:
    """手牌强度（vs **均匀随机手牌**范围）。缺失 → unknown，不猜。

    复用冻结 `EquityCalculator.hand_vs_range`（该路径为 Monte Carlo），
    因此**必须显式传入 seed** 并把 `seed`/`iterations` 写入结果：
    同 seed + 同 iterations → **逐位可复现**；不做无 seed 的随机。
    该值是"对随机手牌的胜率"，**不是** GTO/均衡量。
    """
    hole, h_why = _hero_hole_cards(state)
    board, b_why = _board(state)
    if h_why or b_why:
        return {"present": False, "unknown": [x for x in (h_why, b_why) if x]}
    if len(board) not in (0, 3, 4, 5):
        return {"present": False, "unknown": ["board_size_invalid"]}
    try:
        from .domain.poker.ranges import parse_range
        full = parse_range(
            "22+,A2s+,K2s+,Q2s+,J2s+,T2s+,92s+,82s+,72s+,62s+,52s+,42s+,32s+,"
            "A2o+,K2o+,Q2o+,J2o+,T2o+,92o+,82o+,72o+,62o+,52o+,42o+,32o+",
            name="uniform_random_hand")
        calc = EquityCalculator(iterations=iterations, seed=seed)
        res = calc.hand_vs_range(hole, board, full,
                                 iterations=iterations, seed=seed)
    except Exception as exc:                               # noqa: BLE001
        return {"present": False,
                "unknown": [f"equity_error:{type(exc).__name__}"]}
    return {
        "present": True,
        "hero_equity": round(float(res.hero_equity), 6),
        "tie": round(float(res.tie), 6),
        "samples": int(res.samples),
        "method": res.method,
        "opponent_model": "uniform_random_hand",
        "source": "domain.poker.equity.EquityCalculator.hand_vs_range",
        "seed": int(seed),
        "iterations": int(iterations),
        "deterministic_given_seed": True,
    }


def _pot_odds(to_call_bb: float, pot_bb: float) -> dict:
    """跟注所需最低权益（纯算术；不构成策略主张）。"""
    if to_call_bb <= 0:
        return {"present": True, "required_equity": 0.0,
                "note": "no_call_required"}
    denom = pot_bb + 2.0 * to_call_bb
    if denom <= 0:
        return {"present": False, "unknown": ["pot_geometry_invalid"]}
    return {"present": True,
            "required_equity": round(to_call_bb / denom, 6),
            "source": "arithmetic:to_call/(pot+2*to_call)"}


def _candidate_from_sizing(item: dict, strength: dict, odds: dict) -> dict:
    amt = item.get("amount_bb")
    unknown: list[str] = []
    if not isinstance(amt, (int, float)):
        unknown.append("amount_bb_missing")
    prov = {
        "legality_source": "cash_action_tree.legal_actions",
        "sizing_source": f"cash_action_tree.sizing[{item.get('action')}]",
        "equity_source": strength.get("source", "unavailable"),
        "range_source": "not_supplied",
        "state_integrity": None,
        "mode": "cash",
    }
    comparables: dict[str, Any] = {}
    if strength.get("present"):
        comparables["hero_equity_uniform"] = strength["hero_equity"]
    if odds.get("present") and item.get("action") == "call":
        comparables["required_equity"] = odds["required_equity"]
        if strength.get("present"):
            comparables["equity_minus_required"] = round(
                strength["hero_equity"] - odds["required_equity"], 6)
    if item.get("action") in ("bet", "raise") and isinstance(amt, (int, float)):
        comparables["risk_in_bb"] = float(amt)
    return {
        "action": item.get("action"),
        "sizing_kind": item.get("sizing_kind"),
        "amount_bb": amt if isinstance(amt, (int, float)) else None,
        "amount": item.get("amount"),
        "allin": bool(item.get("allin")),
        "bounds": item.get("bounds"),
        "comparables": comparables,
        "provenance": dict(prov),
        "evidence_grade": EVIDENCE_GRADE,
        "unknown": unknown,
        "is_gto": False,
        "is_equilibrium": False,
    }


def build_action_policy(state: Any, *, action_tree: dict | None = None,
                        range_model: dict | None = None,
                        integrity_context: dict | None = None,
                        equity_seed: int = DEFAULT_EQUITY_SEED,
                        equity_iterations: int = DEFAULT_EQUITY_ITERATIONS) -> dict:
    """产出 CASH 候选动作 + provenance（**不选动作、不接生产**）。

    Parameters
    ----------
    state : CASH state dict（非 CASH → present=False）
    action_tree : **必填**（除非非 CASH 直接 fail closed）。
        由 `cash_action_tree.build_action_tree(state)` 产出；
        本层不隐式导入该层（见模块头"数据接口"）。
    range_model : 可选，仅用于 provenance 标注
    equity_seed / equity_iterations : 手牌强度枚举参数（显式、记入结果）
    """
    if not _is_cash(state):
        return _stub("not_cash_mode_fail_closed")
    if not isinstance(action_tree, dict):
        return _stub("action_tree_required")

    tree = action_tree
    if not tree.get("present"):
        return _stub(f"action_tree_unavailable:{tree.get('reason', 'unknown')}")

    legal = tree.get("legal_actions") or []
    sizing = tree.get("sizing") or []
    if tree.get("terminal") or not legal:
        return _stub("terminal_or_no_legal_actions")

    strength = _hand_strength(state, seed=equity_seed,
                              iterations=equity_iterations)
    odds = _pot_odds(float(tree.get("to_call_bb") or 0.0),
                     float(tree.get("pot_bb") or 0.0))
    integrity = tree.get("integrity") or {}
    grade = integrity.get("evidence_grade")

    ctx = integrity_context
    mode = (ctx.get("mode") if isinstance(ctx, dict)
            else (state.get("game_type") if isinstance(state, dict) else None))

    unknown: list[str] = []
    if not strength.get("present"):
        unknown.extend(strength.get("unknown") or ["hand_strength_unavailable"])
    if not odds.get("present"):
        unknown.extend(odds.get("unknown") or [])
    for u in (tree.get("unknown") or []):
        if u not in unknown:
            unknown.append(u)

    candidates = []
    for item in sizing:
        if not isinstance(item, dict):
            continue
        cand = _candidate_from_sizing(item, strength, odds)
        cand["provenance"]["state_integrity"] = grade
        cand["provenance"]["range_source"] = (
            "cash_range_model" if isinstance(range_model, dict) else "not_supplied")
        cand["provenance"]["provenance_complete"] = all(
            cand["provenance"].get(k) is not None for k in PROVENANCE_KEYS)
        candidates.append(cand)

    cand_actions = sorted({c["action"] for c in candidates if c.get("action")})
    legality_complete = cand_actions == sorted(str(a) for a in legal)

    return {
        "present": True,
        "version": CASH_STRATEGY_ACTION_POLICY_VERSION,
        "mode": mode,
        "street": tree.get("street"),
        # 几何量原样转述（下游排序/评估需要；不重新计算、不默认）
        "pot": tree.get("pot"),
        "pot_bb": tree.get("pot_bb"),
        "to_call": tree.get("to_call"),
        "to_call_bb": tree.get("to_call_bb"),
        "bb": tree.get("bb"),
        "effective_stack_bb": tree.get("effective_stack_bb"),
        "min_raise_to_bb": tree.get("min_raise_to_bb"),
        "legal_actions": list(legal),
        "candidates": candidates,
        "candidate_actions": cand_actions,
        "legality_complete": legality_complete,
        "hand_strength": strength,
        "pot_odds": odds,
        "unknown": unknown,
        "conflicts": list(integrity.get("conflicts") or []),
        "integrity": integrity,
        "evidence_grade": EVIDENCE_GRADE,
        "claim_boundary": dict(CLAIM_BOUNDARY),
        "preferred": None,
        "preferred_note": (
            "selection intentionally not implemented in this layer"
            if not unknown else "suppressed_by_unknown"),
        "selection_implemented": False,
    }
