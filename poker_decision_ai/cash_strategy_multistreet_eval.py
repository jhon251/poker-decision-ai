"""V3H-5 — CASH Multi-Street Evaluation（**只用已知 board**，不发明未来牌）。

定位
----
在**已知公共牌**之上计算"若牌局打到摊牌"的**条件绝对权益**（conditional
absolute equity），作为多街评估的基础量。这是对 V3H-4 限制 **L2**
（仅 flop 手牌强度）的多街延伸。

**核心区分（必须保留）**
------------------------
* **本层不预测未来牌**：不采样 runout、不生成未来牌、不 import `random`；
* 计算的是**条件绝对权益**：在**全部可能的后续公共牌**上取平均——
  即"跑完（run it out）"的枚举期望，**不是**对某一条具体 runout 的预测；
* 该量与冻结 `compute_ev` / 旧 multistreet `sample_runouts` / 叶子 EV 逻辑
  **无关、不复用、不修改**；本层不 import 任何 `research_unfreeze` 模块。

**EV 语义（如实标注；不得措辞为"真实收益"）**
---------------------------------------------
`ev_pot_share = future_equity * (pot + 2*invest) / pot − invest / pot`

* "未来权益"= 跑完后的条件绝对权益（见上）；
* **不含**对手弃牌收益（无对手模型）；
* 因此该值**不是**真实收益、**不是**均衡 EV，只是**模型内**的 pot 份额量。

**river 限制（L1，保留并显式记录）**
-----------------------------------
冻结 V3G-7 把 river 视为 terminal（`terminal_reason="showdown"`），
因此**多街前瞻到 river 即止**：river 上没有可继续的下注决策。
本层**不修改** V3G-7，只把该事实作为一等输出记录
（`river_terminal_limit`）。

数据接口（生产 import 图零新增）
--------------------------------
**不在模块顶层 import 任何 V3G 层**。`action_tree` / `action_policy` /
`belief` 由调用方传入；缺失即 fail closed。
"""
from __future__ import annotations

import itertools
from typing import Any

CASH_STRATEGY_MULTISTREET_EVAL_VERSION = "v3h5-cash-strategy-multistreet-eval-v1"

#: 评估模型身份（显式声明）
EVAL_MODEL: dict[str, Any] = {
    "name": "conditional_absolute_equity_v1",
    "definition": ("average showdown equity over ALL possible remaining board "
                   "cards (run-it-out enumeration); NOT a prediction of one runout"),
    "invents_future_cards": False,
    "samples_runouts": False,
    "uses_randomness": False,
    "reuses_frozen_multistreet_or_compute_ev": False,
    "is_gto": False,
    "is_equilibrium": False,
}

CLAIM_BOUNDARY: dict[str, bool] = {
    "is_gto": False,
    "is_equilibrium": False,
    "is_solver_output": False,
    "is_profit_evidence": False,
    "predicts_specific_runout": False,
    "wired_to_production": False,
}

#: 每街期望公共牌数（用于记录 board 是否完整；不推断未来牌）
_EXPECTED_BOARD = {"preflop": 0, "flop": 3, "turn": 4, "river": 5}

#: river 上按冻结 V3G-7 属于 terminal
RIVER_TERMINAL_LIMIT = {
    "applies": True,
    "source": "frozen V3G-7 cash_action_tree._terminal_reason",
    "fact": "street == river -> terminal (terminal_reason='showdown')",
    "consequence": "no further betting decision is available on the river",
    "handling": "recorded as a first-class output; V3G-7 NOT modified",
}


def _is_cash(state: Any) -> bool:
    if not isinstance(state, dict):
        return False
    gt = state.get("game_type")
    return isinstance(gt, str) and gt.strip().lower() == "cash"


def _stub(reason: str) -> dict:
    return {
        "present": False,
        "version": CASH_STRATEGY_MULTISTREET_EVAL_VERSION,
        "reason": reason,
        "evidence_grade": "NONE",
        "unknown": [reason],
        "equity_by_street": {},
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }


def _hero_hole(state: dict) -> tuple[list[str], str | None]:
    hero_id = state.get("hero_id")
    for p in (state.get("players") or []):
        if not isinstance(p, dict):
            continue
        if bool(p.get("is_hero")) or (
                hero_id is not None and p.get("player_id") == hero_id):
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


def conditional_absolute_equity(hero: list[str], board: list[str],
                                *, iterations: int = 0,
                                seed: int = 42) -> dict:
    """条件绝对权益（跑完枚举）。只读、确定性、不使用随机。

    Parameters
    ----------
    hero : hero 的两张手牌（已知）
    board : **已知**公共牌（0/3/4/5 张）
    iterations : 保留参数（本实现**不采样**；>0 时不改变结果，仅记录）
    seed : 保留参数（本实现不使用随机；仅记录以保持 provenance 一致）
    """
    if len(board) not in (0, 3, 4, 5):
        return {"present": False, "unknown": ["board_size_invalid"]}
    try:
        from .domain.poker.cards import parse_card
        from .domain.poker.equity import EquityCalculator
        from .domain.poker.ranges import parse_range
    except Exception:                                      # noqa: BLE001
        return {"present": False, "unknown": ["equity_engine_unavailable"]}

    needed = 5 - len(board)
    try:
        hero_cards = [parse_card(c) for c in hero]
        board_cards = [parse_card(c) for c in board]
    except Exception:                                      # noqa: BLE001
        return {"present": False, "unknown": ["card_parse_failed"]}
    if len({str(c) for c in hero_cards + board_cards}) != len(hero_cards) + len(board_cards):
        return {"present": False, "unknown": ["duplicate_card_between_hero_and_board"]}

    try:
        full = parse_range(
            "22+,A2s+,K2s+,Q2s+,J2s+,T2s+,92s+,82s+,72s+,62s+,52s+,42s+,32s+,"
            "A2o+,K2o+,Q2o+,J2o+,T2o+,92o+,82o+,72o+,62o+,52o+,42o+,32o+",
            name="uniform_random_hand")
        calc = EquityCalculator()
        res = calc.hand_vs_range([str(c) for c in hero_cards],
                                 [str(c) for c in board_cards], full)
    except Exception as exc:                               # noqa: BLE001
        return {"present": False,
                "unknown": [f"equity_error:{type(exc).__name__}"]}

    return {
        "present": True,
        "hero_equity": round(float(res.hero_equity), 6),
        "villain_equity": round(float(res.villain_equity), 6),
        "tie": round(float(res.tie), 6),
        "samples": int(res.samples),
        "method": res.method,
        "opponent_model": "uniform_random_hand",
        "runouts_remaining": needed,
        "iterations_ignored": int(iterations),
        "seed_recorded": int(seed),
        "source": "domain.poker.equity.EquityCalculator.hand_vs_range",
        "invents_future_cards": False,
    }


def evaluate_multistreet(state: Any, *, action_tree: dict | None = None,
                         action_policy: dict | None = None,
                         belief: dict | None = None) -> dict:
    """多街评估：条件绝对权益 + 每街可行动性（**不接生产、不发明未来牌**）。

    Parameters
    ----------
    state : CASH state（非 CASH → present=False）
    action_tree : 可选；用于转述 terminal 状态（含 river 限制）
    action_policy : 可选；用于转述候选与几何量
    belief : 可选；用于转述 prior-only / likelihood 路径
    """
    if not _is_cash(state):
        return _stub("not_cash_mode_fail_closed")

    street = state.get("street")
    hero, h_why = _hero_hole(state)
    board, b_why = _board(state)

    unknown: list[str] = []
    if h_why:
        unknown.append(h_why)
    if b_why:
        unknown.append(b_why)

    expected = _EXPECTED_BOARD.get(street) if isinstance(street, str) else None
    if expected is None:
        unknown.append(f"unknown_street:{street!r}")
    elif len(board) != expected:
        unknown.append(f"board_size_mismatch:{len(board)}!={expected}")

    equity = ({"present": False, "unknown": ["inputs_unavailable"]}
              if (h_why or b_why) else conditional_absolute_equity(hero, board))

    by_street: dict[str, dict] = {}
    if equity.get("present"):
        by_street[str(street)] = {
            "street": street,
            "board_cards_known": len(board),
            "runouts_remaining": equity["runouts_remaining"],
            "equity": equity,
        }

    # 几何与 EV（仅当上游给出）
    pot = to_call = None
    if isinstance(action_policy, dict) and action_policy.get("present"):
        pot = action_policy.get("pot_bb")
        to_call = action_policy.get("to_call_bb")
    ev_block: dict[str, Any] = {"available": False}
    if equity.get("present") and isinstance(pot, (int, float)) and pot > 0:
        eq = float(equity["hero_equity"])
        invest = float(to_call or 0.0)
        ev = eq * (float(pot) + 2.0 * invest) / float(pot) - invest / float(pot)
        ev_block = {
            "available": True,
            "ev_pot_share": round(ev, 6),
            "unit": "pot_share",
            "invest_bb": round(invest, 6),
            "formula": "equity*(pot+2*invest)/pot - invest/pot",
            "opponent_fold_equity_included": False,
            "is_real_profit": False,
            "is_equilibrium_ev": False,
        }

    terminal = None
    if isinstance(action_tree, dict):
        terminal = {"terminal": action_tree.get("terminal"),
                    "terminal_reason": action_tree.get("terminal_reason")}

    return {
        "present": True,
        "version": CASH_STRATEGY_MULTISTREET_EVAL_VERSION,
        "mode": "cash",
        "street": street,
        "eval_model": dict(EVAL_MODEL),
        "equity_by_street": by_street,
        "ev_by_street": {str(street): ev_block} if ev_block.get("available") else {},
        "terminal": terminal,
        # L1：river 限制作为一等输出
        "river_terminal_limit": dict(RIVER_TERMINAL_LIMIT) if street == "river"
        else {**RIVER_TERMINAL_LIMIT, "applies": False,
              "note": "limit applies on the river street only"},
        "future_board_generated": False,
        "runout_sampling": False,
        "belief_likelihood_source": (belief.get("likelihood_source")
                                     if isinstance(belief, dict) else None),
        "unknown": unknown,
        "level": "UNKNOWN" if unknown else "MODEL_INTERNAL",
        "evidence_grade": "MODEL_INTERNAL",
        "uncertainty": {
            "equity_method": equity.get("method"),
            "samples": equity.get("samples"),
            "note": ("条件绝对权益为跑完枚举期望；Monte Carlo 参数来自上游 "
                     "equity 引擎并原样转述。"),
        },
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }
