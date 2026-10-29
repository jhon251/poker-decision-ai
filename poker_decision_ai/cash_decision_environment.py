"""V3I-P1 Step 1 — 可控 6-max CASH 执行环境（确定性、模型内、零层 import）。

**本模块的定位**
----------------
为 CASH 决策闭环提供**可复现的执行环境**：

* **显式 seed** → 同 seed 同输出（无随机数模块、无 IO、无 wall-clock）；
* **可控 6-max CASH 场景** → 桌容量、盲注、筹码、街道、board 全部由参数决定；
* **模型内 opponent population** → 每个种群是**声明式**的建模参数，
  **不是**真实对手统计、**不是**已观测频率、**不是**外部数据。

**严格边界（必须如实标注）**
----------------------------
* `POPULATIONS` 中的每个取值都是**声明式建模参数**（`DECLARED_MODEL`），
  **不得**被读作真实人类对手的频率或倾向；
* 本模块**不生成未来牌**、**不做 runout 抽样**；
  board 与手牌**仅由显式 seed 索引选择**，且**全部来自牌堆字典序**；
* 本模块**不执行任何动作**、**不接网络/客户端/求解器**、**不写文件**；
* 本模块**不 import 任何 cash_ 层**（含 `cash_integrity_context`）——
  上游闸门与求解器状态一律由**调用方显式传入**，
  以保持 V3G-11 的 `test_only_sanctioned_wiring_exists` 边界语义；
* CASH-only：非 CASH 场景请求一律 fail closed。
"""
from __future__ import annotations

from typing import Any

CASH_DECISION_ENVIRONMENT_VERSION = "v3i-p1-cash-decision-environment-v1"

#: 牌堆（字典序固定，仅按索引取牌，不使用随机）
RANKS: tuple[str, ...] = ("2", "3", "4", "5", "6", "7", "8", "9", "T", "J",
                          "Q", "K", "A")
SUITS: tuple[str, ...] = ("s", "h", "d", "c")
DECK: tuple[str, ...] = tuple(r + s for r in RANKS for s in SUITS)

#: 6-max 座位与位置
SIX_MAX_POSITIONS: tuple[str, ...] = ("SB", "BB", "UTG", "MP", "CO", "BTN")

#: 街道（river 为终局，由冻结 V3G-7 决定，本模块不绕过）
STREETS: tuple[str, ...] = ("preflop", "flop", "turn", "river")

#: 街道 → board 牌数
BOARD_SIZE: dict[str, int] = {"preflop": 0, "flop": 3, "turn": 4, "river": 5}

#: 模型内对手种群（**声明式**，非真实统计）
POPULATIONS: tuple[str, ...] = (
    "DECLARED_TIGHT_PASSIVE",
    "DECLARED_BALANCED",
    "DECLARED_LOOSE_PASSIVE",
    "DECLARED_AGGRESSIVE",
    "DECLARED_MANIAC",
)

#: 证据等级（本环境只产出模型内声明）
ENVIRONMENT_EVIDENCE_LEVEL = "DECLARED_MODEL"

#: 模型内**声明式**种群参数（**不是**真实频率，**不是**已观测数据）
POPULATION_PARAMS: dict[str, dict[str, Any]] = {
    "DECLARED_TIGHT_PASSIVE": {
        "description": "declared model: narrow continuation, low aggression",
        "modeled_preflop_continue_multiplier": 0.7,
        "modeled_postflop_aggression_multiplier": 0.6,
    },
    "DECLARED_BALANCED": {
        "description": "declared model: reference profile",
        "modeled_preflop_continue_multiplier": 1.0,
        "modeled_postflop_aggression_multiplier": 1.0,
    },
    "DECLARED_LOOSE_PASSIVE": {
        "description": "declared model: wide continuation, low aggression",
        "modeled_preflop_continue_multiplier": 1.4,
        "modeled_postflop_aggression_multiplier": 0.5,
    },
    "DECLARED_AGGRESSIVE": {
        "description": "declared model: reference continuation, high aggression",
        "modeled_preflop_continue_multiplier": 1.0,
        "modeled_postflop_aggression_multiplier": 1.8,
    },
    "DECLARED_MANIAC": {
        "description": "declared model: wide continuation, very high aggression",
        "modeled_preflop_continue_multiplier": 1.8,
        "modeled_postflop_aggression_multiplier": 2.6,
    },
}

#: 种群可以施加的动作（与既有 belief 层的 `_last_action` 约定一致）
POPULATION_ACTIONS: tuple[str, ...] = ("fold", "check", "call", "bet", "raise",
                                       "allin")

CLAIM_BOUNDARY: dict[str, bool] = {
    "is_real_opponent_data": False,
    "is_observed_frequency": False,
    "is_external_validation": False,
    "is_gto": False,
    "is_equilibrium": False,
    "is_solver_output": False,
    "is_profit_evidence": False,
    "generates_future_cards": False,
    "samples_runouts": False,
    "executes_action": False,
    "auto_betting": False,
    "accesses_network_client_or_solver": False,
    "wired_to_production": False,
    "modifies_frozen_layers": False,
}
CLAIM_FLAGS_TOTAL = len(CLAIM_BOUNDARY)


def _is_cash(state: Any) -> bool:
    if not isinstance(state, dict):
        return False
    gt = state.get("game_type")
    return isinstance(gt, str) and gt.strip().lower() == "cash"


def _stub(reason: str) -> dict:
    return {
        "present": False,
        "version": CASH_DECISION_ENVIRONMENT_VERSION,
        "reason": reason,
        "evidence_level": "NONE",
        "scenario": None,
        "population_table": [],
        "unknown": [reason],
        "executed": False,
        "result": None,
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }


def _pick(seed: int, offset: int, taken: list[str]) -> str:
    """按 (seed, offset) 确定性选取一张**未被占用**的牌（不抽样、不预测）。"""
    n = len(DECK)
    for probe in range(n):
        card = DECK[(seed * 7 + offset * 13 + probe) % n]
        if card not in taken:
            taken.append(card)
            return card
    raise ValueError("deck_exhausted")


def deck_card_for(seed: int, offset: int) -> dict:
    """返回 seed+offset 对应的**单张**牌（纯索引选择，暴露以便测试固定）。"""
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        return {"present": False, "reason": "seed_must_be_non_negative_int",
                "card": None, "executed": False}
    if not isinstance(offset, int) or isinstance(offset, bool) or offset < 0:
        return {"present": False, "reason": "offset_must_be_non_negative_int",
                "card": None, "executed": False}
    return {"present": True, "version": CASH_DECISION_ENVIRONMENT_VERSION,
            "card": DECK[(seed * 7 + offset * 13) % len(DECK)],
            "index": (seed * 7 + offset * 13) % len(DECK),
            "evidence_level": ENVIRONMENT_EVIDENCE_LEVEL,
            "generates_future_cards": False,
            "samples_runouts": False,
            "executed": False,
            "claim_boundary": dict(CLAIM_BOUNDARY)}


# ══ 场景构建 ═══════════════════════════════════════════════════════════
def build_scenario(*, seed: int = 0, street: str = "flop",
                   table_size: int = 6, bb: float = 1.0,
                   stack_bb: float = 100.0,
                   population_table: Any = None,
                   pot_bb: float = 1.5, to_call_bb: float = 0.0) -> dict:
    """构建一个**确定性**的 CASH 场景（同参数 → 同输出）。

    Parameters
    ----------
    seed : 非负整数（唯一随机性来源；不使用任何随机数模块）
    street : preflop / flop / turn / river
    table_size : 桌容量（本阶段固定 6-max 语义，允许 2..6）
    bb / stack_bb / pot_bb / to_call_bb : 局面参数
    population_table : 对手种群列表（长度须为 table_size - 1）
    """
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        return _stub("seed_must_be_non_negative_int")
    if street not in STREETS:
        return _stub(f"street_not_supported:{street!r}")
    if not isinstance(table_size, int) or isinstance(table_size, bool) \
            or not (2 <= table_size <= 6):
        return _stub("table_size_must_be_int_between_2_and_6")
    for nm, v in (("bb", bb), ("stack_bb", stack_bb), ("pot_bb", pot_bb),
                  ("to_call_bb", to_call_bb)):
        if not isinstance(v, (int, float)) or isinstance(v, bool) or v < 0:
            return _stub(f"{nm}_must_be_non_negative_number")

    pops = list(population_table) if isinstance(
        population_table, (list, tuple)) else ["DECLARED_BALANCED"] * (
        table_size - 1)
    if len(pops) != table_size - 1:
        return _stub("population_table_length_must_equal_table_size_minus_one")
    for p in pops:
        if p not in POPULATIONS:
            return _stub(f"population_not_in_closed_enum:{p!r}")

    taken: list[str] = []
    board = [_pick(seed, 100 + i, taken)
             for i in range(BOARD_SIZE[street])]
    hero_hole = [_pick(seed, 200 + i, taken) for i in range(2)]

    positions = SIX_MAX_POSITIONS[:table_size] if table_size == 6 else \
        SIX_MAX_POSITIONS[:table_size]
    hero_seat = table_size
    players: list[dict] = []
    opp_index = 0
    for seat in range(1, table_size + 1):
        if seat == hero_seat:
            players.append({
                "player_id": "hero", "seat": seat,
                "position": positions[seat - 1], "is_hero": True,
                "status": "active", "hole_cards": list(hero_hole),
                "stack": float(stack_bb),
            })
        else:
            players.append({
                "player_id": f"opp{opp_index + 1}", "seat": seat,
                "position": positions[seat - 1], "status": "active",
                "hole_cards": [],          # 对手手牌不可知（本环境不伪造）
                "stack": float(stack_bb),
            })
            opp_index += 1

    state = {
        "game_type": "cash",
        "street": street,
        "board": board,
        "pot": float(pot_bb) * float(bb),
        "to_call": float(to_call_bb) * float(bb),
        "bb": float(bb),
        "sb": float(bb) / 2.0,
        # 冻结的 cash_action_tree **不设默认值**，这些边界必须由 state 显式给出
        # （本模块只做单位换算，不重算合法性）
        "min_raise_to": (2.0 * float(bb)) + (float(to_call_bb) * float(bb)),
        "min_raise_to_bb": 2.0 + float(to_call_bb),
        "effective_stack_bb": float(stack_bb),
        "pot_bb": float(pot_bb),
        "to_call_bb": float(to_call_bb),
        "hero_id": "hero",
        "button_seat": 2,
        "table_size": table_size,
        "synthetic_test_only": True,
        "players": players,
        "action_history": [],
    }

    return {
        "present": True,
        "version": CASH_DECISION_ENVIRONMENT_VERSION,
        "seed": seed,
        "street": street,
        "table_size": table_size,
        "positions": list(positions),
        "scenario_id": f"scn-seed{seed}-{street}-{table_size}max",
        "state": state,
        "hero_hole_cards": list(hero_hole),
        "board": list(board),
        "population_table": list(pops),
        "population_params": {p: dict(POPULATION_PARAMS[p]) for p in pops},
        "evidence_level": ENVIRONMENT_EVIDENCE_LEVEL,
        "generates_future_cards": False,
        "samples_runouts": False,
        "executed": False,
        "result": None,
        "claim_boundary": dict(CLAIM_BOUNDARY),
        "note": ("确定性场景：同参数同输出；board 与手牌仅按 seed 索引自字典序"
                 "牌堆选取，**不预测未来牌**、**不抽样 runout**。"
                 "对手手牌为空列表，本环境**不伪造**未知信息。"),
        "unknown": [],
    }


def build_scenario_sequence(*, seed: int = 0, streets: Any = None,
                            table_size: int = 6,
                            population_table: Any = None) -> dict:
    """构建一条**同 seed** 的场景序列（供街道级闭环使用）。"""
    st = list(streets) if isinstance(streets, (list, tuple)) else list(STREETS)
    out: list[dict] = []
    for s in st:
        scn = build_scenario(seed=seed, street=s, table_size=table_size,
                             population_table=population_table)
        if not scn["present"]:
            return _stub(f"scenario_failed_at_street:{s}:{scn['reason']}")
        out.append(scn)
    return {
        "present": True,
        "version": CASH_DECISION_ENVIRONMENT_VERSION,
        "seed": seed,
        "n_scenarios": len(out),
        "scenarios": out,
        "scenario_ids": [s["scenario_id"] for s in out],
        "table_size": table_size,
        "population_table": out[0]["population_table"] if out else [],
        "evidence_level": ENVIRONMENT_EVIDENCE_LEVEL,
        "executed": False,
        "result": None,
        "claim_boundary": dict(CLAIM_BOUNDARY),
        "unknown": [],
    }


# ══ 模型内对手动作权重（**声明式**） ═══════════════════════════════════
def population_action_weights(*, population: str = "DECLARED_BALANCED",
                              last_action: str = "check",
                              street: str = "flop") -> dict:
    """按**声明式**种群参数给出 `action_weights`（供既有 belief 层消费）。

    **这不是**已观测频率：仅由 `POPULATION_PARAMS` 的建模乘数导出，
    并如实标注 `DECLARED_MODEL`。
    """
    if population not in POPULATIONS:
        return _stub(f"population_not_in_closed_enum:{population!r}")
    if last_action not in POPULATION_ACTIONS:
        return _stub(f"action_not_in_closed_enum:{last_action!r}")
    if street not in STREETS:
        return _stub(f"street_not_supported:{street!r}")

    p = POPULATION_PARAMS[population]
    if street == "preflop":
        base = p["modeled_preflop_continue_multiplier"]
    else:
        base = p["modeled_postflop_aggression_multiplier"]

    weights: dict[str, float] = {"_last_action": last_action}
    for a in POPULATION_ACTIONS:
        if a in ("fold", "check"):
            weights[a] = 1.0
        elif a in ("call",):
            weights[a] = round(1.0 * base, 6)
        else:                                   # bet / raise / allin
            weights[a] = round(1.0 * base, 6)

    return {
        "present": True,
        "version": CASH_DECISION_ENVIRONMENT_VERSION,
        "population": population,
        "last_action": last_action,
        "street": street,
        "action_weights": weights,
        "evidence_level": "DECLARED_MODEL",
        "is_observed_frequency": False,
        "source": "declared_model_parameter_table",
        "executed": False,
        "result": None,
        "claim_boundary": dict(CLAIM_BOUNDARY),
        "note": ("权重由**声明式建模参数**导出，**不是**真实对手频率、"
                 "**不是**已观测数据、**不构成**外部验证。"),
        "unknown": [],
    }


def environment_contract() -> dict:
    """返回环境契约（纯声明；供测试与后续阶段引用）。"""
    return {
        "success": True,
        "version": CASH_DECISION_ENVIRONMENT_VERSION,
        "streets": list(STREETS),
        "board_size_by_street": dict(BOARD_SIZE),
        "positions_6max": list(SIX_MAX_POSITIONS),
        "populations": list(POPULATIONS),
        "population_params": {k: dict(v) for k, v in
                              POPULATION_PARAMS.items()},
        "population_actions": list(POPULATION_ACTIONS),
        "deck_size": len(DECK),
        "evidence_level": ENVIRONMENT_EVIDENCE_LEVEL,
        "determinism": "explicit_seed_index_selection_no_random_module",
        "generates_future_cards": False,
        "samples_runouts": False,
        "imports_any_cash_layer": False,
        "executed": False,
        "result": None,
        "claim_boundary": dict(CLAIM_BOUNDARY),
        "unknown": [],
    }
