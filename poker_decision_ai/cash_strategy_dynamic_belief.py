"""V3H-4 — CASH Dynamic Belief（序贯 range/belief 更新；UNKNOWN-first）。

定位
----
在**既有** range 模型（初始 range + 已声明动作权重）与对手模型之上，
做**序贯 belief 更新**，并保证：

* 已知公共牌与已知手牌（hero 自己的牌）从对手 range 中**移除**；
* **hero 的动作绝不写入对手历史**（hero 与 opponent 历史严格隔离）；
* 信息不足 → **UNKNOWN**，**绝不**把 UNKNOWN 当作 EMPTY；
* **不引入任何随机性**（不 import `random`、不采样）。

数据接口（审计要求：生产 import 图零新增）
------------------------------------------
本层**不在模块顶层 import 任何 V3G 层**（V3G-11 边界测试按 AST 扫描包内全部
模块）。因此 range/opponent 模型由调用方传入：

* `range_model=` （由 `cash_range_model.build_range_model` 产出）**必填**；
* `opponent_model=` 可选（由 `cash_opponent_model.build_opponent_model` 产出）；
* 未传入 → `present=False`，**不做隐式导入、不猜**。

牌张冲突消除**复用冻结** `domain.poker.ranges`：
把 combo 名重建为 `Range`，再用 `Range.remove_cards` —— **不自己实现**
任何牌张冲突规则、**不新增** range 常数。

关于"似然权重"的严格口径
------------------------
序贯贝叶斯更新需要 P(action | combo)（似然）。**本层不发明这些权重**：
* 若调用方**显式**传入 `action_weights`，则按其更新并记录来源；
* 若未传入，posterior **等于** prior（仅做已知牌移除），
  记录 `likelihood_source="not_supplied"`；
* 已声明权重的应用由 range 模型完成，本层**不重复施加**以免重复计数。

硬约束
------
1. **只读 + 确定性**：不修改输入；无 IO；无 wall-clock；同输入 → 逐字节同输出。
2. **UNKNOWN ≠ EMPTY**：无法解析的动作/缺失字段 → `unknown`；
   `unknown` 非空 → `level="UNKNOWN"`。
3. **不默认**：不默认位置、座位、stack、table size、庄位；缺失即 unknown。
4. **CASH-only，fail closed**：非 CASH → `present=False`。
5. **不声称 GTO/equilibrium**：belief 是**未校准的模型量**。
"""
from __future__ import annotations

from typing import Any

from .domain.poker.ranges import combo_name, parse_single_card

CASH_STRATEGY_DYNAMIC_BELIEF_VERSION = "v3h4-cash-strategy-dynamic-belief-v1"

EVIDENCE_GRADE = "MODEL_INTERNAL"

CLAIM_BOUNDARY: dict[str, bool] = {
    "is_gto": False,
    "is_equilibrium": False,
    "is_calibrated": False,
    "is_solver_output": False,
    "is_profit_evidence": False,
    "selects_action": False,
    "uses_randomness": False,
    "wired_to_production": False,
}

KNOWN_ACTIONS: tuple[str, ...] = (
    "fold", "check", "call", "bet", "raise", "all_in",
)


def _is_cash(state: Any) -> bool:
    if not isinstance(state, dict):
        return False
    gt = state.get("game_type")
    return isinstance(gt, str) and gt.strip().lower() == "cash"


def _stub(reason: str) -> dict:
    return {
        "present": False,
        "version": CASH_STRATEGY_DYNAMIC_BELIEF_VERSION,
        "reason": reason,
        "evidence_grade": "NONE",
        "unknown": [reason],
        "belief": {},
        "hero_actions_excluded": [],
        "known_cards_removed": [],
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }


def _seat_sanity(state: dict) -> list[str]:
    """座位/玩家结构健全性：缺失或不连续 → unknown（不猜、不默认）。"""
    flags: list[str] = []
    players = state.get("players")
    if not isinstance(players, list) or not players:
        return ["players_missing_or_empty"]
    seats: list[int] = []
    for p in players:
        if not isinstance(p, dict):
            flags.append("player_not_a_dict")
            continue
        s = p.get("seat")
        if not isinstance(s, int) or isinstance(s, bool) or s < 1:
            flags.append(f"invalid_seat:{s!r}")
        else:
            seats.append(s)
    if seats and sorted(seats) != list(range(1, len(seats) + 1)):
        flags.append("seats_not_contiguous_from_1")
    ts = state.get("table_size")
    if isinstance(ts, int) and seats and ts != len(seats):
        flags.append(f"table_size_mismatch:{ts}!={len(seats)}")
    return flags


def _hero_id(state: dict) -> tuple[str | None, list[str]]:
    hid = state.get("hero_id")
    players = state.get("players")
    flags: list[str] = []
    found = None
    if isinstance(players, list):
        for p in players:
            if isinstance(p, dict) and p.get("is_hero"):
                found = p.get("player_id")
                break
    if hid is not None and found is not None and hid != found:
        flags.append("hero_identity_conflict")
    hid = hid if hid is not None else found
    if hid is None:
        flags.append("hero_id_missing")
    return hid, flags


def _split_history(state: dict, hero_id: str | None):
    """把 action_history 分成 (opponent_actions, hero_actions, unknown)。

    **hero 动作绝不进入 opponent_actions**（严格隔离）。
    """
    hist = state.get("action_history")
    if hist is None:
        return [], [], ["action_history_missing"]
    if not isinstance(hist, list):
        return [], [], ["action_history_malformed"]
    opp: list[dict] = []
    hero_acts: list[dict] = []
    unknown: list[str] = []
    for i, a in enumerate(hist):
        if not isinstance(a, dict):
            unknown.append(f"action[{i}]_not_a_dict")
            continue
        pid = a.get("player_id")
        is_hero = bool(a.get("is_hero")) or (
            hero_id is not None and pid == hero_id)
        act = a.get("action")
        if not isinstance(act, str) or act.strip().lower() not in KNOWN_ACTIONS:
            unknown.append(f"action[{i}]_unknown_action:{act!r}")
            continue
        rec = {"player_id": pid, "action": act.strip().lower(),
               "street": a.get("street"), "amount_bb": a.get("amount_bb")}
        (hero_acts if is_hero else opp).append(rec)
    return opp, hero_acts, unknown


def _known_cards(state: dict, hero_id: str | None) -> tuple[list[str], list[str]]:
    """已知牌：公共牌 + **hero 自己**的手牌（不含对手未知手牌）。"""
    unknown: list[str] = []
    cards: list[str] = []
    board = state.get("board")
    if board is None:
        unknown.append("board_missing")
    elif not isinstance(board, list):
        unknown.append("board_malformed")
    else:
        for c in board:
            if isinstance(c, str) and c.strip():
                cards.append(str(c))
            else:
                unknown.append("board_card_invalid")
    players = state.get("players")
    if isinstance(players, list):
        for p in players:
            if not isinstance(p, dict):
                continue
            is_hero = bool(p.get("is_hero")) or (
                hero_id is not None and p.get("player_id") == hero_id)
            if not is_hero:
                continue
            hc = p.get("hole_cards")
            if isinstance(hc, list):
                for c in hc:
                    if isinstance(c, str) and c.strip():
                        cards.append(str(c))
            break
    return cards, unknown


def _validate_cards(cards: list[str]) -> list[str]:
    bad: list[str] = []
    for c in cards:
        try:
            parse_single_card(c)
        except Exception:                                  # noqa: BLE001
            bad.append(c)
    return bad


def _posterior(entry: dict, dead: set[str], weights: dict | None) -> dict:
    """给定 range 条目与死牌集合，返回 posterior 摘要（确定性、无随机）。"""
    inferred = entry.get("inferred") or {}
    norm = inferred.get("normalized") or {}
    if not isinstance(norm, dict) or not norm:
        return {"present": False, "unknown": ["range_empty_or_missing"]}
    try:
        from .domain.poker.ranges import Range, parse_range
    except Exception:                                      # noqa: BLE001
        return {"present": False, "unknown": ["range_engine_unavailable"]}

    text = ",".join(sorted(str(k) for k in norm))
    try:
        rebuilt = parse_range(text, name="belief_prior")
    except Exception as exc:                               # noqa: BLE001
        return {"present": False,
                "unknown": [f"range_rebuild_failed:{type(exc).__name__}"]}
    if not rebuilt.weights:
        return {"present": False, "unknown": ["range_rebuild_empty"]}

    for combo in list(rebuilt.weights):
        key = combo_name(combo)
        if key in norm:
            rebuilt.weights[combo] = float(norm[key])

    try:
        dead_cards = [parse_single_card(c) for c in sorted(dead)]
    except Exception:                                      # noqa: BLE001
        return {"present": False, "unknown": ["dead_card_parse_failed"]}

    surviving = rebuilt.remove_cards(dead_cards) if dead_cards else rebuilt
    prior_total = sum(float(v) for v in rebuilt.weights.values())
    post_total = sum(float(v) for v in surviving.weights.values())

    scale = 1.0
    lik_src = "not_supplied"
    if weights:
        last = weights.get("_last_action")
        mult = weights.get(str(last))
        if isinstance(mult, (int, float)) and mult >= 0:
            scale = float(mult)
            lik_src = "explicit_action_weights"

    return {
        "present": True,
        "combos_prior": len(rebuilt.weights),
        "combos_posterior": len(surviving.weights),
        "combos_removed_by_known_cards": len(rebuilt.weights) - len(surviving.weights),
        "mass_prior": round(prior_total, 6),
        "mass_posterior": round(post_total * scale, 6),
        "mass_removed_by_known_cards": round(prior_total - post_total, 6),
        "combo_class_removal_note": (
            "combo 类别按牌面组合计（如 A2s）；已知牌通常按**花色分配**削减权重，"
            "仅在整类全部不可用时才减少 combo 类别数"),
        "mass_conserved_up_to_card_removal": post_total <= prior_total + 1e-9,
        "likelihood_source": lik_src,
        "likelihood_multiplier": round(scale, 6),
        "posterior": {combo_name(c): round(float(w), 6)
                      for c, w in sorted(surviving.weights.items())},
        "source": "domain.poker.ranges.Range.remove_cards",
    }


def build_dynamic_belief(state: Any, *, range_model: dict | None = None,
                         opponent_model: dict | None = None,
                         action_weights: dict | None = None,
                         integrity_context: dict | None = None) -> dict:
    """CASH 序贯 belief 更新（**只读、确定性、UNKNOWN-first**）。

    Parameters
    ----------
    state : CASH state（非 CASH → present=False）
    range_model : **必填**，由 `cash_range_model.build_range_model` 产出
    opponent_model : 可选，由 `cash_opponent_model.build_opponent_model` 产出
    action_weights : 可选，显式似然 {action: multiplier}
    """
    if not _is_cash(state):
        return _stub("not_cash_mode_fail_closed")
    if not isinstance(range_model, dict):
        return _stub("range_model_required")
    rm = range_model
    if not rm.get("present"):
        return _stub(f"range_model_unavailable:{rm.get('reason', 'unknown')}")

    ctx = integrity_context
    mode = (ctx.get("mode") if isinstance(ctx, dict)
            else (state.get("game_type") if isinstance(state, dict) else None))

    hero_id, unknown = _hero_id(state)
    unknown.extend(_seat_sanity(state))
    opp_hist, hero_hist, hist_unknown = _split_history(state, hero_id)
    unknown.extend(hist_unknown)
    known, card_unknown = _known_cards(state, hero_id)
    unknown.extend(card_unknown)

    bad = _validate_cards(known)
    if bad:
        unknown.append(f"invalid_card_literals:{bad}")
    dead = {c for c in known if c not in bad}

    weights = dict(action_weights) if isinstance(action_weights, dict) else None
    if weights is not None and opp_hist:
        weights["_last_action"] = opp_hist[-1]["action"]

    entries = rm.get("model") or {}
    belief: dict[str, dict] = {}
    per_player_unknown: list[str] = []
    for pid, entry in entries.items():
        if not isinstance(entry, dict):
            continue
        post = _posterior(entry, dead, weights)
        if not post.get("present"):
            per_player_unknown.append(f"{pid}:{'|'.join(post.get('unknown') or [])}")
        belief[str(pid)] = {
            "player_id": pid,
            "position": entry.get("position"),
            "initial_range_present": bool(
                (entry.get("initial_range") or {}).get("present")),
            "observed_actions": list((entry.get("observed") or {}).get("actions") or []),
            "posterior": post,
            "evidence_grade": EVIDENCE_GRADE,
            "is_gto": False,
            "is_equilibrium": False,
        }

    if hero_id is not None and not isinstance(hero_id, str):
        unknown.append("hero_id_not_a_string")
    if isinstance(hero_id, str):
        belief.pop(hero_id, None)

    all_unknown = list(unknown) + per_player_unknown
    level = "UNKNOWN" if all_unknown else "MODEL_INTERNAL"

    return {
        "present": True,
        "version": CASH_STRATEGY_DYNAMIC_BELIEF_VERSION,
        "mode": mode,
        "street": state.get("street"),
        "hero_id": hero_id,
        "belief": belief,
        "n_opponents": len(belief),
        "known_cards_removed": sorted(dead),
        "known_cards_count": len(dead),
        "opponent_actions": opp_hist,
        "hero_actions_excluded": hero_hist,
        "hero_actions_in_opponent_history": any(
            hero_id is not None and a.get("player_id") == hero_id
            for a in opp_hist),
        "likelihood_applied": weights is not None,
        "likelihood_source": ("explicit_action_weights" if weights
                              else "not_supplied"),
        "unknown": all_unknown,
        "level": level,
        "conflicts": list((rm.get("conflicts") or [])),
        "integrity": rm.get("integrity") or {},
        "opponent_model_used": bool(
            isinstance(opponent_model, dict) and opponent_model.get("present")),
        "evidence_grade": EVIDENCE_GRADE,
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }
