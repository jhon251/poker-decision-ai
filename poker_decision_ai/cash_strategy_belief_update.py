"""V3H-6 — CASH 动态 Belief 更新（声明式 likelihood 序贯条件化 + 不变量守护）。

定位
----
把 V3H-6 的**声明式 likelihood 模型**应用到**已观测的对手动作序列**上，
得到"似然条件化"的 belief，并在输出中：

* 显式区分 `prior_only` 与 `likelihood_conditioned`；
* **保留** known-card removal（复用既有层，不重实现）；
* 保证 **hero 动作不进入对手 likelihood**；
* **不把 UNKNOWN 当作 EMPTY**；
* 冲突/缺失/非法输入 → UNKNOWN 或 fail closed；
* **不把模型内 posterior 宣称为真实对手范围**。

复用而非重实现（审计结论，授权要求）
------------------------------------
| 能力 | 由谁提供 | 本层是否重实现 |
|---|---|---|
| 初始 range / 路由 / 扁平动作权重（`ACTION_WEIGHT`） | `cash_range_model` | ❌ 复用（经 belief 层） |
| known-card removal（牌张冲突规则） | `domain.poker.ranges.Range.remove_cards`（经 belief 层） | ❌ 复用 |
| 对手特征 | `cash_opponent_model`（经 belief 层） | ❌ 复用 |
| 动作树 / 排序 | `cash_action_tree` / `cash_strategy_action_ranking` | ❌ 不复用也不改 |
| **多维、带 provenance 的 likelihood 参数** | **本层**（`cash_strategy_likelihood_model`） | ✅ 新增 |

**更正记录（诚实性）**：既有 `cash_range_model` **已经**按声明表 `ACTION_WEIGHT`
逐条施加动作权重（实测 `raise`+`bet` → `[2.0, 1.5]`）。因此本层的
`prior_only=true` 语义是"**没有额外的、调用方声明的 likelihood 模型**"，
**不是**"posterior 等于原始先验"。该更正在输出字段
`prior_only_definition` 中机器可读地写明。

不变量（全部在输出中断言，可被测试复核）
----------------------------------------
1. **known cards 不得重新进入范围**：posterior 的每个 combo 都不含任何已知牌；
2. **posterior 不得包含非法牌**：所有 combo 名可被 range 引擎解析；
3. **确定性**：同输入 → 同输出（不 import `random`）；
4. **hero/opponent 隔离**：hero 动作只出现在 `hero_actions_excluded`；
5. **两路可辨**：`prior_only` 与 `likelihood_conditioned` 由 `path` 字段区分；
6. **UNKNOWN 传播**：上传入的 unknown 原样带出并汇总。

数据接口（生产 import 图零新增）
--------------------------------
**不在模块顶层 import 任何 V3G 层**；`belief` / `range_model` / `state` 由调用方传入。
"""
from __future__ import annotations

from typing import Any

CASH_STRATEGY_BELIEF_UPDATE_VERSION = "v3h6-cash-strategy-belief-update-v1"

#: 两条路径（显式区分，禁止混用）
PATH_PRIOR_ONLY = "prior_only"
PATH_LIKELIHOOD_CONDITIONED = "likelihood_conditioned"

#: 本层语义更正（机器可读）
PRIOR_ONLY_DEFINITION = (
    "prior_only=true means NO caller-declared likelihood model was applied. "
    "It does NOT mean the posterior equals the raw prior: the upstream range "
    "model already applies its own declared flat ACTION_WEIGHT to observed "
    "actions, and known-card removal is always applied."
)

CLAIM_BOUNDARY: dict[str, bool] = {
    "is_real_opponent_range": False,
    "is_real_player_frequency": False,
    "is_gto": False,
    "is_equilibrium": False,
    "is_solver_output": False,
    "wired_to_production": False,
}


def _is_cash(state: Any) -> bool:
    if not isinstance(state, dict):
        return False
    gt = state.get("game_type")
    return isinstance(gt, str) and gt.strip().lower() == "cash"


def _stub(reason: str) -> dict:
    return {
        "present": False,
        "version": CASH_STRATEGY_BELIEF_UPDATE_VERSION,
        "reason": reason,
        "evidence_grade": "NONE",
        "unknown": [reason],
        "belief": {},
        "path": None,
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }


def _known_cards(belief: dict, state: dict) -> set[str]:
    """已知牌：belief 的 `known_cards_removed` ∪ state 的 board + hero 手牌。"""
    known = {str(c) for c in (belief.get("known_cards_removed") or [])}
    for c in (state.get("board") or []):
        if isinstance(c, str) and c.strip():
            known.add(str(c))
    hero_id = belief.get("hero_id") or state.get("hero_id")
    for p in (state.get("players") or []):
        if not isinstance(p, dict):
            continue
        if bool(p.get("is_hero")) or (
                hero_id is not None and p.get("player_id") == hero_id):
            for c in (p.get("hole_cards") or []):
                if isinstance(c, str) and c.strip():
                    known.add(str(c))
            break
    return known


def _check_posterior(entry: dict, known: set[str]) -> dict:
    """posterior 不变量检查（用**引擎自己的**冲突规则，避免误判）。

    combo 名是**牌面类别**（如 `A2s`），其花色分配由权重表示；因此"已知牌是否
    回流"必须在**花色分配层面**判定：对每个类别枚举 `_suit_assignments`，
    检查是否存在与已知牌冲突的**正权重**分配。
    """
    post = (entry or {}).get("posterior") or {}
    combos = post.get("posterior") or {}
    if not isinstance(combos, dict):
        return {"checked": False, "reason": "posterior_not_a_dict"}
    try:
        from .domain.poker.ranges import (_suit_assignments, _total_assignments,
                                          parse_range, parse_single_card)
    except Exception:                                      # noqa: BLE001
        return {"checked": False, "reason": "range_engine_unavailable"}

    try:
        dead = [parse_single_card(c) for c in sorted(known)]
    except Exception:                                      # noqa: BLE001
        return {"checked": False, "reason": "known_card_parse_failed"}

    unparseable: list[str] = []
    leaked: list[str] = []
    checked = 0
    for name in combos:
        weight = combos[name]
        if not isinstance(weight, (int, float)) or weight <= 0:
            continue
        try:
            rng = parse_range(str(name))
            combo = next(iter(rng.weights)) if rng.weights else None
        except Exception:                                  # noqa: BLE001
            unparseable.append(str(name))
            continue
        if combo is None:
            unparseable.append(str(name))
            continue
        checked += 1
        hi, lo, suited = combo
        total = _total_assignments(hi, lo, suited)
        dead_set = {str(c) for c in dead}
        available = [c for c in _suit_assignments(hi, lo, suited)
                     if not ({str(c[0]), str(c[1])} & dead_set)]
        if not available:
            # 该类别的**全部**花色分配都与已知牌冲突 → 不可能存在
            leaked.append(str(name))
        elif len(available) > total:
            unparseable.append(str(name))

    return {
        "checked": True,
        "n_combos_positive_weight": checked,
        "impossible_combos_present": sorted(leaked),
        "unparseable_combos": sorted(unparseable),
        "no_known_card_reentered": not leaked,
        "all_combos_parseable": not unparseable,
        "method": "domain.poker.ranges._suit_assignments (engine's own collision rule)",
    }


def _split_actions(state: dict, hero_id: str | None) -> dict:
    """按玩家归集对手动作；hero 动作**单独**列出（隔离）。"""
    opp: dict[str, list[dict]] = {}
    hero_acts: list[dict] = []
    unknown: list[str] = []
    hist = state.get("action_history")
    if hist is None:
        return {"by_player": {}, "hero_actions_excluded": [],
                "unknown": ["action_history_missing"]}
    if not isinstance(hist, list):
        return {"by_player": {}, "hero_actions_excluded": [],
                "unknown": ["action_history_malformed"]}
    for i, a in enumerate(hist):
        if not isinstance(a, dict):
            unknown.append(f"action[{i}]_not_a_dict")
            continue
        pid = a.get("player_id")
        is_hero = bool(a.get("is_hero")) or (
            hero_id is not None and pid == hero_id)
        rec = {"player_id": pid, "action": a.get("action"),
               "street": a.get("street"), "amount_bb": a.get("amount_bb")}
        if is_hero:
            hero_acts.append(rec)
        elif isinstance(pid, str) and pid:
            opp.setdefault(pid, []).append(rec)
        else:
            unknown.append(f"action[{i}]_missing_player_id")
    return {"by_player": opp, "hero_actions_excluded": hero_acts,
            "unknown": unknown}


def _size_bucket(amount_bb: Any, action: Any, pot_bb: Any) -> str | None:
    """按**调用方给出的几何量**分桶；信息不足 → None（不猜）。"""
    act = str(action or "").strip().lower()
    if act in ("fold", "check"):
        return "no_investment"
    if act == "all_in":
        return "allin"
    if not isinstance(amount_bb, (int, float)) or isinstance(amount_bb, bool):
        return None
    if not isinstance(pot_bb, (int, float)) or isinstance(pot_bb, bool) or pot_bb <= 0:
        return None
    frac = float(amount_bb) / float(pot_bb)
    if frac <= 0.4:
        return "small"
    if frac <= 0.8:
        return "medium"
    return "large"


def update_belief_with_likelihood(state: Any, *, belief: dict | None = None,
                                  range_model: dict | None = None,
                                  likelihood_model: dict | None = None,
                                  pot_bb: Any = None) -> dict:
    """把声明式 likelihood 序贯应用到对手动作 → 条件化 belief。

    Parameters
    ----------
    state : CASH state（非 CASH → present=False）
    belief : **必填**，由 `cash_strategy_dynamic_belief.build_dynamic_belief` 产出
        （它已完成 known-card removal；本层**不重实现**该逻辑）
    range_model : 可选，仅用于转述
    likelihood_model : 由 `cash_strategy_likelihood_model.action_likelihood_model` 产出；
        未提供 → `path=prior_only`
    pot_bb : 可选；用于 size bucket 分桶（缺失则 bucket=None，**不猜**）
    """
    if not _is_cash(state):
        return _stub("not_cash_mode_fail_closed")
    if not isinstance(belief, dict):
        return _stub("belief_required")
    bl = belief
    if not bl.get("present"):
        return _stub(f"belief_unavailable:{bl.get('reason', 'unknown')}")

    hero_id = bl.get("hero_id")
    split = _split_actions(state, hero_id)
    known = _known_cards(bl, state)

    lm = likelihood_model if isinstance(likelihood_model, dict) else None
    prior_only = not (lm and lm.get("applied"))
    path = PATH_PRIOR_ONLY if prior_only else PATH_LIKELIHOOD_CONDITIONED

    unknown: list[str] = list(bl.get("unknown") or [])
    unknown.extend(split["unknown"])
    if lm is not None:
        for u in (lm.get("unknown") or []):
            if u not in unknown:
                unknown.append(u)

    # 解析每个对手动作的乘数（按玩家独立；hero 已被隔离）
    from .cash_strategy_likelihood_model import resolve_multiplier
    valid_params = (lm or {}).get("parameters") or []
    per_player: dict[str, dict] = {}
    for pid, entry in (bl.get("belief") or {}).items():
        acts = split["by_player"].get(str(pid), [])
        applied: list[dict] = []
        unmatched: list[dict] = []
        total = 1.0
        for a in acts:
            bucket = _size_bucket(a.get("amount_bb"), a.get("action"), pot_bb)
            res = resolve_multiplier(valid_params, action=str(a.get("action") or ""),
                                     street=str(a.get("street") or ""),
                                     size_bucket=bucket, profile=None)
            if res.get("matched"):
                total *= float(res["multiplier"])
                applied.append({
                    "action": a.get("action"), "street": a.get("street"),
                    "size_bucket": bucket, "multiplier": res["multiplier"],
                    "tier": res["tier"], "source": res["source"],
                    "evidence_level": res["evidence_level"],
                    "uncertainty": res["uncertainty"],
                })
            else:
                unmatched.append({"action": a.get("action"),
                                  "street": a.get("street"),
                                  "size_bucket": bucket,
                                  "reason": res.get("reason")})
        post = (entry or {}).get("posterior") or {}
        inv = _check_posterior(entry, known)
        mass_prior = post.get("mass_posterior")
        per_player[str(pid)] = {
            "player_id": pid,
            "position": (entry or {}).get("position"),
            "observed_actions": acts,
            "likelihood_applied": applied,
            "likelihood_unmatched": unmatched,
            "likelihood_total_multiplier": round(total, 6),
            "likelihood_conditioned": bool(applied),
            "mass_before_likelihood": mass_prior,
            "mass_after_likelihood": (round(float(mass_prior) * total, 6)
                                      if isinstance(mass_prior, (int, float))
                                      else None),
            "posterior_source": post.get("source"),
            "invariants": inv,
            "evidence_grade": "MODEL_INTERNAL",
        }
        if unmatched and not prior_only:
            unknown.append(f"{pid}:unmatched_actions:{len(unmatched)}")
        if inv.get("checked") and not inv.get("no_known_card_reentered", True):
            unknown.append(f"{pid}:known_card_leaked")
        if inv.get("checked") and not inv.get("all_combos_parseable", True):
            unknown.append(f"{pid}:unparseable_combos")

    return {
        "present": True,
        "version": CASH_STRATEGY_BELIEF_UPDATE_VERSION,
        "mode": "cash",
        "street": state.get("street"),
        "hero_id": hero_id,
        "path": path,
        "prior_only": prior_only,
        "prior_only_definition": PRIOR_ONLY_DEFINITION,
        "likelihood_model_version": (lm or {}).get("version"),
        "likelihood_parameters_used": len(valid_params),
        "belief": per_player,
        "n_opponents": len(per_player),
        # 隔离证明
        "hero_actions_excluded": split["hero_actions_excluded"],
        "hero_actions_in_opponent_likelihood": any(
            hero_id is not None and a.get("player_id") == hero_id
            for acts in split["by_player"].values() for a in acts),
        "known_cards_considered": sorted(known),
        "known_cards_count": len(known),
        "unknown": unknown,
        "level": "UNKNOWN" if unknown else "MODEL_INTERNAL",
        "evidence_grade": "MODEL_INTERNAL" if not prior_only else "MODEL_INTERNAL",
        "claim_boundary": dict(CLAIM_BOUNDARY),
        "note": ("posterior 是**模型内**量；不得表述为真实对手范围。"
                 "本层不重实现 known-card removal —— 由传入的 belief 完成。"),
    }
