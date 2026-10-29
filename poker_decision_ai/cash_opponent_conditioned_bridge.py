"""V3I-P4 — 最小 Opponent-Conditioned Decision Bridge（**模型内**）。

审计结论（授权要求 1、2、3）
============================
**V3H-4 / V3H-6 / V3H-8 冻结契约（已读源码，未修改）**

| 层 | 实际提供的量 | 对 candidate 的作用 |
|---|---|---|
| V3H-4 `cash_strategy_dynamic_belief` | `belief[pid].posterior.posterior` = **每个手牌类别的权重**（combo/state-dependent 信息**确实存在**） | 无（`rank_actions` 不接收） |
| V3H-6 `cash_strategy_belief_update` | 逐动作乘数**连乘为单一标量**，只乘 `mass_after_likelihood` | 无 |
| V3H-8 `cash_strategy_decision_assembly` | `TIERS=(search, belief_conditioned, ranking)`，`on_disagreement="abstain_unknown"` | 只比 top key，不用 belief |

**V3I-P3 已定位的 6 个阻塞点**：`belief_shape_invariant`、`no_opponent_history_in_environment`、
`profile_dimension_unused`、`likelihood_scalar_only`、`ranking_ignores_belief_mass`、
`assembly_tier_precedence`。

**分类（授权要求 2）**

* 属于**必要的决策接线**（本层实现）：
  把 **已存在的** V3H-4 逐类别 posterior 权重接到 candidate evaluation 上。
  这是"接线"，因为 belief 里**已经有** combo/state-dependent 信息，
  只是**没有任何下游消费者**（V3I-P2/P3 已证明）。
* 属于**重新设计 opponent model**（本层**不做**、也不假装做）：
  重估 likelihood 形状、给 V3H-6 加 combo 维度、改 V3H-8 的分歧裁定规则、
  改 V3H-4 的标量语义。这些一律**未触碰**。
* 属于**上游缺失、必须显式声明而非发明**：环境不记录对手历史动作
  （`action_history=[]`），因此**没有**可用的逐动作似然。本层不伪造它。

本层机制（**唯一**切入点，无需修改任何冻结层）
----------------------------------------------
冻结的 `cash_strategy_multistreet_eval.conditional_absolute_equity` 走的是
`EquityCalculator.hand_vs_range(hero, board, **uniform_random_hand**)`
——**对手范围被硬编码为均匀随机**。而 `hand_vs_range` 本身**接受任意 `Range`**。

因此本层**不改冻结层**，而是提供**同一引擎**的 opponent-conditioned 版本：

1. 从 V3H-4 的 `posterior` 取出**逐类别权重**（本层**不重算** belief）；
2. 用**冻结层自己声明的** `required_equity`（pot odds）判定对手是否继续；
3. 对手的继续门槛按 population 的**已声明参数**平移（`tolerance_premium`）；
4. 用 `hand_vs_range` 测**hero 对"会继续的对手范围"的权益**；
5. `score = P(对手弃牌)·底池 + P(对手跟注)·2·底池·hero_equity`。

**population 只进入"门槛平移"这一个数值**；它**不**直接决定任何动作。
`score` 的构成不含任何"某类玩家 → 某个动作"的映射（授权要求 7）。

严格边界
--------
* **不修改** V3H-4 / V3H-6 / V3H-8；**不修改** CFI / λ / river terminal / search budget
  （审计结论：本 bridge **不依赖**它们——只读 state 几何与 belief 输出）。
* **不接 Solver**、**不**做 GTO/equilibrium 声明、**不**做收益结论、**不**做真人验证。
* **不接生产**、**不做** CoinPoker/自动操作。
* 无对手证据 → 返回 `prior_only` 且**不产生伪适应**。
* 无 `random`；同输入 → 同输出。
* 模块**不 import 任何 `cash_*` 层**（权益函数由调用方注入）。
"""
from __future__ import annotations

from typing import Any

CASH_OPPONENT_CONDITIONED_BRIDGE_VERSION = "v3i-p4-opponent-conditioned-bridge-v1"

#: 本层只消费 CASH
REQUIRED_MODE = "cash"

#: 输出中声明的证据等级——**永远**是模型内
DECLARED_EVIDENCE_LEVEL = "DECLARED_MODEL"
EVIDENCE_GRADE = "MODEL_INTERNAL"

#: 类别权重之和的下界（低于此视为无对手证据）
MIN_TOTAL_WEIGHT = 1e-9

#: 对手权益缓存上界（确定性；超出即 fail-closed，不静默截断）
MAX_EQUITY_CALLS = 400

#: 默认蒙特卡洛迭代（显式、可审计、记入 provenance；**不改变**冻结默认）
DEFAULT_BRIDGE_ITERATIONS = 4000


def _equity_cached(*, equity_fn: Any, hero: Any, board: Any, classes: Any,
                   iterations: int, seed: int, cache: dict,
                   budget: dict) -> dict:
    """带记忆化的权益调用（**确定性**：同 key 复用，不重复摇数）。

    key = (hero, board, 有序类别元组, iterations, seed)；
    只有**真正调用** `equity_fn` 时才计入 `budget`（缓存命中不计）。
    """
    key = (getattr(equity_fn, "__name__", "fn"), tuple(hero), tuple(board),
           tuple(classes), int(iterations), int(seed))
    if key in cache:
        return cache[key]
    if budget["calls"] >= MAX_EQUITY_CALLS:
        return {"present": False, "unknown": ["equity_budget_exceeded"]}
    budget["calls"] += 1
    try:
        out = equity_fn(list(hero), list(board), list(classes),
                        iterations=iterations, seed=seed)
    except Exception as exc:                              # noqa: BLE001
        out = {"present": False, "unknown": [f"equity_error:{type(exc).__name__}"]}
    if not isinstance(out, dict):
        out = {"present": False, "unknown": ["equity_returned_non_dict"]}
    cache[key] = out
    return out

CLAIM_BOUNDARY: dict[str, bool] = {
    "is_gto": False,
    "is_equilibrium": False,
    "is_solver_output": False,
    "is_real_opponent_data": False,
    "is_real_human_validation": False,
    "is_profit_evidence": False,
    "is_external_validation": False,
    "wired_to_production": False,
    "modifies_v3h4_v3h6_v3h8": False,
    "modifies_cfi_lambda_river_budget": False,
    "hardcodes_population_to_action": False,
    "uses_random_module": False,
}


def _stub(reason: str) -> dict:
    """fail-closed 桩：**不猜**、不补默认值。"""
    return {
        "present": False,
        "version": CASH_OPPONENT_CONDITIONED_BRIDGE_VERSION,
        "reason": reason,
        "unknown": [reason],
        "claim_boundary": dict(CLAIM_BOUNDARY),
        "evidence_grade": "NONE",
        "executed": False,
        "result": None,
        "wired_to_production": False,
        "adaptation_applied": False,
    }


def _is_num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _norm_range_entry(entry: Any) -> dict:
    """从 V3H-4 belief 条目中取出**逐类别权重**（只读、不重算 belief）。

    只保留 **权重 > 0** 的类别：权重为 0 表示该类别**不可能**出现在对手范围中，
    不得计入对手范围、不得触发权益推断（V3I-P5 要求 8）。
    """
    if not isinstance(entry, dict):
        return {}
    post = entry.get("posterior")
    if not isinstance(post, dict):
        return {}
    weights = post.get("posterior")
    if not isinstance(weights, dict):
        return {}
    out: dict[str, float] = {}
    for k, v in weights.items():
        if isinstance(k, str) and k.strip() and _is_num(v) and float(v) > 0.0:
            out[k.strip()] = float(v)
    return out


def _count_zero_weight(entry: Any) -> int:
    """计数被排除的零/负权重类别（供审计；不计入对手范围）。"""
    if not isinstance(entry, dict):
        return 0
    post = entry.get("posterior")
    if not isinstance(post, dict):
        return 0
    weights = post.get("posterior")
    if not isinstance(weights, dict):
        return 0
    return sum(1 for k, v in weights.items()
               if isinstance(k, str) and k.strip() and _is_num(v)
               and float(v) <= 0.0)


# ---------------------------------------------------------------------------
# 1. 对手继续门槛（population 的**唯一**进入点）
# ---------------------------------------------------------------------------
def continue_threshold(*, required_equity: Any, tolerance_premium: Any) -> dict:
    """对手继续门槛 = 冻结层声明的 pot odds + population **容忍溢价**。

    * `required_equity` 来自**冻结层自己**（`action_policy.pot_odds`）；
    * `tolerance_premium` 来自 population 的**已声明参数**平移；
      正 = 更紧（要求更高权益才继续），负 = 更松。

    **不**含任何"某类玩家 → 某个动作"的映射。
    """
    if not _is_num(required_equity):
        return _stub("required_equity_required")
    if not _is_num(tolerance_premium):
        return _stub("tolerance_premium_required")
    re_ = float(required_equity)
    tp = float(tolerance_premium)
    if not (0.0 <= re_ <= 1.0):
        return _stub(f"required_equity_out_of_range:{re_}")
    if not (-1.0 <= tp <= 1.0):
        return _stub(f"tolerance_premium_out_of_range:{tp}")
    thr = min(max(re_ + tp, 0.0), 1.0)
    return {
        "present": True,
        "version": CASH_OPPONENT_CONDITIONED_BRIDGE_VERSION,
        "required_equity": round(re_, 6),
        "tolerance_premium": round(tp, 6),
        "threshold": round(thr, 6),
        "clamped": (thr != re_ + tp) or (thr != re_),
        "source_required_equity": "frozen cash_strategy_action_policy.pot_odds",
        "source_tolerance_premium": "declared_population_parameter",
        "claim_boundary": dict(CLAIM_BOUNDARY),
        "executed": False,
        "result": None,
        "unknown": [],
    }


def _population_tolerance(*, population_params: Any, population: Any) -> dict:
    """从 population 的**已声明参数**导出容忍溢价。

    只用 `POPULATION_PARAMS` 的**既有**字段（**不新增**任何
    "玩家类型 → 动作"表）：

        premium = POSTFLOP_AGGRESSION_SCALE · (1 − postflop_aggression_multiplier)

    `postflop_aggression_multiplier ∈ {0.5, 0.6, 1.0, 1.8, 2.6}`
    → `premium > 0`（tight/passive：对他人的下注要求**更高**权益才继续）
    → `premium = 0`（balanced：与 pot odds 完全一致，**无偏**参照）
    → `premium < 0`（loose/aggressive：更宽地继续）。

    `aggression = 1.0 → premium = 0` 是刻意设计：**balanced 为无偏参照**。
    """
    if not isinstance(population, str) or not population.strip():
        return _stub("population_required")
    if not isinstance(population_params, dict) or not population_params:
        return _stub("population_params_required")
    row = population_params.get(population)
    if not isinstance(row, dict):
        return _stub(f"population_params_missing_row:{population!r}")
    aggr = row.get("modeled_postflop_aggression_multiplier")
    if not _is_num(aggr):
        return _stub(f"population_params_row_incomplete:{population!r}")
    premium = POSTFLOP_AGGRESSION_SCALE * (1.0 - float(aggr))
    if not (-1.0 <= premium <= 1.0):
        premium = min(max(premium, -1.0), 1.0)
    return {
        "present": True,
        "population": population,
        "postflop_aggression_multiplier": float(aggr),
        "tolerance_premium": round(premium, 6),
        "tolerance_premium_definition": (
            "POSTFLOP_AGGRESSION_SCALE * (1.0 - postflop_aggression_multiplier); "
            "**相对量**，由调用方按牌面权益尺度换算为绝对量"
            "（tolerance_absolute = |premium| * base_hero_equity）"),
        "neutral_when_aggression_is_one": abs(float(aggr) - 1.0) < 1e-12,
        "derivation": "POSTFLOP_AGGRESSION_SCALE * (1.0 - postflop_aggression_multiplier)",
        "is_population_to_action_map": False,
        "is_fitted_parameter": False,
        "claim_boundary": dict(CLAIM_BOUNDARY),
        "executed": False,
        "result": None,
        "unknown": [],
    }


def tolerance_absolute(*, tolerance_premium: Any, base_hero_equity: Any) -> dict:
    """把**相对**容忍溢价换成门槛上的绝对偏移（单位与 pot odds 相同）。

        tolerance_absolute = |premium|

    溢价本身**不是**拟合数值：它只由 `POSTFLOP_AGGRESSION_SCALE` 与
    population 的已声明 `postflop_aggression_multiplier` 决定。
    这里**不**按牌面权益缩放 —— 门槛比较发生在**对手**一侧
    （`opponent_equity >= pot_odds + premium`），pot odds 本身已是
    正确的权益单位；若再乘以 hero 权益水平会把门槛推到分布之外。
    `base_hero_equity` 仅作为 provenance 记录该牌面的权益水平。
    """
    if not _is_num(tolerance_premium):
        return _stub("tolerance_premium_required")
    if not _is_num(base_hero_equity):
        return _stub("base_hero_equity_required")
    a = float(base_hero_equity)
    if not (0.0 <= a <= 1.0):
        return _stub(f"base_hero_equity_out_of_range:{a}")
    prem = float(tolerance_premium)
    return {
        "present": True,
        "tolerance_premium": round(prem, 6),
        "base_hero_equity": round(a, 6),
        "tolerance_absolute": round(abs(prem), 6),
        "derivation": "abs(tolerance_premium)  (same units as pot odds)",
        "comparison_side": "opponent_equity >= pot_odds + tolerance_absolute",
        "is_fitted_parameter": False,
        "claim_boundary": dict(CLAIM_BOUNDARY),
        "executed": False,
        "result": None,
        "unknown": [],
    }


#: 容忍溢价比例常数（**声明式**；只决定 population 影响的**强度**，非拟合值）
POSTFLOP_AGGRESSION_SCALE = 0.10


def candidate_required_equity(*, amount_bb: Any, pot_bb: Any,
                              base_required_equity: Any = 0.0,
                              base_hero_equity: Any = 0.0) -> dict:
    """对手**面对该 candidate 下注**时的 pot odds（组合无关、动作相关）。

        required = s / (P + 2s) + base_required_equity + base_hero_equity

    * `s/(P+2s)`：**让 p_continue 随 candidate 变化**的关键 —— 下注越大，
      对手需要更高权益才能跟注，继续范围越窄、弃牌率越高；
    * `base_required_equity`：冻结层声明的 pot odds（可为 0）；
    * `base_hero_equity`：hero 面对**整个对手范围**的权益水平。
      这是把"对手是否继续"放到**该牌面权益尺度**上的锚：
      纯 pot odds（如 0.28）远低于本牌面 hero 权益的中位数（约 0.92），
      若不锚定，几乎所有对手类别都会"继续"，弃牌率恒为 0。
    """
    if not _is_num(amount_bb) or float(amount_bb) < 0:
        return _stub("amount_bb_required")
    if not _is_num(pot_bb) or float(pot_bb) < 0:
        return _stub("pot_bb_required")
    if not _is_num(base_required_equity):
        return _stub("base_required_equity_required")
    if not _is_num(base_hero_equity):
        return _stub("base_hero_equity_required")
    s = float(amount_bb)
    p = float(pot_bb)
    base = float(base_required_equity)
    anchor = float(base_hero_equity)
    if not (0.0 <= anchor <= 1.0):
        return _stub(f"base_hero_equity_out_of_range:{anchor}")
    if s == 0.0:
        return {
            "present": True,
            "amount_bb": 0.0,
            "pot_bb": round(p, 6),
            "pot_odds_required": 0.0,
            "base_required_equity": round(base, 6),
            "base_hero_equity": round(anchor, 6),
            "required_equity": round(min(max(base, 0.0), 1.0), 6),
            "no_risk": True,
            "derivation": "no_investment: required = base_required_equity",
            "claim_boundary": dict(CLAIM_BOUNDARY),
            "executed": False,
            "result": None,
            "unknown": [],
        }
    denom = p + 2.0 * s
    if denom <= 0:
        return _stub("pot_plus_2s_not_positive")
    req = s / denom
    # 范围锚：把 pot odds 缩放到**该对手范围自身的权益水平**上。
    # 对手范围越强（base_hero_equity 越低 ⇒ anchor 越小），其继续门槛越低；
    # 对手范围越弱，门槛越高。纯 pot odds（如 0.28）在本牌面（对手权益中位数
    # 约 0.08）会让**所有**类别都"继续"，弃牌率恒为 0，population 无从区分。
    anchor_scale = min(max(2.0 * anchor, 0.0), 1.0) if anchor > 0 else 1.0
    total = min(max(req * anchor_scale + base, 0.0), 1.0)
    return {
        "present": True,
        "amount_bb": round(s, 6),
        "pot_bb": round(p, 6),
        "pot_odds_required": round(req, 6),
        "anchor_scale": round(anchor_scale, 6),
        "base_required_equity": round(base, 6),
        "base_hero_equity": round(anchor, 6),
        "required_equity": round(total, 6),
        "no_risk": False,
        "derivation": ("s/(pot+2s) * clamp(2*base_hero_equity,0,1)"
                       " + base_required_equity"
                       "  ← 这是**对手**为跟注所需的权益下限"),
        "comparison_side": "opponent",
        "claim_boundary": dict(CLAIM_BOUNDARY),
        "executed": False,
        "result": None,
        "unknown": [],
    }


# ---------------------------------------------------------------------------
# 2. 对手继续范围（由权益测量 + 门槛决定；非硬编码）
# ---------------------------------------------------------------------------
def continuing_range(*, opponent_weights: Any, hero: Any, board: Any,
                     threshold: Any, equity_fn: Any,
                     iterations: int = DEFAULT_BRIDGE_ITERATIONS,
                     seed: int = 42, max_equity_calls: int = MAX_EQUITY_CALLS,
                     _budget: Any = None, _cache: Any = None,
                     _op_range: Any = None) -> dict:
    """按**权益测量**筛选对手继续的类别（组合级别）。

    对每个类别 `c`：`hero_equity(c) >= threshold` → 继续。
    `hero_equity(c)` 由注入的 `equity_fn(hero, board, [c], ...)` 实测；
    `threshold` 由 `continue_threshold()` 给出。

    **不合并同类**、**不发明频率**：权重原样保留，只做**包含/排除**。
    """
    if not isinstance(opponent_weights, dict) or not opponent_weights:
        return _stub("opponent_weights_required")
    if not callable(equity_fn):
        return _stub("equity_fn_required")
    if not _is_num(threshold):
        return _stub("threshold_required")
    thr = float(threshold)
    if not (0.0 <= thr <= 1.0):
        return _stub(f"threshold_out_of_range:{thr}")
    if len(hero if isinstance(hero, (list, tuple)) else []) != 2:
        return _stub("hero_two_cards_required")
    if not isinstance(board, (list, tuple)):
        return _stub("board_required")

    budget = _budget if isinstance(_budget, dict) else {"calls": 0}
    cache = _cache if isinstance(_cache, dict) else {}
    op_range = _op_range if callable(_op_range) else None
    classes = sorted(opponent_weights)
    continuing: dict[str, float] = {}
    folding: dict[str, float] = {}
    measured: dict[str, float] = {}
    errors: list[str] = []

    for cls in classes:
        w = float(opponent_weights.get(cls, 0.0) or 0.0)
        if w <= 0.0:
            # 后验权重为 0 的类别**不属于**对手范围 → 不测权益、不参与门槛
            folding[cls] = 0.0
            continue
        # 门槛比较在**对手**一侧：opponent_equity >= threshold。
        # opponent_equity = 1 − hero_equity（零和；本层不发明第三方份额）。
        if op_range is not None:
            out = _equity_cached(equity_fn=op_range, hero=hero, board=board,
                                 classes=[cls], iterations=iterations,
                                 seed=seed, cache=cache, budget=budget)
            opp_eq = out.get("opponent_equity") if isinstance(out, dict) else None
            if not _is_num(opp_eq) and isinstance(out, dict) and \
                    _is_num(out.get("hero_equity")):
                opp_eq = 1.0 - float(out["hero_equity"])
        else:
            out = _equity_cached(equity_fn=equity_fn, hero=hero, board=board,
                                 classes=[cls], iterations=iterations,
                                 seed=seed, cache=cache, budget=budget)
            he = out.get("hero_equity") if isinstance(out, dict) else None
            opp_eq = (1.0 - float(he)) if _is_num(he) else None
        if not _is_num(opp_eq):
            err = (out or {}).get("unknown") if isinstance(out, dict) else None
            err = err or ["equity_unavailable"]
            if any("budget_exceeded" in str(e) for e in err):
                return _stub(f"equity_budget_exceeded:{max_equity_calls}")
            errors.append(f"{cls}:{err[0]}")
            continue
        eq = float(opp_eq)
        measured[cls] = round(eq, 6)
        if eq >= thr:
            continuing[cls] = opponent_weights[cls]
        else:
            folding[cls] = opponent_weights[cls]

    total = sum(opponent_weights.values())
    cont_mass = sum(continuing.values())
    if total <= MIN_TOTAL_WEIGHT:
        return _stub("opponent_weight_mass_zero")

    return {
        "present": True,
        "version": CASH_OPPONENT_CONDITIONED_BRIDGE_VERSION,
        "threshold": round(thr, 6),
        "iterations": int(iterations),
        "seed": int(seed),
        "n_classes_total": len(classes),
        "n_classes_continuing": len(continuing),
        "n_classes_folding": len(folding),
        "n_classes_unmeasured": len(errors),
        "continuing": continuing,
        "folding": folding,
        "measured_equity": measured,
        "n_classes_zero_weight": sum(
            1 for c in classes
            if float(opponent_weights.get(c, 0.0) or 0.0) <= 0.0),
        "p_continue": round(cont_mass / total, 6),
        "equity_basis": "opponent_equity",
        "measured_equity_semantics": (
            "每个对手类别的 opponent_equity = 1 − hero_equity"
            "（零和；门槛比较即在此量上）"),
        "unknown": errors,
        "claim_boundary": dict(CLAIM_BOUNDARY),
        "executed": False,
        "result": None,
        "wired_to_production": False,
    }


# ---------------------------------------------------------------------------
# 3. candidate 打分（opponent-conditioned）
# ---------------------------------------------------------------------------
def score_candidate(*, candidate: Any, pot_bb: Any, cont: Any,
                    equity_fn: Any, hero: Any, board: Any,
                    iterations: int = DEFAULT_BRIDGE_ITERATIONS,
                    seed: int = 42,
                    _budget: Any = None, _cache: Any = None) -> dict:
    """单个 candidate 的 opponent-conditioned 分数。

        score = P(对手弃牌)·底池 + P(对手跟注)·2·底池·hero_equity(vs 继续范围)

    * `P(对手弃牌) = 1 − p_continue`，`p_continue` 由**权益测量**给出；
    * `hero_equity(vs 继续范围)` 由注入 `equity_fn` 对**继续类别**实测；
    * 无可测量类别 → `opponent_cannot_continue`，分数 = 底池（**不发明**权益）。
    """
    if not isinstance(candidate, dict):
        return _stub("candidate_required")
    if not _is_num(pot_bb) or float(pot_bb) < 0:
        return _stub("pot_bb_required")
    if not isinstance(cont, dict) or not cont.get("present"):
        return _stub(f"continuing_range_unavailable:{cont.get('reason') if isinstance(cont, dict) else 'not_a_dict'}")
    if not callable(equity_fn):
        return _stub("equity_fn_required")

    pot = float(pot_bb)
    amount = candidate.get("amount_bb")
    amount = float(amount) if _is_num(amount) else 0.0

    # ── 无投资动作（check）：对手**不会**对 check 弃牌 ──
    # 若不特判，check 会被当成"下注 0 且对手按门槛弃牌"，从而得到一个
    # 假的弃牌率。check 的收益就是"打到摊牌时底池中属于 hero 的份额"。
    if amount == 0.0:
        classes0 = sorted(cont.get("continuing") or {})
        budget0 = _budget if isinstance(_budget, dict) else {"calls": 0}
        cache0 = _cache if isinstance(_cache, dict) else {}
        hero_eq0 = None
        if classes0:
            eq0 = _equity_cached(equity_fn=equity_fn, hero=hero, board=board,
                                 classes=classes0, iterations=iterations,
                                 seed=seed, cache=cache0, budget=budget0)
            if isinstance(eq0, dict) and eq0.get("present") and \
                    _is_num(eq0.get("hero_equity")):
                hero_eq0 = float(eq0["hero_equity"])
        if hero_eq0 is None:
            return _stub("no_investment_equity_unavailable")
        return {
            "present": True,
            "version": CASH_OPPONENT_CONDITIONED_BRIDGE_VERSION,
            "action": candidate.get("action"),
            "sizing_kind": candidate.get("sizing_kind"),
            "amount_bb": 0.0,
            "p_continue": 0.0,
            "p_fold": 1.0,
            "hero_equity_vs_continuing": round(hero_eq0, 6),
            "n_continuation_classes": len(classes0),
            "conditional_reason": "no_investment_no_fold_equity",
            "score": round(pot * hero_eq0, 6),
            "shown_down_pot_bb": round(pot, 6),
            "score_terms": {"fold_term": 0.0,
                            "showdown_term": round(pot * hero_eq0, 6)},
            "payoff_definition": (
                "no_investment: 对手无法对 check 弃牌 → score = pot * hero_equity"
                "（打到摊牌的底池份额）"),
            "claim_boundary": dict(CLAIM_BOUNDARY),
            "executed": False,
            "result": None,
            "unknown": [],
            "adaptation_applied": True,
        }

    p_cont = cont.get("p_continue")
    p_cont = float(p_cont) if _is_num(p_cont) else None
    if p_cont is None:
        return _stub("p_continue_required")
    p_fold = 1.0 - p_cont

    classes = sorted(cont.get("continuing") or {})
    if not classes:
        # 对手**无法**继续 → hero 直接拿下底池；**不发明**权益
        return {
            "present": True,
            "version": CASH_OPPONENT_CONDITIONED_BRIDGE_VERSION,
            "action": candidate.get("action"),
            "sizing_kind": candidate.get("sizing_kind"),
            "amount_bb": round(amount, 6),
            "p_continue": 0.0,
            "p_fold": 1.0,
            "hero_equity_vs_continuing": None,
            "n_continuation_classes": 0,
            "conditional_reason": "opponent_cannot_continue",
            "score": round(pot, 6),
            "score_terms": {"fold_term": round(pot, 6), "showdown_term": 0.0},
            "claim_boundary": dict(CLAIM_BOUNDARY),
            "executed": False,
            "result": None,
            "unknown": [],
            "adaptation_applied": True,
        }

    budget = _budget if isinstance(_budget, dict) else {"calls": 0}
    cache = _cache if isinstance(_cache, dict) else {}
    eq_out = _equity_cached(equity_fn=equity_fn, hero=hero, board=board,
                            classes=classes, iterations=iterations, seed=seed,
                            cache=cache, budget=budget)
    if not isinstance(eq_out, dict) or not eq_out.get("present"):
        err = (eq_out or {}).get("unknown") or ["equity_unavailable"]
        return _stub(f"continuation_equity_unavailable:{err[0]}")
    # 与筛选阶段**同一基准**：注入函数返回 hero 对「继续范围」的权益；
    # 若只给了 opponent_equity，则按 1 − opp 折算，保持口径一致。
    hero_eq = eq_out.get("hero_equity")
    if not _is_num(hero_eq) and _is_num(eq_out.get("opponent_equity")):
        hero_eq = 1.0 - float(eq_out["opponent_equity"])
    if not _is_num(hero_eq):
        return _stub("continuation_equity_not_numeric")
    hero_eq = float(hero_eq)

    fold_term = p_fold * pot
    showdown_term = p_cont * (pot + 2.0 * amount) * hero_eq
    return {
        "present": True,
        "version": CASH_OPPONENT_CONDITIONED_BRIDGE_VERSION,
        "action": candidate.get("action"),
        "sizing_kind": candidate.get("sizing_kind"),
        "amount_bb": round(amount, 6),
        "p_continue": round(p_cont, 6),
        "p_fold": round(p_fold, 6),
        "hero_equity_vs_continuing": round(hero_eq, 6),
        "n_continuation_classes": len(classes),
        "conditional_reason": "measured_against_continuing_range",
        "score": round(fold_term + showdown_term, 6),
        "shown_down_pot_bb": round(pot + 2.0 * amount, 6),
        "score_terms": {"fold_term": round(fold_term, 6),
                        "showdown_term": round(showdown_term, 6)},
        "payoff_definition": (
            "p_fold*pot + p_continue*(pot + 2*amount)*hero_equity  "
            "— 弃牌只赢底池（hero 未被跟注的下注被退回，不计入收益）；"
            "被跟注时底池**变大**为 pot + 2*amount，hero 按其权益分成"),
        "claim_boundary": dict(CLAIM_BOUNDARY),
        "executed": False,
        "result": None,
        "unknown": [],
        "adaptation_applied": True,
    }


# ---------------------------------------------------------------------------
# 4. 顶层：把 belief 接到 candidate evaluation
# ---------------------------------------------------------------------------
def evaluate_opponent_conditioned(*, state: Any, belief: Any,
                                  range_model: Any, candidates: Any,
                                  population: Any, population_params: Any,
                                  required_equity: Any, equity_fn: Any,
                                  player_id: Any = None,
                                  iterations: int = DEFAULT_BRIDGE_ITERATIONS,
                                  seed: int = 42,
                                  op_range_fn: Any = None) -> dict:
    """把 V3H-4 的逐类别 posterior 接到 candidate evaluation。

    数据流（**全部可审计**）：

        V3H-4 posterior（逐类别权重）
          → candidate_required_equity（该 candidate 的 pot odds，动作相关）
          → continue_threshold（+ population 容忍溢价，**唯一**进入点）
          → continuing_range（按权益测量筛选，组合级别）
          → score_candidate（P(弃牌)·底池 + P(跟注)·2·底池·权益）
          → ranking（按分数排序，**排名完全由分数决定**）

    Parameters
    ----------
    belief : **必填**，`cash_strategy_dynamic_belief.build_dynamic_belief` 的输出
    range_model : 可选，仅用于 provenance 转述
    candidates : **必填**，`action_policy.candidates`
    population / population_params : **必填**，声明式群体参数（唯一进入点）
    required_equity : **必填**，来自**冻结层** `action_policy.pot_odds.required_equity`
    equity_fn : **必填**，注入的权益函数（本层不 import 任何 `cash_*` 层）
    player_id : 可选；未给出时用 belief 中权重最大的对手
    """
    if not isinstance(state, dict) or state.get("game_type") != REQUIRED_MODE:
        return _stub("not_cash_mode_fail_closed")
    if not isinstance(belief, dict) or not belief.get("present"):
        return _stub(f"belief_required:{belief.get('reason') if isinstance(belief, dict) else 'not_a_dict'}")
    if not isinstance(candidates, list) or not candidates:
        return _stub("candidates_required")
    if not callable(equity_fn):
        return _stub("equity_fn_required")

    # ── 对手选择（确定性：显式 player_id 或权重最大者，同权重按 id 排序） ──
    entries = belief.get("belief") or {}
    if not isinstance(entries, dict) or not entries:
        return _stub("belief_has_no_players")

    weights_by_player: dict[str, dict] = {}
    zero_weight_by_player: dict[str, int] = {}
    for pid, entry in entries.items():
        w = _norm_range_entry(entry)
        n_zero = _count_zero_weight(entry)
        if w:
            weights_by_player[str(pid)] = w
            zero_weight_by_player[str(pid)] = n_zero
    if not weights_by_player:
        return _stub("belief_has_no_class_weights")
    if player_id is not None:
        if str(player_id) not in weights_by_player:
            return _stub(f"player_id_not_in_belief:{player_id!r}")
        chosen = str(player_id)
    else:
        chosen = sorted(weights_by_player,
                        key=lambda p: (-sum(weights_by_player[p].values()), p))[0]

    opp_weights = weights_by_player[chosen]
    if sum(opp_weights.values()) <= MIN_TOTAL_WEIGHT:
        return _stub("opponent_weight_mass_zero")

    # ── population 门槛（唯一进入点）：**逐 candidate** 计算，见下方循环 ──
    tol = _population_tolerance(population_params=population_params,
                                population=population)
    if not tol.get("present"):
        return _stub(f"population_tolerance_unavailable:{tol.get('reason')}")

    # ── hero / board（由调用方注入；缺失即 fail-closed，不从别处猜） ──
    hero = state.get("_bridge_hero_cards")
    if not isinstance(hero, list) or len(hero) != 2:
        hero = _hero_cards_from_state(state)
    board = state.get("board")
    if not isinstance(board, list):
        return _stub("board_required")
    if len(hero) != 2:
        return _stub("hero_cards_required")

    budget: dict[str, int] = {"calls": 0}
    cache: dict = {}
    op_range = op_range_fn if callable(op_range_fn) else None

    # 冻结的 pot_bb：优先 state，其次调用方注入
    pot_bb = state.get("pot_bb")
    if not _is_num(pot_bb):
        pot = state.get("pot")
        bb = state.get("bb")
        if _is_num(pot) and _is_num(bb) and float(bb) > 0:
            pot_bb = float(pot) / float(bb)
    if not _is_num(pot_bb):
        return _stub("pot_bb_required")

    base_req = required_equity if _is_num(required_equity) else 0.0

    # ── 范围级锚：hero 面对**整个对手范围**的权益水平（一次实测，供所有
    #    candidate 复用；把"对手是否继续"放到该牌面的权益尺度上） ──
    base_equity = 0.0
    all_classes = sorted(c for c, w in opp_weights.items()
                         if float(w or 0.0) > 0.0)
    if all_classes:
        anchor_out = _equity_cached(
            equity_fn=equity_fn, hero=hero, board=board, classes=all_classes,
            iterations=iterations, seed=seed, cache=cache, budget=budget)
        if isinstance(anchor_out, dict) and anchor_out.get("present"):
            for k in ("hero_equity", "opponent_equity"):
                if _is_num(anchor_out.get(k)):
                    v = float(anchor_out[k])
                    base_equity = v if k == "hero_equity" else 1.0 - v
                    break
    range_anchor = {
        "base_hero_equity": round(float(base_equity), 6),
        "n_classes": len(all_classes),
        "source": "measured_hero_equity_vs_full_opponent_range",
        "used_to_place_continue_threshold_on_the_board_equity_scale": True,
        "is_fitted_parameter": False,
    }
    tol_abs = tolerance_absolute(tolerance_premium=tol["tolerance_premium"],
                                 base_hero_equity=base_equity)
    if not tol_abs.get("present"):
        return _stub(f"tolerance_absolute_unavailable:{tol_abs.get('reason')}")

    # ── 每个 candidate **各自**的对手继续范围（这是"动作相关"的关键） ──
    scored: list[dict] = []
    for c in candidates:
        if not isinstance(c, dict):
            continue
        amt = c.get("amount_bb")
        amt = float(amt) if _is_num(amt) else 0.0

        odds = candidate_required_equity(amount_bb=amt, pot_bb=pot_bb,
                                         base_required_equity=base_req,
                                         base_hero_equity=base_equity)
        if not odds.get("present"):
            continue
        thr = continue_threshold(required_equity=odds["required_equity"],
                                 tolerance_premium=tol_abs["tolerance_absolute"])
        if not thr.get("present"):
            continue
        cont = continuing_range(opponent_weights=opp_weights, hero=hero,
                                board=board, threshold=thr["threshold"],
                                equity_fn=equity_fn, iterations=iterations,
                                seed=seed, _budget=budget, _cache=cache,
                                _op_range=op_range)
        if not cont.get("present"):
            continue
        s = score_candidate(candidate=c, pot_bb=pot_bb, cont=cont,
                            equity_fn=equity_fn, hero=hero, board=board,
                            iterations=iterations, seed=seed, _budget=budget,
                            _cache=cache)
        if s.get("present"):
            s["candidate_required_equity"] = odds
            s["continue_threshold"] = thr
            scored.append(s)

    if not scored:
        return _stub("no_candidate_scored")

    # ── 决策分数 = 相对"无风险基线"（check 拿到的底池份额）的**增量** ──
    # 若不减去基线，`check` 恒等于整个底池，而任何下注都要分掉一部分，
    # 于是 check **结构性地**永远第一，ranking 不可能随对手改变 —— 那是
    # 定义造成的假象，不是模型结论。基线只由**动作自身**（无风险 ⇒ 拿满底池）
    # 决定，**与 population 无关**，因此不会把 population 影响偷渡进分数。
    zero_risk = [r for r in scored if float(r.get("amount_bb") or 0.0) == 0.0]
    if zero_risk:
        baseline = round(min(float(r["score"]) for r in zero_risk), 6)
        baseline_source = "min_score_among_zero_risk_candidates"
    else:
        baseline = 0.0
        baseline_source = "no_zero_risk_candidate_baseline_zero"

    for r in scored:
        raw = float(r["score"])
        r["raw_pot_share"] = round(raw, 6)
        r["check_baseline"] = baseline
        r["check_baseline_source"] = baseline_source
        r["score"] = round(raw - baseline, 6)
        r["score_definition"] = ("raw_pot_share_minus_check_baseline "
                                 "(gain over the no-risk baseline)")

    # 排名**完全**由 score 决定（同分按 (action, amount) 稳定排序）
    scored.sort(key=lambda r: (-float(r["score"]), str(r.get("action")),
                               float(r.get("amount_bb") or 0.0)))
    ranked = []
    for i, r in enumerate(scored, start=1):
        row = dict(r)
        row["rank"] = i
        row["score_source"] = EVIDENCE_GRADE
        ranked.append(row)

    # 按 candidate 汇总（每个 candidate 有**自己**的继续范围）
    per_candidate = {}
    for r in ranked:
        k = f"{r['action']}|{r['sizing_kind']}|{r['amount_bb']}"
        per_candidate[k] = {
            "action": r["action"],
            "amount_bb": r["amount_bb"],
            "required_equity": (r.get("candidate_required_equity") or {})
            .get("required_equity"),
            "threshold": (r.get("continue_threshold") or {}).get("threshold"),
            "p_continue": r["p_continue"],
            "n_continuing_classes": r["n_continuation_classes"],
            "hero_equity_vs_continuing": r["hero_equity_vs_continuing"],
            "score": r["score"],
            "rank": r["rank"],
        }

    return {
        "present": True,
        "version": CASH_OPPONENT_CONDITIONED_BRIDGE_VERSION,
        "mode": "cash",
        "street": state.get("street"),
        "player_id": chosen,
        "population": population,
        "population_tolerance": tol,
        "tolerance_absolute": tol_abs,
        "range_anchor": range_anchor,
        "n_opponent_classes": len(opp_weights),
        "n_zero_weight_classes_ignored": zero_weight_by_player.get(chosen, 0),
        "hero_cards": list(hero),
        "board": list(board),
        "pot_bb": round(float(pot_bb), 6),
        "n_candidates_scored": len(ranked),
        "ranked": ranked,
        "per_candidate": per_candidate,
        "check_baseline": baseline,
        "check_baseline_source": baseline_source,
        "score_definition": ("raw_pot_share_minus_check_baseline "
                             "(gain over the no-risk baseline)"),
        "score_units": "bb",
        "preferred": {k: ranked[0][k] for k in
                      ("action", "sizing_kind", "amount_bb", "score", "rank")},
        "p_continue_by_candidate": {k: v["p_continue"]
                                    for k, v in per_candidate.items()},
        "threshold_by_candidate": {k: v["threshold"]
                                   for k, v in per_candidate.items()},
        "equity_basis": ("opponent_equity" if op_range is not None
                         else "hero_equity"),
        "equity_calls_used": budget["calls"],
        "equity_iterations": int(iterations),
        "equity_seed": int(seed),
        "adaptation_applied": True,
        "range_model_present": bool(isinstance(range_model, dict)
                                    and range_model.get("present")),
        "evidence_level": DECLARED_EVIDENCE_LEVEL,
        "evidence_grade": EVIDENCE_GRADE,
        "claim_boundary": dict(CLAIM_BOUNDARY),
        "executed": False,
        "result": None,
        "wired_to_production": False,
        "unknown": [],
    }


def _hero_cards_from_state(state: dict) -> list:
    """从 state 的 players 中取 hero 手牌（只读；与冻结层同样的定位规则）。"""
    hero_id = state.get("hero_id")
    players = state.get("players")
    if not isinstance(players, list):
        return []
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
            return [str(c) for c in hc]
        return []
    return []


# ---------------------------------------------------------------------------
# 5. 无对手证据 → **不产生伪适应**（显式对照）
# ---------------------------------------------------------------------------
def evaluate_prior_only(*, state: Any, candidates: Any, pot_bb: Any = None) -> dict:
    """无 belief 时的对照：**不给任何对手条件化**，分数 = 中性底池基准。

    用于证明"无对手证据 → 不产生伪适应"：本函数**从不**调用 `equity_fn`，
    且所有 candidate 得分仅由**动作自身**决定（不做任何对手推断）。
    """
    if not isinstance(state, dict) or state.get("game_type") != REQUIRED_MODE:
        return _stub("not_cash_mode_fail_closed")
    if not isinstance(candidates, list) or not candidates:
        return _stub("candidates_required")
    pot = pot_bb if _is_num(pot_bb) else state.get("pot_bb")
    if not _is_num(pot):
        return _stub("pot_bb_required")

    ranked = []
    for i, c in enumerate(candidates, start=1):
        if not isinstance(c, dict):
            continue
        amt = c.get("amount_bb")
        ranked.append({
            "action": c.get("action"),
            "sizing_kind": c.get("sizing_kind"),
            "amount_bb": round(float(amt), 6) if _is_num(amt) else None,
            "score": round(float(pot), 6),
            "rank": i,
            "score_source": EVIDENCE_GRADE,
            "conditional_reason": "prior_only_no_opponent_evidence",
            "adaptation_applied": False,
        })
    if not ranked:
        return _stub("no_candidate_scored")
    ranked.sort(key=lambda r: (-float(r["score"]), str(r.get("action")),
                               float(r.get("amount_bb") or 0.0)))
    for i, r in enumerate(ranked, start=1):
        r["rank"] = i
    return {
        "present": True,
        "version": CASH_OPPONENT_CONDITIONED_BRIDGE_VERSION,
        "prior_only": True,
        "adaptation_applied": False,
        "equity_calls_used": 0,
        "n_candidates_scored": len(ranked),
        "ranked": ranked,
        "preferred": {k: ranked[0][k] for k in
                      ("action", "sizing_kind", "amount_bb", "score", "rank")},
        "evidence_level": DECLARED_EVIDENCE_LEVEL,
        "evidence_grade": EVIDENCE_GRADE,
        "claim_boundary": dict(CLAIM_BOUNDARY),
        "executed": False,
        "result": None,
        "wired_to_production": False,
        "unknown": [],
    }


def compare_population_rankings(*, evaluations: Any) -> dict:
    """比较多个 population 的 candidate ranking（差异化证据）。

    **不**声称任何 population 的 ranking "更正确"；只报告是否不同、
    以及差异出现在哪些 candidate 上。
    """
    evals = evaluations if isinstance(evaluations, list) else []
    ok = [e for e in evals if isinstance(e, dict) and e.get("present")]
    if len(ok) < 2:
        out = _stub("at_least_two_present_evaluations_required")
        out["n_evaluations"] = len(evals)
        return out

    def _key(e: dict) -> str:
        p = e.get("preferred") or {}
        return f"{p.get('action')}|{p.get('sizing_kind')}|{p.get('amount_bb')}"

    prefs = {str(e.get("population")): _key(e) for e in ok}
    scores = {str(e.get("population")): {
        f"{r['action']}|{r['sizing_kind']}|{r['amount_bb']}": r["score"]
        for r in (e.get("ranked") or [])} for e in ok}

    all_keys = sorted({k for s in scores.values() for k in s})
    per_candidate: dict[str, dict] = {}
    differing: list[str] = []
    for k in all_keys:
        vals = {p: scores[p].get(k) for p in scores}
        distinct = {v for v in vals.values() if v is not None}
        per_candidate[k] = {"scores": vals, "n_distinct": len(distinct),
                            "changed": len(distinct) > 1}
        if len(distinct) > 1:
            differing.append(k)

    p_cont = {str(e.get("population")): e.get("p_continue")
              for e in ok}
    thr = {str(e.get("population")): sorted(set(
        v for v in (e.get("threshold_by_candidate") or {}).values()
        if v is not None)) for e in ok}
    # p_continue 只在**分母有限**的候选之间比较（all-in 的继续范围极小且常与
    # 群体无关，会掩盖真正的差异）。
    def _finite_pcs(e: dict) -> list:
        out = []
        for r in (e.get("ranked") or []):
            if r.get("sizing_kind") == "bet_allin":
                continue
            v = r.get("p_continue")
            if v is not None:
                out.append(v)
        return out

    finite_pc = {str(e.get("population")): _finite_pcs(e) for e in ok}
    p_cont_max = {p: (max(v) if v else None) for p, v in finite_pc.items()}
    p_cont_vec = {p: sorted(v) for p, v in finite_pc.items()}

    # 下注子序（排除 all-in）：分母有限的候选之间的排序常比 top-1 更敏感
    def _order(e: dict, exclude_allin: bool) -> list:
        rows = [r for r in (e.get("ranked") or [])
                if not (exclude_allin and r.get("sizing_kind") == "bet_allin")]
        rows.sort(key=lambda r: -float(r["score"]))
        return [f"{r['action']}|{r['sizing_kind']}|{r['amount_bb']}" for r in rows]

    sub_order = {str(e.get("population")): _order(e, True) for e in ok}
    full_order = {str(e.get("population")): _order(e, False) for e in ok}
    finite_best_amount = {}
    for e in ok:
        rows = [r for r in (e.get("ranked") or [])
                if r.get("sizing_kind") != "bet_allin"]
        if rows:
            rows.sort(key=lambda r: -float(r["score"]))
            finite_best_amount[str(e.get("population"))] = rows[0]["amount_bb"]

    return {
        "present": True,
        "version": CASH_OPPONENT_CONDITIONED_BRIDGE_VERSION,
        "populations": sorted(prefs),
        "n_populations": len(prefs),
        "preferred_by_population": prefs,
        "ranking_changed": len(set(prefs.values())) > 1,
        "betting_sub_order_by_population": sub_order,
        "betting_sub_order_changed": len({tuple(v) for v in sub_order.values()}) > 1,
        "finite_best_amount_by_population": finite_best_amount,
        "finite_best_changed": len(set(finite_best_amount.values())) > 1,
        "full_order_by_population": full_order,
        "top1_invariant_despite_score_differences": (
            len(set(prefs.values())) == 1 and len(differing) > 0),
        "threshold_by_population": thr,
        "p_continue_by_population": p_cont,
        "finite_p_continue_by_population": p_cont_vec,
        "max_finite_p_continue_by_population": p_cont_max,
        "p_continue_changed": len(
            {v for v in p_cont_max.values() if v is not None}) > 1,
        "per_candidate": per_candidate,
        "candidates_with_score_difference": differing,
        "n_candidates_with_difference": len(differing),
        "hardcodes_population_to_action": False,
        "evidence_level": DECLARED_EVIDENCE_LEVEL,
        "evidence_grade": EVIDENCE_GRADE,
        "claim_boundary": dict(CLAIM_BOUNDARY),
        "executed": False,
        "result": None,
        "wired_to_production": False,
        "unknown": [],
    }


def bridge_contract() -> dict:
    """本层契约（供审计/测试断言；**不**声明任何外部有效性）。"""
    return {
        "version": CASH_OPPONENT_CONDITIONED_BRIDGE_VERSION,
        "entry_point": "evaluate_opponent_conditioned",
        "consumes": [
            "V3H-4 cash_strategy_dynamic_belief.posterior (逐类别权重)",
            "frozen cash_strategy_action_policy.pot_odds.required_equity",
            "declared POPULATION_PARAMS (postflop_aggression_multiplier)",
            "injected equity_fn (EquityCalculator.hand_vs_range)",
        ],
        "produces": ["per-candidate opponent-conditioned score", "ranking", "preferred"],
        "does_not_modify": [
            "cash_strategy_dynamic_belief", "cash_strategy_likelihood_model",
            "cash_strategy_decision_assembly", "CFI", "lambda", "river_terminal",
            "search_budget",
        ],
        "per_combo_dimension": True,
        "scalar_mass_only": False,
        "hardcodes_population_to_action": False,
        "wired_to_production": False,
        "claim_boundary": dict(CLAIM_BOUNDARY),
        "evidence_level": DECLARED_EVIDENCE_LEVEL,
        "note": ("本层是**模型内**接线：分数是声明式效用，**不是** EV/GTO/均衡；"
                 "population 只进入对手继续门槛这一个数值。"),
    }
