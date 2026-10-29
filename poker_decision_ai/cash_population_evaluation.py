"""V3I-P3 — 多对手类型 / 多行为群体的**模型内**评估能力。

授权目标
--------
建立 "多对手类型 / 多行为群体 → belief/range → candidate/decision" 的
**模型内**评估能力，验证当前 Agent 是否**至少**能在不同 opponent population 下
产生**可解释**的策略差异。

本层新增什么
------------
V3I-P1 的 `cash_decision_environment.POPULATION_PARAMS` 只声明了
"建模的先验乘数"（`modeled_preflop_continue_multiplier` /
`modeled_postflop_aggression_multiplier`），且 V3I-P1 的
`population_action_weights` 把**所有非 fold/check 动作**赋成**同一个标量**。

本层把"群体 → 观测先验"这一步**显式化、可审计化**：

1. `population_propensity()` —— 把声明的群体参数投影成 **per-action 倾向**
   （动作被区分：fold / check / call / bet / raise / all_in **不再是同一个数**）；
2. `population_observation_prior()` —— 由倾向导出
   * `action_weights`（交给冻结的 `cash_strategy_dynamic_belief`），
   * 声明式 likelihood 参数（交给 V3H-6 schema 校验后生成模型，
     再交给冻结的 `cash_strategy_belief_update`）。
3. `evaluate_population()` —— 复用 **V3I-P1 环境 + V3I-P2 真实 pipeline**
   （`cash_decision_loop.run_decision_loop`）跑完整链路，并逐层记录指纹：
   `population → 观测先验 → belief/range → candidate ranking → assembly decision`。
4. `compare_populations()` —— 逐层比较多个群体，回答
   "belief / range / candidate / decision 是否改变"，并在**未改变时定位冻结层**。
5. `likelihood_conflict_probe()` —— UNKNOWN / 冲突语义（fail-closed）验证入口。

严格边界
--------
* **不修改** V3H-4 `cash_strategy_dynamic_belief`、V3H-6
  `cash_strategy_likelihood_model`、V3H-8 `cash_strategy_decision_assembly`
  —— 本层只**调用**它们（经 V3I-P1 的 `run_decision_loop`）。
* **不新增** combo 级 likelihood（冻结 schema 无 combo 维度）；
  本层只填 `(action, street, profile)` 维度，`size_bucket=None`。
* 本模块 **不 import 任何 `cash_*` 层**，**不新增生产接线**（`wired=False`）。
* 倾向表**区分动作**，但仍是**声明式模型**：**不是**观测频率、
  **不是**真实玩家数据、**不是**外部验证。
* 不修改冻结策略 / CFI / λ / river 语义 / 搜索预算；不使用 `random`。
"""
from __future__ import annotations

from typing import Any

CASH_POPULATION_EVALUATION_VERSION = "v3i-p3-cash-population-evaluation-v1"

#: 受限群体枚举（必须与 V3I-P1 `cash_decision_environment.POPULATIONS` 一致）
DECLARED_POPULATIONS: tuple[str, ...] = (
    "DECLARED_TIGHT_PASSIVE",
    "DECLARED_BALANCED",
    "DECLARED_LOOSE_PASSIVE",
    "DECLARED_AGGRESSIVE",
    "DECLARED_MANIAC",
)

#: P3 授权要求的三类群体（保守/激进/平衡）——评估的**最小必测集**
MINIMUM_REQUIRED_POPULATIONS: tuple[str, ...] = (
    "DECLARED_TIGHT_PASSIVE",
    "DECLARED_AGGRESSIVE",
    "DECLARED_BALANCED",
)

#: 作为 ratio 归一化基准的群体（"无信息"参照）
BASELINE_POPULATION = "DECLARED_BALANCED"

DECLARED_STREETS: tuple[str, ...] = ("preflop", "flop", "turn", "river")

#: 参与倾向投影的动作（与 V3H-6 `KNOWN_ACTIONS` 词表一致）
DECLARED_ACTIONS: tuple[str, ...] = (
    "fold", "check", "call", "bet", "raise", "all_in",
)

#: 保守动作（倾向 = 继续/被动倾向）
PASSIVE_ACTIONS: tuple[str, ...] = ("fold", "check", "call")
#: 激进动作（倾向 = 后翻牌激进乘数）
AGGRESSIVE_ACTIONS: tuple[str, ...] = ("bet", "raise", "all_in")

#: 倾向上界（防止数值爆炸；超界即记录 `propensity_clamped`）
MAX_PROPENSITY = 100.0

#: 输出中声明的证据等级——**永远**是模型内
DECLARED_EVIDENCE_LEVEL = "DECLARED_MODEL"
MODE_INTERNAL_ONLY = "MODEL_INTERNAL"

CLAIM_BOUNDARY: dict[str, bool] = {
    "is_real_player_frequency": False,
    "is_real_human_validation": False,
    "is_gto": False,
    "is_equilibrium": False,
    "is_solver_output": False,
    "is_profit_evidence": False,
    "is_external_validation": False,
    "wired_to_production": False,
    "modifies_frozen_layers": False,
}

#: 链条逐层指纹的**顺序**（population → 观测先验 → belief/range → candidate → decision）
CHAIN_ARTIFACTS: tuple[str, ...] = (
    "propensity",            # 声明式群体倾向（观测先验的输入）
    "likelihood_multipliers",  # 观测先验（V3H-6 schema 中的乘数）
    "dynamic_belief",        # belief（V3H-4）
    "belief_update",         # belief 条件化（V3H-6 应用结果）
    "range_model",           # range（V3G 层）
    "ranking",               # candidate ranking（V3H-7 ranking 层）
    "belief_conditioned",    # belief 条件化候选评估
    "search_eval",           # 搜索层候选评估
    "assembly",              # V3H-8 决策装配
)

#: 判定"决策是否改变"时看的**决策级**工件
DECISION_LEVEL_ARTIFACTS: tuple[str, ...] = ("assembly", "search_eval")

#: 冻结层定位：检测到"上游变了但下游没变"时可归因的冻结层
FROZEN_ATTRIBUTION: dict[str, str] = {
    "belief_shape_invariant": (
        "frozen V3H-4 cash_strategy_dynamic_belief._posterior: "
        "posterior 的每个 combo 权重**直接取自** range 条目的 normalized 数值"
        "（norm[key]），action_weights 只经 `scale` 作为**单一标量**乘到 "
        "mass_posterior；形状不随观测/群体改变"
    ),
    "likelihood_scalar_only": (
        "frozen V3H-6 cash_strategy_belief_update: 每个玩家把逐动作乘数**连乘**成"
        " `likelihood_total_multiplier` 单一标量，仅乘到 `mass_after_likelihood`；"
        "不做 combo 级重加权"
    ),
    "profile_dimension_unused": (
        "frozen V3H-6 cash_strategy_belief_update 调用 resolve_multiplier 时传 "
        "profile=None，故参数中的 profile 维度**永远不参与匹配**，只走 "
        "action+street 兜底层"
    ),
    "no_opponent_history_in_environment": (
        "V3I-P1 cash_decision_environment 生成的状态 `action_history=[]`，"
        "因此 belief_update 的逐动作连乘为空 → total=1.0，"
        "群体似然在该路径上**完全不可达**"
    ),
    "ranking_ignores_action_weights": (
        "frozen V3H-7 cash_strategy_action_ranking.rank_actions 签名不含 "
        "action_weights / likelihood，故候选排序**无法**接收群体信息"
    ),
    "assembly_tier_precedence": (
        "frozen V3H-8 cash_strategy_decision_assembly 按 tier 优先级裁定且"
        " on_disagreement='abstain_unknown'：**不平均、不投票**；"
        "任一 tier 缺 top candidate 即弃权"
    ),
}


def _stub(reason: str) -> dict:
    """fail-closed 桩：**不猜**、不补默认值。"""
    return {
        "present": False,
        "version": CASH_POPULATION_EVALUATION_VERSION,
        "reason": reason,
        "unknown": [reason],
        "claim_boundary": dict(CLAIM_BOUNDARY),
        "evidence_grade": "NONE",
        "executed": False,
        "result": None,
        "wired_to_production": False,
    }


def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _stable_number(v: Any, ndigits: int = 6) -> Any:
    """确定性数值化（**不做**四舍五入以外的任何变换）。"""
    if _is_number(v):
        return round(float(v), ndigits)
    return v


# ---------------------------------------------------------------------------
# 1. 群体 → 声明式 per-action 倾向
# ---------------------------------------------------------------------------
def population_propensity(*, population: str, street: str,
                          population_params: Any) -> dict:
    """把声明的群体参数投影为 **per-action** 倾向（确定性、无随机）。

    Parameters
    ----------
    population : 受限枚举中的群体名
    street : preflop / flop / turn / river
    population_params : V3I-P1 `POPULATION_PARAMS` 表（**由调用方注入**，
        本层不 import 该模块）

    映射（**声明式**，非观测频率）
    ------------------------------
    * `fold` / `check` : 倾向 1.0（基线动作，不随群体变化）
    * `call`           : `modeled_preflop_continue_multiplier`
    * `bet`/`raise`/`all_in` :
        - `preflop` → `modeled_preflop_continue_multiplier`
        - 其他街   → `modeled_postflop_aggression_multiplier`
    """
    if population not in DECLARED_POPULATIONS:
        return _stub(f"population_not_in_closed_enum:{population!r}")
    if street not in DECLARED_STREETS:
        return _stub(f"street_not_supported:{street!r}")
    if not isinstance(population_params, dict) or not population_params:
        return _stub("population_params_required")

    row = population_params.get(population)
    if not isinstance(row, dict):
        return _stub(f"population_params_missing_row:{population!r}")

    cont = row.get("modeled_preflop_continue_multiplier")
    aggr = row.get("modeled_postflop_aggression_multiplier")
    if not _is_number(cont) or not _is_number(aggr):
        return _stub(f"population_params_row_incomplete:{population!r}")

    cont = float(cont)
    aggr = float(aggr)
    if cont < 0 or aggr < 0:
        return _stub(f"population_params_negative:{population!r}")

    agg_base = cont if street == "preflop" else aggr
    clamped: list[str] = []

    propensity: dict[str, float] = {}
    for a in DECLARED_ACTIONS:
        if a in ("fold", "check"):
            v = 1.0
        elif a == "call":
            v = cont
        else:                                   # bet / raise / all_in
            v = agg_base
        if v > MAX_PROPENSITY:
            v = MAX_PROPENSITY
            clamped.append(a)
        propensity[a] = round(v, 6)

    return {
        "present": True,
        "version": CASH_POPULATION_EVALUATION_VERSION,
        "population": population,
        "street": street,
        "propensity": propensity,
        "passive_actions": list(PASSIVE_ACTIONS),
        "aggressive_actions": list(AGGRESSIVE_ACTIONS),
        "declared_parameters": {
            "modeled_preflop_continue_multiplier": cont,
            "modeled_postflop_aggression_multiplier": aggr,
        },
        "applied_aggression_source": ("preflop_continue_multiplier"
                                      if street == "preflop"
                                      else "postflop_aggression_multiplier"),
        "clamped_actions": clamped,
        "distinguishes_actions": len(set(propensity.values())) > 1,
        "evidence_level": DECLARED_EVIDENCE_LEVEL,
        "is_observed_frequency": False,
        "source": "declared_population_parameter_table",
        "claim_boundary": dict(CLAIM_BOUNDARY),
        "executed": False,
        "result": None,
        "wired_to_production": False,
        "unknown": [],
    }


def _baseline_propensity(*, street: str, population_params: Any) -> dict:
    """基准群体（`DECLARED_BALANCED`）倾向——用于 ratio 归一化。"""
    return population_propensity(population=BASELINE_POPULATION,
                                 street=street,
                                 population_params=population_params)


# ---------------------------------------------------------------------------
# 2. 观测先验（action_weights + 声明式 likelihood 参数）
# ---------------------------------------------------------------------------
def population_observation_prior(*, population: str, street: str,
                                 last_action: str,
                                 population_params: Any,
                                 observation_factor: float = 1.0) -> dict:
    """由群体倾向导出**观测先验**：`action_weights` + likelihood 参数。

    * `action_weights` : 交给冻结 V3H-4（它只读 `_last_action` 与
      `weights[str(last)]`，本层**显式声明**这一点，不假装它是 per-combo 权重）；
    * `likelihood_parameters` : 声明式条目
      `(action, street, profile=population, size_bucket=None)`，
      每项带 `source` / `evidence_level` / `provenance` / `uncertainty`，
      交给冻结 V3H-6 schema 校验。

    `observation_factor` 是"观测本身"的强度（默认 1.0 = 只表达群体假设，
    不注入额外观测），**不是**频率估计。
    """
    if population not in DECLARED_POPULATIONS:
        return _stub(f"population_not_in_closed_enum:{population!r}")
    if street not in DECLARED_STREETS:
        return _stub(f"street_not_supported:{street!r}")
    if last_action not in DECLARED_ACTIONS:
        return _stub(f"action_not_in_closed_enum:{last_action!r}")
    if not _is_number(observation_factor) or float(observation_factor) < 0:
        return _stub("observation_factor_invalid")

    prop = population_propensity(population=population, street=street,
                                 population_params=population_params)
    if not prop.get("present"):
        return _stub(f"propensity_unavailable:{prop.get('reason')}")
    base = _baseline_propensity(street=street,
                                population_params=population_params)
    if not base.get("present"):
        return _stub(f"baseline_propensity_unavailable:{base.get('reason')}")

    p_map = prop["propensity"]
    b_map = base["propensity"]

    ratio: dict[str, float] = {}
    for a in DECLARED_ACTIONS:
        denom = b_map[a]
        ratio[a] = round((p_map[a] / denom) if denom else 0.0, 6)

    obs = float(observation_factor)

    def _mult(a: str) -> float:
        """群体动作乘数 = 倾向比 × 观测量；有界（≤ MAX_PROPENSITY）。"""
        v = ratio[a] * obs
        return round(min(v, MAX_PROPENSITY), 6)

    likelihood_parameters: list[dict] = []
    for a in DECLARED_ACTIONS:
        likelihood_parameters.append({
            "action": a,
            "street": street,
            "size_bucket": None,          # 显式：不按尺寸匹配（观测尺寸未知）
            "profile": population,
            "multiplier": _mult(a),
            "source": "declared_population_parameter_table",
            "evidence_level": DECLARED_EVIDENCE_LEVEL,
            "provenance": {
                "layer": "v3i_p3_population_evaluation",
                "population": population,
                "street": street,
                "derivation": "propensity_ratio_vs_baseline_x_observation_factor",
                "propensity": p_map[a],
                "baseline_propensity": b_map[a],
                "ratio": ratio[a],
                "observation_factor": obs,
                "declared_parameter_source": prop["applied_aggression_source"],
            },
            "uncertainty": {
                "quantified": False,
                "note": ("**未量化**：这是声明式模型参数，不是频率估计；"
                         "本层不发明置信区间"),
            },
        })

    # 供冻结 V3H-4 使用的显式 action_weights（照实声明其语义）
    action_weights: dict[str, Any] = {"_last_action": last_action}
    for a in DECLARED_ACTIONS:
        action_weights[a] = ratio[a]

    return {
        "present": True,
        "version": CASH_POPULATION_EVALUATION_VERSION,
        "population": population,
        "street": street,
        "last_action": last_action,
        "observation_factor": obs,
        "propensity": p_map,
        "baseline_propensity": b_map,
        "propensity_ratio": ratio,
        "applied_aggression_source": prop["applied_aggression_source"],
        "declared_parameters": dict(prop["declared_parameters"]),
        "distinguishes_actions": prop["distinguishes_actions"],
        "likelihood_multiplier_by_action": {a: _mult(a)
                                            for a in DECLARED_ACTIONS},
        "action_weights": action_weights,
        "likelihood_parameters": likelihood_parameters,
        "action_weights_semantics": (
            "冻结 V3H-4 仅读取 action_weights['_last_action'] 与 "
            "action_weights[str(last)]，并把该值作为**单一标量** scale 乘到 "
            "mass_posterior；其余键不参与 combo 级重加权"
        ),
        "likelihood_dimensions_used": ["action", "street", "profile"],
        "likelihood_dimensions_declared_unused": ["size_bucket", "position"],
        "profile_tagged": True,
        "profile_tag_semantics": (
            "条目带 `profile` 标记以**声明来源群体**（provenance 完整）；"
            "冻结 `cash_strategy_belief_update` 调用 `resolve_multiplier(profile=None)`，"
            "因此带 profile 的条目**不会**被命中 → 见 `profile_free_parameters()`"
            " 提供的等价无 profile 对照组"),
        "per_combo_likelihood": False,
        "evidence_level": DECLARED_EVIDENCE_LEVEL,
        "is_observed_frequency": False,
        "claim_boundary": dict(CLAIM_BOUNDARY),
        "executed": False,
        "result": None,
        "wired_to_production": False,
        "unknown": [],
    }


def likelihood_conflict_probe(*, street: str, population_params: Any,
                              populations: Any = None) -> dict:
    """UNKNOWN / 冲突语义探针（**fail-closed** 证据）。

    构造"两个群体对同一
    `(action, street, size_bucket, profile)` 给出**不同**乘数"的参数集，
    交给冻结 V3H-6 schema 校验，验证：
    * 冲突键**不产出**乘数（`valid` 中被剔除）；
    * `conflicts` 非空、`unknown` 记录；
    * **不静默择一、不静默覆盖**。

    同时给出 UNKNOWN 输入（群体不在枚举内 / street 不支持 / 参数非 list）
    的 fail-closed 结果，证明本层**不猜**。
    """
    if street not in DECLARED_STREETS:
        return _stub(f"street_not_supported:{street!r}")

    pops = list(populations) if populations is not None else list(
        MINIMUM_REQUIRED_POPULATIONS)

    # 同一 profile+action 上放两个不同乘数 → 必须被判为冲突
    rows: list[dict] = []
    for pop in pops:
        prior = population_observation_prior(population=pop, street=street,
                                             last_action="raise",
                                             population_params=population_params)
        if not prior.get("present"):
            return _stub(f"prior_unavailable:{pop}:{prior.get('reason')}")
        for row in prior["likelihood_parameters"]:
            rows.append(dict(row))

    # 人为制造冲突：把第一条复制成同 key、不同乘数
    conflict_created = False
    if rows:
        dup = dict(rows[0])
        dup["multiplier"] = round(float(dup["multiplier"]) + 1.0, 6)
        rows.append(dup)
        conflict_created = True

    return {
        "present": True,
        "version": CASH_POPULATION_EVALUATION_VERSION,
        "street": street,
        "populations": pops,
        "n_parameters_supplied": len(rows),
        "conflict_injected": conflict_created,
        "first_parameter": dict(rows[0]) if rows else None,
        "duplicate_parameter": dict(rows[-1]) if conflict_created else None,
        "unknown_inputs": {
            "unknown_population": population_observation_prior(
                population="DECLARED_NOT_A_POPULATION", street=street,
                last_action="raise", population_params=population_params),
            "unknown_street": population_observation_prior(
                population=pops[0] if pops else BASELINE_POPULATION,
                street="not_a_street", last_action="raise",
                population_params=population_params),
            "unknown_action": population_observation_prior(
                population=pops[0] if pops else BASELINE_POPULATION,
                street=street, last_action="not_an_action",
                population_params=population_params),
            "missing_params_table": population_observation_prior(
                population=pops[0] if pops else BASELINE_POPULATION,
                street=street, last_action="raise", population_params=None),
        },
        "evidence_level": DECLARED_EVIDENCE_LEVEL,
        "claim_boundary": dict(CLAIM_BOUNDARY),
        "executed": False,
        "result": None,
        "wired_to_production": False,
        "unknown": [],
    }


def profile_free_parameters(prior: Any) -> list[dict]:
    """把带 `profile` 标记的条目转成**等价的无 profile** 条目。

    用途：冻结 `cash_strategy_belief_update` 调用
    `resolve_multiplier(..., profile=None)`，带 profile 的条目**永远**命中不了；
    去掉 profile 后走 `action+street` 兜底层即可命中，从而让群体的逐动作乘数
    **真的**进入 belief 条件化（仍只作用于标量 mass）。

    这是**机制可达性对照**，不是新策略：乘数数值完全不变，只去掉一个
    冻结层不会读取的维度标记。原始群体归属仍完整保留在 `provenance.population`。
    """
    rows = prior.get("likelihood_parameters") if isinstance(prior, dict) else None
    if not isinstance(rows, list):
        return []
    out: list[dict] = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        q = dict(r)
        q.pop("profile", None)
        prov = dict(q.get("provenance") or {})
        prov["profile_tag_removed"] = True
        prov["profile_tag_removed_reason"] = (
            "frozen belief_update resolves with profile=None; "
            "profile-free rows match at the action+street tier"
        )
        prov["population_attribution"] = r.get("profile")
        q["provenance"] = prov
        out.append(q)
    return out


# ---------------------------------------------------------------------------
# 3. 单群体完整链路评估（复用 V3I-P1 环境 + V3I-P2 真实 pipeline）
# ---------------------------------------------------------------------------
def _artifact_fingerprint(value: Any) -> str:
    """确定性指纹（仅用 stdlib json；冻结输出可能含 tuple 键 → 先规范化）。"""
    import hashlib
    import json

    def _norm(v: Any) -> Any:
        if isinstance(v, dict):
            return {str(k): _norm(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [_norm(x) for x in v]
        return v

    try:
        text = json.dumps(_norm(value), sort_keys=True, separators=(",", ":"),
                          ensure_ascii=True, default=str)
    except Exception:                                   # noqa: BLE001
        text = repr(value)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _step(present: Any, fingerprint: Any, extra: Any = None) -> dict:
    out = {"present": bool(present), "fingerprint": fingerprint}
    if isinstance(extra, dict):
        out.update(extra)
    return out


def _stage_present(value: Any) -> bool:
    """`run_decision_loop` 的阶段可用性判定（已核对源码 `_stage_ok`）。

    * `None` → 阶段**未执行**（如 likelihood 未提供时 belief_update 被跳过）；
    * 非 dict → 不可用；
    * dict → 有 `present` 键则**照实**读，无该键（如 V3H-8 assembly）视为可用。
    """
    if not isinstance(value, dict):
        return False
    if "present" in value:
        return bool(value.get("present"))
    return True


def evaluate_population(*, scenario: Any, population: str, street: str,
                        last_action: str, population_params: Any,
                        pipeline: Any, integrity_context: Any,
                        opponent_model: Any, range_model: Any,
                        action_tree: Any, transition: Any,
                        solver: Any, budget: Any, policy: Any,
                        run_decision_loop: Any,
                        action_likelihood_model: Any,
                        validate_parameters: Any = None,
                        max_root_candidates: int = 3,
                        equity_seed: int = 42,
                        equity_iterations: int = 20000,
                        observation_factor: float = 1.0,
                        profile_free: bool = False,
                        state_override: Any = None) -> dict:
    """跑一个群体的完整链路并逐层留证（**复用** V3I-P1 环境 + V3I-P2 pipeline）。

    所有上游对象（scenario / pipeline / 冻结层函数）都由**调用方注入**，
    本层不 import 任何 `cash_*` 模块、不新增接线。

    Parameters
    ----------
    profile_free : 若为 True，用 `profile_free_parameters()` 的等价无 profile 条目
        （机制可达性对照；乘数数值不变）。默认 False = 冻结层实际行为。
    state_override : 可选，替换 scenario 中的 state（用于"声明式历史"对照）。
        **不修改**冻结层，只提供不同输入。
    """
    if population not in DECLARED_POPULATIONS:
        return _stub(f"population_not_in_closed_enum:{population!r}")
    if street not in DECLARED_STREETS:
        return _stub(f"street_not_supported:{street!r}")
    if not isinstance(scenario, dict) or not isinstance(scenario.get("state"), dict):
        return _stub("scenario_required")
    if not callable(run_decision_loop):
        return _stub("run_decision_loop_required")
    if not callable(action_likelihood_model):
        return _stub("action_likelihood_model_required")
    if not isinstance(pipeline, dict):
        return _stub("pipeline_required")

    state = scenario["state"] if state_override is None else state_override
    if not isinstance(state, dict):
        return _stub("state_override_not_a_dict")

    prior = population_observation_prior(
        population=population, street=street, last_action=last_action,
        population_params=population_params,
        observation_factor=observation_factor)
    if not prior.get("present"):
        return _stub(f"observation_prior_unavailable:{prior.get('reason')}")

    params_used = (profile_free_parameters(prior) if profile_free
                   else prior["likelihood_parameters"])
    likelihood_model = action_likelihood_model(params_used)
    if not isinstance(likelihood_model, dict):
        return _stub("likelihood_model_not_a_dict")

    loop_out = run_decision_loop(
        state,
        pipeline=pipeline,
        integrity_context=integrity_context,
        opponent_model=opponent_model,
        range_model=range_model,
        action_tree=action_tree,
        transition=transition,
        belief=None,                     # 由 pipeline 的 build_dynamic_belief 产出
        likelihood_model=likelihood_model,
        action_weights=prior["action_weights"],
        solver=solver,
        budget=budget,
        max_root_candidates=max_root_candidates,
        equity_seed=equity_seed,
        equity_iterations=equity_iterations,
        policy=policy,
        claimed_evidence_level=MODE_INTERNAL_ONLY,
    )
    if not isinstance(loop_out, dict):
        return _stub("decision_loop_returned_non_dict")

    stages = loop_out.get("stage_outputs") or {}
    decision = loop_out.get("decision_record") or {}

    # `run_decision_loop` 的 `stage_outputs` 使用**自己的**键名（已核对源码）：
    # action_policy / belief / belief_update / ranking / belief_conditioned /
    # search_tree / search_eval / decision
    dyn = stages.get("belief") or {}
    bel_upd = stages.get("belief_update") or {}
    rank = stages.get("ranking") or {}
    bel_cond = stages.get("belief_conditioned") or {}
    search = stages.get("search_eval") or {}
    search_tree = stages.get("search_tree") or {}

    # belief 摘要
    bel_entries = (dyn.get("belief") or {}) if isinstance(dyn, dict) else {}
    posterior_summary: dict[str, Any] = {}
    mass_summary: dict[str, Any] = {}
    combo_weights: dict[str, Any] = {}
    for pid, entry in (bel_entries or {}).items():
        post = ((entry or {}).get("posterior") or {})
        posterior_summary[str(pid)] = post.get("posterior") or {}
        mass_summary[str(pid)] = {
            "mass_prior": post.get("mass_prior"),
            "mass_removed_by_known_cards": post.get("mass_removed_by_known_cards"),
            "mass_posterior": post.get("mass_posterior"),
            "combos_prior": post.get("combos_prior"),
            "combos_posterior": post.get("combos_posterior"),
            "likelihood_multiplier": post.get("likelihood_multiplier"),
            "likelihood_source": post.get("likelihood_source"),
        }
        for ck, cv in (post.get("posterior") or {}).items():
            combo_weights[str(ck)] = cv

    bel_upd_players = (bel_upd.get("belief") or {}) if isinstance(bel_upd, dict) else {}
    bel_upd_summary = {
        str(pid): {
            "likelihood_total_multiplier": (e or {}).get("likelihood_total_multiplier"),
            "mass_before_likelihood": (e or {}).get("mass_before_likelihood"),
            "mass_after_likelihood": (e or {}).get("mass_after_likelihood"),
            "likelihood_conditioned": (e or {}).get("likelihood_conditioned"),
            "n_likelihood_applied": len((e or {}).get("likelihood_applied") or []),
            "n_likelihood_unmatched": len((e or {}).get("likelihood_unmatched") or []),
        }
        for pid, e in (bel_upd_players or {}).items()
    }

    # ── 群体信息的**投递路径**诊断（逐门槛如实记录，不做推测）──
    n_hist = len(state.get("action_history") or [])
    n_applied_total = sum(v["n_likelihood_applied"] for v in bel_upd_summary.values())
    n_unmatched_total = sum(v["n_likelihood_unmatched"] for v in bel_upd_summary.values())
    totals = {v["likelihood_total_multiplier"] for v in bel_upd_summary.values()}
    applied_multipliers = sorted({t for t in totals if t is not None})
    delivery = {
        "state_action_history_length": n_hist,
        "gate1_history_present": n_hist > 0,
        "gate1_note": FROZEN_ATTRIBUTION["no_opponent_history_in_environment"],
        "gate2_parameters_profiled": bool(prior.get("profile_tagged")),
        "gate2_note": FROZEN_ATTRIBUTION["profile_dimension_unused"],
        "per_action_multipliers_applied": n_applied_total,
        "per_action_multipliers_unmatched": n_unmatched_total,
        "distinct_total_multipliers": applied_multipliers,
        "population_reaches_belief_update": (
            n_applied_total > 0 and len(applied_multipliers) > 1),
        "population_reaches_belief_shape": False,   # 由 compare_populations 判定
        "path": ("population_scalar_applied_to_mass"
                 if n_applied_total > 0 else
                 "population_not_delivered_to_belief_update"),
    }

    chain: dict[str, Any] = {
        "propensity": _step(
            bool((prior.get("propensity") or {})),
            _artifact_fingerprint(prior.get("propensity")),
            {"values": prior.get("propensity")}),
        "likelihood_multipliers": _step(
            bool(prior.get("likelihood_multiplier_by_action")),
            _artifact_fingerprint(prior.get("likelihood_multiplier_by_action")),
            {"values": prior.get("likelihood_multiplier_by_action")}),
        "dynamic_belief": _step(
            _stage_present(dyn),
            _artifact_fingerprint({
                "posterior": posterior_summary,
                "version": (dyn or {}).get("version"),
                "likelihood_applied": (dyn or {}).get("likelihood_applied"),
            }),
            {"mass": mass_summary,
             "likelihood_applied": (dyn or {}).get("likelihood_applied"),
             "likelihood_source": (dyn or {}).get("likelihood_source")}),
        "belief_update": _step(
            _stage_present(bel_upd),
            _artifact_fingerprint({
                "path": (bel_upd or {}).get("path"),
                "players": bel_upd_summary,
                "hero_actions_excluded": (bel_upd or {}).get("hero_actions_excluded"),
            }),
            {"path": (bel_upd or {}).get("path"),
             "players": bel_upd_summary,
             "unknown": (bel_upd or {}).get("unknown")}),
        "range_model": _step(
            _stage_present(range_model),
            _artifact_fingerprint(range_model),
            {"n_opponents": (range_model or {}).get("n_opponents")
             if isinstance(range_model, dict) else None}),
        "ranking": _step(
            _stage_present(rank),
            _artifact_fingerprint({
                "preferred": (rank or {}).get("preferred"),
                "ranked": (rank or {}).get("ranked"),
            }),
            {"preferred": (rank or {}).get("preferred")}),
        "belief_conditioned": _step(
            _stage_present(bel_cond),
            _artifact_fingerprint({
                "preferred": (bel_cond or {}).get("preferred"),
                "ranked": (bel_cond or {}).get("ranked"),
            }),
            {"preferred": (bel_cond or {}).get("preferred")}),
        "search_eval": _step(
            _stage_present(search),
            _artifact_fingerprint({
                "preferred": (search or {}).get("preferred"),
                "ranked": (search or {}).get("ranked"),
                "n_candidates": (search or {}).get("n_candidates"),
            }),
            {"preferred": (search or {}).get("preferred"),
             "n_candidates": (search or {}).get("n_candidates")}),
        "assembly": _step(
            _stage_present(decision),
            _artifact_fingerprint({
                "preferred": (decision or {}).get("preferred"),
                "abstained": (decision or {}).get("abstained"),
                "disagreements": (decision or {}).get("disagreements"),
            }),
            {"preferred": (decision or {}).get("preferred"),
             "abstained": (decision or {}).get("abstained"),
             "disagreements": (decision or {}).get("disagreements")}),
    }

    return {
        "present": True,
        "version": CASH_POPULATION_EVALUATION_VERSION,
        "mode": "cash",
        "population": population,
        "street": street,
        "last_action": last_action,
        "seed": scenario.get("seed"),
        "state_shared": {
            "board": state.get("board"),
            "pot_bb": state.get("pot_bb"),
            "effective_stack_bb": state.get("effective_stack_bb"),
            "hero_id": state.get("hero_id"),
            "n_players": len(state.get("players") or []),
        },
        "observation_prior": {
            "propensity": prior.get("propensity"),
            "propensity_ratio": prior.get("propensity_ratio"),
            "likelihood_multiplier_by_action": prior.get("likelihood_multiplier_by_action"),
            "action_weights": prior.get("action_weights"),
            "applied_aggression_source": prior.get("applied_aggression_source"),
            "profile_tagged": prior.get("profile_tagged"),
            "profile_free_variant": bool(profile_free),
            "per_combo_likelihood": False,
        },
        "likelihood_model": {
            "present": likelihood_model.get("present"),
            "applied": likelihood_model.get("applied"),
            "prior_only": likelihood_model.get("prior_only"),
            "n_parameters": len(likelihood_model.get("parameters") or []),
            "n_invalid": len(likelihood_model.get("invalid") or []),
            "n_conflicts": len(likelihood_model.get("conflicts") or []),
            "evidence_grade": likelihood_model.get("evidence_grade"),
        },
        "chain": chain,
        "combo_weight_values": sorted(set(combo_weights.values())),
        "n_combo_weights": len(combo_weights),
        "population_delivery": delivery,
        "loop_status": loop_out.get("loop_status"),
        "loop_present": loop_out.get("present"),
        "loop_reason": loop_out.get("reason"),
        "loop_unknown": loop_out.get("unknown"),
        "decision_level_flags": {
            "belief_update_present": (decision or {}).get("belief_update_present"),
            "belief_update_level": (decision or {}).get("belief_update_level"),
            "search_eval_present": (decision or {}).get("search_eval_present"),
            "search_tree_present": (decision or {}).get("search_tree_present"),
            "selected_by_tier": (decision or {}).get("selected_by_tier"),
            "stage_order": (decision or {}).get("stage_order"),
        },
        "search_tree_leaves": (search_tree or {}).get("n_leaves") if isinstance(
            search_tree, dict) else None,
        "evidence_level": DECLARED_EVIDENCE_LEVEL,
        "claimed_evidence_level": MODE_INTERNAL_ONLY,
        "executed": False,
        "result": None,
        "claim_boundary": dict(CLAIM_BOUNDARY),
        "wired_to_production": False,
        "unknown": list(loop_out.get("unknown") or []),
    }


# ---------------------------------------------------------------------------
# 4. 多群体比较 + 冻结层定位
# ---------------------------------------------------------------------------
def compare_populations(*, evaluations: Any,
                        artifacts: Any = None) -> dict:
    """逐层比较多个群体，回答"哪一层变了"，并在未变时定位冻结层。

    **同名评估按"重复运行"处理**：若多个评估的 `population` 相同（即同一群体的
    多次运行），它们在每个工件上**累积**为运行列表，取**不同值的个数**判定。
    这样"同群体 3 次运行是否完全一致"才是**真检验**，不会因按名覆盖而空转。
    """
    names = list(artifacts) if artifacts is not None else list(CHAIN_ARTIFACTS)
    evals = evaluations if isinstance(evaluations, list) else []
    ok = [e for e in evals if isinstance(e, dict) and e.get("present")]

    if len(ok) < 2:
        out = _stub("at_least_two_present_evaluations_required")
        out["n_populations"] = len(evals)
        out["n_present"] = len(ok)
        return out

    labels = [str(e.get("population")) for e in ok]
    populations = sorted(set(labels))
    unable = [e.get("population") for e in evals
              if not (isinstance(e, dict) and e.get("present"))]
    repeats = {p: labels.count(p) for p in populations}

    per_artifact: dict[str, Any] = {}
    changed: list[str] = []
    invariant: list[str] = []
    for art in names:
        by_pop: dict[str, list] = {p: [] for p in populations}
        present_flags: dict[str, Any] = {}
        for e, lab in zip(ok, labels):
            step = (e.get("chain") or {}).get(art) or {}
            by_pop[lab].append(step.get("fingerprint"))
            present_flags[lab] = step.get("present")
        # 判定：把**所有运行**的指纹放在一起取不同值个数（同名重复自然累积比较）
        all_values = [v for p in populations for v in by_pop[p]]
        distinct = sorted({str(v) for v in all_values})
        n_distinct = len(distinct)
        did_change = n_distinct > 1
        (changed if did_change else invariant).append(art)
        per_artifact[art] = {
            "fingerprints_by_population": {p: list(by_pop[p]) for p in populations},
            # 单群体时保持旧形状（向后兼容）；多群体时给出"每群体 1 个值"的摘要
            "fingerprints": ({p: (by_pop[p][0] if by_pop[p] else None)
                              for p in populations} if len(populations) > 1
                             else None),
            "present": present_flags,
            "n_distinct": n_distinct,
            "changed": did_change,
        }

    # 决策级是否改变
    decision_changed = any(per_artifact.get(a, {}).get("changed")
                           for a in DECISION_LEVEL_ARTIFACTS)

    # 冻结层定位：只在"有依据"时给出
    blocked: list[dict] = []
    if per_artifact.get("likelihood_multipliers", {}).get("changed") and \
            not per_artifact.get("dynamic_belief", {}).get("changed"):
        blocked.append({
            "artifact": "dynamic_belief",
            "code": "belief_shape_invariant",
            "detail": FROZEN_ATTRIBUTION["belief_shape_invariant"],
        })
    if not per_artifact.get("belief_conditioned", {}).get("changed") and \
            per_artifact.get("belief_update", {}).get("changed"):
        blocked.append({
            "artifact": "belief_conditioned",
            "code": "likelihood_scalar_only",
            "detail": FROZEN_ATTRIBUTION["likelihood_scalar_only"],
        })
    if not decision_changed and per_artifact.get("ranking", {}).get("changed"):
        blocked.append({
            "artifact": "assembly",
            "code": "assembly_tier_precedence",
            "detail": FROZEN_ATTRIBUTION["assembly_tier_precedence"],
        })
    if per_artifact.get("belief_conditioned", {}).get("changed") is False and \
            per_artifact.get("likelihood_multipliers", {}).get("changed"):
        blocked.append({
            "artifact": "belief_update",
            "code": "profile_dimension_unused",
            "detail": FROZEN_ATTRIBUTION["profile_dimension_unused"],
        })

    prior_changed = bool(per_artifact.get("propensity", {}).get("changed")) or \
        bool(per_artifact.get("likelihood_multipliers", {}).get("changed"))

    return {
        "present": True,
        "version": CASH_POPULATION_EVALUATION_VERSION,
        "populations": populations,
        "n_populations": len(populations),
        "n_evaluations": len(ok),
        "repeats_per_population": repeats,
        "populations_unavailable": unable,
        "artifacts_compared": names,
        "per_artifact": per_artifact,
        "changed_artifacts": changed,
        "invariant_artifacts": invariant,
        "observation_prior_changed": prior_changed,
        "belief_changed": bool(per_artifact.get("dynamic_belief", {}).get("changed")),
        "belief_update_changed": bool(per_artifact.get("belief_update", {}).get("changed")),
        "range_changed": bool(per_artifact.get("range_model", {}).get("changed")),
        "candidate_changed": bool(per_artifact.get("ranking", {}).get("changed")),
        "decision_changed": decision_changed,
        "blocked_at": blocked,
        "blocked_codes": [b["code"] for b in blocked],
        "explainable_strategy_difference": bool(prior_changed or decision_changed),
        "difference_scope": (
            "decision_level" if decision_changed else
            ("observation_prior_and_mass" if prior_changed else "none")),
        "evidence_level": DECLARED_EVIDENCE_LEVEL,
        "claimed_evidence_level": MODE_INTERNAL_ONLY,
        "executed": False,
        "result": None,
        "claim_boundary": dict(CLAIM_BOUNDARY),
        "wired_to_production": False,
        "unknown": [],
    }


def determinism_probe(*, evaluations_by_repeat: Any) -> dict:
    """同群体重复运行 → 位移为零（**确定性**证据，非新语义）。"""
    pairs = evaluations_by_repeat if isinstance(evaluations_by_repeat, dict) else {}
    per_population: dict[str, Any] = {}
    all_stable = True
    for pop, runs in pairs.items():
        runs = list(runs) if isinstance(runs, (list, tuple)) else []
        cmp = compare_populations(evaluations=runs)
        stable = bool(cmp.get("present")) and not cmp.get("changed_artifacts")
        all_stable = all_stable and stable
        per_population[str(pop)] = {
            "n_repeats": len(runs),
            "identical": stable,
            "changed_artifacts": cmp.get("changed_artifacts"),
        }
    return {
        "present": True,
        "version": CASH_POPULATION_EVALUATION_VERSION,
        "all_populations_deterministic": all_stable,
        "per_population": per_population,
        "evidence_level": DECLARED_EVIDENCE_LEVEL,
        "claim_boundary": dict(CLAIM_BOUNDARY),
        "executed": False,
        "result": None,
        "wired_to_production": False,
        "unknown": [],
    }


def population_evaluation_contract() -> dict:
    """本层契约（供审计/测试断言；**不**声明任何外部有效性）。"""
    return {
        "version": CASH_POPULATION_EVALUATION_VERSION,
        "declared_populations": list(DECLARED_POPULATIONS),
        "minimum_required_populations": list(MINIMUM_REQUIRED_POPULATIONS),
        "baseline_population": BASELINE_POPULATION,
        "declared_streets": list(DECLARED_STREETS),
        "declared_actions": list(DECLARED_ACTIONS),
        "passive_actions": list(PASSIVE_ACTIONS),
        "aggressive_actions": list(AGGRESSIVE_ACTIONS),
        "chain_artifacts": list(CHAIN_ARTIFACTS),
        "decision_level_artifacts": list(DECISION_LEVEL_ARTIFACTS),
        "frozen_attribution": dict(FROZEN_ATTRIBUTION),
        "claim_boundary": dict(CLAIM_BOUNDARY),
        "per_combo_likelihood": False,
        "modifies_frozen_layers": False,
        "wired_to_production": False,
        "evidence_level": DECLARED_EVIDENCE_LEVEL,
        "note": ("本层是**模型内**评估能力：可解释差异**不等于**真实玩家验证、"
                 "GTO、Solver 输出或盈利证据。"),
    }
