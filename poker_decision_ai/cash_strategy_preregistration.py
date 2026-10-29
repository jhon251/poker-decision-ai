"""V3H-12 — CASH 预注册分析基础（设计面）＋ 结果生成**闸门**。

**为什么这是 V3H-12 的最小必要能力（源码级缺口审计结论）**
------------------------------------------------------------
对**既有** 15 个 V3H 层做源码探针（排除本模块）：

```
pre_regist / preregist / hypothes / statistic / estimator     0 个文件
dataset / dataframe / frame / cohort / stratif                0 个文件
unit_of_analysis / endpoint / outcome / confound              0 个文件
power / effect_size / bootstrap / ci_ / interval / alpha       0 个文件
significan / protocol / estimand / missingness / attrition     0 个文件
sampl   7 个文件 —— 但全部是 V3H-5/6/7 的**蒙特卡洛抽样**，不是实验抽样框
design  1 个文件 —— V3H-9 的 `"design"` 字段，与预先注册无关
```

即：**描述性证据已可聚合（V3H-11），但把它转成"预注册统计评估所需的数据结构 /
分析基础"的**任何**构件都不存在**——没有分析单元定义、没有预先注册的设计面、
没有数据集 schema、没有"何时**才**允许产生结果"的闸门。

**严格区分：设计 ≠ 结果**
--------------------------
本层是**设计面（design-only）**，并自带**结果生成闸门**：

* 输出**恒含** `result=None`、`results_available=False`、
  `analysis_executed=False`、`executed=False`；
* 只要缺少真实执行数据 / 环境事实 / 求解器均衡参考 / 独立 blocker 未解，
  `run_analysis_preconditions()` 一律返回 `would_run=False`，**不产生任何结果**；
* 调用方**主动注入**结果数据时，本层**拒绝消费**：
  `result_generation_refused=True` + `result_refusal_reason`
  （不建立任何可能被误用的真实结果入口）；
* `evidence_level = "DESIGN_DECLARATION"`（**声明的设计**，不是实验结果）。

**不新增评分/EV 公式**
----------------------
模块内**没有**任何统计分析实现：不计算效应量、不计算区间、不做功效计算、
不计算比率与百分比。数值运算**仅**为计数与阈值比较（`Sub` / `Add`），
由测试以 AST 断言（**无** `Mult` / `Div` / `Pow` / `FloorDiv` / `Mod`）。

硬约束
------
* CASH-only，fail closed；非 dict / 非 CASH → `present=False`；
* **不 import 任何 V3G / V3H 层**（任何位置）；不 import 伪随机数模块；无 IO；
* 同输入 → 同输出（键序、列表序固定）；
* UNKNOWN / UNAVAILABLE / abstention / disagreement 原样传播并阻断"可分析"；
* Solver `BLOCKED` → `equilibrium=None`、`gto=None`、`exploitability=None`
  （**不猜**），且 `analysis_ready=False`；
* M3 / B1 / B3 / B4 / B5 / solver **一个都不突破**：全部登记为未解阻塞，
  且声明**非本层测量**；
* `MODEL_INTERNAL` / `DESIGN_DECLARATION` 仅描述**证据等级**，
  **不是**真实对手证据，也**不是**真实盈利证据。
"""
from __future__ import annotations

from typing import Any

CASH_STRATEGY_PREREGISTRATION_VERSION = (
    "v3h12-cash-strategy-preregistration-v1")

EVIDENCE_LEVELS: tuple[str, ...] = ("DESIGN_DECLARATION", "MODEL_INTERNAL",
                                    "UNAVAILABLE")

#: 本层消费的上游设计事实（全部由**调用方显式**提供）
REQUIRED_UPSTREAM_KEYS: tuple[str, ...] = (
    "closure",       # V3H-11 audit_evidence_closure 输出
)

#: 数据集 schema（**精确**字段集，来自 V3H-10 冻结输出面）
#: 只声明"每一行必须有什么"，**不**声明任何统计量。
DATASET_SCHEMA_VERSION = "v3h12-cash-shadow-dataset-schema-v1"

REQUIRED_FIELDS: tuple[str, ...] = (
    "record_id",            # 调用方分配，稳定且唯一（本层只校验非空字符串）
    "shadow_version",       # 必须等于调用方传入的期望版本
    "state_hash",           # 调用方提供，用于去重与来源追溯
    "street",               # preflop / flop / turn / river
    "divergence",           # aligned / different / unavailable
    "recommended_action",   # 或 None（不可对账时）
    "recommended_amount_bb",  # 或 None
    "observed_action",      # 或 None
    "observed_amount_bb",   # 或 None
    "available",            # bool
    "level",                # MODEL_INTERNAL / UNKNOWN / ...
    "evidence_grade",       # MODEL_INTERNAL / NONE
    "snapshot_present",     # bool
    "snapshot_method",      # 或 None
    "snapshot_samples",     # 或 None（int）
    "unknown_tokens",       # list[str]
    "executed",             # 必须为 False
)

#: 分析单元定义（**声明性**，不含任何量）
ANALYSIS_UNIT: dict[str, Any] = {
    "unit": "one reconciled decision point (one V3H-10 shadow result)",
    "independence_claim": False,
    "independence_note": ("本层**不**声称独立；独立计数（B1 >= 1200）"
                          "**不由本层测量**。"),
    "one_row_per": "decision point",
}

#: 缺失与排除规则（**声明性**）
MISSINGNESS_POLICY: dict[str, Any] = {
    "unavailable_records": ("保留为行，标记 excluded_reason="
                            "'divergence_unavailable'；**不**插补、**不**丢弃"),
    "unknown_tokens_present": ("保留为行，标记 excluded_reason="
                              "'unknown_tokens_present'"),
    "abstention": "保留为行，标记 excluded_reason='decision_abstained'",
    "disagreement": "保留为行，标记 excluded_reason='decision_disagreements'",
    "imputation": "forbidden",
    "silent_drop": "forbidden",
    "model_internal_only": ("evidence_grade 非 MODEL_INTERNAL 的行必须被标记，"
                            "**不**得当作真实对手证据"),
}

#: 结局（endpoint）声明：**可测量性**受真实执行数据约束
ENDPOINTS: tuple[dict[str, Any], ...] = (
    {
        "id": "E1",
        "name": "reconciliation_availability",
        "kind": "descriptive_count",
        "measurable_without_execution_data": True,
        "note": "可由 V3H-10/V3H-11 计数得到；**不是**收益、**不是**优劣。",
    },
    {
        "id": "E2",
        "name": "divergence_class_distribution",
        "kind": "descriptive_count",
        "measurable_without_execution_data": True,
        "note": "仅分类计数；**禁止**折算为比率或对齐率。",
    },
    {
        "id": "E3",
        "name": "realised_outcome",
        "kind": "outcome_requires_real_execution_data",
        "measurable_without_execution_data": False,
        "note": ("需要**真实执行数据**；本层**不**生成、**不**读取、**不**推断。"
                 "B3 未解 → 不可评估。"),
    },
)

#: 预先声明的假设（**声明性**，`claim` 不含任何数值估计或优越性断言）
HYPOTHESES: tuple[dict[str, Any], ...] = (
    {
        "id": "H1",
        "statement": ("在满足纳入条件且仅使用 MODEL_INTERNAL 证据的决策点上，"
                      "V3H-8 的建议与 hero 已观测动作的描述性差异分布"
                      "可被稳定复现。"),
        "endpoints": ["E2"],
        "falsifiable": True,
        "direction_claimed": False,
        "superiority_claimed": False,
        "note": ("**不**声明方向、**不**声明优劣、**不**声明收益；"
                 "复现性检验需要真实执行数据，因而当前不可评估。"),
    },
)

#: 独立阻塞（引自既有阶段，**非**本层测量）
BLOCKERS: dict[str, dict] = {
    "M3": {"status": "BLOCKED", "measured_by_this_layer": False,
           "description": "M3 gate not cleared (carried from earlier stages)"},
    "B1": {"status": "BLOCKED", "measured_by_this_layer": False,
           "description": ("independent decision points >= 1200; independently "
                           "measured as unavailable (0/1200) - this layer does "
                           "NOT measure independence")},
    "B3": {"status": "BLOCKED", "measured_by_this_layer": False,
           "description": "real profit evidence unavailable"},
    "B4": {"status": "BLOCKED", "measured_by_this_layer": False,
           "description": "reserved downstream blocker"},
    "B5": {"status": "BLOCKED", "measured_by_this_layer": False,
           "description": "reserved downstream blocker"},
    "solver": {"status": "BLOCKED", "measured_by_this_layer": False,
               "description": ("6-max CASH exact solver / equilibrium "
                               "reference; discovery_state=UNAVAILABLE")},
}

#: 结果生成的**全部**前置条件（缺一即不产生结果）
RESULT_PRECONDITIONS: tuple[str, ...] = (
    "closure_possible_is_true",
    "independent_blockers_all_resolved",
    "solver_not_blocked_and_equilibrium_available",
    "real_execution_data_supplied_and_verified",
    "environment_facts_all_true",
    "dataset_has_zero_excluded_rows",
    "dataset_rows_at_or_above_declared_minimum",
    "no_unknown_tokens",
    "no_forbidden_claim_flags",
)

CLAIM_BOUNDARY: dict[str, bool] = {
    "is_gto": False,
    "is_equilibrium": False,
    "is_solver_output": False,
    "is_profit_evidence": False,
    "is_causal_claim": False,
    "is_strategy_quality_claim": False,
    "is_real_opponent_evidence": False,
    "is_experimental_result": False,
    "computes_effect_size_or_interval": False,
    "computes_rates_or_percentages": False,
    "performs_power_or_sample_size_calculation": False,
    "adds_scoring_or_ev_formula": False,
    "produces_results_without_execution_data": False,
    "executes_action": False,
    "auto_betting": False,
    "claims_experiment_ready": False,
    "wired_to_production": False,
}
CLAIM_FLAGS_TOTAL = len(CLAIM_BOUNDARY)

DEFAULT_MIN_ROWS = 1200


def _is_cash(state: Any) -> bool:
    if not isinstance(state, dict):
        return False
    gt = state.get("game_type")
    return isinstance(gt, str) and gt.strip().lower() == "cash"


def _stub(reason: str) -> dict:
    return {
        "present": False,
        "version": CASH_STRATEGY_PREREGISTRATION_VERSION,
        "reason": reason,
        "evidence_level": "UNAVAILABLE",
        "evidence_grade": "NONE",
        "design_only": True,
        "result": None,
        "results_available": False,
        "analysis_executed": False,
        "executed": False,
        "analysis_ready": False,
        "claim_boundary": dict(CLAIM_BOUNDARY),
        "unknown": [reason],
    }


def _is_int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _norm_str(v: Any) -> str | None:
    if isinstance(v, str) and v.strip():
        return v.strip()
    return None


# ══ 1. 数据集 schema（只声明"每行必须有什么"） ════════════════════════
def dataset_schema() -> dict:
    """返回冻结的数据集 schema（**声明性**；不含任何统计量）。"""
    return {
        "success": True,
        "version": CASH_STRATEGY_PREREGISTRATION_VERSION,
        "schema_version": DATASET_SCHEMA_VERSION,
        "required_fields": list(REQUIRED_FIELDS),
        "n_required_fields": len(REQUIRED_FIELDS),
        "field_kinds": {
            "record_id": "non_empty_str_unique",
            "shadow_version": "str_equal_to_expected",
            "state_hash": "non_empty_str",
            "street": "str",
            "divergence": "aligned|different|unavailable",
            "recommended_action": "str|None",
            "recommended_amount_bb": "number|None",
            "observed_action": "str|None",
            "observed_amount_bb": "number|None",
            "available": "bool",
            "level": "str",
            "evidence_grade": "str",
            "snapshot_present": "bool",
            "snapshot_method": "str|None",
            "snapshot_samples": "int|None",
            "unknown_tokens": "list[str]",
            "executed": "bool_must_be_false",
        },
        "missingness_policy": dict(MISSINGNESS_POLICY),
        "analysis_unit": dict(ANALYSIS_UNIT),
        "statistics_declared": [],
        "statistics_note": ("schema 只声明列与纳入/排除规则；"
                            "**不**声明任何统计量、效应量或区间。"),
        "evidence_level": "DESIGN_DECLARATION",
        "result": None,
        "executed": False,
    }


def validate_candidate_records(state: Any, records: Any = None, *,
                               expected_shadow_version: str | None = None) -> dict:
    """**结构校验**候选行是否满足 schema（只判定，不统计、不执行）。"""
    if not _is_cash(state):
        return _stub("not_cash_mode_fail_closed")
    if not isinstance(records, (list, tuple)):
        return _stub("records_required")
    if _norm_str(expected_shadow_version) is None:
        return _stub("expected_shadow_version_required")

    per_row: list[dict] = []
    seen_ids: dict[str, int] = {}
    counts = {"valid": 0, "invalid": 0}
    field_missing: dict[str, int] = {}
    field_bad: dict[str, int] = {}

    for i, row in enumerate(records):
        issues: list[str] = []
        if not isinstance(row, dict):
            counts["invalid"] += 1
            per_row.append({"index": i, "valid": False,
                            "issues": ["row_not_a_dict"]})
            field_bad["row_not_a_dict"] = field_bad.get("row_not_a_dict", 0) + 1
            continue
        for f in REQUIRED_FIELDS:
            if f not in row:
                issues.append(f"missing_field:{f}")
                field_missing[f] = field_missing.get(f, 0) + 1
        # schema 声明的"非空字符串"字段必须真正被强制
        for f in ("record_id", "state_hash", "street"):
            if f in row and _norm_str(row.get(f)) is None:
                issues.append(f"{f}_not_non_empty_str")
        rid = _norm_str(row.get("record_id"))
        if rid is not None:
            if rid in seen_ids:
                issues.append("duplicate_record_id")
            else:
                seen_ids[rid] = i
        if _norm_str(row.get("shadow_version")) != \
                expected_shadow_version.strip():
            issues.append("shadow_version_mismatch")
        if _norm_str(row.get("divergence")) not in ("aligned", "different",
                                                    "unavailable"):
            issues.append("divergence_not_in_closed_enum")
        if row.get("executed") is not False:
            issues.append("executed_must_be_false")
        if not isinstance(row.get("available"), bool):
            issues.append("available_not_bool")
        if not isinstance(row.get("snapshot_present"), bool):
            issues.append("snapshot_present_not_bool")
        tok = row.get("unknown_tokens")
        if not isinstance(tok, list) or any(not isinstance(t, str) for t in tok):
            issues.append("unknown_tokens_not_list_of_str")
        ss = row.get("snapshot_samples")
        if ss is not None and not _is_int(ss):
            issues.append("snapshot_samples_not_int_or_none")
        for amt in ("recommended_amount_bb", "observed_amount_bb"):
            v = row.get(amt)
            if v is not None and (isinstance(v, bool)
                                  or not isinstance(v, (int, float))):
                issues.append(f"{amt}_not_number_or_none")
        for act in ("recommended_action", "observed_action"):
            v = row.get(act)
            if v is not None and _norm_str(v) is None:
                issues.append(f"{act}_not_str_or_none")
        if issues:
            counts["invalid"] += 1
        else:
            counts["valid"] += 1
        per_row.append({"index": i, "record_id": rid, "valid": not issues,
                        "issues": issues})

    return {
        "present": True,
        "version": CASH_STRATEGY_PREREGISTRATION_VERSION,
        "schema_version": DATASET_SCHEMA_VERSION,
        "mode": "cash",
        "n_records": len(records),
        "counts": counts,
        "field_missing_counts": dict(sorted(field_missing.items())),
        "field_bad_counts": dict(sorted(field_bad.items())),
        "rows": per_row,
        "all_valid": counts["invalid"] == 0 and len(records) > 0,
        "design_only": True,
        "result": None,
        "results_available": False,
        "analysis_executed": False,
        "executed": False,
        "evidence_level": "DESIGN_DECLARATION",
        "evidence_grade": "NONE",
        "claim_boundary": dict(CLAIM_BOUNDARY),
        "unknown": [],
        "note": ("本函数**只做结构校验**：不插补、不丢弃、不统计、不推断结果。"),
    }


# ══ 2. 结果生成闸门 ══════════════════════════════════════════════════
def run_analysis_preconditions(state: Any, *, closure: Any = None,
                               solver: Any = None,
                               real_execution_data: Any = None,
                               environment: Any = None,
                               min_rows: int = DEFAULT_MIN_ROWS) -> dict:
    """判定**是否允许产生结果**；任何一项不满足 → `would_run=False`。

    **本函数永不产生结果**：即使全部满足，也只返回 `would_run=True` 与
    `would_run_if`，并把 `result` 保持为 `None`（真实执行数据与均衡参考
    在当前阶段均不可得；Solver 为 `BLOCKED`）。
    """
    if not _is_cash(state):
        return _stub("not_cash_mode_fail_closed")
    if not _is_int(min_rows) or min_rows < 1:
        return _stub("min_rows_invalid")

    unmet: list[str] = []
    met: list[str] = []

    def _mark(i: int, ok: bool) -> None:
        (met if ok else unmet).append(RESULT_PRECONDITIONS[i])

    # 1. V3H-11 关闭判定
    cl_ok = isinstance(closure, dict) and closure.get("present") is True \
        and closure.get("closure_possible") is True
    _mark(0, cl_ok)

    # 2. 独立 blocker 全部解除（本层不测量，只读调用方给定事实）
    blockers = closure.get("closure_test", {}).get("established_blockers") \
        if isinstance(closure, dict) else None
    blockers_resolved = (isinstance(blockers, list) and not blockers)
    _mark(1, blockers_resolved)

    # 3. 求解器可用且提供均衡参考（当前恒为 BLOCKED → None）
    solver_ok = (isinstance(solver, dict)
                 and solver.get("solver_status") == "AVAILABLE"
                 and solver.get("equilibrium") is True
                 and solver.get("available") is True)
    _mark(2, solver_ok)

    # 4. 真实执行数据（**主动拒绝消费**）
    _mark(3, False)

    # 5. 环境事实齐全
    env_ok = isinstance(environment, dict) and environment and all(
        v is True for v in environment.values())
    _mark(4, env_ok)

    # 6-9. 数据集面（本函数未接收数据集 → 一律不满足，不猜）
    for i in (5, 6, 7, 8):
        _mark(i, False)

    refused = real_execution_data is not None
    would_run = False      # 硬编码：任何未满足项存在时不产生结果

    return {
        "present": True,
        "version": CASH_STRATEGY_PREREGISTRATION_VERSION,
        "mode": "cash",
        "would_run": would_run,
        "met": met,
        "unmet": unmet,
        "would_run_if": list(RESULT_PRECONDITIONS),
        "n_met": len(met),
        "n_unmet": len(unmet),
        "result": None,
        "results_available": False,
        "analysis_executed": False,
        "executed": False,
        "result_generation_refused": refused,
        "result_refusal_reason": (
            "real execution data must not be consumed by a design-only layer; "
            "no real result may be produced while M3/B1/B3/B4/B5/solver remain "
            "BLOCKED" if refused else
            "no real execution data; no solver equilibrium reference; "
            "independent blockers unresolved"),
        "design_only": True,
        "analysis_ready": False,
        "evidence_level": "DESIGN_DECLARATION",
        "evidence_grade": "NONE",
        "uncertainty": {
            "measured": False,
            "note": ("未运行任何分析 → **没有**可报告的统计不确定性；"
                     "此处只记录**未决条件**。"),
            "unknowns": list(unmet),
        },
        "claim_boundary": dict(CLAIM_BOUNDARY),
        "unknown": [],
        "note": ("闸门本身**不**产生结果；`result` 恒为 `None`。"),
    }


# ══ 3. 预注册文档（设计面） ═══════════════════════════════════════════
def build_preregistration(state: Any, *, closure: Any = None,
                          eligibility: Any = None,
                          solver: Any = None,
                          environment: Any = None,
                          min_rows: int = DEFAULT_MIN_ROWS) -> dict:
    """构建 CASH 预注册分析基础（**设计面**；无结果）。

    Parameters
    ----------
    state : CASH state（非 CASH → present=False）
    closure : **必填**，V3H-11 `audit_evidence_closure` 输出
        （`analysis_ready` 直接取它的 `closure_possible`，**不重算**）
    eligibility : 可选，V3H-9 输出序列（只记录**计数**）
    solver : 可选，调用方给定的求解器状态（只读；`BLOCKED` → 各量 `None`）
    environment : 可选，调用方环境事实（只计数 missing）
    min_rows : 显式的最小行数（默认引用既有 B1 阈值 1200）
    """
    if not _is_cash(state):
        return _stub("not_cash_mode_fail_closed")
    if not isinstance(closure, dict) or closure.get("present") is not True:
        return _stub("closure_required")
    if not _is_int(min_rows) or min_rows < 1:
        return _stub("min_rows_invalid")

    # ── 转述 V3H-11（不重算）──
    ct = closure.get("closure_test") if isinstance(
        closure.get("closure_test"), dict) else {}
    closure_possible = closure.get("closure_possible") is True
    established = ct.get("established_blockers") or []
    n_records = closure.get("n_records") if _is_int(closure.get("n_records")) \
        else 0
    rows_short = ct.get("records_short_by") if _is_int(
        ct.get("records_short_by")) else None

    # ── 求解器（只读；BLOCKED → None，不猜）──
    solver_blocked = not (isinstance(solver, dict)
                          and solver.get("solver_status") == "AVAILABLE")
    solver_view = {
        "solver_status": (solver.get("solver_status")
                          if isinstance(solver, dict) else "UNKNOWN"),
        "available": bool(solver.get("available")) if isinstance(solver, dict)
        else False,
        "discovery_state": (solver.get("discovery_state")
                            if isinstance(solver, dict) else None),
        "blocked": solver_blocked,
        "equilibrium": None,
        "gto": None,
        "exploitability": None,
        "note": ("求解器不可用 / 未提供均衡参考 → 相关量记 None（**不猜**）。"
                 "本层不调用任何求解器。"),
    }

    # ── 资格结果只计数 ──
    n_elig = n_elig_ineligible = 0
    if isinstance(eligibility, (list, tuple)):
        for e in eligibility:
            if isinstance(e, dict) and e.get("present") is True:
                n_elig += 1
                if e.get("eligible") is not True:
                    n_elig_ineligible += 1

    environment_missing: list[str] = []
    if isinstance(environment, dict):
        environment_missing = sorted(k for k, v in environment.items()
                                     if v is not True)

    analysis_ready = bool(closure_possible)      # 恒为 False（当前阶段）
    unknown: list[str] = []
    if not closure_possible:
        unknown.append("closure_not_possible")
    if unknown_tokens := (closure.get("counts", {}) or {}).get(
            "unknown_reasons"):
        unknown.append(f"upstream_unknown_reasons:{len(unknown_tokens)}")
    if not isinstance(solver, dict):
        unknown.append("solver_status_not_supplied")
    if environment_missing:
        unknown.append(f"environment_missing:{len(environment_missing)}")

    return {
        "present": True,
        "version": CASH_STRATEGY_PREREGISTRATION_VERSION,
        "mode": "cash",
        "design_only": True,
        "design_version_frozen": True,
        "analysis_ready": analysis_ready,
        "analysis_ready_source": "V3H-11 closure.closure_possible (transcribed)",
        "schema": dataset_schema(),
        "analysis_unit": dict(ANALYSIS_UNIT),
        "hypotheses": [dict(h) for h in HYPOTHESES],
        "endpoints": [dict(e) for e in ENDPOINTS],
        "missingness_policy": dict(MISSINGNESS_POLICY),
        "multiple_comparisons": {
            "policy": "declare_all_endpoints_upfront",
            "n_endpoints": len(ENDPOINTS),
            "correction_method": None,
            "note": ("未运行任何比较 → 不选择、不实现任何多重比较校正方法"
                     "（避免事后选择）。"),
        },
        "stopping_rule": {
            "kind": "no_early_stopping_declared",
            "min_rows": int(min_rows),
            "min_rows_source": ("declared by the caller; the conventional 1200 "
                                "is the pre-existing B1 threshold, NOT measured "
                                "by this layer"),
            "rows_observed": n_records,
            "rows_short_by": rows_short,
            "note": "本层**不**做功效计算、**不**估计所需样本量。",
        },
        "blockers": {k: dict(v) for k, v in BLOCKERS.items()},
        "blockers_from_upstream": list(established),
        "blockers_measured_by_this_layer": False,
        "eligibility_counts": {
            "n_supplied": n_elig,
            "n_not_eligible": n_elig_ineligible,
            "note": "资格结果仅计数；本层**不**复制 V3H-9 判定逻辑。",
        },
        "environment_missing": environment_missing,
        "provenance": {
            "evidence_level": "DESIGN_DECLARATION",
            "evidence_grade": "NONE",
            "source_components": [
                {"stage": "V3H-11", "role": "closure / gap facts",
                 "version": closure.get("version"),
                 "consumed_as": "transcribed"},
                {"stage": "V3H-9", "role": "eligibility results",
                 "consumed_as": "counted_only"},
                {"stage": "V3H-12", "role": "design surface",
                 "version": CASH_STRATEGY_PREREGISTRATION_VERSION},
            ],
            "frozen_multistreet_or_sample_runouts_reused": False,
            "future_cards_generated": False,
            "runout_sampling_performed": False,
            "real_execution_data_consumed": False,
            "solver_invoked": False,
        },
        "solver": solver_view,
        "preconditions": run_analysis_preconditions(
            state, closure=closure, solver=solver, environment=environment,
            min_rows=min_rows),
        "result": None,
        "results_available": False,
        "analysis_executed": False,
        "executed": False,
        "result_generation_refused": True,
        "result_refusal_reason": (
            "design-only layer: no result may be produced without real "
            "execution data AND a solver equilibrium reference AND resolution "
            "of M3/B1/B3/B4/B5"),
        "evidence_level": "DESIGN_DECLARATION",
        "evidence_grade": "NONE",
        "evidence_levels": list(EVIDENCE_LEVELS),
        "uncertainty": {
            "measured": False,
            "note": ("未运行任何分析 → **无**统计不确定性可报告；"
                     "仅记录设计面的未决条件。"),
            "design_unknowns": unknown,
        },
        "claim_boundary": dict(CLAIM_BOUNDARY),
        "unknown": unknown,
        "note": ("本层只提供**预注册分析基础**：不产生结果、不计算统计量、"
                 "不判优劣、不称盈利、不做因果主张、不执行任何动作。"),
    }
