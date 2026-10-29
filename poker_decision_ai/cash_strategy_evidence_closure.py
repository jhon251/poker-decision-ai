"""V3H-11 — CASH Shadow 证据闭合审计（对账记录**总体**的描述性聚合 + 缺口清单）。

**为什么这是 V3H-11 的最小下一能力（审计结论）**
--------------------------------------------------
实测现状（14 个 V3H 策略层）：

| 已有 | 实测缺口 |
|---|---|
| V3H-8 `assemble_decision` 产出可审计建议 | — |
| V3H-9 `audit_experiment_eligibility` 逐决策判断实验入口 | — |
| V3H-10 `evaluate_shadow_decision` 产出**单条**"建议 ↔ 已观测动作"对账记录 | — |
| — | **没有任何层把多条 shadow 记录聚合起来**（探针实测：`aggregate` / `n_records` / `cohort` / `batch` / `accumulate` / `alignment_rate` / `coverage_over` / `confidence_interval` 在 14 层中命中 **全为 0**；`divergence` 只命中 V3H-10 自身） |

因此**缺"总体视图"**：单条记录无法回答"一批决策点里有多少可对账、有多少因
UNKNOWN/弃权/分歧而 fail closed、以及**要形成可验证闭环还缺什么**"。
本层补这一环，**不新增任何评分/EV 公式**，也**不**做任何收益推断。

**严格定位（必须如实标注）**
----------------------------
本层输出的是**描述性计数**：

* 只**计数**，不计算比率、百分比、均值、收益、EV 差或因果效应
  （`counts_only=True`、`rates_forbidden=True`，由测试 AST 断言）；
* `divergence` 计数**不是**策略优劣，也**不是**收益证据；
* `uncertainty` 逐条如实汇总样本量与方法；
* 缺口清单只说"**还缺什么**"，**不**声称已经可实验、**不**声称能盈利。

硬约束
------
* CASH-only，fail closed；非 dict / 非 CASH → `present=False`；
* **不 import 任何 V3G / V3H 层**（任何位置）；不 import 伪随机数模块；无 IO；
* 同输入 → 同输出（reason 排序、键序固定）；
* UNKNOWN / UNAVAILABLE / abstention / disagreement 原样传播并计入缺口；
* 不执行任何动作；不产生收益/可利用度结论；
* Solver `BLOCKED` 时 `exploitability=None`、`gto=None`、`equilibrium=None`
  （**不猜**），且 `closure_test` 记 `solver_blocked_and_no_equilibrium_reference`；
* `MODEL_INTERNAL` 仅描述**证据等级**，**不是**真实对手证据，也**不是**真实盈利证据。
"""
from __future__ import annotations

from typing import Any

CASH_STRATEGY_EVIDENCE_CLOSURE_VERSION = (
    "v3h11-cash-strategy-evidence-closure-v1")

#: V3H-10 冻结的输出键（本层只读这些；缺失 → UNKNOWN，不猜）
SHADOW_RESULT_KEYS: tuple[str, ...] = (
    "present", "version", "mode", "street", "divergence", "divergence_classes",
    "record", "available", "level", "evidence_grade", "uncertainty", "unknown",
    "executed", "claim_boundary", "note",
)

#: 分歧类别固定顺序（与 V3H-10 冻结枚举一致）
DIVERGENCE_KEYS: tuple[str, ...] = ("aligned", "different", "unavailable")

#: 缺口维度（封闭）
GAP_DIMENSIONS: tuple[str, ...] = (
    "divergence",      # 记录不可对账 / 不对齐
    "abstention",      # 上游弃权
    "disagreement",    # 上游 scorer 分歧
    "unknown_reason",  # 未解析的 UNKNOWN 原因
    "evidence_count",  # 记录数不足
    "solver",          # 求解器不可用且无均衡参考
)

#: 关闭判定所依据的既有缺口（V3H-1～V3H-9 实测，不由本层测量）
ESTABLISHED_GAPS: dict[str, dict] = {
    "B1": {
        "description": ("independent decision points for experiment evidence "
                        "(>= 1200 required) - independently measured as "
                        "UNAVAILABLE at V3H-2/V3H-3; this layer counts only "
                        "how many of the supplied records can be reconciled, "
                        "it does NOT measure B1"),
        "status": "BLOCKED",
        "measured_by_this_layer": False,
    },
    "B3": {"description": "real profit evidence",
           "status": "BLOCKED", "measured_by_this_layer": False},
    "B4": {"description": "reserved downstream blocker",
           "status": "BLOCKED", "measured_by_this_layer": False},
    "B5": {"description": "reserved downstream blocker",
           "status": "BLOCKED", "measured_by_this_layer": False},
    "solver": {"description": ("6-max CASH exact solver / equilibrium "
                               "reference; discovery_state=UNAVAILABLE"),
               "status": "BLOCKED", "measured_by_this_layer": False},
}

DEFAULT_MIN_RECORDS = 1200

CLAIM_BOUNDARY: dict[str, bool] = {
    "is_gto": False,
    "is_equilibrium": False,
    "is_solver_output": False,
    "is_profit_evidence": False,
    "is_causal_claim": False,
    "is_strategy_quality_claim": False,
    "is_real_opponent_evidence": False,
    "computes_rates_or_percentages": False,
    "adds_scoring_or_ev_formula": False,
    "executes_action": False,
    "auto_betting": False,
    "claims_experiment_ready": False,
    "wired_to_production": False,
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
        "version": CASH_STRATEGY_EVIDENCE_CLOSURE_VERSION,
        "reason": reason,
        "evidence_grade": "NONE",
        "n_records": 0,
        "unknown": [reason],
        "gaps": [],
        "closure_test": None,
        "closure_possible": False,
        "executed": False,
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }


def _norm_divergence(v: Any) -> str:
    if isinstance(v, str) and v.strip().lower() in DIVERGENCE_KEYS:
        return v.strip().lower()
    return "unavailable"


def _bump(d: dict, k: Any) -> None:
    key = k if isinstance(k, str) and k.strip() else "<empty>"
    d[key] = d.get(key, 0) + 1


def _reason_token(u: Any) -> str:
    if not isinstance(u, str) or not u.strip():
        return "<empty>"
    s = u.strip()
    return s.split(":", 1)[0]


def audit_evidence_closure(state: Any, *, shadow_records: Any = None,
                           shadow_version: str | None = None,
                           eligibility: Any = None,
                           environment: Any = None,
                           min_records: int = DEFAULT_MIN_RECORDS) -> dict:
    """对一批 V3H-10 shadow 结果做**描述性计数聚合**并列出缺口。

    Parameters
    ----------
    state : CASH state（非 CASH → present=False）
    shadow_records : **必填**，`evaluate_shadow_decision(...)` 返回值的**序列**
    shadow_version : 可选，期望的 V3H-10 版本串；提供则逐条比对（不符 → 拒绝）
    eligibility : 可选，V3H-9 `audit_experiment_eligibility` 输出的序列
        （只用于**计数**，不重算、不复制其判定逻辑）
    environment : 可选，**调用方提供**的环境事实字典 → **仅**用于**计数**
        （不调用任何闸门；本层不 import V3G-10）
    min_records : 关闭判定所需的独立决策点数量（显式；默认 1200）

    Returns
    -------
    dict  仅含计数、缺口与关闭条件；**不含**任何比率 / 收益 / EV 字段。
    """
    if not _is_cash(state):
        return _stub("not_cash_mode_fail_closed")
    if not isinstance(shadow_records, (list, tuple)):
        return _stub("shadow_records_required")
    if not isinstance(min_records, int) or isinstance(min_records, bool) \
            or min_records < 1:
        return _stub("min_records_invalid")
    # 显式拒绝调用方把上限口径搞混：min_records 必须是期望的独立决策点数
    if isinstance(state.get("shadow_records"), (list, tuple)):
        return _stub("state_shadow_records_not_accepted_use_explicit_arg")

    unknown: list[str] = []

    # ── 逐条消费 V3H-10 输出（只读，不重算）──
    n_present = n_stub = 0
    divergence: dict[str, int] = {k: 0 for k in DIVERGENCE_KEYS}
    stub_reasons: dict[str, int] = {}
    unknown_reasons: dict[str, int] = {}
    streets: dict[str, int] = {}
    recommended: dict[str, int] = {}
    observed: dict[str, int] = {}
    evidence_grades: dict[str, int] = {}
    levels: dict[str, int] = {}
    snapshot_present = snapshot_absent = 0
    methods: dict[str, int] = {}
    samples: list[int] = []
    bounds: list[float] = []
    n_executed_true = 0
    n_forbidden_true = 0
    n_unreconcilable = 0
    n_eligibility_not_eligible = 0
    n_abstained = 0
    n_disagreement = 0
    n_version_mismatch = 0
    n_non_dict = 0

    for item in shadow_records:
        if not isinstance(item, dict):
            n_non_dict += 1
            divergence["unavailable"] += 1
            _bump(stub_reasons, "shadow_result_not_a_dict")
            continue

        if shadow_version is not None and item.get("version") != shadow_version:
            n_version_mismatch += 1
            divergence["unavailable"] += 1
            _bump(stub_reasons, "shadow_version_mismatch")
            continue

        if item.get("present") is not True:
            n_stub += 1
            divergence["unavailable"] += 1
            _bump(stub_reasons, item.get("reason") or "unknown")
            for u in (item.get("unknown") or []):
                _bump(unknown_reasons, _reason_token(u))
            continue

        n_present += 1
        div = _norm_divergence(item.get("divergence"))
        divergence[div] += 1
        _bump(streets, item.get("street") or "<none>")
        _bump(evidence_grades, str(item.get("evidence_grade") or "<none>"))
        _bump(levels, str(item.get("level") or "<none>"))
        for u in (item.get("unknown") or []):
            tok = _reason_token(u)
            _bump(unknown_reasons, tok)
            if tok.startswith("decision_abstained"):
                n_abstained += 1
            if tok.startswith("decision_disagreements"):
                n_disagreement += 1

        rec = item.get("record") if isinstance(item.get("record"), dict) else {}
        if not rec:
            if "record_missing" not in unknown:
                unknown.append("record_missing")
            n_unreconcilable += 1
        recd = rec.get("recommended")
        obsd = rec.get("observed")
        _bump(recommended,
              (recd or {}).get("action") if isinstance(recd, dict)
              else "<non-dict>" if recd is not None else "<none>")
        _bump(observed,
              (obsd or {}).get("action") if isinstance(obsd, dict)
              else "<non-dict>" if obsd is not None else "<none>")
        if isinstance(recd, dict) and recd.get("amount_bb") is not None:
            bounds.append(float(recd["amount_bb"]))
        if rec.get("executed") is True:
            n_executed_true += 1
        for flag in ("is_gto", "is_equilibrium", "is_solver_output",
                     "is_profit_evidence", "is_causal_claim"):
            if rec.get(flag) is True:
                n_forbidden_true += 1

        snap = rec.get("shadow_snapshot") if isinstance(
            rec.get("shadow_snapshot"), dict) else {}
        if snap.get("present") is True:
            snapshot_present += 1
            _bump(methods, snap.get("method") or "<none>")
            if isinstance(snap.get("samples"), int) and not isinstance(
                    snap.get("samples"), bool):
                samples.append(int(snap["samples"]))
        else:
            snapshot_absent += 1

    for it in (eligibility if isinstance(eligibility, (list, tuple)) else []):
        if isinstance(it, dict) and it.get("present") is True:
            if it.get("eligible") is not True:
                n_eligibility_not_eligible += 1
            if it.get("status") == "BLOCKED":
                _bump(unknown_reasons, "eligibility_blocked")

    # ── 关闭判定（**只**返回条件，不声称已可实验、不声称能盈利）──
    n_records = len(shadow_records)
    count_short = max(0, int(min_records) - n_records)
    established_blockers = [k for k, v in ESTABLISHED_GAPS.items()
                            if v.get("status") == "BLOCKED"]
    closure_reasons: list[str] = []
    gap_counts: dict[str, int] = {k: 0 for k in GAP_DIMENSIONS}
    gap_counts["divergence"] = divergence["different"] + \
        divergence["unavailable"]
    gap_counts["abstention"] = n_abstained
    gap_counts["disagreement"] = n_disagreement
    gap_counts["unknown_reason"] = len(unknown_reasons)
    gap_counts["evidence_count"] = count_short
    gap_counts["solver"] = 1
    if gap_counts["divergence"]:
        closure_reasons.append(
            f"unreconciled_or_divergent_records:{gap_counts['divergence']}")
    if gap_counts["abstention"]:
        closure_reasons.append(f"abstentions:{gap_counts['abstention']}")
    if gap_counts["disagreement"]:
        closure_reasons.append(f"disagreements:{gap_counts['disagreement']}")
    if unknown_reasons:
        closure_reasons.append(
            f"unknown_reasons_present:{len(unknown_reasons)}")
    if n_non_dict:
        closure_reasons.append(f"malformed_shadow_results:{n_non_dict}")
    if n_version_mismatch:
        closure_reasons.append(
            f"version_mismatched_shadow_results:{n_version_mismatch}")
    if n_unreconcilable:
        closure_reasons.append(f"unreconcilable_records:{n_unreconcilable}")
    if n_eligibility_not_eligible:
        closure_reasons.append(
            f"eligibility_not_eligible:{n_eligibility_not_eligible}")
    if count_short:
        closure_reasons.append(f"records_below_minimum:{count_short}")
    if n_executed_true:
        closure_reasons.append("an_upstream_record_claims_executed")
    if n_forbidden_true:
        closure_reasons.append("a_forbidden_claim_flag_is_true")

    environment_missing: tuple[str, ...] = ()
    if isinstance(environment, dict):
        environment_missing = tuple(sorted(
            k for k, v in environment.items() if v is not True))
    else:
        closure_reasons.append("environment_facts_not_supplied")

    closure_reasons.append("solver_blocked_and_no_equilibrium_reference")
    closure_reasons.append(
        "established_blockers_unresolved:" + ",".join(established_blockers))

    closure_test = {
        "records_supplied": n_records,
        "records_expected_min": int(min_records),
        "records_short_by": count_short,
        "counts_by_divergence": dict(divergence),
        "n_abstained": n_abstained,
        "n_disagreement": n_disagreement,
        "n_unknown_reasons": len(unknown_reasons),
        "n_unreconcilable": n_unreconcilable,
        "n_eligibility_not_eligible": n_eligibility_not_eligible,
        "n_malformed": n_non_dict + n_version_mismatch,
        "environment_missing": list(environment_missing),
        "established_blockers": established_blockers,
        "established_blockers_measured_by_this_layer": False,
        "gap_counts": gap_counts,
        "would_close_if": [
            "records_supplied >= records_expected_min",
            "counts_by_divergence['different'] == 0",
            "counts_by_divergence['unavailable'] == 0",
            "n_abstained == 0",
            "n_disagreement == 0",
            "n_unknown_reasons == 0",
            "n_unreconcilable == 0",
            "n_malformed == 0",
            "environment_missing == []",
            "no_independent_blocker_remains (B1 >= 1200 independent decision "
            "points measured OUTSIDE this layer; solver discovers an "
            "equilibrium)",
        ],
        "closure_reasons": [
            "a_solver_with_an_equilibrium_reference_is_still_required"] +
            closure_reasons,
        "note": ("本清单只说明**还缺什么**；**不**表示已可实验，"
                 "**不**表示可获得收益，**不**构成策略优劣结论。"),
    }
    closure_possible = False      # solver BLOCKED ⇒ 恒为 False（不猜、不翻转）

    # 全零计数 + 证据充足 + 无缺口时，is_complete 记录"聚合本身完整"，
    # 但 closure_possible 仍为 False，因为存在独立 blocker。
    is_complete = bool(
        not closure_reasons and count_short == 0
        and divergence["different"] == 0 and divergence["unavailable"] == 0
        and n_abstained == 0 and n_disagreement == 0
        and not unknown_reasons and n_unreconcilable == 0
        and n_eligibility_not_eligible == 0
        and n_non_dict == 0 and n_version_mismatch == 0)

    return {
        "present": True,
        "version": CASH_STRATEGY_EVIDENCE_CLOSURE_VERSION,
        "mode": "cash",
        "n_records": n_records,
        "n_shadow_present": n_present,
        "n_shadow_stub": n_stub,
        "counts_only": True,
        "rates_forbidden": True,
        "counts": {
            "by_divergence": dict(divergence),
            "shadow_stub_reasons": dict(sorted(stub_reasons.items())),
            "unknown_reasons": dict(sorted(unknown_reasons.items())),
            "by_street": dict(sorted(streets.items())),
            "recommended_actions": dict(sorted(recommended.items())),
            "observed_actions": dict(sorted(observed.items())),
            "by_evidence_grade": dict(sorted(evidence_grades.items())),
            "by_levels": dict(sorted(levels.items())),
            "snapshot_present": snapshot_present,
            "snapshot_absent": snapshot_absent,
            "snapshot_methods": dict(sorted(methods.items())),
            "n_records_claiming_executed": n_executed_true,
            "n_forbidden_claim_flags_true": n_forbidden_true,
        },
        "observed_range": {
            "recommended_amount_bb": {
                "n": len(bounds),
                "min": (min(bounds) if bounds else None),
                "max": (max(bounds) if bounds else None),
            },
            "note": ("仅记录**已观测范围**（min/max），**不**计算均值 / 中位数 / "
                     "分布统计，**不**推断收益。"),
        },
        "tolerance_bb": {
            "note": ("amount_tolerance_bb 由**调用方**显式给出；"
                     "本层不校验、不聚合该取值。"),
        },
        "uncertainty": {
            "snapshot_samples_count": len(samples),
            "snapshot_samples_min": (min(samples) if samples else None),
            "snapshot_samples_max": (max(samples) if samples else None),
            "snapshot_methods": dict(sorted(methods.items())),
            "monte_carlo_inherited_from_v3h5": True,
            "solver": {"solver_status": "BLOCKED", "available": False,
                       "equilibrium": None, "discovery_state": "UNAVAILABLE",
                       "uncertainty_is_declared_not_measured_by_this_layer": True},
            "exploitability": None,
            "gto": None,
            "equilibrium": None,
            "note": ("求解器不可用且无均衡参考：相关量记为 None（**不猜**）。"
                     "`MODEL_INTERNAL` 仅为证据等级。"),
        },
        "rate": None,
        "pct": None,
        "rate_computed": False,
        "rate_refusal_reason": ("禁止把对账记录折算为比率 / 百分比："
                                "对齐率会被误读为策略质量或收益指标。"),
        "gaps": [
            {"dimension": k, "count_or_flag": gap_counts[k]}
            for k in GAP_DIMENSIONS
        ],
        "closure_test": closure_test,
        "closure_possible": closure_possible,
        "is_complete": is_complete,
        "evidence_grade": ("MODEL_INTERNAL" if n_present else "NONE"),
        "executed": False,
        "claim_boundary": dict(CLAIM_BOUNDARY),
        "unknown": unknown,
        "note": ("本层只做**描述性计数**与**缺口列举**：不计算比率、不判优劣、"
                 "不称盈利、不做因果主张、不执行任何动作。"),
    }
