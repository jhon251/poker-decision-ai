"""V3H-14 — CASH 外部就绪度审计（验证前检查 + 外部依赖清单）。

**为什么这是 V3H-14 的最小必要能力（源码级缺口审计结论）**
------------------------------------------------------------
对**既有 17 个** V3H 层做源码探针（排除本阶段新模块）：

```
readiness / resolvable / resolution_class / blocked_by         0 个文件
who_can_resolve / owner / required_input / prerequisite        0 个文件
satisf / unsatisf / remaining_gap / resource_required          0 个文件
no_internal_substitute / externally / preflight / chain_audit  0 个文件

resolution  1 个文件 —— V3H-12 中作为普通 dict 键，非"由谁解决"的分类
out_of_band 1 个文件 —— V3H-13 的信任锚说明文字，非依赖清单
```

即：既有各层**各自**知道自己缺什么（V3H-11 缺口清单 / V3H-13 前置条件），
但**没有任何层**回答最关键的问题：

* 当前阻塞项**分别由谁解决**（内部 vs **外部资源**）；
* 哪些是**只能靠外部**、**不可能**用内部代码替代的；
* 整条链到"真实验证"还差什么；
* 因此下一步是**可内部实现**，还是**依赖外部资源而必须在法律/合规意义上等待**。

**本层的核心结论（诚实返回，不制造内部替代品）**
------------------------------------------------
经审计：**4 项外部依赖全部未满足**（精确解算器与均衡参考、真实执行记录、
B1 独立性测量、真实盈亏证据），且**内部代码无法替代任何一项**。
因此 `verdict = "BLOCKED"`：**下一步依赖外部资源，无法在内部合法实现**。

本层**不**提供、**不**模拟、**不**伪造任何外部资源，
**不**声称已就绪，**不**生成任何结果。

硬约束
------
* CASH-only，fail closed；非 dict / 非 CASH → `present=False`；
* **不 import 任何 V3G / V3H 层**（任何位置）；不 import 伪随机数模块；**无 IO**；
* 同输入 → 同输出（键序、列表序固定）；
* UNKNOWN / UNAVAILABLE / abstention / disagreement 原样传播并阻断就绪；
* Solver `BLOCKED` → `equilibrium` / `gto` / `exploitability` 恒为 `None`（**不猜**）；
* **不**把 SHA-256 / 内容指纹当作真实性、认证或来源证明；
* 外部信任锚**必须**保持带外；本层**不**伪造、**不**代填、**不**自行判定；
* **不**生成真实执行结果 / 收益结果 / 统计结果 / 因果结论；
* `MODEL_INTERNAL` 等内部证据**不得**被当作外部依赖的替代品。
"""
from __future__ import annotations

from typing import Any

CASH_STRATEGY_EXTERNAL_READINESS_VERSION = (
    "v3h14-cash-strategy-external-readiness-v1")

#: 已构建的 CASH 决策链（**静态**声明；本层不重算上游任何结果）
PIPELINE_STAGES: tuple[dict[str, str], ...] = (
    {"stage": "V3H-4", "layer": "cash_strategy_action_policy",
     "role": "action policy / candidate actions"},
    {"stage": "V3H-5", "layer": "cash_strategy_action_ranking",
     "role": "action ranking over the policy"},
    {"stage": "V3H-6", "layer": "cash_strategy_likelihood_model",
     "role": "declarative likelihood model + belief update"},
    {"stage": "V3H-7", "layer": "cash_strategy_search_tree",
     "role": "bounded multistreet search"},
    {"stage": "V3H-8", "layer": "cash_strategy_decision_assembly",
     "role": "cross-scorer reconciled recommendation"},
    {"stage": "V3H-9", "layer": "cash_strategy_experiment_eligibility",
     "role": "per-decision experiment eligibility"},
    {"stage": "V3H-10", "layer": "cash_strategy_shadow_evaluation",
     "role": "single-decision shadow reconciliation"},
    {"stage": "V3H-11", "layer": "cash_strategy_evidence_closure",
     "role": "descriptive aggregate + gap list"},
    {"stage": "V3H-12", "layer": "cash_strategy_preregistration",
     "role": "pre-registration design surface + result gate"},
    {"stage": "V3H-13", "layer": "cash_strategy_execution_intake",
     "role": "pre-execution intake gate + integrity digest"},
)

#: 就绪判定所需的**输入**（全部由调用方显式提供；本层不自行获取）
REQUIRED_INPUTS: tuple[str, ...] = (
    "closure",           # V3H-11 audit_evidence_closure 输出
    "preregistration",   # V3H-12 build_preregistration 输出
    "intake",            # V3H-13 intake_execution_records 输出
    "solver",            # 调用方给定的求解器状态（只读）
    "environment",       # 调用方环境事实
)

#: 解析类别（封闭枚举）
RESOLUTION_CLASSES: tuple[str, ...] = (
    "SATISFIED",                # 已满足
    "BLOCKED_INTERNAL",         # 内部可解但当前未解
    "BLOCKED_EXTERNAL_RESOURCE",  # 依赖**外部资源**，内部无法合法替代
    "NOT_APPLICABLE",           # 不适用
)

#: 就绪判定（封闭枚举）
READINESS_VERDICTS: tuple[str, ...] = (
    "READY",      # 全部满足（当前**不可能**达到）
    "BLOCKED",    # 至少一项依赖外部资源且未满足
    "PARTIAL",    # 无外部依赖未满足，但仍有内部可解项未完成
)

#: 来源等级（**仅**用于标注证据等级，不用于替代外部资源）
EVIDENCE_LEVELS: tuple[str, ...] = (
    "EXTERNAL_RESOURCE_FACT",   # 外部资源事实（由带外提供）
    "INTERNAL_DECLARATION",     # 内部声明（设计/机制）
    "UNAVAILABLE",              # 不可用
)

#: **外部依赖清单**（本层核心产物；每一项都**不可能**用内部代码替代）
EXTERNAL_REQUIREMENTS: tuple[dict[str, Any], ...] = (
    {
        "id": "R1",
        "blocker": "solver",
        "title": "exact 6-max CASH solver with an equilibrium reference",
        "kind": "external_resource",
        "resolution_class": "BLOCKED_EXTERNAL_RESOURCE",
        "satisfiable_by_this_project_internally": False,
        "no_internal_substitute": True,
        "internal_substitutes_forbidden": [
            "fabricating equilibrium or GTO values",
            "declaring exploitability without a solver",
            "approximating an equilibrium and presenting it as exact",
        ],
        "required_input": ("an installed postflop solver reachable through the "
                           "existing discovery channel (POSTFLOP_SOLVER_PATH) "
                           "plus a completed solve"),
        "observable_check": "solver_status(...).solver_status == 'AVAILABLE'",
        "current_state": "discovery_state=UNAVAILABLE; solver_status=BLOCKED",
        "evidence_level": "EXTERNAL_RESOURCE_FACT",
        "measured_by_this_layer": False,
    },
    {
        "id": "R2",
        "blocker": "real_execution_data",
        "title": "real execution records captured from actual play",
        "kind": "external_resource",
        "resolution_class": "BLOCKED_EXTERNAL_RESOURCE",
        "satisfiable_by_this_project_internally": False,
        "no_internal_substitute": True,
        "internal_substitutes_forbidden": [
            "synthetic or simulated execution environments",
            "mock executors or paper records presented as real",
            "self-generated records fed back into the intake gate",
            "inferring realised outcomes from model-internal data",
        ],
        "required_input": ("records captured by an external executor under a "
                           "real environment, delivered with an out-of-band "
                           "trust anchor"),
        "observable_check": ("intake_execution_records(...).intake_permitted "
                             "is true with non-zero admitted records"),
        "current_state": "not supplied; the intake gate refuses everything",
        "evidence_level": "EXTERNAL_RESOURCE_FACT",
        "measured_by_this_layer": False,
    },
    {
        "id": "R3",
        "blocker": "B1",
        "title": "at least 1200 independent decision points",
        "kind": "external_resource",
        "resolution_class": "BLOCKED_EXTERNAL_RESOURCE",
        "satisfiable_by_this_project_internally": False,
        "no_internal_substitute": True,
        "internal_substitutes_forbidden": [
            "counting correlated records as independent",
            "inflating the count with generated or repeated records",
            "asserting independence without an external measurement",
        ],
        "required_input": ("an independent measurement procedure over real "
                           "decision points (independently measured at V3H-2 / "
                           "V3H-3 as unavailable, 0/1200)"),
        "observable_check": ("a measured independence count at or above 1200 "
                             "supplied by an external process"),
        "current_state": "independently measured as UNAVAILABLE (0/1200)",
        "evidence_level": "EXTERNAL_RESOURCE_FACT",
        "measured_by_this_layer": False,
    },
    {
        "id": "R4",
        "blocker": "B3",
        "title": "real profit evidence",
        "kind": "external_resource",
        "resolution_class": "BLOCKED_EXTERNAL_RESOURCE",
        "satisfiable_by_this_project_internally": False,
        "no_internal_substitute": True,
        "internal_substitutes_forbidden": [
            "deriving profit from shadow divergence counts",
            "converting alignment or divergence rates into PnL",
            "presenting model-internal equity as realised profit",
        ],
        "required_input": ("realised monetary outcomes from real play, joined "
                           "to the decision points"),
        "observable_check": "an externally audited realised-outcome dataset",
        "current_state": "not supplied; B3 remains BLOCKED",
        "evidence_level": "EXTERNAL_RESOURCE_FACT",
        "measured_by_this_layer": False,
    },
    {
        "id": "R5",
        "blocker": "trust_anchor",
        "title": "out-of-band trust anchor for externally produced records",
        "kind": "external_resource",
        "resolution_class": "BLOCKED_EXTERNAL_RESOURCE",
        "satisfiable_by_this_project_internally": False,
        "no_internal_substitute": True,
        "internal_substitutes_forbidden": [
            "treating a SHA-256 content digest as proof of authenticity",
            "self-issuing a trust anchor inside this project",
            "treating the caller flag trust_anchor_verified=True as verified "
            "by this layer",
        ],
        "required_input": ("an out-of-band attestation or provenance chain "
                           "established outside this project"),
        "observable_check": ("trust_anchor_verified supplied by the caller AND "
                             "independently attributable to an external party"),
        "current_state": "no out-of-band anchor exists",
        "evidence_level": "EXTERNAL_RESOURCE_FACT",
        "measured_by_this_layer": False,
    },
    {
        "id": "R6",
        "blocker": "M3",
        "title": "M3 gate",
        "kind": "declared_blocker",
        "resolution_class": "BLOCKED_INTERNAL",
        "satisfiable_by_this_project_internally": False,
        "no_internal_substitute": True,
        "internal_substitutes_forbidden": [
            "declaring M3 cleared without the required external evidence",
        ],
        "required_input": "carried blocker from earlier stages; unresolved",
        "observable_check": "M3 declared cleared by its own authority",
        "current_state": "BLOCKED",
        "evidence_level": "INTERNAL_DECLARATION",
        "measured_by_this_layer": False,
    },
)

#: 内部已就绪项（**只声明角色**；本层不重算、不评分）
INTERNAL_READY_CAPABILITIES: tuple[dict[str, str], ...] = (
    {"id": "C1", "capability": "decision assembly",
     "stage": "V3H-8", "rationale": "produces an auditable recommendation"},
    {"id": "C2", "capability": "experiment eligibility",
     "stage": "V3H-9", "rationale": "per-decision entry audit"},
    {"id": "C3", "capability": "shadow evaluation",
     "stage": "V3H-10", "rationale": "descriptive reconciliation record"},
    {"id": "C4", "capability": "evidence closure",
     "stage": "V3H-11", "rationale": "aggregate + gap list"},
    {"id": "C5", "capability": "pre-registration / result gate",
     "stage": "V3H-12", "rationale": "design surface; refuses result generation"},
    {"id": "C6", "capability": "execution intake / integrity digest",
     "stage": "V3H-13", "rationale": "pre-execution gate; digest is not proof"},
)

CLAIM_BOUNDARY: dict[str, bool] = {
    "is_gto": False,
    "is_equilibrium": False,
    "is_solver_output": False,
    "is_profit_evidence": False,
    "is_causal_claim": False,
    "is_strategy_quality_claim": False,
    "is_experimental_result": False,
    "is_authenticity_proof": False,
    "is_certification": False,
    "digest_treated_as_provenance_proof": False,
    "fabricates_trust_anchor": False,
    "supplies_internal_substitute_for_external_resource": False,
    "generates_execution_or_profit_or_statistical_results": False,
    "computes_rates_or_statistics": False,
    "adds_scoring_or_ev_formula": False,
    "executes_action": False,
    "auto_betting": False,
    "claims_experiment_ready": False,
    "wired_to_production": False,
}
CLAIM_FLAGS_TOTAL = len(CLAIM_BOUNDARY)

#: **禁止的内部替代品**（调用方若注入，一律判为越界并阻断）
FORBIDDEN_SUBSTITUTIONS: tuple[str, ...] = (
    "internal_substitute",
    "synthetic_execution",
    "simulated_execution",
    "mock_solver",
    "approximated_equilibrium",
    "declared_exploitability",
    "model_internal_as_external",
    "self_issued_trust_anchor",
    "digest_as_proof",
    "generated_profit",
    "inferred_realised_outcome",
)


def _is_cash(state: Any) -> bool:
    if not isinstance(state, dict):
        return False
    gt = state.get("game_type")
    return isinstance(gt, str) and gt.strip().lower() == "cash"


def _is_int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _stub(reason: str) -> dict:
    return {
        "present": False,
        "version": CASH_STRATEGY_EXTERNAL_READINESS_VERSION,
        "reason": reason,
        "verdict": "BLOCKED",
        "ready": False,
        "evidence_level": "UNAVAILABLE",
        "evidence_grade": "NONE",
        "n_blocked_external": 0,
        "requirements": [],
        "external_dependency_manifest": [],
        "executed": False,
        "result": None,
        "results_available": False,
        "analysis_executed": False,
        "claim_boundary": dict(CLAIM_BOUNDARY),
        "unknown": [reason],
    }


def external_dependency_manifest() -> dict:
    """返回**外部依赖清单**（本层核心产物；**静态**、无 IO、无结果）。

    每一项都标注：由谁解决、能否由本项目内部满足、
    以及**被禁止的内部替代品**。
    """
    return {
        "success": True,
        "version": CASH_STRATEGY_EXTERNAL_READINESS_VERSION,
        "n_requirements": len(EXTERNAL_REQUIREMENTS),
        "requirements": [dict(r) for r in EXTERNAL_REQUIREMENTS],
        "external_requirement_ids": [r["id"] for r in EXTERNAL_REQUIREMENTS
                                     if r["kind"] == "external_resource"],
        "n_external": sum(1 for r in EXTERNAL_REQUIREMENTS
                          if r["kind"] == "external_resource"),
        "all_unsatisfied": True,
        "internally_satisfiable": [],
        "no_internal_substitute": all(
            r["no_internal_substitute"] for r in EXTERNAL_REQUIREMENTS),
        "forbidden_substitutions": list(FORBIDDEN_SUBSTITUTIONS),
        "resolution_classes": list(RESOLUTION_CLASSES),
        "trust_anchor_policy": {
            "must_be_out_of_band": True,
            "issued_by_this_project": False,
            "digest_is_not_proof": True,
            "note": ("外部信任锚**必须**由带外提供；本项目**不**签发、"
                     "**不**代填、**不**自行判定；SHA-256 内容指纹"
                     "**不是**真实性证明。"),
        },
        "evidence_level": "INTERNAL_DECLARATION",
        "evidence_grade": "NONE",
        "executed": False,
        "result": None,
        "results_available": False,
        "analysis_executed": False,
        "claim_boundary": dict(CLAIM_BOUNDARY),
        "unknown": [],
        "note": ("清单本身是**声明**：它说明实现真实验证缺少哪些**外部资源**，"
                 "**不**声明已就绪、**不**声明可盈利。"),
    }


def _classify(req: dict, available: dict) -> tuple[str, str, list[str]]:
    """返回 (resolution_class, current_state, missing_evidence)。"""
    rid = req["id"]
    key = {"R1": "solver", "R2": "real_execution_data", "R3": "independent_"
           "decision_points", "R4": "real_profit_evidence",
           "R5": "trust_anchor", "R6": "m3"}[rid]
    entry = available.get(key) if isinstance(available, dict) else None
    if isinstance(entry, dict) and entry.get("satisfied") is True:
        return "SATISFIED", "satisfied by an externally supplied fact", []
    missing = [f"{key}:{req['required_input']}"]
    cls = req["resolution_class"]
    if cls == "BLOCKED_EXTERNAL_RESOURCE":
        state = (f"{req['current_state']}; cannot be satisfied by internal "
                 f"code (missing: {req['observable_check']})")
    else:
        state = f"{req['current_state']}; internal declaration still required"
    return cls, state, missing


def audit_external_readiness(state: Any, *, closure: Any = None,
                             preregistration: Any = None,
                             intake: Any = None,
                             solver: Any = None,
                             environment: Any = None,
                             available_external_resources: Any = None
                             ) -> dict:
    """审计"到真实验证还缺什么"，并**诚实**给出 `BLOCKED` / `PARTIAL` / `READY`。

    **本层不获取、不模拟、不伪造任何外部资源**；
    调用方若通过 `available_external_resources` 注入**被禁止的内部替代品**
    （见 `FORBIDDEN_SUBSTITUTIONS`），该项一律**不予采信**并阻断。
    """
    if not _is_cash(state):
        return _stub("not_cash_mode_fail_closed")

    missing_inputs = [k for k, v in (("closure", closure),
                                     ("preregistration", preregistration),
                                     ("intake", intake),
                                     ("solver", solver),
                                     ("environment", environment))
                      if not isinstance(v, dict)]
    if "closure" in missing_inputs or "preregistration" in missing_inputs:
        return _stub("closure_and_preregistration_required")

    # ── 外部资源事实（**只读**；不获取、不推断）──
    available = (available_external_resources
                 if isinstance(available_external_resources, dict) else {})
    refused_substitutions: list[str] = []
    for k in FORBIDDEN_SUBSTITUTIONS:
        if available.get(k) is not None:
            refused_substitutions.append(k)
    if refused_substitutions:
        return _stub("forbidden_internal_substitution_detected:" +
                     ",".join(sorted(refused_substitutions)))

    unknown: list[str] = []
    for k in missing_inputs:
        unknown.append(f"{k}_not_supplied")

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
        "solver_invoked": False,
        "note": ("求解器不可用 / 未提供均衡参考 → 相关量记 None（**不猜**）。"
                 "本层不调用任何求解器、不近似、不声明。"),
    }

    # ── 逐项分类 ──
    requirements: list[dict] = []
    counts = {c: 0 for c in RESOLUTION_CLASSES}
    missing_evidence: list[str] = []
    for req in EXTERNAL_REQUIREMENTS:
        cls, cstate, miss = _classify(req, available)
        counts[cls] = counts.get(cls, 0) + 1
        missing_evidence.extend(miss)
        requirements.append({
            "id": req["id"],
            "blocker": req["blocker"],
            "title": req["title"],
            "kind": req["kind"],
            "resolution_class": cls,
            "satisfied": cls == "SATISFIED",
            "satisfiable_by_this_project_internally":
                req["satisfiable_by_this_project_internally"],
            "no_internal_substitute": req["no_internal_substitute"],
            "internal_substitutes_forbidden":
                list(req["internal_substitutes_forbidden"]),
            "required_input": req["required_input"],
            "observable_check": req["observable_check"],
            "current_state": cstate,
            "evidence_level": req["evidence_level"],
            "measured_by_this_layer": False,
        })

    # ── 上游事实转述（**不重算**）──
    upstream_facts = {
        "closure_closure_possible": closure.get("closure_possible") is True,
        "closure_established_blockers": (
            (closure.get("closure_test") or {}).get("established_blockers")
            if isinstance(closure.get("closure_test"), dict) else None),
        "preregistration_analysis_ready":
            preregistration.get("analysis_ready") is True,
        "preregistration_result_generation_refused":
            preregistration.get("result_generation_refused") is True,
        "preregistration_blockers": preregistration.get("blockers_from_upstream"),
        "intake_present": intake.get("present") is True if isinstance(
            intake, dict) else False,
        "intake_permitted": intake.get("intake_permitted") is True
        if isinstance(intake, dict) else False,
        "intake_admitted": (intake.get("n_admitted")
                            if isinstance(intake, dict) else None),
        "note": "全部**转述自上游**，本层不重算任何上游结果。",
    }

    n_blocked_external = counts.get("BLOCKED_EXTERNAL_RESOURCE", 0)
    n_blocked_internal = counts.get("BLOCKED_INTERNAL", 0)
    n_satisfied = counts.get("SATISFIED", 0)

    if n_blocked_external:
        verdict = "BLOCKED"
    elif n_blocked_internal:
        verdict = "PARTIAL"
    else:
        verdict = "READY"
    ready = verdict == "READY"

    external_reqs = [r for r in requirements
                     if r["kind"] == "external_resource"]
    unsatisfied_external = [r["id"] for r in external_reqs
                            if not r["satisfied"]]

    return {
        "present": True,
        "version": CASH_STRATEGY_EXTERNAL_READINESS_VERSION,
        "mode": "cash",
        "verdict": verdict,
        "ready": ready,
        "stops_internally": verdict != "READY",
        "n_requirements": len(requirements),
        "n_satisfied": n_satisfied,
        "n_blocked_external": n_blocked_external,
        "n_blocked_internal": n_blocked_internal,
        "unsatisfied_external_requirement_ids": unsatisfied_external,
        "requirements": requirements,
        "external_dependency_manifest": [dict(r) for r in EXTERNAL_REQUIREMENTS
                                         if r["kind"] == "external_resource"],
        "requirement_counts_by_class": dict(counts),
        "resolution_classes": list(RESOLUTION_CLASSES),
        "readiness_verdicts": list(READINESS_VERDICTS),
        "missing_evidence": sorted(set(missing_evidence)),
        "pipeline_stages": [dict(s) for s in PIPELINE_STAGES],
        "internal_ready_capabilities": [dict(c) for c
                                        in INTERNAL_READY_CAPABILITIES],
        "internally_implementable_next_capability": None,
        "next_step_depends_on_external_resources": n_blocked_external > 0,
        "honest_conclusion": (
            "下一步依赖**外部资源**（精确解算器与均衡参考、真实执行记录、"
            "B1 独立性测量、真实盈亏证据、带外信任锚），"
            "本项目**无法**在内部合法实现，且**不制造**内部替代品。"
            if n_blocked_external else
            "无外部依赖未满足；仍有内部可解项未完成。" if n_blocked_internal
            else "全部满足。"),
        "substitution_policy": {
            "internal_substitutes_offered": [],
            "substitution_forbidden": True,
            "forbidden_substitutions": list(FORBIDDEN_SUBSTITUTIONS),
            "refused_substitutions_detected": sorted(refused_substitutions),
            "note": ("**禁止**用内部产物替代外部资源；本层不提供任何替代品，"
                     "检测到即 fail closed。"),
        },
        "trust_anchor_policy": {
            "must_be_out_of_band": True,
            "issued_by_this_project": False,
            "verified_by_this_layer": False,
            "digest_is_not_proof": True,
            "note": ("外部信任锚**必须**带外提供；本层**不**签发、**不**代填、"
                     "**不**自行判定；SHA-256 内容指纹**不是**真实性证明、"
                     "**不是**认证、**不是**来源证明。"),
        },
        "upstream_facts": upstream_facts,
        "blockers_measured_by_this_layer": False,
        "solver": solver_view,
        "environment": {
            "supplied": isinstance(environment, dict),
            "missing_facts": (sorted(k for k, v in environment.items()
                                     if v is not True)
                              if isinstance(environment, dict) else None),
            "counted_only": True,
        },
        "provenance": {
            "evidence_level": ("INTERNAL_DECLARATION" if not ready
                               else "EXTERNAL_RESOURCE_FACT"),
            "evidence_grade": "NONE",
            "evidence_levels": list(EVIDENCE_LEVELS),
            "source_components": [
                {"stage": "V3H-11", "role": "closure / gap facts",
                 "version": closure.get("version"),
                 "consumed_as": "transcribed"},
                {"stage": "V3H-12", "role": "design surface / result gate",
                 "version": preregistration.get("version"),
                 "consumed_as": "transcribed"},
                {"stage": "V3H-13", "role": "intake gate facts",
                 "version": intake.get("version") if isinstance(intake, dict)
                 else None, "consumed_as": "transcribed"},
                {"stage": "V3H-14", "role": "external readiness audit",
                 "version": CASH_STRATEGY_EXTERNAL_READINESS_VERSION},
            ],
            "external_resources_obtained_by_this_layer": False,
            "external_resources_simulated_or_fabricated": False,
            "digest_used_as_proof": False,
            "wrote_any_file": False,
            "opened_any_client": False,
            "real_money_moved": False,
            "frozen_multistreet_or_sample_runouts_reused": False,
            "future_cards_generated": False,
            "runout_sampling_performed": False,
        },
        "result": None,
        "results_available": False,
        "analysis_executed": False,
        "executed": False,
        "evidence_level": "INTERNAL_DECLARATION" if not ready else \
            "EXTERNAL_RESOURCE_FACT",
        "evidence_grade": "NONE",
        "uncertainty": {
            "measured": False,
            "note": ("未执行任何动作、未运行任何分析 → **无**统计不确定性；"
                     "仅记录**外部依赖未满足**这一事实。"),
            "unsatisfied_external_requirements": unsatisfied_external,
            "unresolved_inputs": missing_inputs,
        },
        "claim_boundary": dict(CLAIM_BOUNDARY),
        "unknown": sorted(set(unknown)),
        "note": ("本层是**验证前检查 + 外部依赖清单**：不获取外部资源、"
                 "不伪造信任锚、不生成任何结果、不执行任何动作。"),
    }
