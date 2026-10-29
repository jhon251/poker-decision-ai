"""V3I-P1 Step 3/4 — CASH 决策闭环编排器（**零层 import**、证据等级强制）。

**本模块的定位**
----------------
把既有 CASH 链路**真正跑起来**：State → ActionTree/Range/Opponent → ActionPolicy
→ Belief → **Likelihood 观测更新** → Ranking → BeliefConditioned → SearchTree
→ SearchEval → DecisionAssembly。

**关键架构约束（V3G-11 边界）**
------------------------------
上游层函数**一律由调用方通过 `pipeline=` 显式注入**，本模块
**不 import 任何 `cash_*` 层**（含 `cash_integrity_context`）。
这与 V3H-9 `gate=` / V3H-10 `equity_snapshot=` / V3H-13 `preregistration=`
的既有惯例一致，并保持：

* `test_only_sanctioned_wiring_exists` 仍通过（不新增包装配）；
* 7 个未接线 V3G 层在生产侧**仍零 import**。

**证据等级强制（Step 4）**
--------------------------
* 本闭环**只允许** `MODEL_INTERNAL`；
* 任何 `EXTERNAL_VALIDATION` 主张 **fail closed**（返回 `BLOCKED`），
  理由：无 RC-2 语料、无求解器均衡参考、无带外信任锚；
* 输出**恒**带 `evidence_level = "MODEL_INTERNAL"`（或 `NONE`）。

**严格禁止**
------------
不新增策略 / 不重算上游 / 不伪造 equilibrium / 不自动操作 / 不写文件 / 无 IO / 无随机。
"""
from __future__ import annotations

from typing import Any

CASH_DECISION_LOOP_VERSION = "v3i-p1-cash-decision-loop-v1"

#: 闭环状态（封闭枚举）
LOOP_STATUSES: tuple[str, ...] = (
    "COMPLETE",      # 装配出可审计建议
    "ABSTAINED",     # 上游弃权（分歧/无 scorer）→ 不透传为建议
    "UNKNOWN",       # 链路不完整或存在未解 UNKNOWN
    "BLOCKED",       # 证据等级越界或必需阶段失败
)

#: 允许的证据等级（**只有**模型内）
ALLOWED_EVIDENCE_LEVELS: tuple[str, ...] = ("MODEL_INTERNAL",)

#: 被拒绝的证据等级（fail closed）
REJECTED_EVIDENCE_LEVELS: tuple[str, ...] = (
    "EXTERNAL_VALIDATION",
    "REAL_OPPONENT_DATA",
    "OBSERVED_FREQUENCY",
    "SOLVER_EQUILIBRIUM",
    "PROFIT_EVIDENCE",
)

#: pipeline 必须提供的阶段（显式注入；本模块不 import 它们）
REQUIRED_PIPELINE_STAGES: tuple[str, ...] = (
    "build_action_policy",
    "build_dynamic_belief",
    "update_belief",
    "rank_actions",
    "evaluate_belief_conditioned",
    "build_search_tree",
    "evaluate_search",
    "assemble_decision",
)

#: 阶段调用顺序（仅用于记录与复现）
STAGE_ORDER: tuple[str, ...] = (
    "action_policy",
    "dynamic_belief",
    "belief_update",
    "action_ranking",
    "belief_conditioned",
    "search_tree",
    "search_eval",
    "decision_assembly",
)

CLAIM_BOUNDARY: dict[str, bool] = {
    "is_external_validation": False,
    "is_real_opponent_data": False,
    "is_gto": False,
    "is_equilibrium": False,
    "is_solver_output": False,
    "is_profit_evidence": False,
    "is_causal_claim": False,
    "is_strategy_quality_claim": False,
    "adds_scoring_or_ev_formula": False,
    "recomputes_upstream_results": False,
    "executes_action": False,
    "auto_betting": False,
    "wired_to_production": False,
    "modifies_frozen_layers": False,
}
CLAIM_FLAGS_TOTAL = len(CLAIM_BOUNDARY)


def _is_cash(state: Any) -> bool:
    if not isinstance(state, dict):
        return False
    gt = state.get("game_type")
    return isinstance(gt, str) and gt.strip().lower() == "cash"


def _stub(reason: str, status: str = "BLOCKED") -> dict:
    return {
        "present": False,
        "version": CASH_DECISION_LOOP_VERSION,
        "reason": reason,
        "loop_status": status,
        "evidence_level": "NONE",
        "decision_record": None,
        "stage_trace": [],
        "unknown": [reason],
        "executed": False,
        "result": None,
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }


def _unknowns_of(obj: Any) -> list[str]:
    if isinstance(obj, dict) and isinstance(obj.get("unknown"), list):
        return [str(u) for u in obj["unknown"]]
    return []


def _stage_ok(obj: Any) -> bool:
    return isinstance(obj, dict) and obj.get("present") is True


def run_decision_loop(state: Any, *, pipeline: Any = None,
                      integrity_context: Any = None,
                      opponent_model: Any = None,
                      range_model: Any = None,
                      action_tree: Any = None,
                      transition: Any = None,
                      belief: Any = None,
                      likelihood_model: Any = None,
                      action_weights: Any = None,
                      solver: Any = None,
                      budget: Any = None,
                      max_root_candidates: int = 3,
                      equity_seed: int = 42,
                      equity_iterations: int = 20000,
                      policy: Any = None,
                      allow_higher_tier_override: bool = False,
                      override_justification: Any = None,
                      claimed_evidence_level: str = "MODEL_INTERNAL") -> dict:
    """执行一次**完整** CASH 决策闭环，返回 `decision_record` + `loop_status`。

    所有上游层函数经 `pipeline=` 注入；`pipeline` 是
    `{stage_name: callable}` 映射，键见 `REQUIRED_PIPELINE_STAGES`。
    """
    if not _is_cash(state):
        return _stub("not_cash_mode_fail_closed")

    # ── Step 4：证据等级强制（fail closed，先于任何计算）──
    if not isinstance(claimed_evidence_level, str) \
            or claimed_evidence_level not in ALLOWED_EVIDENCE_LEVELS:
        return _stub(
            "evidence_level_not_permitted_internal_only:"
            f"{claimed_evidence_level!r}")
    if isinstance(pipeline, dict):
        pip_claim = pipeline.get("claimed_evidence_level")
        if isinstance(pip_claim, str) and pip_claim not in \
                ALLOWED_EVIDENCE_LEVELS:
            return _stub(
                "pipeline_claims_forbidden_evidence_level:"
                f"{pip_claim!r}")

    if not isinstance(pipeline, dict):
        return _stub("pipeline_required")
    missing = [s for s in REQUIRED_PIPELINE_STAGES
               if not callable(pipeline.get(s))]
    if missing:
        return _stub("pipeline_missing_stages:" + ",".join(sorted(missing)))

    def _call(stage: str, **kw):
        fn = pipeline[stage]
        try:
            return fn(state, **kw), None
        except Exception as exc:                            # noqa: BLE001
            return None, f"{stage}_raised:{type(exc).__name__}"

    trace: list[dict] = []
    unknown: list[str] = []

    def _rec(stage_name: str, out: Any, err: str | None) -> None:
        trace.append({
            "stage": stage_name,
            "present": _stage_ok(out),
            "error": err,
            "unknown": _unknowns_of(out),
            "level": (out.get("level") if isinstance(out, dict) else None),
        })
        if err:
            unknown.append(err)
        for u in _unknowns_of(out):
            if u not in unknown:
                unknown.append(u)

    # ── 1. action policy ──
    ap, err = _call("build_action_policy", action_tree=action_tree,
                    range_model=range_model,
                    integrity_context=integrity_context,
                    equity_seed=equity_seed,
                    equity_iterations=equity_iterations)
    _rec("action_policy", ap, err)
    if not _stage_ok(ap):
        return _finish(state, "BLOCKED", trace, unknown,
                       "stage_failed:action_policy")

    # ── 2. dynamic belief ──
    bel = belief if _stage_ok(belief) else None
    if bel is None:
        bel, err = _call("build_dynamic_belief", range_model=range_model,
                         opponent_model=opponent_model,
                         action_weights=action_weights,
                         integrity_context=integrity_context)
        _rec("dynamic_belief", bel, err)
        if not _stage_ok(bel):
            return _finish(state, "BLOCKED", trace, unknown,
                           "stage_failed:dynamic_belief")
    else:
        trace.append({"stage": "dynamic_belief", "present": True,
                      "error": None, "unknown": _unknowns_of(bel),
                      "level": bel.get("level"), "note": "caller_supplied"})

    # ── 3. likelihood 观测更新（Step 2 的核心：**真正执行**）──
    bu = None
    if likelihood_model is not None:
        pot_bb = None
        if isinstance(state.get("pot"), (int, float)) \
                and isinstance(state.get("bb"), (int, float)) \
                and state.get("bb"):
            pot_bb = float(state["pot"]) / float(state["bb"])
        bu, err = _call("update_belief", belief=bel,
                        range_model=range_model,
                        likelihood_model=likelihood_model, pot_bb=pot_bb)
        _rec("belief_update", bu, err)
    else:
        trace.append({"stage": "belief_update", "present": False,
                      "error": None,
                      "unknown": ["likelihood_model_not_supplied"],
                      "level": None})
        unknown.append("likelihood_model_not_supplied")

    # ── 4. ranking ──
    rk, err = _call("rank_actions", action_policy=ap, belief=bel)
    _rec("action_ranking", rk, err)
    if not _stage_ok(rk):
        return _finish(state, "BLOCKED", trace, unknown,
                       "stage_failed:action_ranking")

    # ── 5. belief-conditioned eval（仅当观测更新可用）──
    bc = None
    if _stage_ok(bu):
        bc, err = _call("evaluate_belief_conditioned", action_policy=ap,
                        ranking=rk, belief_update=bu, solver=solver)
        _rec("belief_conditioned", bc, err)
    else:
        trace.append({"stage": "belief_conditioned", "present": False,
                      "error": None,
                      "unknown": ["belief_update_unavailable"],
                      "level": None})

    # ── 6. search tree ──
    stree, err = _call("build_search_tree", action_policy=ap, belief=bel,
                       belief_update=bu, transition=transition,
                       budget=budget, max_root_candidates=max_root_candidates)
    _rec("search_tree", stree, err)

    # ── 7. search eval ──
    se = None
    if _stage_ok(stree):
        se, err = _call("evaluate_search", search_tree=stree, solver=solver)
        _rec("search_eval", se, err)
    else:
        trace.append({"stage": "search_eval", "present": False, "error": None,
                      "unknown": ["search_tree_unavailable"], "level": None})

    # ── 8. decision assembly ──
    dec, err = _call("assemble_decision", ranking=rk, belief_conditioned=bc,
                     search_eval=se, solver=solver, policy=policy,
                     allow_higher_tier_override=allow_higher_tier_override,
                     override_justification=override_justification)
    _rec("decision_assembly", dec, err)
    if not _stage_ok(dec):
        return _finish(state, "BLOCKED", trace, unknown,
                       "stage_failed:decision_assembly")

    for u in _unknowns_of(dec):
        if u not in unknown:
            unknown.append(u)

    # ── 状态判定：弃权 / UNKNOWN 原样传播，不透传为建议 ──
    abstained = dec.get("abstained") is True
    if abstained:
        status = "ABSTAINED"
    elif not dec.get("preferred"):
        status = "UNKNOWN"
    elif unknown:
        status = "UNKNOWN"
    else:
        status = "COMPLETE"

    return _finish(state, status, trace, unknown, None, decision=dec,
                   action_policy=ap, belief=bel, belief_update=bu,
                   ranking=rk, belief_conditioned=bc, search_tree=stree,
                   search_eval=se)


def _finish(state: dict, status: str, trace: list[dict],
            unknown: list[str], reason: str | None, decision: Any = None,
            action_policy: Any = None, belief: Any = None,
            belief_update: Any = None, ranking: Any = None,
            belief_conditioned: Any = None, search_tree: Any = None,
            search_eval: Any = None) -> dict:
    """组装返回值；证据等级恒为 `MODEL_INTERNAL`（或 `NONE`）。"""
    available = status in ("COMPLETE", "ABSTAINED", "UNKNOWN")
    record = {
        "stage_order": list(STAGE_ORDER),
        "stage_trace": trace,
        "decision": decision,
        "preferred": (decision or {}).get("preferred") if isinstance(
            decision, dict) else None,
        "abstained": bool(isinstance(decision, dict)
                          and decision.get("abstained") is True),
        "disagreements": (decision or {}).get("disagreements", []) if isinstance(
            decision, dict) else [],
        "selected_by_tier": ((decision or {}).get("preferred") or {}).get(
            "selected_by_tier") if isinstance(decision, dict) else None,
        "belief_update_present": _stage_ok(belief_update),
        "belief_update_level": (belief_update or {}).get("level")
        if isinstance(belief_update, dict) else None,
        "search_tree_present": _stage_ok(search_tree),
        "search_eval_present": _stage_ok(search_eval),
        "evidence_level": "MODEL_INTERNAL",
        "is_external_validation": False,
        "unknown": sorted(set(unknown)),
        "executed": False,
        "solver_status": (decision or {}).get("solver", {}).get("solver_status")
        if isinstance(decision, dict)
        and isinstance(decision.get("solver"), dict) else None,
    }
    return {
        "present": available,
        "version": CASH_DECISION_LOOP_VERSION,
        "reason": reason,
        "loop_status": status,
        "street": state.get("street"),
        "evidence_level": "MODEL_INTERNAL" if available else "NONE",
        "decision_record": record,
        "stage_outputs": {
            "action_policy": action_policy,
            "belief": belief,
            "belief_update": belief_update,
            "ranking": ranking,
            "belief_conditioned": belief_conditioned,
            "search_tree": search_tree,
            "search_eval": search_eval,
            "decision": decision,
        },
        "stage_trace": trace,
        "unknown": sorted(set(unknown)),
        "result": None,
        "results_available": False,
        "analysis_executed": False,
        "executed": False,
        "claim_boundary": dict(CLAIM_BOUNDARY),
        "provenance": {
            "version": CASH_DECISION_LOOP_VERSION,
            "orchestration": "explicit_pipeline_injection_zero_layer_import",
            "evidence_level": "MODEL_INTERNAL" if available else "NONE",
            "external_validation_available": False,
            "recomputes_upstream_results": False,
            "adds_scoring_or_ev_formula": False,
            "modifies_frozen_layers": False,
            "auto_operation": False,
        },
        "note": ("闭环**只**产出模型内决策记录：**不是**外部验证、"
                 "**不是**真实对手数据、**不是**收益证据、**不执行**动作。"),
    }


def loop_contract() -> dict:
    """返回闭环契约（纯声明）。"""
    return {
        "success": True,
        "version": CASH_DECISION_LOOP_VERSION,
        "loop_statuses": list(LOOP_STATUSES),
        "required_pipeline_stages": list(REQUIRED_PIPELINE_STAGES),
        "stage_order": list(STAGE_ORDER),
        "allowed_evidence_levels": list(ALLOWED_EVIDENCE_LEVELS),
        "rejected_evidence_levels": list(REJECTED_EVIDENCE_LEVELS),
        "imports_any_cash_layer": False,
        "injection_style": "pipeline_kwarg_same_convention_as_v3h9_gate",
        "external_validation_permitted": False,
        "recomputes_upstream_results": False,
        "executed": False,
        "result": None,
        "claim_boundary": dict(CLAIM_BOUNDARY),
        "unknown": [],
    }
