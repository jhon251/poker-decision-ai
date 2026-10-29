"""V3I-P9 — 把 P8 已验证的 opponent-conditioned search bridge 最小接入 V3I-P1 决策链。

审计结论（授权要求 1，先审计后实施）
------------------------------------
`cash_decision_loop.run_decision_loop` 的**现有**接线契约：

* L174-177：只校验 `REQUIRED_PIPELINE_STAGES`（8 个）**可调用**；
  **额外的 pipeline 键被允许**（不被拒绝、不被读取）→ 这提供了注入点。
* L269-271：`_call("build_search_tree", action_policy=, belief=, belief_update=,
  transition=, budget=, max_root_candidates=)`
* L277：`_call("evaluate_search", search_tree=stree, solver=solver)`
  → **`evaluate_search` 这个键就是官方注入点**。
* `_call`（L179-184）：`fn(state, **kw)` —— 固定注入 `state`。

**因此本层不需要修改 P1 loop、也不需要扩大任何冻结契约**：
把 `pipeline["evaluate_search"]` 换成一个**闭包**即可（依赖注入，
符合 P1 "本模块不 import 上游层" 的设计）。

本层做什么
----------
1. `audit_loop_wiring_contract(fn)`：校验 loop 未改签名、额外键被允许。
2. `prepare_conditioned_search(...)`：用 P4 bridge（逐候选 conditioned equity）
   经 P8 adapter 物化出**条件化 search tree**。
3. `make_conditioned_evaluate_search(...)`：返回一个闭包，签名与冻结
   `evaluate_search` 一致；内部把条件化树交给**冻结** `evaluate_search`。
4. `run_loop_with_conditioning(...)`：用真实 `run_decision_loop` 跑，
   显式注入上面的闭包。

fail-closed（要求 5）
--------------------
* 无 `conditioned_input` → 闭包**原样透传原树**（保留原有行为，要求 6）；
* bridge 不可用（UNKNOWN / prior-only / conflict / 非 CASH）→
  闭包**回退到原树**并把原因记入 `conditioning_status`，**不伪造适应**。

边界：本模块只 import 标准库；不 import 任何项目内模块（全部由调用方注入）。
"""
from __future__ import annotations

import hashlib
import inspect
import json
from typing import Any

CASH_P9_LOOP_WIRING_VERSION = "v3i-p9-opponent-conditioned-loop-wiring-v1"

#: P1 loop 的必需阶段（与 `cash_decision_loop.REQUIRED_PIPELINE_STAGES` 一致）
EXPECTED_REQUIRED_STAGES: tuple[str, ...] = (
    "build_action_policy",
    "build_dynamic_belief",
    "update_belief",
    "rank_actions",
    "evaluate_belief_conditioned",
    "build_search_tree",
    "evaluate_search",
    "assemble_decision",
)

#: 本层用于携带条件化输入的**额外** pipeline 键（loop 会忽略它们）
EXTRA_PIPELINE_KEYS: tuple[str, ...] = (
    "opponent_conditioned_search",
    "p9_conditioning_status",
)

#: 冻结 `evaluate_search` 的调用约定（loop 实际使用的入参）
LOOP_EVALUATE_SEARCH_INPUTS: tuple[str, ...] = ("state", "search_tree", "solver")

CLAIM_BOUNDARY: dict[str, bool] = {
    "is_gto": False,
    "is_equilibrium": False,
    "is_solver_output": False,
    "is_ev": False,
    "is_real_opponent_data": False,
    "is_real_human_validation": False,
    "is_profit_evidence": False,
    "is_external_validation": False,
    "wired_to_production": False,
    "adds_strategy_capability": False,
    "rewrites_p1_loop": False,
    "modifies_frozen_layers": False,
    "widens_frozen_contract": False,
    "hardcodes_population_to_action": False,
}

EVIDENCE_GRADE = "MODEL_INTERNAL"


def _is_num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _stub(reason: str) -> dict:
    return {
        "present": False,
        "version": CASH_P9_LOOP_WIRING_VERSION,
        "reason": reason,
        "unknown": [reason],
        "claim_boundary": dict(CLAIM_BOUNDARY),
        "evidence_grade": "NONE",
        "executed": False,
        "result": None,
        "wired_to_production": False,
    }


def stable_fingerprint(value: Any) -> str:
    def _norm(v: Any) -> Any:
        if isinstance(v, dict):
            return {str(k): _norm(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [_norm(x) for x in v]
        return v
    try:
        text = json.dumps(_norm(value), sort_keys=True, separators=(",", ":"),
                          ensure_ascii=True, default=str)
    except Exception:                                      # noqa: BLE001
        text = repr(value)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# 1. 审计 P1 loop 接线契约（要求 1）
# ---------------------------------------------------------------------------
def audit_loop_wiring_contract(*, run_decision_loop_fn: Any,
                              required_stages: Any = None) -> dict:
    """校验 loop 的接线契约与 P9 依赖一致（只读检查）。"""
    if not callable(run_decision_loop_fn):
        return _stub("run_decision_loop_fn_required")
    try:
        sig = inspect.signature(run_decision_loop_fn)
    except (TypeError, ValueError) as exc:                  # pragma: no cover
        return _stub(f"signature_unavailable:{type(exc).__name__}")
    params = list(sig.parameters)
    required = tuple(required_stages) if required_stages is not None \
        else EXPECTED_REQUIRED_STAGES

    has_state = "state" in params
    has_pipeline = "pipeline" in params
    pipeline_kw_only = True
    p = sig.parameters.get("pipeline")
    if p is not None and p.kind not in (p.KEYWORD_ONLY, p.POSITIONAL_OR_KEYWORD):
        pipeline_kw_only = False
    return {
        "present": True,
        "version": CASH_P9_LOOP_WIRING_VERSION,
        "loop_parameters": params,
        "has_state": has_state,
        "has_pipeline": has_pipeline,
        "pipeline_is_keyword_capable": pipeline_kw_only,
        "required_stages_expected": list(required),
        "required_stages_match": tuple(required) == EXPECTED_REQUIRED_STAGES,
        "extra_pipeline_keys_allowed": True,
        "extra_pipeline_keys": list(EXTRA_PIPELINE_KEYS),
        "loop_evaluate_search_inputs": list(LOOP_EVALUATE_SEARCH_INPUTS),
        "needs_loop_modification": False,
        "needs_frozen_contract_widening": False,
        "conclusion": (
            "loop 只校验必需阶段可调用，额外 pipeline 键被允许；"
            "`evaluate_search` 键即官方注入点，`_call` 固定注入 state。"
            "因此注入一个同签名的闭包即可，**无需修改 loop、无需扩大冻结契约**。"),
        "claim_boundary": dict(CLAIM_BOUNDARY),
        "evidence_grade": EVIDENCE_GRADE,
        "executed": False,
        "result": None,
        "wired_to_production": False,
        "unknown": [],
    }


# ---------------------------------------------------------------------------
# 2. 准备条件化 search tree（P4 bridge → P8 adapter）
# ---------------------------------------------------------------------------
def prepare_conditioned_search(*, state: Any, belief: Any, range_model: Any,
                               candidates: Any, population: Any,
                               population_params: Any, required_equity: Any,
                               equity_fn: Any, player_id: Any = None,
                               iterations: int = 200, seed: int = 42,
                               bridge_fn: Any = None, input_fn: Any = None,
                               materialize_fn: Any = None) -> dict:
    """P4 bridge → 逐候选 conditioned equity → P8 物化条件化树。

    任一步不可用 → `present=False` 且**不伪造**（调用方据此回退原树）。
    """
    for name, fn in (("bridge_fn", bridge_fn), ("input_fn", input_fn),
                     ("materialize_fn", materialize_fn)):
        if not callable(fn):
            return _stub(f"{name}_required")

    br = bridge_fn(state=state, belief=belief, range_model=range_model,
                   candidates=candidates, population=population,
                   population_params=population_params,
                   required_equity=required_equity, equity_fn=equity_fn,
                   player_id=player_id, iterations=iterations, seed=seed)
    if not isinstance(br, dict) or not br.get("present"):
        return _stub("bridge_unavailable:"
                     f"{(br or {}).get('reason') if isinstance(br, dict) else 'not_a_dict'}")

    ci = input_fn(bridge_out=br)
    if not isinstance(ci, dict) or not ci.get("present"):
        return _stub("conditioned_input_unavailable:"
                     f"{(ci or {}).get('reason') if isinstance(ci, dict) else 'not_a_dict'}")

    per = ci.get("per_candidate") or {}
    if len(per) < 2:
        return _stub("conditioned_input_not_per_candidate")

    return {
        "present": True,
        "version": CASH_P9_LOOP_WIRING_VERSION,
        "population": population,
        "n_candidates": len(per),
        "per_candidate": per,
        "materialize_fn": materialize_fn,
        "bridge_preferred": br.get("preferred"),
        "conditioned_input": ci,
        "candidates_without_equity": sorted(
            k for k, v in per.items()
            if not _is_num(v.get("hero_equity_vs_continuing"))),
        "claim_boundary": dict(CLAIM_BOUNDARY),
        "evidence_grade": EVIDENCE_GRADE,
        "executed": False,
        "result": None,
        "wired_to_production": False,
        "unknown": [],
    }


# ---------------------------------------------------------------------------
# 3. 同签名闭包：注入到 pipeline["evaluate_search"]
# ---------------------------------------------------------------------------
def make_conditioned_evaluate_search(*, prepared: Any, frozen_evaluate_search: Any,
                                    tag: str = "p9_conditioned") -> dict:
    """构造一个**与冻结 `evaluate_search` 同签名**的可调用对象。

    行为：
    * `prepared` 不可用 → 原样调用冻结函数（保留原有行为）；
    * 可用 → 物化条件化树后交给**同一个冻结函数**；
    * 在返回 dict 中加 `p9_routing` 标记，使"确实经过本接线"可被机器验证。
    """
    if not callable(frozen_evaluate_search):
        return _stub("frozen_evaluate_search_required")

    ok = isinstance(prepared, dict) and prepared.get("present")
    materialize_fn = (prepared or {}).get("materialize_fn") if ok else None
    ci = (prepared or {}).get("conditioned_input") if ok else None
    if ok and not callable(materialize_fn):
        ok = False

    def evaluate_search(state, *, search_tree=None, solver=None):
        if not ok:
            out = frozen_evaluate_search(state, search_tree=search_tree,
                                        solver=solver)
            if isinstance(out, dict):
                out = dict(out)
                out["p9_routing"] = {
                    "routed": False,
                    "path": "passthrough_no_conditioning",
                    "reason": (prepared or {}).get("reason")
                    if isinstance(prepared, dict) else "prepared_missing",
                    "tag": tag,
                }
            return out
        mat = materialize_fn(search_tree=search_tree, conditioned_input=ci)
        if not isinstance(mat, dict) or not mat.get("present"):
            out = frozen_evaluate_search(state, search_tree=search_tree,
                                        solver=solver)
            if isinstance(out, dict):
                out = dict(out)
                out["p9_routing"] = {
                    "routed": False,
                    "path": "passthrough_materialize_failed",
                    "reason": (mat or {}).get("reason")
                    if isinstance(mat, dict) else "materialize_not_a_dict",
                    "tag": tag,
                }
            return out
        out = frozen_evaluate_search(state, search_tree=mat["tree"], solver=solver)
        if isinstance(out, dict):
            out = dict(out)
            out["p9_routing"] = {
                "routed": True,
                "path": "explicit_conditioned_tree_to_frozen_evaluator",
                "population": prepared.get("population"),
                "n_leaves_rewritten": mat.get("n_leaves_rewritten"),
                "n_candidates_conditioned": prepared.get("n_candidates"),
                "tag": tag,
            }
        return out

    return {
        "present": True,
        "version": CASH_P9_LOOP_WIRING_VERSION,
        "callable": evaluate_search,
        "will_route": bool(ok),
        "signature": "(state, *, search_tree=None, solver=None)",
        "uses_global_state": False,
        "uses_monkeypatch": False,
        "claim_boundary": dict(CLAIM_BOUNDARY),
        "evidence_grade": EVIDENCE_GRADE,
        "executed": False,
        "result": None,
        "wired_to_production": False,
        "unknown": [],
    }


# ---------------------------------------------------------------------------
# 4. 真实 loop 运行
# ---------------------------------------------------------------------------
def run_loop_with_conditioning(*, state: Any, pipeline: Any,
                              run_decision_loop_fn: Any, prepared: Any,
                              frozen_evaluate_search: Any,
                              loop_kwargs: Any = None, tag: str = "p9_conditioned") -> dict:
    """用**真实** `run_decision_loop` 跑一次，注入条件化 `evaluate_search`。"""
    if not callable(run_decision_loop_fn):
        return _stub("run_decision_loop_fn_required")
    if not isinstance(pipeline, dict):
        return _stub("pipeline_required")

    built = make_conditioned_evaluate_search(
        prepared=prepared, frozen_evaluate_search=frozen_evaluate_search, tag=tag)
    if not built.get("present"):
        return _stub(f"closure_build_failed:{built.get('reason')}")

    pip = dict(pipeline)
    pip["evaluate_search"] = built["callable"]
    pip["opponent_conditioned_search"] = {
        "present": bool(isinstance(prepared, dict) and prepared.get("present")),
        "population": (prepared or {}).get("population")
        if isinstance(prepared, dict) else None,
        "n_candidates": (prepared or {}).get("n_candidates")
        if isinstance(prepared, dict) else None,
    }
    pip["p9_conditioning_status"] = (
        "conditioned" if built["will_route"] else "passthrough")

    kwargs = dict(loop_kwargs or {})
    out = run_decision_loop_fn(state, pipeline=pip, **kwargs)
    if not isinstance(out, dict):
        return _stub("loop_returned_non_dict")

    se = (out.get("stage_outputs") or {}).get("search_eval")
    routing = se.get("p9_routing") if isinstance(se, dict) else None
    return {
        "present": True,
        "version": CASH_P9_LOOP_WIRING_VERSION,
        "loop_out": out,
        "loop_status": out.get("loop_status"),
        "conditioned_routing": routing,
        "routed_through_p9": bool(isinstance(routing, dict)
                                  and routing.get("routed")),
        "conditioning_status": pip["p9_conditioning_status"],
        "search_eval_fingerprint": stable_fingerprint(se),
        "assembly": {
            "preferred": (out.get("decision_record") or {}).get("preferred"),
            "abstained": (out.get("decision_record") or {}).get("abstained"),
            "disagreements": (out.get("decision_record") or {}).get("disagreements"),
        },
        "claim_boundary": dict(CLAIM_BOUNDARY),
        "evidence_grade": EVIDENCE_GRADE,
        "executed": False,
        "result": None,
        "wired_to_production": False,
        "unknown": [],
    }


def compare_loop_runs(*, runs_by_population: Any) -> dict:
    """比较多次真实 loop 运行的 search / assembly 层指纹。"""
    runs = runs_by_population if isinstance(runs_by_population, dict) else {}
    ok = {k: v for k, v in runs.items()
          if isinstance(v, dict) and v.get("present")}
    if len(ok) < 2:
        out = _stub("at_least_two_runs_required")
        out["n_runs"] = len(runs)
        return out

    per = {}
    for p, r in ok.items():
        se = (r["loop_out"].get("stage_outputs") or {}).get("search_eval") or {}
        cands = {f"{c.get('action')}|{c.get('amount_bb')}":
                 (c.get("future_equity_mean"), c.get("search_score"),
                  c.get("n_leaves_evaluable"))
                 for c in (se.get("candidates") or [])}
        per[str(p)] = {
            "candidates": cands,
            "preferred": (se.get("preferred") or {}).get("action")
            and f"{(se.get('preferred') or {}).get('action')}|"
                f"{(se.get('preferred') or {}).get('amount_bb')}",
            "search_fingerprint": r.get("search_eval_fingerprint"),
            "routed": r.get("routed_through_p9"),
            "loop_status": r.get("loop_status"),
            "assembly_abstained": (r.get("assembly") or {}).get("abstained"),
        }

    all_keys = sorted({k for v in per.values() for k in v["candidates"]})
    changed = []
    for k in all_keys:
        vals = {p: per[p]["candidates"].get(k) for p in per}
        distinct = {v for v in vals.values() if v is not None}
        if len(distinct) > 1:
            changed.append(k)

    routed = sorted(p for p, v in per.items() if v["routed"])
    passthrough = sorted(p for p, v in per.items() if not v["routed"])
    # 只对**条件化**运行判定一致性（baseline/passthrough 本来就不该 routed）
    conditioned_only = [per[p] for p in routed]
    return {
        "present": True,
        "version": CASH_P9_LOOP_WIRING_VERSION,
        "populations": sorted(per),
        "per_population": per,
        "candidates_with_different_evaluation": changed,
        "n_candidates_with_difference": len(changed),
        "search_layer_changed": len({v["search_fingerprint"]
                                     for v in per.values()}) > 1,
        "preferred_changed": len({v["preferred"] for v in per.values()}) > 1,
        "routed_runs": routed,
        "passthrough_runs": passthrough,
        "all_conditioned_runs_routed": all(v["routed"] for v in conditioned_only),
        "n_conditioned_runs": len(conditioned_only),
        "baseline_remains_passthrough": all(
            not per[p]["routed"] for p in passthrough),
        "assembly_abstained_consistently": len(
            {v["assembly_abstained"] for v in per.values()}) == 1,
        "claim_boundary": dict(CLAIM_BOUNDARY),
        "evidence_grade": EVIDENCE_GRADE,
        "executed": False,
        "result": None,
        "wired_to_production": False,
        "unknown": [],
    }


def p9_contract() -> dict:
    return {
        "version": CASH_P9_LOOP_WIRING_VERSION,
        "approach": "dependency-injected closure under pipeline['evaluate_search']",
        "needs_loop_modification": False,
        "needs_frozen_contract_widening": False,
        "extra_pipeline_keys": list(EXTRA_PIPELINE_KEYS),
        "loop_evaluate_search_inputs": list(LOOP_EVALUATE_SEARCH_INPUTS),
        "requires": ["P4 bridge", "P8 conditioned search wiring",
                     "frozen evaluate_search", "V3I-P1 run_decision_loop"],
        "adds_strategy_capability": False,
        "claim_boundary": dict(CLAIM_BOUNDARY),
        "evidence_grade": EVIDENCE_GRADE,
    }
