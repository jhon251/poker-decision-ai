"""V3I-P13 — 把 `feasible_keys` 最小契约接入 P1/P9 实际决策链。

接入方式（与 P9 同构，**不修改 P1 loop**）
------------------------------------------
P1 `run_decision_loop` 以 `pipeline["assemble_decision"]` 调用装配层；因此本层提供
一个**同签名闭包**注入该键，闭包内部把从 **search tree 实际展开的 root 候选**
导出的 canonical keys 作为 `feasible_keys` 传给**已实施契约的** V3H-8。

可行域来源（要求 2）：**只**取 `search_tree["root"]["children"]` 的
`(action, amount_bb)` —— 即 search **实际可评估**的候选，**不伪造、不扩大**。

本层只 import 标准库；不修改任何冻结层。
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

CASH_P13_FEASIBLE_DOMAIN_WIRING_VERSION = "v3i-p13-feasible-domain-wiring-v1"

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
    "enables_override": False,
    "relaxes_abstain": False,
    "expands_search_domain": False,
}

EVIDENCE_GRADE = "MODEL_INTERNAL"


def _is_num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _stub(reason: str) -> dict:
    return {
        "present": False,
        "version": CASH_P13_FEASIBLE_DOMAIN_WIRING_VERSION,
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


def feasible_keys_from_search_tree(*, search_tree: Any) -> dict:
    """从 search tree **实际展开的 root 候选**导出可行域（要求 2）。

    只读 `root.children`；若结构缺失 → `present=False`（不发明可行域）。
    """
    if not isinstance(search_tree, dict) or not search_tree.get("present"):
        return _stub("present_search_tree_required")
    root = search_tree.get("root")
    if not isinstance(root, dict):
        return _stub("search_tree_root_missing")
    children = [c for c in (root.get("children") or []) if isinstance(c, dict)]
    if not children:
        return _stub("no_root_candidates_expanded")
    keys: list[list] = []
    for c in children:
        act = c.get("action")
        amt = c.get("amount_bb")
        if not isinstance(act, str):
            continue
        keys.append([act, round(float(amt), 6) if _is_num(amt) else None])
    if not keys:
        return _stub("no_canonical_keys_derivable")
    return {
        "present": True,
        "version": CASH_P13_FEASIBLE_DOMAIN_WIRING_VERSION,
        "feasible_keys": keys,
        "n_keys": len(keys),
        "source": "search_tree.root.children",
        "derived_from_actual_search_expansion": True,
        "expands_search_domain": False,
        "fabricated": False,
        "claim_boundary": dict(CLAIM_BOUNDARY),
        "evidence_grade": EVIDENCE_GRADE,
        "executed": False, "result": None, "wired_to_production": False,
        "unknown": [],
    }


def make_feasible_domain_assembler(*, frozen_assemble: Any,
                                   feasible_keys: Any = None) -> dict:
    """构造与冻结 `assemble_decision` **同签名**的闭包，注入 `feasible_keys`。

    * `feasible_keys is None` → 原样调用冻结函数（**默认路径不变**）；
    * 否则 → 透传 `feasible_keys` 给**已实施契约**的同一函数。
    """
    if not callable(frozen_assemble):
        return _stub("frozen_assemble_required")

    def assemble_decision(state, *, ranking=None, belief_conditioned=None,
                          search_eval=None, solver=None, policy=None,
                          allow_higher_tier_override=False,
                          override_justification=None):
        if feasible_keys is None:
            return frozen_assemble(
                state, ranking=ranking, belief_conditioned=belief_conditioned,
                search_eval=search_eval, solver=solver, policy=policy,
                allow_higher_tier_override=allow_higher_tier_override,
                override_justification=override_justification)
        return frozen_assemble(
            state, ranking=ranking, belief_conditioned=belief_conditioned,
            search_eval=search_eval, solver=solver, policy=policy,
            allow_higher_tier_override=allow_higher_tier_override,
            override_justification=override_justification,
            feasible_keys=feasible_keys)

    return {
        "present": True,
        "version": CASH_P13_FEASIBLE_DOMAIN_WIRING_VERSION,
        "callable": assemble_decision,
        "signature": ("(state, *, ranking=None, belief_conditioned=None, "
                      "search_eval=None, solver=None, policy=None, "
                      "allow_higher_tier_override=False, "
                      "override_justification=None)"),
        "will_apply_feasible_domain": feasible_keys is not None,
        "n_feasible_keys": (len(feasible_keys)
                            if isinstance(feasible_keys, (list, tuple)) else 0),
        "uses_global_state": False,
        "uses_monkeypatch": False,
        "enables_override": False,
        "claim_boundary": dict(CLAIM_BOUNDARY),
        "evidence_grade": EVIDENCE_GRADE,
        "executed": False, "result": None, "wired_to_production": False,
        "unknown": [],
    }


def run_loop_with_feasible_domain(*, state: Any, pipeline: Any,
                                  run_decision_loop_fn: Any,
                                  frozen_assemble: Any,
                                  feasible_keys: Any,
                                  loop_kwargs: Any = None) -> dict:
    """用**真实** `run_decision_loop` 跑一次，注入限域版 assemble_decision。"""
    if not callable(run_decision_loop_fn):
        return _stub("run_decision_loop_fn_required")
    if not isinstance(pipeline, dict):
        return _stub("pipeline_required")
    built = make_feasible_domain_assembler(frozen_assemble=frozen_assemble,
                                           feasible_keys=feasible_keys)
    if not built.get("present"):
        return _stub(f"assembler_build_failed:{built.get('reason')}")
    pip = dict(pipeline)
    pip["assemble_decision"] = built["callable"]
    out = run_decision_loop_fn(state, pipeline=pip, **(loop_kwargs or {}))
    if not isinstance(out, dict):
        return _stub("loop_returned_non_dict")
    dec = out.get("decision_record") or {}
    # 注意：`feasible_domain_*` 由装配层输出，位于 decision_record["decision"]；
    # decision_record 顶层没有该字段（此前读错层级导致误报 False）。
    asm = dec.get("decision") if isinstance(dec.get("decision"), dict) else {}
    return {
        "present": True,
        "version": CASH_P13_FEASIBLE_DOMAIN_WIRING_VERSION,
        "loop_out": out,
        "loop_status": out.get("loop_status"),
        "feasible_domain_applied": bool(asm.get("feasible_domain_applied")),
        "feasible_domain_size": asm.get("feasible_domain_size"),
        "assembly": {
            "preferred": dec.get("preferred"),
            "abstained": dec.get("abstained"),
            "n_disagreements": len(dec.get("disagreements") or []),
            "selected_by_tier": dec.get("selected_by_tier"),
        },
        "claim_boundary": dict(CLAIM_BOUNDARY),
        "evidence_grade": EVIDENCE_GRADE,
        "executed": False, "result": None, "wired_to_production": False,
        "unknown": [],
    }


def p13_contract() -> dict:
    return {
        "version": CASH_P13_FEASIBLE_DOMAIN_WIRING_VERSION,
        "implements": "P12 设计的 V3H-8 feasible_keys 最小契约",
        "contract_target": "cash_strategy_decision_assembly.assemble_decision",
        "injection": "pipeline['assemble_decision'] 同签名闭包（不修改 P1 loop）",
        "feasible_domain_source": "search_tree.root.children（search 实际展开的候选）",
        "expands_search_domain": False,
        "enables_override": False,
        "modifies_scoring_formula_or_tier_precedence": False,
        "modifies_solver_v3h4_v3h6_cfi_lambda_river_budget": False,
        "claim_boundary": dict(CLAIM_BOUNDARY),
        "evidence_grade": EVIDENCE_GRADE,
    }
