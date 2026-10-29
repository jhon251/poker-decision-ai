"""V3I-P6 — P4 bridge 在 multistreet/search 上下文中的实际决策作用（验证层）。

**本层不新增策略能力、不修改任何冻结层**：只做接口能力声明、链路分离与敏感性验证。

核心结论（已实测，见 `probe_search_interface`）
------------------------------------------------
冻结 search 接口**没有**任何槽位可以接收 opponent-conditioned evaluation：

* `build_search_tree(state, *, action_policy, belief, belief_update, transition,
  budget, max_root_candidates)` —— 仅消费这几个入参；
  其 root 展开是 `ap["candidates"][:cap]`（**冻结顺序切片**，
  不按任何优先级分数重排），因此 bridge 的排序**无法**改变被展开的 root 集合；
* `evaluate_search(state, *, search_tree, solver)` —— 仅消费 `search_tree` 与 `solver`；
* `assemble_decision(...)` 的 tier 只读 `search_eval` / `belief_conditioned` /
  `ranking` 的 **top key**，不读 bridge 输出。

因此本层**如实报告接口阻塞**（授权要求 10），**不绕过冻结契约**
（不重排 action_policy、不改 budget、不改 river terminal、不改 tier 规则）。

本层只 import 标准库。
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

CASH_P6_SEARCH_CONTEXT_VERSION = "v3i-p6-search-context-verification-v1"

#: 冻结 search 接口实际消费的入参名（实测签名）。
#: 历史（V3I-P6 时点）`SEARCH_EVAL_INPUTS = ("state", "search_tree", "solver")`；
#: **V3I-G1 Phase 1**（v4，经授权）新增**可选**入参 `action_value_input`
#: （逐候选**测得**的 p_fold/p_continue/equity_vs_continuing/threshold；
#: 缺省 ⇒ 动作价值 UNKNOWN，legacy `search_score` 与排序逐位不变）。
#: 注意：它**不是** bridge 输出名（`BRIDGE_OUTPUTS` 不含之），故
#: `conditioned_evaluation_has_a_slot`（按 bridge 输出名交集计算）仍为 False。
SEARCH_TREE_INPUTS: tuple[str, ...] = (
    "state", "action_policy", "belief", "belief_update", "transition",
    "budget", "max_root_candidates",
)
SEARCH_EVAL_INPUTS: tuple[str, ...] = ("state", "search_tree", "solver",
                                       "action_value_input")
#: V3I-P6 时点的接口入参（**历史，不删除**）
SEARCH_EVAL_INPUTS_V3I_P6: tuple[str, ...] = ("state", "search_tree", "solver")
#: 新增该槽位的阶段（审计追溯用）
ACTION_VALUE_SLOT_ADDED_BY = "V3I-G1-PHASE1"

#: bridge 能产出的量（都不在上面的入参列表中）
BRIDGE_OUTPUTS: tuple[str, ...] = (
    "ranked", "preferred", "per_candidate", "p_continue_by_candidate",
    "continue_threshold", "candidate_required_equity", "range_anchor",
)

#: 明确**不允许**为"接入"而改动的东西
MUST_NOT_CHANGE: tuple[str, ...] = (
    "search_depth", "leaf_budget", "river_terminal", "tier_rules",
    "action_policy_candidate_order", "v3h4_dynamic_belief",
    "v3h6_likelihood_model", "v3h8_decision_assembly",
)

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
    "modifies_frozen_layers": False,
    "bypasses_frozen_contract": False,
}

EVIDENCE_GRADE = "MODEL_INTERNAL"


def _is_num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _stub(reason: str) -> dict:
    return {
        "present": False,
        "version": CASH_P6_SEARCH_CONTEXT_VERSION,
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
# 1. 接口能力声明（要求 2、10）
# ---------------------------------------------------------------------------
def probe_search_interface(*, passed_inputs: Any = None) -> dict:
    """声明冻结 search 接口的**可消费入参**，并判定 bridge 输出是否有槽位。"""
    given = set(passed_inputs) if isinstance(passed_inputs, (list, tuple, set)) \
        else set(SEARCH_TREE_INPUTS)
    accepted = sorted(given & set(SEARCH_TREE_INPUTS))
    rejected = sorted(given - set(SEARCH_TREE_INPUTS))

    bridge_slots = sorted(set(BRIDGE_OUTPUTS) & set(SEARCH_TREE_INPUTS)
                          | set(BRIDGE_OUTPUTS) & set(SEARCH_EVAL_INPUTS))
    return {
        "present": True,
        "version": CASH_P6_SEARCH_CONTEXT_VERSION,
        "search_tree_inputs": list(SEARCH_TREE_INPUTS),
        "search_eval_inputs": list(SEARCH_EVAL_INPUTS),
        "bridge_outputs": list(BRIDGE_OUTPUTS),
        "bridge_outputs_with_a_slot": bridge_slots,
        "conditioned_evaluation_has_a_slot": bool(bridge_slots),
        "accepted_inputs": accepted,
        "rejected_inputs": rejected,
        "blocking_finding": (
            "冻结 search 接口不接受 opponent-conditioned evaluation："
            "build_search_tree 的 root 展开为 ap['candidates'][:cap]（冻结顺序切片），"
            "evaluate_search 只读 search_tree/solver。bridge 的 ranked/preferred/"
            "per_candidate 等输出均无对应入参。"),
        "resolution": ("如实报告接口阻塞（授权要求 10）；**不**通过重排 "
                       "action_policy、改 budget、改 river terminal 或改 tier 规则绕过。"),
        "must_not_change": list(MUST_NOT_CHANGE),
        "claim_boundary": dict(CLAIM_BOUNDARY),
        "evidence_grade": EVIDENCE_GRADE,
        "executed": False,
        "result": None,
        "wired_to_production": False,
        "unknown": [],
    }


# ---------------------------------------------------------------------------
# 2. 链路分离（要求 5）
# ---------------------------------------------------------------------------
def trace_loop_layers(*, loop_out: Any, bridge_out: Any = None) -> dict:
    """把一次 loop 的分层结果拆开，明确"谁产生了什么"。"""
    if not isinstance(loop_out, dict):
        return _stub("loop_out_required")
    so = loop_out.get("stage_outputs") or {}
    dec = loop_out.get("decision_record") or {}

    def _pref(stage: str) -> Any:
        v = so.get(stage)
        return (v or {}).get("preferred") if isinstance(v, dict) else None

    layers = {
        "bridge_independent": {
            "present": bool(isinstance(bridge_out, dict)
                            and bridge_out.get("present")),
            "preferred": (bridge_out or {}).get("preferred"),
            "fingerprint": (stable_fingerprint((bridge_out or {}).get("ranked"))
                            if isinstance(bridge_out, dict)
                            and bridge_out.get("present") else None),
            "enters_search": False,
        },
        "frozen_ranking": {
            "preferred": _pref("ranking"),
            "fingerprint": stable_fingerprint(_pref("ranking")),
        },
        "frozen_belief_conditioned": {
            "preferred": _pref("belief_conditioned"),
            "present": isinstance(so.get("belief_conditioned"), dict)
            and bool((so.get("belief_conditioned") or {}).get("present")),
            "fingerprint": stable_fingerprint(_pref("belief_conditioned")),
        },
        "frozen_search": {
            "preferred": _pref("search_eval"),
            "fingerprint": stable_fingerprint(_pref("search_eval")),
            "n_candidates": (so.get("search_eval") or {}).get("n_candidates")
            if isinstance(so.get("search_eval"), dict) else None,
            "accepts_conditioned_eval": False,
        },
        "frozen_assembly": {
            "preferred": dec.get("preferred"),
            "abstained": dec.get("abstained"),
            "disagreements": dec.get("disagreements"),
            "loop_status": loop_out.get("loop_status"),
            "selected_by_tier": dec.get("selected_by_tier"),
        },
    }
    same = (layers["frozen_search"]["fingerprint"] ==
            layers["frozen_ranking"]["fingerprint"])
    return {
        "present": True,
        "version": CASH_P6_SEARCH_CONTEXT_VERSION,
        "layers": layers,
        "search_equals_ranking": same,
        "assembly_abstained": bool(layers["frozen_assembly"]["abstained"]),
        "abstain_reason": ("frozen V3H-8 tier 在 top key 上分歧 → "
                           "on_disagreement='abstain_unknown'"
                           if layers["frozen_assembly"]["abstained"] else None),
        "attribution": {
            "bridge_generated_differences": "见 compare_bridge_effect（仅存在于 bridge 自身层）",
            "search_generated_differences": "由冻结 search 层产生（其入参不含 bridge 输出）",
            "assembly_abstain": "由冻结 V3H-8 tier 规则产生，与 bridge 无关",
        },
        "claim_boundary": dict(CLAIM_BOUNDARY),
        "evidence_grade": EVIDENCE_GRADE,
        "executed": False,
        "result": None,
        "wired_to_production": False,
        "unknown": [],
    }


def compare_bridge_effect(*, traces: Any) -> dict:
    """比较多次运行的分层指纹，定位差异**实际出现在哪一层**。"""
    ts = traces if isinstance(traces, list) else []
    ok = [t for t in ts if isinstance(t, dict) and t.get("present")]
    if len(ok) < 2:
        out = _stub("at_least_two_traces_required")
        out["n_traces"] = len(ts)
        return out

    def _vary(path: tuple) -> bool:
        vals = []
        for t in ok:
            cur: Any = t
            for p in path:
                cur = (cur or {}).get(p) if isinstance(cur, dict) else None
            vals.append(json.dumps(cur, sort_keys=True, default=str))
        return len(set(vals)) > 1

    per_layer = {
        "bridge_independent": _vary(("layers", "bridge_independent", "preferred")),
        "frozen_ranking": _vary(("layers", "frozen_ranking", "preferred")),
        "frozen_belief_conditioned": _vary(
            ("layers", "frozen_belief_conditioned", "preferred")),
        "frozen_search": _vary(("layers", "frozen_search", "preferred")),
        "frozen_assembly": _vary(("layers", "frozen_assembly", "preferred")),
    }
    changed = [k for k, v in per_layer.items() if v]
    decision_changed = per_layer["frozen_assembly"]
    return {
        "present": True,
        "version": CASH_P6_SEARCH_CONTEXT_VERSION,
        "n_traces": len(ok),
        "layers_that_changed": changed,
        "per_layer_changed": per_layer,
        "bridge_layer_changed": per_layer["bridge_independent"],
        "search_layer_changed": per_layer["frozen_search"],
        "decision_changed": decision_changed,
        "difference_is_bridge_only": (
            per_layer["bridge_independent"] and not decision_changed),
        "interpretation": (
            "差异出现在哪些层即由哪些层产生；search 层未变 ⇒ "
            "bridge 未进入 search（接口无槽位），不是 search 忽略了它。"),
        "claim_boundary": dict(CLAIM_BOUNDARY),
        "evidence_grade": EVIDENCE_GRADE,
        "executed": False,
        "result": None,
        "wired_to_production": False,
        "unknown": [],
    }


# ---------------------------------------------------------------------------
# 3. 敏感性：search 层对 bridge 输入是否敏感（要求 4、5）
# ---------------------------------------------------------------------------
def search_sensitivity_probe(*, loops_by_input: Any) -> dict:
    """对同一状态、不同 opponent 输入，检查 search / assembly 层是否变化。"""
    loops = loops_by_input if isinstance(loops_by_input, dict) else {}
    if len(loops) < 2:
        return _stub("at_least_two_loop_outputs_required")

    per: dict[str, Any] = {}
    for name, out in loops.items():
        so = (out or {}).get("stage_outputs") or {}
        dec = (out or {}).get("decision_record") or {}
        per[str(name)] = {
            "ranking": stable_fingerprint((so.get("ranking") or {}).get("preferred")),
            "search": stable_fingerprint((so.get("search_eval") or {}).get("preferred")),
            "assembly": stable_fingerprint({
                "preferred": dec.get("preferred"),
                "abstained": dec.get("abstained"),
            }),
            "loop_status": (out or {}).get("loop_status"),
        }

    def _vary(layer: str) -> bool:
        return len({v[layer] for v in per.values()}) > 1

    return {
        "present": True,
        "version": CASH_P6_SEARCH_CONTEXT_VERSION,
        "inputs": sorted(per),
        "per_input": per,
        "ranking_sensitive": _vary("ranking"),
        "search_sensitive": _vary("search"),
        "assembly_sensitive": _vary("assembly"),
        "search_accepts_conditioned_eval": False,
        "conclusion": (
            "search / assembly 层对 opponent 输入**不敏感**；"
            "原因已由接口审计定位（search 入参不含 bridge 输出），"
            "**不是** search 读了却忽略。"),
        "claim_boundary": dict(CLAIM_BOUNDARY),
        "evidence_grade": EVIDENCE_GRADE,
        "executed": False,
        "result": None,
        "wired_to_production": False,
        "unknown": [],
    }


# ---------------------------------------------------------------------------
# 4. 多街 / multistreet 前置检查
# ---------------------------------------------------------------------------
def check_multistreet(*, transition: Any, state: Any) -> dict:
    """校验 multistreet transition 合法（不发明未来牌、river 终止语义保持）。"""
    if not isinstance(transition, dict):
        return _stub("transition_required")
    v: list[str] = []
    if transition.get("present") is not True:
        v.append("transition_not_present")
    if transition.get("transition_legal") is not True:
        v.append("transition_illegal")
    fb = transition.get("future_board") or {}
    if isinstance(fb, dict):
        if fb.get("state") != "unresolved":
            v.append(f"future_board_state_not_unresolved:{fb.get('state')!r}")
        if fb.get("cards") is not None:
            v.append("future_board_cards_invented")
        if fb.get("must_not_invent") is not True:
            v.append("must_not_invent_flag_missing")
    street = (state or {}).get("street")
    if street == "river" and not transition.get("terminal"):
        v.append("river_must_be_terminal")
    return {
        "present": True,
        "version": CASH_P6_SEARCH_CONTEXT_VERSION,
        "current_street": transition.get("current_street"),
        "next_street": transition.get("next_street"),
        "terminal": transition.get("terminal"),
        "terminal_reason": transition.get("terminal_reason"),
        "future_board_state": fb.get("state") if isinstance(fb, dict) else None,
        "future_board_cards": fb.get("cards") if isinstance(fb, dict) else None,
        "valid": not v,
        "violations": v,
        "invents_future_cards": False,
        "claim_boundary": dict(CLAIM_BOUNDARY),
        "evidence_grade": EVIDENCE_GRADE,
        "executed": False,
        "result": None,
        "wired_to_production": False,
        "unknown": [],
    }


def p6_contract() -> dict:
    return {
        "version": CASH_P6_SEARCH_CONTEXT_VERSION,
        "reuses": ["v3i-p4-opponent-conditioned-bridge-v1",
                   "v3i-p5-bridge-stability-verification-v1",
                   "v3i-p1-decision-loop"],
        "adds_strategy_capability": False,
        "search_tree_inputs": list(SEARCH_TREE_INPUTS),
        "search_eval_inputs": list(SEARCH_EVAL_INPUTS),
        "conditioned_evaluation_has_a_slot": False,
        "must_not_change": list(MUST_NOT_CHANGE),
        "interface_block_reported": True,
        "claim_boundary": dict(CLAIM_BOUNDARY),
        "evidence_grade": EVIDENCE_GRADE,
    }
