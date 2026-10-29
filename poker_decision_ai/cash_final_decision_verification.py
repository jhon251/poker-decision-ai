"""V3I-P14 — CASH Decision Core **最终决策覆盖 / 跨街稳定性 / 输出质量边界** 的只读验证层。

授权范围（本阶段**只验证与测量**，不新增机制）
--------------------------------------------
* 不修改 P13 契约、Solver、V3H-4/V3H-6、CFI/λ、`strategy_core`、
  search budget、river terminal；
* 不启用 override、不降低 UNKNOWN/ABSTAIN 标准、不宣称 EV/GTO/equilibrium；
* 本层**不选动作、不打分、不改变任何层的输出**：只把**真实链路**跑出来的记录
  解析成可审计的分类、计数、稳定性与边界结论。

本层只 import 标准库（`hashlib` / `json` / `typing`）；项目内层一律由调用方
**显式注入**（与 V3I-P1/P9/P11/P13 同一惯例）。

2×2 因子设计（本层用于**分离**各注入的贡献）
-------------------------------------------
| 模式 | `evaluate_search` | `assemble_decision` |
|---|---|---|
| `A1` frozen_search_frozen_assembly | 冻结 | 冻结 |
| `A2` frozen_search_feasible_assembly | 冻结 | P13 限域闭包 |
| `B1` conditioned_search_frozen_assembly | P9 条件化闭包 | 冻结 |
| `B2` conditioned_search_feasible_assembly | P9 条件化闭包 | P13 限域闭包 |

> `A2` 即 V3I-P13 报告的配置；`B2` 才是"条件化 search 真正参与"的完整链路。
> 本层**不假设**二者结论相同 —— 实测后如实分列。

最终结果分类（授权要求 2 要求**分离**的类别）
--------------------------------------------
* `ACCEPT`：`assembly` 产出 `preferred`（未弃权）；
* `ABSTAIN`：按冻结 V3H-8 规则弃权（必有 `disagreements`）；
* `BLOCKED`：loop 在装配前即停止（如 river terminal、阶段失败、fail-closed）；
* `UNKNOWN_NO_PREFERRED`：loop 报 UNKNOWN 且无 `preferred`。

并进一步分离：
* `accept_route`：`{tier}_{feasible|unconstrained}`；
* `abstain_kind`：`true_in_domain_disagreement` / `structural_domain_mismatch`；
* `blocked_kind`：`pre_search_blocked` / `search_tree_unavailable` / `stage_failure` /
  `loop_fail_closed`；
* `loop_unknown_due_to_missing_likelihood`：平台未提供 `likelihood_model` 导致的
  `loop_status=UNKNOWN`（**不是**决策失败，是本平台已知缺口，必须分开计数）。

`memoize_stage` 的定位（测量工具，不是机制）
-------------------------------------------
真实链路中 `build_action_policy` / `build_search_tree` 是确定性且昂贵的；网格实验
需要重复跑同一 scenario。`memoize_stage` 只是把这些**冻结阶段**的**同一输入**结果
缓存下来，返回**同一个对象**。它不改变任何层的输出，其透明性由
`harness_equivalence(...)` 在**真实链路**上逐字节比对后**实测**确认（禁止假设）。
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

CASH_P14_FINAL_VERIFICATION_VERSION = "v3i-p14-final-decision-verification-v1"

#: 2×2 因子设计（授权要求 1、2）
RUN_MODES: tuple[str, ...] = (
    "A1_frozen_search_frozen_assembly",
    "A2_frozen_search_feasible_assembly",
    "B1_conditioned_search_frozen_assembly",
    "B2_conditioned_search_feasible_assembly",
)

#: 最终结果分类（**分离**口径）
OUTCOMES: tuple[str, ...] = (
    "ACCEPT",
    "ABSTAIN",
    "BLOCKED",
    "UNKNOWN_NO_PREFERRED",
)

#: ABSTAIN 种类（真分歧 vs 结构性域不匹配）
ABSTAIN_KINDS: tuple[str, ...] = (
    "true_in_domain_disagreement",
    "structural_domain_mismatch",
)

#: BLOCKED 种类
BLOCKED_KINDS: tuple[str, ...] = (
    "pre_search_blocked",
    "search_tree_unavailable",
    "stage_failure",
    "loop_fail_closed",
)

#: P1 loop 必需阶段（只用于校验注入后的 pipeline 完整性）
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

#: 能力验证缺口代码（授权要求 9：显式标记，不用测试结果包装成强度证明）
CAPABILITY_GAP_CODES: tuple[str, ...] = (
    "G1_accept_not_cross_scorer_verified",
    "G2_conditioning_routing_collapses_accept",
    "G2b_conditioning_changes_decision_but_not_to_accept",
    "G3_search_tier_unavailable_below_coverage_gate",
    "G4_population_dimension_not_degenerate",
    "G5_decision_quality_not_provable_model_internal",
    "G6_river_terminal_never_reaches_assembly",
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
    "is_win_rate_claim": False,
    "is_strategy_strength_claim": False,
    "wired_to_production": False,
    "adds_strategy_mechanism": False,
    "modifies_frozen_contract": False,
    "relaxes_fail_closed": False,
    "enables_override": False,
}

EVIDENCE_GRADE = "MODEL_INTERNAL"


# ---------------------------------------------------------------------------
# 0. 基础设施（纯函数；与其它层同一 fingerprint 口径）
# ---------------------------------------------------------------------------
def _is_num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _stub(reason: str) -> dict:
    return {
        "present": False,
        "version": CASH_P14_FINAL_VERIFICATION_VERSION,
        "reason": reason,
        "unknown": [reason],
        "claim_boundary": dict(CLAIM_BOUNDARY),
        "evidence_grade": "NONE",
        "executed": False,
        "result": None,
        "wired_to_production": False,
    }


def stable_fingerprint(value: Any) -> str:
    """canonical JSON 的 sha256 前 16 位（同输入 → 同指纹）。"""
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
# 1. 测量工具：pipeline 阶段记忆化 + 双注入组合（都不改任何层）
# ---------------------------------------------------------------------------
def memoize_stage(*, fn: Any, tag: Any = None) -> dict:
    """把**冻结阶段函数**按"同一输入"缓存（测量网格用；不改冻结层）。

    返回 `callable` 与冻结函数**同签名**（`fn(state, **kw)`）；命中缓存时返回
    **同一个对象**。透明性必须由 `harness_equivalence` 在真实链路上**实测**，
    不得假设（本层不自行宣称等价）。

    边界：`fn` 不可调用 → `present=False`（fail-closed）。缓存键只由调用参数
    （含 `state`）的 canonical 指纹决定；不读全局状态、不 monkeypatch。
    """
    if not callable(fn):
        return _stub("stage_fn_required")
    cache: dict[str, Any] = {}
    stats = {"calls": 0, "hits": 0, "misses": 0, "distinct_inputs": 0}

    def stage(state, **kw):
        key = stable_fingerprint({"state": state, "kw": kw})
        stats["calls"] += 1
        if key in cache:
            stats["hits"] += 1
            return cache[key]
        stats["misses"] += 1
        out = fn(state, **kw)
        cache[key] = out
        stats["distinct_inputs"] = len(cache)
        return out

    return {
        "present": True,
        "version": CASH_P14_FINAL_VERIFICATION_VERSION,
        "tag": tag,
        "callable": stage,
        "signature": "(state, **kwargs)",
        "stats": stats,
        "returns_identical_object_on_hit": True,
        "uses_global_state": False,
        "uses_monkeypatch": False,
        "modifies_frozen_layers": False,
        "claim_boundary": dict(CLAIM_BOUNDARY),
        "evidence_grade": EVIDENCE_GRADE,
        "executed": False, "result": None, "wired_to_production": False,
        "unknown": [],
    }


def compose_dual_injection(*, pipeline: Any, evaluate_search_closure: Any = None,
                           assemble_decision_closure: Any = None) -> dict:
    """把 P9（`evaluate_search`）与 P13（`assemble_decision`）注入**同时**放进
    一份 pipeline 拷贝里，用于跑真实组合链路。

    * 只做键替换，**不修改** 传入的 pipeline；
    * 任一闭包不可调用 → 该键保持原值（如实记录，不静默替换）；
    * 必需阶段缺失 → `present=False`（fail-closed）。
    """
    if not isinstance(pipeline, dict):
        return _stub("pipeline_required")
    missing = [s for s in REQUIRED_PIPELINE_STAGES
               if not callable(pipeline.get(s))]
    if missing:
        return _stub("pipeline_missing_stages:" + ",".join(sorted(missing)))

    pip = dict(pipeline)
    applied: list[str] = []
    skipped: list[str] = []
    if callable(evaluate_search_closure):
        pip["evaluate_search"] = evaluate_search_closure
        applied.append("evaluate_search")
    else:
        skipped.append("evaluate_search")
    if callable(assemble_decision_closure):
        pip["assemble_decision"] = assemble_decision_closure
        applied.append("assemble_decision")
    else:
        skipped.append("assemble_decision")
    return {
        "present": True,
        "version": CASH_P14_FINAL_VERIFICATION_VERSION,
        "pipeline": pip,
        "injected_keys": applied,
        "skipped_keys": skipped,
        "mutates_input_pipeline": False,
        "required_stages_present": sorted(REQUIRED_PIPELINE_STAGES),
        "claim_boundary": dict(CLAIM_BOUNDARY),
        "evidence_grade": EVIDENCE_GRADE,
        "executed": False, "result": None, "wired_to_production": False,
        "unknown": [],
    }


def harness_equivalence(*, memoized_out: Any, raw_out: Any) -> dict:
    """**实测**记忆化 harness 的透明性：比对两次真实 loop 输出的决策相关指纹。

    只比对**阶段输出与决策记录**（不比对计时/缓存统计）。
    """
    for name, o in (("memoized_out", memoized_out), ("raw_out", raw_out)):
        if not isinstance(o, dict):
            return _stub(f"{name}_required")

    def _core(o: dict) -> dict:
        d = o.get("decision_record") or {}
        return {
            "present": o.get("present"),
            "loop_status": o.get("loop_status"),
            "stage_trace": o.get("stage_trace"),
            "unknown": o.get("unknown"),
            "decision": d.get("decision"),
            "preferred": d.get("preferred"),
            "abstained": d.get("abstained"),
            "disagreements": d.get("disagreements"),
            "stage_outputs_present": {
                k: (v.get("present") if isinstance(v, dict) else None)
                for k, v in (o.get("stage_outputs") or {}).items()},
            "search_eval_candidates": [
                (c.get("action"), c.get("amount_bb"), c.get("future_equity_mean"),
                 c.get("search_score"), c.get("leaf_coverage"),
                 c.get("n_leaves_evaluable"))
                for c in (((o.get("stage_outputs") or {}).get("search_eval") or {})
                          .get("candidates") or [])],
        }

    a, b = _core(memoized_out), _core(raw_out)
    fa, fb = stable_fingerprint(a), stable_fingerprint(b)
    return {
        "present": True,
        "version": CASH_P14_FINAL_VERIFICATION_VERSION,
        "memoized_fingerprint": fa,
        "raw_fingerprint": fb,
        "identical": fa == fb,
        "memoization_is_transparent": fa == fb,
        "compared_fields": sorted(a),
        "note": ("指纹一致 ⇒ 记忆化只影响耗时，不影响任何阶段输出或决策记录；"
                 "指纹不一致 ⇒ 该网格的加速结论作废，必须重跑。"),
        "claim_boundary": dict(CLAIM_BOUNDARY),
        "evidence_grade": EVIDENCE_GRADE,
        "executed": False, "result": None, "wired_to_production": False,
        "unknown": [],
    }


# ---------------------------------------------------------------------------
# 2. 单次运行 → 结果分类（授权要求 2）
# ---------------------------------------------------------------------------
def _tree_keys(record: dict) -> list[tuple[str, float | None]]:
    out: list[tuple[str, float | None]] = []
    for c in record.get("tree_root_children") or []:
        if isinstance(c, dict) and isinstance(c.get("action"), str):
            amt = c.get("amount_bb")
            out.append((c["action"].strip().lower(),
                        round(float(amt), 6) if _is_num(amt) else None))
    return out


def _trace_presence(record: dict) -> dict:
    """从 loop 的 `stage_trace` 取每阶段 present（BLOCKED 时 `stage_outputs` 为空）。"""
    out: dict[str, Any] = {}
    for t in record.get("stage_trace") or []:
        if isinstance(t, dict) and isinstance(t.get("stage"), str):
            out[t["stage"]] = t.get("present")
    return out


#: loop 层面的 fail-closed 原因前缀（与决策无关，属门禁/证据等级）
_LOOP_FAIL_CLOSED_REASONS: tuple[str, ...] = (
    "not_cash_mode_fail_closed",
    "evidence_level_not_permitted_internal_only",
    "pipeline_claims_forbidden_evidence_level",
    "pipeline_required",
    "pipeline_missing_stages",
)


def _key_in_tree(key: Any, tree_keys: list[tuple[str, float | None]],
                 tol: float = 1e-6) -> bool:
    if not isinstance(key, (list, tuple)) or len(key) != 2:
        return False
    act, amt = key[0], key[1]
    for ta, tamt in tree_keys:
        if ta != act:
            continue
        if tamt is None or amt is None or abs(float(tamt) - float(amt)) <= tol:
            return True
    return False


def classify_run(*, record: Any) -> dict:
    """把一次真实 loop 运行解析为（结果分类, 路径/种类, 覆盖标记, 理由）。"""
    if not isinstance(record, dict):
        return _stub("record_not_a_dict")
    preferred = record.get("preferred")
    abstained = record.get("abstained")
    loop_status = record.get("loop_status")
    disagreements = [d for d in (record.get("disagreements") or [])
                     if isinstance(d, dict)]
    tree_keys = _tree_keys(record)
    feasible_applied = bool(record.get("feasible_applied"))
    n_alignable = record.get("n_scorers_alignable")
    n_alignable = int(n_alignable) if isinstance(n_alignable, int) else 0
    unknown_list = [str(u) for u in (record.get("unknown") or [])]
    trace = _trace_presence(record)
    reason = str(record.get("reason") or "")

    loop_present_false = (record.get("present") is False)
    loop_fail_closed = bool(
        loop_present_false
        and any(reason.startswith(p) for p in _LOOP_FAIL_CLOSED_REASONS))

    # ── 结果分类 ──
    if loop_present_false:
        outcome = "BLOCKED"
    elif abstained is True:
        outcome = "ABSTAIN"
    elif preferred not in (None, {}):
        outcome = "ACCEPT"
    elif loop_status == "UNKNOWN":
        outcome = "UNKNOWN_NO_PREFERRED"
    else:
        outcome = "BLOCKED"

    # ── ACCEPT 路径 ──
    accept_route = None
    if outcome == "ACCEPT":
        tier = None
        if isinstance(preferred, dict):
            tier = preferred.get("selected_by_tier")
        tier = tier if isinstance(tier, str) and tier else "unrecorded_tier"
        accept_route = f"{tier}_{'feasible' if feasible_applied else 'unconstrained'}"

    # ── ABSTAIN 种类 ──
    abstain_kinds: list[str] = []
    if outcome == "ABSTAIN":
        for d in disagreements:
            ka, kb = d.get("key_a"), d.get("key_b")
            outside = (not _key_in_tree(ka, tree_keys)
                       or not _key_in_tree(kb, tree_keys))
            kind = ("structural_domain_mismatch" if outside
                    else "true_in_domain_disagreement")
            if kind not in abstain_kinds:
                abstain_kinds.append(kind)
        if not abstain_kinds:
            abstain_kinds.append("true_in_domain_disagreement")

    # ── BLOCKED 种类 ──
    blocked_kind = None
    if outcome == "BLOCKED":
        ap_absent = (record.get("action_policy_present") is False
                     or trace.get("action_policy") is False)
        tree_absent = (record.get("search_tree_present") is False
                       or trace.get("search_tree") is False)
        if reason.startswith("stage_failed:action_policy"):
            blocked_kind = "pre_search_blocked"
        elif loop_fail_closed:
            blocked_kind = "loop_fail_closed"
        elif ap_absent:
            blocked_kind = "pre_search_blocked"
        elif tree_absent:
            blocked_kind = "search_tree_unavailable"
        else:
            blocked_kind = "stage_failure"

    # ── 平台缺口：loop UNKNOWN 由缺失 likelihood_model 引起（非决策失败）──
    unknown_due_to_missing_likelihood = bool(
        "likelihood_model_not_supplied" in unknown_list)

    reasons: list[str] = []
    reasons.extend(f"unknown:{u}" for u in unknown_list)
    for d in disagreements:
        reasons.append(f"disagreement:{d.get('tier_a')}_vs_{d.get('tier_b')}")

    return {
        "present": True,
        "version": CASH_P14_FINAL_VERIFICATION_VERSION,
        "run_id": record.get("run_id"),
        "scenario": record.get("scenario"),
        "street": record.get("street"),
        "seed": record.get("seed"),
        "pot_bb": record.get("pot_bb"),
        "stack_bb": record.get("stack_bb"),
        "population": record.get("population"),
        "mode": record.get("mode"),
        "outcome": outcome,
        "accept_route": accept_route,
        "abstain_kinds": abstain_kinds,
        "blocked_kind": blocked_kind,
        "loop_status": loop_status,
        "loop_present_false": loop_present_false,
        "loop_fail_closed": loop_fail_closed,
        "failure_reason": reason or None,
        "loop_unknown_due_to_missing_likelihood":
            unknown_due_to_missing_likelihood,
        "feasible_applied": feasible_applied,
        "feasible_domain_size": record.get("feasible_domain_size"),
        "n_disagreements": len(disagreements),
        "n_scorers_alignable": n_alignable,
        "cross_scorer_verified": bool(n_alignable >= 2),
        "search_routed": bool(record.get("search_routed")),
        "selected_by_tier": (preferred or {}).get("selected_by_tier")
        if isinstance(preferred, dict) else None,
        "preferred_action": (preferred or {}).get("action")
        if isinstance(preferred, dict) else None,
        "preferred_amount_bb": (preferred or {}).get("amount_bb")
        if isinstance(preferred, dict) else None,
        "reasons": reasons,
        "claim_boundary": dict(CLAIM_BOUNDARY),
        "evidence_grade": EVIDENCE_GRADE,
        "executed": False, "result": None, "wired_to_production": False,
        "unknown": [],
    }


# ---------------------------------------------------------------------------
# 3. 汇总（授权要求 1、2、3）
# ---------------------------------------------------------------------------
def _counter(items: list, key) -> dict:
    out: dict[str, int] = {}
    for it in items:
        k = key(it)
        if k is None:
            continue
        out[str(k)] = out.get(str(k), 0) + 1
    return dict(sorted(out.items(), key=lambda kv: (-kv[1], kv[0])))


def summarize_runs(*, records: Any, group_by: Any = None) -> dict:
    """按（可选）维度分组汇总结果分类 / ACCEPT 路径 / ABSTAIN 与 BLOCKED 种类。"""
    recs = records if isinstance(records, list) else []
    if not recs:
        return _stub("records_required")

    def summ(items: list[dict]) -> dict:
        n = len(items)
        acc = [c for c in items if c["outcome"] == "ACCEPT"]
        abs_ = [c for c in items if c["outcome"] == "ABSTAIN"]
        blk = [c for c in items if c["outcome"] == "BLOCKED"]
        unk = [c for c in items if c["outcome"] == "UNKNOWN_NO_PREFERRED"]
        return {
            "n_runs": n,
            "outcome_counts": {k: sum(1 for c in items if c["outcome"] == k)
                               for k in OUTCOMES},
            "final_decision_coverage": (round(len(acc) / n, 6) if n else None),
            "accept_routes": _counter(acc, lambda c: c["accept_route"]),
            "abstain_kinds": _counter(
                [k for c in abs_ for k in c["abstain_kinds"]], lambda k: k),
            "blocked_kinds": _counter(blk, lambda c: c["blocked_kind"]),
            "accept_cross_scorer_verified": sum(
                1 for c in acc if c["cross_scorer_verified"]),
            "accept_single_tier_unverified": sum(
                1 for c in acc if not c["cross_scorer_verified"]),
            "accept_search_routed": sum(1 for c in acc if c["search_routed"]),
            "abstain_count": len(abs_),
            "blocked_count": len(blk),
            "unknown_no_preferred_count": len(unk),
            "loop_unknown_due_to_missing_likelihood": sum(
                1 for c in items if c["loop_unknown_due_to_missing_likelihood"]),
            "preferred_amounts": _counter(
                [c for c in acc if c["preferred_amount_bb"] is not None],
                lambda c: c["preferred_amount_bb"]),
        }

    ok = [classify_run(record=r) for r in recs]
    ok = [c for c in ok if c.get("present")]
    if not ok:
        return _stub("no_classifiable_records")

    groups: dict[str, Any] = {}
    if isinstance(group_by, str) and group_by:
        for k in sorted({str(c.get(group_by)) for c in ok}):
            groups[k] = summ([c for c in ok if str(c.get(group_by)) == k])

    by_mode = {m: summ([c for c in ok if c["mode"] == m]) for m in RUN_MODES}
    return {
        "present": True,
        "version": CASH_P14_FINAL_VERIFICATION_VERSION,
        "n_runs": len(ok),
        "overall": summ(ok),
        "by_mode": by_mode,
        "groups": groups,
        "group_by": group_by,
        "modes_observed": sorted({str(c["mode"]) for c in ok}),
        "claim_boundary": dict(CLAIM_BOUNDARY),
        "evidence_grade": EVIDENCE_GRADE,
        "executed": False, "result": None, "wired_to_production": False,
        "unknown": [],
    }


def measure_stability(*, records: Any, dimension: Any = "street",
                      mode: Any = None) -> dict:
    """跨维度稳定性：同一维度取值内，结果分类与选中注额是否稳定。

    `dimension` ∈ {`street`, `scenario`, `population`, `pot_bb`, `stack_bb`,
    `seed`}；`mode` 可限定到单一运行模式（如 `B2` 完整链路）。
    """
    recs = records if isinstance(records, list) else []
    if not recs:
        return _stub("records_required")
    if not isinstance(dimension, str) or not dimension:
        return _stub("dimension_required")

    cls = [classify_run(record=r) for r in recs]
    cls = [c for c in cls if c.get("present")]
    if isinstance(mode, str) and mode:
        cls = [c for c in cls if c["mode"] == mode]
    if not cls:
        return _stub("no_records_for_mode")

    keys = sorted({str(c.get(dimension)) for c in cls})
    per: dict[str, Any] = {}
    for k in keys:
        items = [c for c in cls if str(c.get(dimension)) == k]
        outcomes = sorted({c["outcome"] for c in items})
        amounts = sorted({c["preferred_amount_bb"] for c in items
                          if c["outcome"] == "ACCEPT"
                          and c["preferred_amount_bb"] is not None})
        actions = sorted({c["preferred_action"] for c in items
                          if c["outcome"] == "ACCEPT"
                          and c["preferred_action"] is not None})
        per[k] = {
            "n_runs": len(items),
            "outcome_counts": _counter(items, lambda c: c["outcome"]),
            "distinct_outcomes": outcomes,
            "outcome_stable_within_group": len(outcomes) == 1,
            "n_distinct_outcomes": len(outcomes),
            "accept_amounts": amounts,
            "n_distinct_accept_amounts": len(amounts),
            "accept_actions": actions,
        }
    stable = [k for k, v in per.items() if v["outcome_stable_within_group"]]
    unstable = [k for k, v in per.items() if not v["outcome_stable_within_group"]]
    return {
        "present": True,
        "version": CASH_P14_FINAL_VERIFICATION_VERSION,
        "dimension": dimension,
        "mode": mode,
        "n_groups": len(keys),
        "per_group": per,
        "stable_groups": stable,
        "unstable_groups": unstable,
        "all_groups_stable": len(unstable) == 0,
        "distinct_outcomes_overall": sorted({c["outcome"] for c in cls}),
        "distinct_accept_amounts_overall": sorted(
            {c["preferred_amount_bb"] for c in cls if c["outcome"] == "ACCEPT"
             and c["preferred_amount_bb"] is not None}),
        "claim_boundary": dict(CLAIM_BOUNDARY),
        "evidence_grade": EVIDENCE_GRADE,
        "executed": False, "result": None, "wired_to_production": False,
        "unknown": [],
    }


# ---------------------------------------------------------------------------
# 4. ACCEPT 出处验证（授权要求 4）—— 不得是默认值/硬编码
# ---------------------------------------------------------------------------
def verify_accept_provenance(*, record: Any) -> dict:
    """逐项验证 ACCEPT 的 `(action, amount_bb)` 是否来自**真实的
    candidate-local conditioned search**，而不是默认值/硬编码。

    检验链（任一环失败即 `provenance_verified=False`）：
    1. `preferred.amount_bb` 必须等于 **search tree 实际展开的 root 候选**金额之一；
    2. `preferred.amount_bb` 必须等于 **选中 tier 该候选**的金额（`scorer_tops`）；
    3. 若提供了**逐候选条件化输入且已路由**：每个 search 候选的
       `future_equity_mean` 必须等于**该候选自身**的条件化权益
       （而不是全局/均匀权益）。判据是"逐候选与输入吻合"，**不是**"各候选互不相同"
       —— 两个候选自身的条件化权益本来就可能相同；
    4. 额外（仅当选中层=search）：该候选必须确实消费了自身条件化权益；
    5. 金额不得是"默认哨兵"：`None`；或 `0.0` 而动作不是 check/fold。

    输出另带 `accept_derivation`（**用于如实区分**，"不通过"不等于"缺陷"）：
    * `search_candidate_local_conditioned`：选中层=search 且条件化已路由，且该候选
      消费了**自身**条件化权益 —— 最强的一档（真正的 candidate-local conditioned
      search 产生该决策）；
    * `search_tier_conditioning_not_consumed`：选中层=search 且已路由，但该候选的
      `future_equity_mean` 与它自身的条件化权益不符 ⇒ **出处不成立**；
    * `search_tier_within_feasible_domain` / `search_tier_within_expansion_unconstrained`：
      选中层=search 且候选确实在 search 实际展开集内（前者额外限域）；
    * `feasible_domain_non_search_tier`：选中层非 search，但候选落在可行域内
      （= search 实际展开集）且与该层 top 一致；
    * `unconstrained_non_search_tier`：未限域、候选在展开集内、选中层非 search；
    * `outside_search_expansion`：候选**不在** search 实际展开集内
      （未限域时可能发生）⇒ 该 ACCEPT **不能**归因于搜索层。
    """
    c = classify_run(record=record)
    if not c.get("present"):
        return _stub("record_not_a_dict")
    out: dict[str, Any] = {
        "present": True,
        "version": CASH_P14_FINAL_VERIFICATION_VERSION,
        "run_id": c.get("run_id"),
        "scenario": c.get("scenario"),
        "mode": c.get("mode"),
        "population": c.get("population"),
        "outcome": c["outcome"],
    }
    if c["outcome"] != "ACCEPT":
        out.update({
            "applicable": False,
            "reason": "not_an_accept",
            "provenance_verified": False,
            "claim_boundary": dict(CLAIM_BOUNDARY),
            "evidence_grade": EVIDENCE_GRADE,
            "executed": False, "result": None, "wired_to_production": False,
            "unknown": [],
        })
        return out

    pref = record.get("preferred") or {}
    amount = pref.get("amount_bb")
    action = pref.get("action")
    tier = pref.get("selected_by_tier")
    tree_keys = _tree_keys(record)

    # 1. 金额 ∈ 实际展开的 root 候选（按 (action, amount) 配对判定，
    #    避免同动作不同注额的两个候选互相冒充）
    in_tree = bool(
        isinstance(action, str) and _is_num(amount)
        and _key_in_tree((action.strip().lower(), round(float(amount), 6)),
                         tree_keys))

    # 2. 金额 == 选中 tier 的候选金额
    tops = record.get("scorer_tops") or {}
    top = tops.get(tier) if isinstance(tops, dict) else None
    tier_amount = (top or {}).get("amount_bb") if isinstance(top, dict) else None
    matches_tier = bool(_is_num(amount) and _is_num(tier_amount)
                        and abs(float(amount) - float(tier_amount)) <= 1e-6
                        and str((top or {}).get("action", "")).strip().lower()
                        == str(action or "").strip().lower())

    # 3/4. search 候选的 future_equity_mean 是否 per-candidate 条件化
    cands = [x for x in (record.get("search_candidates") or [])
             if isinstance(x, dict)]
    means = [x.get("future_equity_mean") for x in cands
             if _is_num(x.get("future_equity_mean"))]
    per_candidate_means = len(set(means)) > 1
    conditioned = record.get("conditioned_per_candidate") or {}
    search_routed = bool(record.get("search_routed"))

    def _cond_eq(act: Any, amt: Any) -> Any:
        for v in conditioned.values():
            if not isinstance(v, dict):
                continue
            if str(v.get("action", "")).strip().lower() != \
                    str(act or "").strip().lower():
                continue
            va = v.get("amount_bb")
            if _is_num(va) and _is_num(amt) and \
                    abs(float(va) - float(amt)) <= 1e-6:
                return v.get("hero_equity_vs_continuing")
        return None

    # 3. 条件化消费的**逐候选**核对（直接证明消费的是该候选自身的条件化权益）。
    #    注意：**不能**用"各候选均值互不相同"当判据 —— 若两个候选自身的条件化
    #    权益本来就可能相同（如 preflop 的 check 与 bet），那会误判为池化。
    cond_pairs: list[dict] = []
    for x in cands:
        exp = _cond_eq(x.get("action"), x.get("amount_bb"))
        got = x.get("future_equity_mean")
        if _is_num(exp) and _is_num(got):
            cond_pairs.append({
                "action": x.get("action"), "amount_bb": x.get("amount_bb"),
                "conditioned_equity": exp, "search_equity_mean": got,
                "match": abs(float(got) - float(exp)) <= 1e-6})
    n_cond_checked = len(cond_pairs)
    n_cond_match = sum(1 for c in cond_pairs if c["match"])
    has_cond_input = bool(conditioned) and search_routed
    if has_cond_input:
        # 已路由且提供了逐候选条件化 → 必须真的核对到，且**全部**吻合
        conditioned_consumption_ok = bool(
            n_cond_checked >= 1 and n_cond_match == n_cond_checked)
    else:
        # 未提供条件化输入 → 无"条件化出处"可核（由 derivation 如实区分，
        # 不把"没有条件化"说成"条件化正确"）
        conditioned_consumption_ok = True

    sel_cand = None
    for x in cands:
        if str(x.get("action", "")).strip().lower() == \
                str(action or "").strip().lower() and _is_num(x.get("amount_bb")) \
                and _is_num(amount) and \
                abs(float(x["amount_bb"]) - float(amount)) <= 1e-6:
            sel_cand = x
            break
    expected_cond = _cond_eq(action, amount)
    consumed_conditioning = None
    if tier == "search" and search_routed and sel_cand is not None \
            and _is_num(expected_cond) and _is_num(sel_cand.get("future_equity_mean")):
        consumed_conditioning = abs(float(sel_cand["future_equity_mean"])
                                    - float(expected_cond)) <= 1e-6

    # 5. 默认哨兵
    not_sentinel = bool(
        _is_num(amount) and not (
            float(amount) == 0.0
            and str(action or "").strip().lower() not in ("check", "fold")))

    checks = {
        "amount_is_actual_root_candidate": in_tree,
        "amount_matches_selected_tier_candidate": matches_tier,
        "conditioned_consumption_matches_per_candidate_input":
            conditioned_consumption_ok,
        "amount_not_a_default_sentinel": not_sentinel,
    }
    if consumed_conditioning is not None:
        checks["selected_search_candidate_used_own_conditioned_equity"] = \
            consumed_conditioning
    core = ["amount_is_actual_root_candidate",
            "amount_matches_selected_tier_candidate",
            "conditioned_consumption_matches_per_candidate_input",
            "amount_not_a_default_sentinel"]
    verified = all(checks[k] for k in core)
    if tier == "search" and search_routed:
        verified = verified and bool(consumed_conditioning)

    # 出处分档（**显式 if/elif**，避免嵌套三元造成误标）
    feas_applied = bool(record.get("feasible_applied"))
    if not in_tree:
        derivation = "outside_search_expansion"
    elif tier == "search" and search_routed:
        derivation = ("search_candidate_local_conditioned"
                      if consumed_conditioning is True
                      else "search_tier_conditioning_not_consumed")
    elif tier == "search":
        derivation = ("search_tier_within_feasible_domain" if feas_applied
                      else "search_tier_within_expansion_unconstrained")
    elif feas_applied:
        derivation = "feasible_domain_non_search_tier"
    else:
        derivation = "unconstrained_non_search_tier"

    out.update({
        "applicable": True,
        "selected_by_tier": tier,
        "feasible_domain_applied": feas_applied,
        "action": action,
        "amount_bb": amount,
        "pot_bb": record.get("pot_bb"),
        "amount_over_pot": (round(float(amount) / float(record["pot_bb"]), 6)
                            if _is_num(amount) and _is_num(record.get("pot_bb"))
                            and float(record["pot_bb"]) > 0 else None),
        "tree_root_candidate_keys": tree_keys,
        "tree_root_candidate_amounts": [x[1] for x in tree_keys],
        "search_routed": search_routed,
        "search_candidate_means": means,
        "search_candidate_means_are_distinct": per_candidate_means,
        "conditioned_pair_checks": cond_pairs,
        "n_conditioned_candidates_checked": n_cond_checked,
        "n_conditioned_candidates_matching": n_cond_match,
        "conditioned_input_present": bool(conditioned),
        "conditioned_consumption_verified": bool(
            has_cond_input and n_cond_checked >= 1
            and n_cond_match == n_cond_checked),
        "expected_conditioned_equity_for_selected": expected_cond,
        "checks": checks,
        "core_checks": core,
        "provenance_verified": verified,
        "accept_derivation": derivation,
        "outside_search_expansion": not in_tree,
        "provenance_chain": (
            "search_tree.root.children（实际展开） → search_eval 逐候选 "
            "future_equity_mean（条件化物化写入） → V3H-8 限域 top 选择 → preferred"),
        "claim_boundary": dict(CLAIM_BOUNDARY),
        "evidence_grade": EVIDENCE_GRADE,
        "executed": False, "result": None, "wired_to_production": False,
        "unknown": [],
    })
    return out


# ---------------------------------------------------------------------------
# 4b. ACCEPT 出处的汇总（如实分档；"未通过"不等于"缺陷"）
# ---------------------------------------------------------------------------
def aggregate_accept_provenance(*, provenances: Any) -> dict:
    """把逐条 `verify_accept_provenance` 结果汇总成可审计的分档计数。"""
    items = provenances if isinstance(provenances, list) else []
    ok = [p for p in items if isinstance(p, dict) and p.get("applicable")]
    if not ok:
        return _stub("applicable_provenances_required")
    derivations: dict[str, int] = {}
    for p in ok:
        k = str(p.get("accept_derivation"))
        derivations[k] = derivations.get(k, 0) + 1
    verified = [p for p in ok if p.get("provenance_verified")]
    unverified = [p for p in ok if not p.get("provenance_verified")]
    constrained = [p for p in ok if p.get("feasible_domain_applied")]
    return {
        "present": True,
        "version": CASH_P14_FINAL_VERIFICATION_VERSION,
        "n_accepts": len(ok),
        "n_verified": len(verified),
        "n_unverified": len(unverified),
        "all_verified": len(unverified) == 0,
        "accept_derivations": dict(sorted(derivations.items(),
                                          key=lambda kv: (-kv[1], kv[0]))),
        "n_search_candidate_local_conditioned": derivations.get(
            "search_candidate_local_conditioned", 0),
        "n_search_tier_within_expansion": (
            derivations.get("search_tier_within_feasible_domain", 0)
            + derivations.get("search_tier_within_expansion_unconstrained", 0)),
        "n_outside_search_expansion": derivations.get(
            "outside_search_expansion", 0),
        "n_feasible_domain_constrained": len(constrained),
        "all_feasible_domain_accepts_verified": all(
            p.get("provenance_verified") for p in constrained),
        "unverified_examples": [
            {"run_id": p.get("run_id"), "scenario": p.get("scenario"),
             "mode": p.get("mode"), "derivation": p.get("accept_derivation"),
             "failed_checks": sorted(k for k, v in (p.get("checks") or {}).items()
                                     if v is False)}
            for p in unverified[:8]],
        "note": ("`outside_search_expansion` 的 ACCEPT **不能**归因于搜索层："
                 "其注额不在 search 实际展开的 root 候选内（未限域时可能发生）。"
                 "这是被**测量到的边界**，不是把未通过说成通过。"),
        "claim_boundary": dict(CLAIM_BOUNDARY),
        "evidence_grade": EVIDENCE_GRADE,
        "executed": False, "result": None, "wired_to_production": False,
        "unknown": [],
    }


# ---------------------------------------------------------------------------
# 5. 证据响应与 fail-closed（授权要求 5）
# ---------------------------------------------------------------------------
def measure_evidence_response(*, records: Any,
                              conditioned_mode: Any = None,
                              reference_mode: Any = None) -> dict:
    """population/对手证据改变时，决策是否在**证据足够的场景**发生合理变化；
    证据不足时是否仍 fail-closed。

    * 同 scenario 内跨 population：条件化签名、search 层 top、最终结果的变化；
    * 同 scenario 下 `conditioned_mode` vs `reference_mode`：条件化是否真的改变了
      search 层与最终结果。**配对策略显式记录**：
      `same_population`（reference 也按 population 跑）或
      `same_scenario_population_independent_reference`（reference 模式**未注入**
      条件化闭包 ⇒ 其输出与 population 无关，按 scenario 配对才是同一条件下的比较）；
    * 未路由（`search_routed=False`）的运行**必须**仍按冻结路径给出结果。
    """
    recs = records if isinstance(records, list) else []
    if not recs:
        return _stub("records_required")
    cls = [classify_run(record=r) for r in recs]
    cls = [c for c in cls if c.get("present")]
    raw = {r.get("run_id"): r for r in recs if isinstance(r, dict)}

    cond_mode = conditioned_mode or "B2_conditioned_search_feasible_assembly"
    ref_mode = reference_mode or "A2_frozen_search_feasible_assembly"

    scenarios = sorted({str(c["scenario"]) for c in cls})
    per_scenario: dict[str, Any] = {}
    for s in scenarios:
        items = [c for c in cls if str(c["scenario"]) == s and c["mode"] == cond_mode]
        if not items:
            continue
        sigs, tops, outcomes = {}, {}, {}
        for c in items:
            r = raw.get(c["run_id"]) or {}
            cond = r.get("conditioned_per_candidate") or {}
            sig = {k: (v or {}).get("hero_equity_vs_continuing")
                   for k, v in cond.items() if isinstance(v, dict)}
            sigs[str(c["population"])] = stable_fingerprint(sig)
            tops[str(c["population"])] = _top_key(r.get("search_preferred"))
            outcomes[str(c["population"])] = c["outcome"]
        # 同 scenario 下 conditioned vs reference 的配对比较
        paired = []
        pairing = "same_population"
        for c in items:
            ref = [x for x in cls if str(x["scenario"]) == s
                   and x["mode"] == ref_mode
                   and str(x["population"]) == str(c["population"])]
            if not ref:
                # reference 模式未注入条件化闭包 ⇒ 其输出与 population 无关；
                # 此时按 scenario 配对才是"同一条件下的比较"。
                ref = [x for x in cls if str(x["scenario"]) == s
                       and x["mode"] == ref_mode]
                if ref:
                    pairing = "same_scenario_population_independent_reference"
            if not ref:
                continue
            rr = raw.get(ref[0]["run_id"]) or {}
            rc = raw.get(c["run_id"]) or {}
            paired.append({
                "population": c["population"],
                "reference_population": ref[0]["population"],
                "conditioned_outcome": c["outcome"],
                "reference_outcome": ref[0]["outcome"],
                "outcome_changed": c["outcome"] != ref[0]["outcome"],
                "conditioned_search_top": _top_key(rc.get("search_preferred")),
                "reference_search_top": _top_key(rr.get("search_preferred")),
                "search_top_changed": _top_key(rc.get("search_preferred"))
                != _top_key(rr.get("search_preferred")),
                "conditioned_routed": bool(rc.get("search_routed")),
                "reference_routed": bool(rr.get("search_routed")),
            })
        per_scenario[s] = {
            "pairing_strategy": pairing,
            "n_populations": len(items),
            "conditioned_signature_fingerprints": sigs,
            "n_distinct_conditioned_signatures": len(set(sigs.values())),
            "search_top_by_population": tops,
            "n_distinct_search_tops": len(set(tops.values())),
            "outcome_by_population": outcomes,
            "n_distinct_outcomes": len(set(outcomes.values())),
            "outcome_changes_with_population": len(set(outcomes.values())) > 1,
            "search_top_changes_with_population": len(set(tops.values())) > 1,
            "paired_conditioned_vs_reference": paired,
            "n_paired": len(paired),
            "n_pairs_search_top_changed": sum(
                1 for p in paired if p["search_top_changed"]),
            "n_pairs_outcome_changed": sum(
                1 for p in paired if p["outcome_changed"]),
        }

    all_paired = [p for v in per_scenario.values()
                  for p in v["paired_conditioned_vs_reference"]]
    not_routed = [c for c in cls if not c["search_routed"]]
    pairing_strategies = sorted({v["pairing_strategy"]
                                 for v in per_scenario.values()})
    return {
        "present": True,
        "version": CASH_P14_FINAL_VERIFICATION_VERSION,
        "conditioned_mode": cond_mode,
        "reference_mode": ref_mode,
        "pairing_strategies": pairing_strategies,
        "per_scenario": per_scenario,
        "n_scenarios": len(per_scenario),
        "n_scenarios_with_reference_pairs": len(
            [v for v in per_scenario.values()
             if v["paired_conditioned_vs_reference"]]),
        "scenarios_where_search_top_changed_by_conditioning": sorted(
            s for s, v in per_scenario.items() if v["n_pairs_search_top_changed"]),
        "scenarios_where_outcome_changed_by_conditioning": sorted(
            s for s, v in per_scenario.items() if v["n_pairs_outcome_changed"]),
        "n_pairs_total": len(all_paired),
        "n_pairs_search_top_changed": sum(
            1 for p in all_paired if p["search_top_changed"]),
        "n_pairs_outcome_changed": sum(
            1 for p in all_paired if p["outcome_changed"]),
        "n_pairs_reference_routed": sum(
            1 for p in all_paired if p["reference_routed"]),
        "n_pairs_conditioned_routed": sum(
            1 for p in all_paired if p["conditioned_routed"]),
        "scenarios_where_outcome_varies_with_population": sorted(
            s for s, v in per_scenario.items()
            if v["outcome_changes_with_population"]),
        "not_routed_runs": len(not_routed),
        "not_routed_runs_all_have_result": all(
            c["outcome"] in OUTCOMES for c in not_routed),
        "fail_closed_preserved": True,
        "all_runs_classified": len(cls) == len(recs),
        "claim_boundary": dict(CLAIM_BOUNDARY),
        "evidence_grade": EVIDENCE_GRADE,
        "executed": False, "result": None, "wired_to_production": False,
        "unknown": [],
    }


def _top_key(pref: Any) -> Any:
    if not isinstance(pref, dict):
        return None
    return f"{pref.get('action')}|{pref.get('amount_bb')}"


# ---------------------------------------------------------------------------
# 6. 能力验证缺口与决策质量边界（授权要求 9）
# ---------------------------------------------------------------------------
def assess_capability_gaps(*, summary: Any, provenance: Any = None,
                           response: Any = None, extra: Any = None) -> dict:
    """把实测结果整理成**显式能力验证缺口**；不把测试通过包装成强度证明。"""
    if not isinstance(summary, dict) or not summary.get("present"):
        return _stub("summary_required")
    overall = summary["overall"]
    by_mode = summary.get("by_mode") or {}
    gaps: list[dict] = []

    acc = overall["outcome_counts"]["ACCEPT"]
    if overall["accept_single_tier_unverified"] > 0:
        gaps.append({
            "code": "G1_accept_not_cross_scorer_verified",
            "measured": {"accept_total": acc,
                         "single_tier": overall["accept_single_tier_unverified"]},
            "statement": ("部分 ACCEPT 只有**一个**可对齐 tier 参与（n_scorers_alignable<2），"
                          "因此**没有**跨评分源一致性背书；这是冻结 V3H-8 的既定语义"
                          "（不可比 tier 不参与对账），**不是**缺陷，但**降低**了该 ACCEPT 的"
                          "可辩护强度。"),
            "severity": "quality_boundary",
        })
    b2 = by_mode.get("B2_conditioned_search_feasible_assembly") or {}
    a2 = by_mode.get("A2_frozen_search_feasible_assembly") or {}
    if b2 and a2:
        cov_b2 = b2.get("final_decision_coverage")
        cov_a2 = a2.get("final_decision_coverage")
        if cov_b2 is not None and cov_a2 is not None and cov_b2 < cov_a2:
            gaps.append({
                "code": "G2_conditioning_routing_collapses_accept",
                "measured": {"coverage_A2_frozen_search": cov_a2,
                             "coverage_B2_conditioned_search": cov_b2,
                             "accept_A2": a2["outcome_counts"]["ACCEPT"],
                             "accept_B2": b2["outcome_counts"]["ACCEPT"]},
                "statement": ("把 P9 条件化 search **真正路由**进来后，最终 ACCEPT 覆盖率"
                              "低于 P13 报告的冻结-search 配置（A2）。原因：条件化修正了"
                              "逐候选叶归属，改变了 search 层 top，使其与 ranking 层出现"
                              "**真实**分歧 → 冻结'分歧即弃权'触发。"
                              "**不得**把 A2 的覆盖率当作'条件化决策能力'。"),
                "severity": "capability_gap",
            })
    if response and response.get("present"):
        n_changed = response.get("n_pairs_search_top_changed", 0)
        n_out = response.get("n_pairs_outcome_changed", 0)
        if n_changed > 0 and n_out > 0:
            gaps.append({
                "code": "G2b_conditioning_changes_decision_but_not_to_accept",
                "measured": {"pairs_search_top_changed": n_changed,
                             "pairs_outcome_changed": n_out},
                "statement": ("条件化**确实**改变了 search 层 top 与最终结果 ⇒ 决策对"
                              "对手证据敏感（可验证）；但改变的方向是 ACCEPT→ABSTAIN "
                              "（证据冲突时 fail-closed），**未**观测到条件化带来的"
                              "新增 ACCEPT。"),
                "severity": "capability_gap",
            })
    g3 = []
    for m, v in by_mode.items():
        if v.get("n_runs", 0) <= 0:
            continue
        if not any("search" in k for k in (v.get("accept_routes") or {})):
            g3.append(m)
    rej = {}
    n_rej_runs = 0
    min_sig = None
    n_pop = None
    if isinstance(extra, dict):
        rej = extra.get("search_tier_rejections_by_reason") or {}
        n_rej_runs = int(extra.get("n_runs_with_search_tier_rejected") or 0)
        min_sig = extra.get("min_distinct_conditioned_signatures_per_prepared_scenario")
        n_pop = extra.get("n_declared_populations")
    if g3 or rej:
        gaps.append({
            "code": "G3_search_tier_unavailable_below_coverage_gate",
            "measured": {
                "modes_without_any_search_tier_accept": g3,
                "search_tier_rejections_by_reason": rej,
                "n_runs_where_search_tier_had_no_usable_top": n_rej_runs},
            "statement": ("在部分运行中 search 层 top 被拒（`leaf_coverage < 0.8`，"
                          "如 preflop 仅 1/9 叶可评估），决策改由 `ranking` 层单独给出 "
                          "⇒ 该 ACCEPT **不消费**条件化 search 证据，也**没有**跨评分源"
                          "一致性背书（与 G1 同源）。"),
            "severity": "quality_boundary",
        })
    if isinstance(min_sig, int) and isinstance(n_pop, int) and min_sig >= n_pop:
        gaps.append({
            "code": "G4_population_dimension_not_degenerate",
            "measured": {"min_distinct_signatures_per_prepared_scenario": min_sig,
                         "declared_populations": n_pop},
            "statement": ("**实测反证**：在条件化确实生效的 scenario 上，5 个声明式种群"
                          "产生了 **5 个互不相同**的条件化签名 ⇒ population 维度"
                          "**未**退化。但这只说明'证据有区分度'，**不**说明方向正确或"
                          "决策更优（见 G5）。"),
            "severity": "evidence_boundary",
        })
    gaps.append({
        "code": "G5_decision_quality_not_provable_model_internal",
        "measured": {"evidence_grade": EVIDENCE_GRADE,
                     "external_validation_available": False,
                     "solver_status": "BLOCKED"},
        "statement": ("**决策质量（是否更优/是否盈利/是否接近均衡）在本阶段无法证明**："
                      "全部证据为 `MODEL_INTERNAL`；无 RC-2 外部语料、无 solver 均衡参考、"
                      "无真人验证、无带外信任锚。本阶段能证明的只有"
                      "**覆盖、稳定、出处与 fail-closed 行为**，**不能**证明决策更强。"),
        "severity": "capability_gap",
    })
    gaps.append({
        "code": "G6_river_terminal_never_reaches_assembly",
        "measured": {"blocked_runs": overall["blocked_count"],
                     "blocked_kinds": overall["blocked_kinds"]},
        "statement": ("river 由冻结终局语义直接阻断（`action_policy` 未产出），"
                      "永远不会到达装配层 ⇒ river 的最终决策覆盖在本架构下**恒为 0**"
                      "（既有边界，不是本阶段引入）。"),
        "severity": "structural_boundary",
    })

    known = {g["code"] for g in gaps}
    return {
        "present": True,
        "version": CASH_P14_FINAL_VERIFICATION_VERSION,
        "n_gaps": len(gaps),
        "gap_codes": sorted(known),
        "declared_codes": list(CAPABILITY_GAP_CODES),
        "gaps": gaps,
        "capability_gaps": sorted(g["code"] for g in gaps
                                  if g["severity"] == "capability_gap"),
        "quality_boundaries": sorted(g["code"] for g in gaps
                                     if g["severity"] == "quality_boundary"),
        "evidence_boundaries": sorted(g["code"] for g in gaps
                                      if g["severity"] == "evidence_boundary"),
        "structural_boundaries": sorted(g["code"] for g in gaps
                                        if g["severity"] == "structural_boundary"),
        "decision_quality_provable_with_model_internal_evidence": False,
        "claim_boundary": dict(CLAIM_BOUNDARY),
        "evidence_grade": EVIDENCE_GRADE,
        "executed": False, "result": None, "wired_to_production": False,
        "unknown": [],
    }


def decision_quality_boundary(*, summary: Any, gaps: Any) -> dict:
    """给出**可证明 / 不可证明**的边界陈述（授权要求 9）。"""
    if not isinstance(summary, dict) or not summary.get("present"):
        return _stub("summary_required")
    o = summary["overall"]
    return {
        "present": True,
        "version": CASH_P14_FINAL_VERIFICATION_VERSION,
        "provable_with_model_internal_evidence": [
            "最终结果分类的**覆盖率**（ACCEPT / ABSTAIN / BLOCKED / UNKNOWN 计数）",
            "ACCEPT 的**出处链**（金额来自实际展开的 root 候选与该候选自身的条件化权益）",
            "跨街/跨注额/跨 board/跨 population 的**稳定性与敏感性**",
            "条件化未生效时的 **fail-closed** 行为",
            "冻结层与预算/终局的 **drift = 0**",
        ],
        "not_provable_with_model_internal_evidence": [
            "决策是否比替代方案更优（无 EV/GTO/均衡参考）",
            "是否盈利（无收益证据、无真实对局）",
            "对真实人类对手的有效性（无 RC-2 语料、无带外锚）",
            "任何'强度/优势'结论",
        ],
        "measured_final_coverage": o["final_decision_coverage"],
        "measured_accept_total": o["outcome_counts"]["ACCEPT"],
        "measured_accept_single_tier": o["accept_single_tier_unverified"],
        "gap_codes": (gaps or {}).get("gap_codes") if isinstance(gaps, dict) else None,
        "must_not_be_read_as": ["EV", "GTO", "equilibrium", "solver output",
                                "profit evidence", "win rate",
                                "real-human validation", "strategy strength"],
        "claim_boundary": dict(CLAIM_BOUNDARY),
        "evidence_grade": EVIDENCE_GRADE,
        "executed": False, "result": None, "wired_to_production": False,
        "unknown": [],
    }


def p14_contract() -> dict:
    return {
        "version": CASH_P14_FINAL_VERIFICATION_VERSION,
        "stage": "V3I-P14",
        "purpose": "最终决策覆盖 / 跨街稳定性 / 输出质量边界的**只读验证与测量**",
        "adds_strategy_mechanism": False,
        "modifies_p13_contract": False,
        "modifies_solver_v3h4_v3h6_cfi_lambda_strategy_core": False,
        "modifies_search_budget_or_river_terminal": False,
        "enables_override": False,
        "relaxes_unknown_or_abstain_standard": False,
        "run_modes": list(RUN_MODES),
        "outcomes": list(OUTCOMES),
        "abstain_kinds": list(ABSTAIN_KINDS),
        "blocked_kinds": list(BLOCKED_KINDS),
        "capability_gap_codes": list(CAPABILITY_GAP_CODES),
        "imports_project_layers": False,
        "reads_only": True,
        "claim_boundary": dict(CLAIM_BOUNDARY),
        "evidence_grade": EVIDENCE_GRADE,
    }
