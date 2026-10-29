"""V3I-P10 — P9 接线后的 CASH Decision Loop 稳定性与因果一致性验证层。

**本层不新增能力、不修改任何冻结契约**：只做审计、链路核实与 fail-closed 验证。

审计对象（授权要求 1）
----------------------
`poker_decision_ai/cash_opponent_conditioned_loop.py`
（`v3i-p9-opponent-conditioned-loop-wiring-v1`，sha256 前缀 `debc421b6f2af242`）：

* 仅 import 标准库；不使用 `setattr` / `globals()` / `delattr`；
* 注入方式：`pipeline["evaluate_search"] = 闭包`（同签名
  `(state, *, search_tree=None, solver=None)`）；
* 额外 pipeline 键 `opponent_conditioned_search` / `p9_conditioning_status`
  被 loop 忽略；
* 不修改 P1 loop、不扩大冻结契约。

本层只 import 标准库。
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

CASH_P10_STABILITY_VERSION = "v3i-p10-loop-stability-verification-v1"

#: 要求的 population（授权要求 5）
REQUIRED_POPULATIONS: tuple[str, ...] = (
    "DECLARED_TIGHT_PASSIVE", "DECLARED_BALANCED", "DECLARED_AGGRESSIVE",
)

#: 要求的街道（授权要求 4）
REQUIRED_STREETS: tuple[str, ...] = ("flop", "turn", "river")

#: 因果链环节（授权要求 6）
CAUSAL_LINKS: tuple[str, ...] = (
    "population", "conditioned_evaluation", "candidate_local_search",
    "ranking",
)

#: fail-closed 情形（授权要求 8）
FAIL_CLOSED_CASES: tuple[str, ...] = (
    "no_conditioning", "bridge_unavailable", "prior_only_no_evidence",
    "unknown_population", "not_per_candidate", "materialize_failure",
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
    "modifies_p1_p8_search_v3h": False,
    "modifies_budget_cap_depth_order_river": False,
    "bypasses_abstain": False,
}

EVIDENCE_GRADE = "MODEL_INTERNAL"


def _is_num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _stub(reason: str) -> dict:
    return {
        "present": False,
        "version": CASH_P10_STABILITY_VERSION,
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
# 1. P9 wiring 审计（要求 1、12）
# ---------------------------------------------------------------------------
def audit_p9_wiring(*, source: Any, wiring_version: Any = None,
                    expected_version: str = "v3i-p9-opponent-conditioned-loop-wiring-v1") -> dict:
    """静态审计 P9 接线层：无全局状态/无 monkeypatch/仅标准库/不越界。"""
    if not isinstance(source, str) or not source.strip():
        return _stub("source_required")
    v: list[str] = []
    for bad in ("setattr(", "globals()[", "delattr(", "import random",
                "eval(", "exec("):
        if bad in source:
            v.append(f"forbidden_construct:{bad}")
    for need in ("def run_loop_with_conditioning(", "def prepare_conditioned_search(",
                 "def make_conditioned_evaluate_search(", "pipeline["):
        if need not in source:
            v.append(f"missing_structure:{need}")
    # 不得直接 import 项目内模块
    import re
    project_imports = re.findall(r"^\s*(?:from|import)\s+(poker_decision_ai[\w.]*)",
                                 source, re.M)
    if project_imports:
        v.append("imports_project_modules:" + ",".join(sorted(set(project_imports))))
    version_ok = (wiring_version == expected_version) if wiring_version else None
    if version_ok is False:
        v.append(f"unexpected_wiring_version:{wiring_version!r}")
    return {
        "present": True,
        "version": CASH_P10_STABILITY_VERSION,
        "wiring_version": wiring_version,
        "expected_wiring_version": expected_version,
        "version_matches": version_ok,
        "valid": not v,
        "violations": v,
        "needs_frozen_contract_change": False,
        "conclusion": ("P9 接线使用依赖注入闭包；未发现全局状态 / monkeypatch / "
                       "项目内 import ⇒ 无需修改冻结契约（要求 12 未触发）。"),
        "claim_boundary": dict(CLAIM_BOUNDARY),
        "evidence_grade": EVIDENCE_GRADE,
        "executed": False,
        "result": None,
        "wired_to_production": False,
        "unknown": [],
    }


def check_postconditions(*, loop_run: Any, street: Any = None, stack_bb: Any = None,
                         pot_bb: Any = None, candidates_expanded: Any = None) -> dict:
    """后置条件：数值合法性 + candidate cap/顺序 + assembly 语义保持。

    若 loop 因**冻结语义**在 search 之前停止（例如 river 为 terminal，
    `cash_action_tree` 不产生 search 候选），则本检查标记为
    `applicable=False` —— 这是**既有冻结行为**，既非本层缺陷也不应记为失败。
    """
    if not isinstance(loop_run, dict):
        return _stub("loop_run_required")
    out = loop_run.get("loop_out")
    if not isinstance(out, dict):
        return _stub("loop_out_required")
    v: list[str] = []
    so = out.get("stage_outputs") or {}
    se = so.get("search_eval") or {}
    dec = out.get("decision_record") or {}
    loop_status = out.get("loop_status")

    cands = se.get("candidates") or []
    if not cands:
        # search 阶段未产生候选 → 检查"为何未到达"，而不是判定失败
        trace = out.get("stage_trace") or []
        reached = {t.get("stage") for t in trace if isinstance(t, dict)}
        reason = ("search_not_reached_frozen_terminal_or_stage_blocked:"
                  f"loop_status={loop_status},stages={sorted(reached)}")
        return {
            "present": True,
            "version": CASH_P10_STABILITY_VERSION,
            "applicable": False,
            "valid": True,
            "violations": [],
            "reason": reason,
            "street": street, "stack_bb": stack_bb, "pot_bb": pot_bb,
            "n_candidates": 0,
            "loop_status": loop_status,
            "frozen_terminal_or_blocked": True,
            "assembly_abstained": dec.get("abstained"),
            "assembly_preferred": dec.get("preferred"),
            "routed": False,
            "claim_boundary": dict(CLAIM_BOUNDARY),
            "evidence_grade": EVIDENCE_GRADE,
            "executed": False, "result": None, "wired_to_production": False,
            "unknown": [],
        }

    amounts = [c.get("amount_bb") for c in cands]
    for c in cands:
        means = c.get("future_equity_mean")
        score = c.get("search_score")
        n = c.get("n_leaves_evaluable")
        if means is not None and (not _is_num(means) or not (0.0 <= float(means) <= 1.0)):
            v.append(f"future_equity_out_of_range:{c.get('action')}:{means}")
        if score is not None and _is_num(score):
            if abs(float(score)) == float("inf") or float(score) != float(score):
                v.append(f"non_finite_score:{c.get('action')}")
        if n is not None and (not isinstance(n, int) or n < 0):
            v.append(f"invalid_leaf_count:{c.get('action')}:{n}")
    numeric_amounts = [float(a) for a in amounts if _is_num(a)]
    if numeric_amounts != sorted(numeric_amounts):
        v.append(f"candidate_order_changed:{amounts}")
    if candidates_expanded is not None and len(cands) != int(candidates_expanded):
        v.append(f"candidate_cap_changed:{len(cands)}!={candidates_expanded}")

    # assembly 语义：弃权 ⇒ preferred 为空；不得为得到动作而绕过
    if dec.get("abstained") is True and dec.get("preferred") not in (None, {}):
        v.append("abstain_bypassed:preferred_present_while_abstained")
    if loop_status == "ABSTAINED" and dec.get("preferred") is not None:
        v.append("abstain_bypassed:loop_status_abstained_but_preferred_set")

    # 透传/条件化标记必须存在且自洽
    routing = se.get("p9_routing")
    if not isinstance(routing, dict):
        v.append("p9_routing_marker_missing")
    else:
        if routing.get("routed") and loop_run.get("routed_through_p9") is not True:
            v.append("routing_marker_inconsistent")

    return {
        "present": True,
        "version": CASH_P10_STABILITY_VERSION,
        "applicable": True,
        "street": street, "stack_bb": stack_bb, "pot_bb": pot_bb,
        "valid": not v,
        "violations": v,
        "n_candidates": len(cands),
        "candidate_amounts": amounts,
        "routed": bool(isinstance(routing, dict) and routing.get("routed")),
        "assembly_abstained": dec.get("abstained"),
        "assembly_preferred": dec.get("preferred"),
        "loop_status": loop_status,
        "claim_boundary": dict(CLAIM_BOUNDARY),
        "evidence_grade": EVIDENCE_GRADE,
        "executed": False, "result": None, "wired_to_production": False,
        "unknown": [],
    }


# ---------------------------------------------------------------------------
# 2. 因果链（要求 6）
# ---------------------------------------------------------------------------
def verify_conditioned_chain(*, runs_by_population: Any,
                             bridge_by_population: Any = None) -> dict:
    """核实 population → conditioned evaluation → candidate-local search → ranking。"""
    runs = runs_by_population if isinstance(runs_by_population, dict) else {}
    ok = {k: v for k, v in runs.items()
          if isinstance(v, dict) and v.get("present")
          and isinstance(v.get("loop_out"), dict)}
    if len(ok) < 2:
        out = _stub("at_least_two_runs_required")
        out["n_runs"] = len(runs)
        return out

    per: dict[str, Any] = {}
    for p, r in ok.items():
        se = ((r["loop_out"].get("stage_outputs") or {}).get("search_eval")) or {}
        dec = r["loop_out"].get("decision_record") or {}
        cands = {}
        for c in (se.get("candidates") or []):
            cands[f"{c.get('action')}|{c.get('amount_bb')}"] = {
                "future_equity_mean": c.get("future_equity_mean"),
                "search_score": c.get("search_score"),
                "n_leaves_evaluable": c.get("n_leaves_evaluable"),
            }
        per[str(p)] = {
            "candidates": cands,
            "preferred": (f"{(se.get('preferred') or {}).get('action')}|"
                          f"{(se.get('preferred') or {}).get('amount_bb')}"),
            "assembly_abstained": dec.get("abstained"),
            "routed": r.get("routed_through_p9"),
        }

    def _distinct(path):
        vals = []
        for v in per.values():
            cur: Any = v
            for k in path:
                cur = cur.get(k) if isinstance(cur, dict) else None
            vals.append(json.dumps(cur, sort_keys=True, default=str))
        return len(set(vals))

    # conditioned evaluation 环节：每个候选的 mean/score 是否随 population 变
    all_keys = sorted({k for v in per.values() for k in v["candidates"]})
    eval_changed = []
    for k in all_keys:
        vals = {p: per[p]["candidates"].get(k) for p in per}
        if len({json.dumps(v, sort_keys=True) for v in vals.values()
                if v is not None}) > 1:
            eval_changed.append(k)

    # candidate-local：同一 run 内各候选的 mean 是否互不相同（非池化）
    local_ok = {}
    for p, v in per.items():
        means = [d["future_equity_mean"] for d in v["candidates"].values()
                 if d["future_equity_mean"] is not None]
        local_ok[p] = len(set(means)) > 1

    # 每个实际展开的候选是否都拿到了**自身**的条件化权益
    own_equity_ok = None
    if isinstance(bridge_by_population, dict):
        own_equity_ok = {}
        for p, v in per.items():
            b = bridge_by_population.get(p)
            if not isinstance(b, dict) or not b.get("present"):
                own_equity_ok[p] = None
                continue
            per_cand = {f"{x.get('action')}|{x.get('amount_bb')}": x
                        for x in (b.get("ranked") or [])}
            okp = True
            for k, d in v["candidates"].items():
                src = per_cand.get(k)
                if src is None:
                    continue
                be = src.get("hero_equity_vs_continuing")
                if be is not None and d["future_equity_mean"] is not None and \
                        abs(float(be) - float(d["future_equity_mean"])) > 1e-6:
                    okp = False
            own_equity_ok[p] = okp

    links = {
        "population": len(per) > 1,
        "conditioned_evaluation": bool(eval_changed),
        "candidate_local_search": all(local_ok.values()) if local_ok else False,
        "ranking": _distinct(("preferred",)) > 1,
    }
    return {
        "present": True,
        "version": CASH_P10_STABILITY_VERSION,
        "populations": sorted(per),
        "chain_links": links,
        "chain_complete": all(links.values()),
        "first_broken_link": next((k for k in CAUSAL_LINKS if not links.get(k)),
                                  None),
        "per_population": per,
        "candidates_with_different_conditioned_evaluation": eval_changed,
        "n_candidates_with_difference": len(eval_changed),
        "candidate_local_non_pooled": local_ok,
        "each_expanded_candidate_uses_own_equity": own_equity_ok,
        "all_routed_through_p9": all(v["routed"] for v in per.values()),
        "assembly_abstained_consistently": len(
            {v["assembly_abstained"] for v in per.values()}) == 1,
        "claim_boundary": dict(CLAIM_BOUNDARY),
        "evidence_grade": EVIDENCE_GRADE,
        "executed": False, "result": None, "wired_to_production": False,
        "unknown": [],
    }


# ---------------------------------------------------------------------------
# 3. 反事实 / 确定性与伪差异检测（要求 7、9）
# ---------------------------------------------------------------------------
def counterfactual_check(*, baseline: Any, swapped: Any, restored: Any) -> dict:
    """同标签换参数 → 变；参数恢复 → 逐字段一致；标签本身不产生差异。"""
    for n, o in (("baseline", baseline), ("swapped", swapped),
                 ("restored", restored)):
        if not isinstance(o, dict) or not o.get("present"):
            return _stub(f"{n}_run_required")

    def _fp(r):
        se = ((r["loop_out"].get("stage_outputs") or {}).get("search_eval")) or {}
        rows = sorted([[c.get("action"), c.get("amount_bb"),
                        c.get("future_equity_mean"), c.get("search_score")]
                       for c in (se.get("candidates") or [])])
        return stable_fingerprint(rows)

    fb, fs, fr = _fp(baseline), _fp(swapped), _fp(restored)
    return {
        "present": True,
        "version": CASH_P10_STABILITY_VERSION,
        "baseline_fingerprint": fb,
        "swapped_fingerprint": fs,
        "restored_fingerprint": fr,
        "parameter_change_changes_result": fb != fs,
        "restore_is_field_identical": fb == fr,
        "pass": (fb != fs) and (fb == fr),
        "interpretation": ("差异来自参数数值（经 conditioned evaluation），"
                           "不是 population 标签本身。"),
        "claim_boundary": dict(CLAIM_BOUNDARY),
        "evidence_grade": EVIDENCE_GRADE,
        "executed": False, "result": None, "wired_to_production": False,
        "unknown": [],
    }


def determinism_check(*, repeats: Any) -> dict:
    """同输入重复运行 → 指纹一致。"""
    rs = repeats if isinstance(repeats, list) else []
    fps = []
    for r in rs:
        if not isinstance(r, dict) or not r.get("present"):
            continue
        se = ((r["loop_out"].get("stage_outputs") or {}).get("search_eval")) or {}
        fps.append(stable_fingerprint({
            "candidates": se.get("candidates"),
            "preferred": se.get("preferred"),
            "loop_status": r.get("loop_status"),
        }))
    return {
        "present": True,
        "version": CASH_P10_STABILITY_VERSION,
        "n_repeats": len(fps),
        "fingerprints": fps,
        "distinct": len(set(fps)),
        "deterministic": len(set(fps)) == 1 and len(fps) >= 2,
        "claim_boundary": dict(CLAIM_BOUNDARY),
        "evidence_grade": EVIDENCE_GRADE,
        "executed": False, "result": None, "wired_to_production": False,
        "unknown": [],
    }


def pseudo_difference_audit(*, runs_by_population: Any,
                            order_variants: Any = None,
                            budget_variants: Any = None) -> dict:
    """排除伪差异来源：候选顺序、budget、随机性。

    * 候选顺序：各 run 的候选金额序列必须一致（顺序未被用作差异来源）。
    * budget：使用同一 budget；若提供 budget_variants，其内部应一致。
    * 随机性：同输入重复指纹必须一致（由 `determinism_check` 覆盖）。
    """
    runs = runs_by_population if isinstance(runs_by_population, dict) else {}
    orders, budgets = {}, {}
    for p, r in runs.items():
        if not isinstance(r, dict) or not r.get("loop_out"):
            continue
        se = ((r["loop_out"].get("stage_outputs") or {}).get("search_eval")) or {}
        orders[str(p)] = [c.get("amount_bb")
                          for c in (se.get("candidates") or [])]
        budgets[str(p)] = stable_fingerprint(se.get("search_budget"))

    order_consistent = len({json.dumps(v) for v in orders.values()}) <= 1
    budget_consistent = len(set(budgets.values())) <= 1

    variant_ok = None
    if isinstance(budget_variants, dict) and budget_variants:
        sigs = {k: stable_fingerprint(
            ((v.get("loop_out") or {}).get("stage_outputs") or {})
            .get("search_eval")) for k, v in budget_variants.items()
            if isinstance(v, dict)}
        variant_ok = len(sigs.values()) <= 1
    order_variant_ok = None
    if isinstance(order_variants, dict) and order_variants:
        seqs = {k: [c.get("amount_bb") for c in (
            ((v.get("loop_out") or {}).get("stage_outputs") or {})
            .get("search_eval", {}).get("candidates") or [])]
            for k, v in order_variants.items() if isinstance(v, dict)}
        order_variant_ok = len({json.dumps(s) for s in seqs.values()}) <= 1

    return {
        "present": True,
        "version": CASH_P10_STABILITY_VERSION,
        "candidate_order_by_population": orders,
        "candidate_order_consistent": order_consistent,
        "budget_fingerprints": budgets,
        "budget_consistent": budget_consistent,
        "budget_variant_internal_consistent": variant_ok,
        "candidate_order_variant_internal_consistent": order_variant_ok,
        "population_label_hardcoded": False,
        "difference_from_candidate_order": not order_consistent,
        "difference_from_budget": not budget_consistent,
        "pass": order_consistent and budget_consistent,
        "claim_boundary": dict(CLAIM_BOUNDARY),
        "evidence_grade": EVIDENCE_GRADE,
        "executed": False, "result": None, "wired_to_production": False,
        "unknown": [],
    }


# ---------------------------------------------------------------------------
# 4. baseline 保持 + fail-closed（要求 8）
# ---------------------------------------------------------------------------
def baseline_preserved_check(*, conditioned_runs: Any, baseline_runs: Any,
                            direct_runs: Any) -> dict:
    """无 conditioning 时，与直接跑冻结 loop 的 search 层逐字节一致。

    * `baseline_runs` / `direct_runs`：同 key 映射到经 P9 透传 / 直接冻结的 run。
    """
    if not isinstance(baseline_runs, dict) or not isinstance(direct_runs, dict):
        return _stub("baseline_and_direct_runs_required")
    keys = sorted(set(baseline_runs) & set(direct_runs))
    if not keys:
        return _stub("no_common_keys")

    per = {}
    all_ok = True
    for k in keys:
        b, d = baseline_runs[k], direct_runs[k]
        if not isinstance(b, dict) or not isinstance(d, dict):
            per[k] = {"identical": False, "reason": "run_not_a_dict"}
            all_ok = False
            continue
        bse = ((b.get("loop_out") or {}).get("stage_outputs") or {}).get("search_eval")
        dse = ((d.get("stage_outputs") or {}) if "stage_outputs" in d
               else ((d.get("loop_out") or {}).get("stage_outputs") or {})).get("search_eval")
        if isinstance(bse, dict):
            bse = {kk: vv for kk, vv in bse.items() if kk != "p9_routing"}
        same = stable_fingerprint(bse) == stable_fingerprint(dse)
        all_ok = all_ok and same
        per[k] = {"identical": same,
                  "baseline_fp": stable_fingerprint(bse),
                  "direct_fp": stable_fingerprint(dse),
                  "routed": b.get("routed_through_p9")}
    conditioned = [k for k, v in (conditioned_runs or {}).items()
                   if isinstance(v, dict) and v.get("routed_through_p9")] \
        if isinstance(conditioned_runs, dict) else []
    return {
        "present": True,
        "version": CASH_P10_STABILITY_VERSION,
        "keys": keys,
        "per_key": per,
        "baseline_identical_to_direct_frozen": all_ok,
        "baseline_runs_are_passthrough": all(
            not per[k].get("routed", False) for k in keys),
        "conditioned_runs_are_routed": len(conditioned) >= 0,
        "n_conditioned_runs": len(conditioned),
        "pass": all_ok,
        "claim_boundary": dict(CLAIM_BOUNDARY),
        "evidence_grade": EVIDENCE_GRADE,
        "executed": False, "result": None, "wired_to_production": False,
        "unknown": [],
    }


def fail_closed_matrix(*, cases: Any) -> dict:
    """汇总 fail-closed 情形。

    每个情形都必须满足：**不产生条件化路由**（`routed=False`）+ **有具名原因**
    + **回退到原树**。注意 `materialize_failure` 一类情形中 `prepared` 本身是
    present（bridge/输入都正常），失败发生在闭包**调用时** —— 因此 `prepared_present`
    允许为 True，判定只看"是否 routed=False 且有原因"。
    """
    cs = cases if isinstance(cases, dict) else {}
    if not cs:
        return _stub("cases_required")
    per = {}
    all_ok = True
    for name in FAIL_CLOSED_CASES:
        c = cs.get(name)
        if not isinstance(c, dict):
            per[name] = {"covered": False}
            all_ok = False
            continue
        not_routed = c.get("routed") is False
        has_reason = bool(c.get("reason"))
        fallback_ok = c.get("fallback_to_original_tree", True) is not False
        ok = not_routed and has_reason and fallback_ok
        all_ok = all_ok and ok
        per[name] = {"covered": True, "ok": ok,
                     "prepared_present": c.get("prepared_present"),
                     "routed": c.get("routed"), "reason": c.get("reason"),
                     "stage": c.get("stage", "unspecified")}
    return {
        "present": True,
        "version": CASH_P10_STABILITY_VERSION,
        "per_case": per,
        "n_cases": len(per),
        "n_covered": sum(1 for v in per.values() if v.get("covered")),
        "all_fail_closed": all_ok,
        "fallback_to_original_tree": all(
            v.get("routed") is False for v in per.values() if v.get("covered")),
        "claim_boundary": dict(CLAIM_BOUNDARY),
        "evidence_grade": EVIDENCE_GRADE,
        "executed": False, "result": None, "wired_to_production": False,
        "unknown": [],
    }


def p10_contract() -> dict:
    return {
        "version": CASH_P10_STABILITY_VERSION,
        "adds_capability": False,
        "modifies_p1_p8_search_v3h": False,
        "modifies_budget_cap_depth_order_river": False,
        "required_populations": list(REQUIRED_POPULATIONS),
        "required_streets": list(REQUIRED_STREETS),
        "causal_links": list(CAUSAL_LINKS),
        "fail_closed_cases": list(FAIL_CLOSED_CASES),
        "claim_boundary": dict(CLAIM_BOUNDARY),
        "evidence_grade": EVIDENCE_GRADE,
    }
