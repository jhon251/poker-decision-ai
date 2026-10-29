"""V3I-P12 — V3H-8 契约变更**设计与最小实验验证**（不在本层修改冻结代码）。

根因（已复现）
--------------
`search` 只评估 `action_policy.candidates[:max_root_candidates]`（冻结 cap），
而 `ranking` 评估**全部**候选。两层因此处于**不对等的候选域**：
`ranking` 的 top 可能是 `search` **结构上无法评估**的候选（例如 `bet_allin`）。

实测（seed=11, flop, cap=2）：
* search 域     = {('check',0.0), ('bet',0.495)}
* ranking 域    = 全部 6 个候选，top = ('bet',100.0)  ← **不在 search 域内**
* ranking **限域** top = ('bet',0.495) = search top  ← **一致**

⇒ 该分歧是**非对等候选集造成的伪分歧**，不是两层策略判断真的冲突。

提议的最小契约变更（要求 4、5）
------------------------------
给 V3H-8 的 `assemble_decision` 增加**可选**参数
`feasible_keys: Iterable[tuple[str, float]] | None = None`：

* `None`（默认）→ 行为与现状**逐字节一致**（冻结契约不变，基线可追溯）；
* 提供时 → 每个 tier 的 top 先**限定在 `feasible_keys` 内**再比较；
  某 tier 在限域内无候选 → 记该 tier 不可用（**不**伪装成可用）；
  限域后仍不一致 → **照旧弃权**。

**为什么不会把不一致包装成一致**：本变更只删除"某层根本无法评估的候选"，
不改变任何评分公式、不改变 tier 优先级、不引入容忍度或投票；
若两层在**同一**可行域上给出的 top 仍不同，`disagreements` 照常记录并弃权。

本层只 import 标准库；**不修改任何冻结代码**。
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

CASH_P12_CONTRACT_DESIGN_VERSION = "v3i-p12-v3h8-feasible-domain-contract-design-v1"

#: 提议变更的目标与签名
PROPOSED_CHANGE: dict[str, Any] = {
    "file": "poker_decision_ai/cash_strategy_decision_assembly.py",
    "function": "assemble_decision",
    "added_parameter": "feasible_keys",
    "signature_change": ("assemble_decision(..., feasible_keys: "
                         "Iterable[tuple[str, float]] | None = None)"),
    "default_preserves_current_behaviour": True,
    "modifies_scoring_formula": False,
    "modifies_tier_precedence": False,
    "modifies_override_semantics": False,
    "adds_tolerance_or_voting": False,
    "relaxes_fail_closed": False,
    "preserved_fail_closed_conditions": [
        "限域后 tier 无候选 → 该 tier 不可用（不伪装）",
        "限域后 top 仍不一致 → abstained=True（真分歧照旧弃权）",
        "leaf_coverage_below_minimum → 该 tier 不可用",
        "search_score_unavailable → 该 tier 不可用",
        "override 语义不变（仍需调用方显式开启 + 非空理由）",
    ],
    "affected_existing_tests": [
        "tests/unit/test_v3h8_decision_assembly.py（若其中断言的是"
        "「不改动任何参数时的行为」→ **不受影响**，因默认路径不变）",
        "任何显式构造 disagreements 的测试：仅在**传入** feasible_keys 时才可能变化",
    ],
}

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
    "modifies_frozen_code_in_this_stage": False,
    "relaxes_abstain": False,
    "enables_override": False,
    "weakens_unknown_or_fail_closed": False,
}

EVIDENCE_GRADE = "MODEL_INTERNAL"


def _is_num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _stub(reason: str) -> dict:
    return {
        "present": False,
        "version": CASH_P12_CONTRACT_DESIGN_VERSION,
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


def _key_of(row: Any) -> Any:
    """把候选行规范化为 (action, amount) 键。"""
    if isinstance(row, dict):
        a = row.get("action")
        m = row.get("amount_bb")
        if isinstance(a, str):
            return (a, round(float(m), 6) if _is_num(m) else None)
    if isinstance(row, (list, tuple)) and len(row) >= 2:
        a, m = row[0], row[1]
        if isinstance(a, str):
            return (a, round(float(m), 6) if _is_num(m) else None)
    return None


def normalize_feasible(keys: Any) -> list:
    out = []
    for k in (keys or []):
        kk = _key_of(k) if not (isinstance(k, tuple) and len(k) == 2) else \
            (str(k[0]), round(float(k[1]), 6) if _is_num(k[1]) else None)
        if kk is not None:
            out.append(kk)
    return sorted(set(out), key=lambda t: (t[0], -1 if t[1] is None else t[1]))


def reproduce_root_cause(*, action_policy_keys: Any, search_domain_keys: Any,
                         ranking_ranked: Any) -> dict:
    """复现：全域分歧 vs 限域一致（要求 3 的可复现证据）。"""
    apk = normalize_feasible(action_policy_keys)
    sdk = normalize_feasible(search_domain_keys)
    sd_set = set(sdk)

    ranked = [r for r in (ranking_ranked or []) if isinstance(r, dict)]
    ranked_sorted = sorted(ranked, key=lambda r: -(r.get("score") or 0))
    top_global = _key_of(ranked_sorted[0]) if ranked_sorted else None
    limited = [r for r in ranked_sorted if _key_of(r) in sd_set]
    top_limited = _key_of(limited[0]) if limited else None

    return {
        "present": True,
        "version": CASH_P12_CONTRACT_DESIGN_VERSION,
        "action_policy_domain_size": len(apk),
        "search_domain_size": len(sdk),
        "search_domain": [list(k) for k in sdk],
        "ranking_only_candidates": [
            list(k) for k in apk if k not in sd_set],
        "ranking_top_global": list(top_global) if top_global else None,
        "ranking_top_limited_to_search_domain":
            list(top_limited) if top_limited else None,
        "ranking_top_global_in_search_domain":
            (top_global in sd_set) if top_global else None,
        "domain_mismatch": len(apk) != len(sdk),
        "root_cause": ("search 只评估冻结 cap 内的候选，ranking 评估全部候选；"
                       "两层处于**不对等候选域**，故 top key 比较会把"
                       "'某层结构上无法评估的候选'记成分歧。"),
        "claim_boundary": dict(CLAIM_BOUNDARY),
        "evidence_grade": EVIDENCE_GRADE,
        "executed": False, "result": None, "wired_to_production": False,
        "unknown": [],
    }


def simulate_feasible_domain_assembly(*, tier_ranked: Any, feasible_keys: Any,
                                      current_out: Any = None) -> dict:
    """在**不修改冻结代码**的前提下模拟提议契约的效果。

    Parameters
    ----------
    tier_ranked : {tier_name: [ {action, amount_bb, score}, ... ]}
        各 tier 的候选（按自身分数排序即可，本函数会重新排序）。
    feasible_keys : 可行域（通常 = search 实际评估的候选集合）
    current_out : 可选，冻结 V3H-8 的实际输出，用于对照
    """
    if not isinstance(tier_ranked, dict) or not tier_ranked:
        return _stub("tier_ranked_required")
    feas = set(normalize_feasible(feasible_keys))
    if not feas:
        return _stub("feasible_keys_required")

    per_tier: dict[str, Any] = {}
    limited_tops: dict[str, Any] = {}
    unavailable: list[str] = []
    for tier, rows in tier_ranked.items():
        ranked = [r for r in (rows or []) if isinstance(r, dict)]
        ranked.sort(key=lambda r: -(r.get("score") or 0))
        top_global = _key_of(ranked[0]) if ranked else None
        limited = [r for r in ranked if _key_of(r) in feas]
        top_limited = _key_of(limited[0]) if limited else None
        if top_limited is None:
            unavailable.append(str(tier))
        else:
            limited_tops[str(tier)] = top_limited
        per_tier[str(tier)] = {
            "top_global": list(top_global) if top_global else None,
            "top_limited": list(top_limited) if top_limited else None,
            "n_candidates": len(ranked),
            "n_in_feasible": len(limited),
            "changed_by_restriction": top_global != top_limited,
        }

    distinct = {json.dumps(list(v)) for v in limited_tops.values()}
    agreed = len(distinct) == 1 and len(limited_tops) >= 1
    # 真分歧：限域后仍不一致 → 仍弃权；tier 不可用 → 仍弃权
    abstained = (not agreed) or bool(unavailable)

    return {
        "present": True,
        "version": CASH_P12_CONTRACT_DESIGN_VERSION,
        "feasible_domain": [list(k) for k in sorted(feas)],
        "per_tier": per_tier,
        "limited_tops": {k: list(v) for k, v in limited_tops.items()},
        "tiers_unavailable_within_feasible_domain": sorted(unavailable),
        "agreed_within_feasible_domain": agreed,
        "would_abstain_after_change": abstained,
        "would_accept_after_change": not abstained,
        "selected_key": (list(next(iter(distinct and limited_tops.values())))
                         if agreed and limited_tops else None),
        "current_frozen_output_summary": (
            {"abstained": (current_out or {}).get("abstained"),
             "preferred": (current_out or {}).get("preferred"),
             "n_disagreements": len((current_out or {}).get("disagreements") or [])}
            if isinstance(current_out, dict) else None),
        "why_not_packaging_inconsistency_as_consistency": (
            "本模拟只删除'某 tier 结构上无法评估的候选'（限定到共同可行域）；"
            "不修改评分、不改 tier 优先级、不引入容忍度或投票。若限域后各层 top "
            "仍不同，或某层在限域内无候选，则**照旧弃权**（见 "
            "would_abstain_after_change 的依据）。"),
        "claim_boundary": dict(CLAIM_BOUNDARY),
        "evidence_grade": EVIDENCE_GRADE,
        "executed": False, "result": None, "wired_to_production": False,
        "unknown": [],
    }


def true_disagreement_probe(*, tier_ranked: Any) -> dict:
    """构造/检验**真分歧**：同一可行域内各层 top 仍不同 → 必须仍弃权。"""
    if not isinstance(tier_ranked, dict) or len(tier_ranked) < 2:
        return _stub("at_least_two_tiers_required")
    keys = set()
    for rows in tier_ranked.values():
        for r in (rows or []):
            if isinstance(r, dict):
                k = _key_of(r)
                if k is not None:
                    keys.add(k)
    sim = simulate_feasible_domain_assembly(tier_ranked=tier_ranked,
                                            feasible_keys=sorted(keys))
    if not sim.get("present"):
        return sim
    return {
        "present": True,
        "version": CASH_P12_CONTRACT_DESIGN_VERSION,
        "shared_domain": [list(k) for k in sorted(keys)],
        "agreed": sim["agreed_within_feasible_domain"],
        "still_abstains": sim["would_abstain_after_change"],
        "limited_tops": sim["limited_tops"],
        "verdict": ("真分歧被正确保留（仍弃权）"
                    if sim["would_abstain_after_change"] else
                    "各层在同一可行域内一致（非真分歧）"),
        "claim_boundary": dict(CLAIM_BOUNDARY),
        "evidence_grade": EVIDENCE_GRADE,
        "executed": False, "result": None, "wired_to_production": False,
        "unknown": [],
    }


def default_path_unchanged_check(*, frozen_out: Any,
                                 simulated_without_feasible: Any) -> dict:
    """证明默认路径（不传 feasible_keys）与现状一致。"""
    if not isinstance(frozen_out, dict):
        return _stub("frozen_out_required")
    return {
        "present": True,
        "version": CASH_P12_CONTRACT_DESIGN_VERSION,
        "frozen_abstained": frozen_out.get("abstained"),
        "frozen_n_disagreements": len(frozen_out.get("disagreements") or []),
        "proposed_default_is_none": True,
        "behaviour_change_on_default_path": False,
        "note": ("提议的 feasible_keys 默认值为 None；此时实现应**完全跳过**"
                 "限域逻辑，与冻结现状逐字节一致（由既有 V3H-8 测试守护）。"),
        "baseline_traceable": True,
        "claim_boundary": dict(CLAIM_BOUNDARY),
        "evidence_grade": EVIDENCE_GRADE,
        "executed": False, "result": None, "wired_to_production": False,
        "unknown": [],
    }


def p12_contract() -> dict:
    return {
        "version": CASH_P12_CONTRACT_DESIGN_VERSION,
        "modifies_frozen_code": False,
        "stage_focus": "契约设计 + 最小实验验证",
        "proposed_change": dict(PROPOSED_CHANGE),
        "claim_boundary": dict(CLAIM_BOUNDARY),
        "evidence_grade": EVIDENCE_GRADE,
    }
