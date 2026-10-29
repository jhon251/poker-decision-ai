"""V3I-P18 — 冻结后的**能力验证计划与可执行性审计**层（只读、只 import 标准库）。

阶段定位（要求 1、2、3）
------------------------
自 V3I-P17 起，CASH Decision Core 视为**冻结基线**：**禁止**新增策略机制、
**禁止**修改链路契约（State / Belief / Population / Conditioning / Search /
Ranking / Assembly / P13 feasible-domain / 0.8 gate / budget / `strategy_core` /
CFI·λ / river terminal 一律不动）。本阶段把目标改为
**「冻结后的能力验证计划 + 可执行性审计」**。

本层只做三件事
--------------
1. 对 G1 / G2 / G2b / G5 / G6 / G7 / G8 **逐项**给出：需要什么外部证据、
   当前已有什么证据、缺什么、最小闭合实验是什么（`audit_gap`）；
2. 判定每个缺口**能否在无 Solver / 无 RC-2 的前提下闭合**
   （`CLOSURE_CATEGORIES`），并明确 **RC-2 / Solver / 真人验证**的依赖关系
   （`external_dependency_graph`）与**已确认不可得的公开数据**清单
   （`EXHAUSTED_PUBLIC_SOURCES`，用于**避免重复寻找**）；
3. 给出**最小 CASH 能力验收矩阵**（`acceptance_matrix`），
   并对其施加**禁止替代**守卫（`validate_claim`）：
   `MODEL_INTERNAL` 证据**不得**用于满足需要外部证据的验收项。

硬边界：不接 CoinPoker、不自动下注、不进入 MTT/PKO；
不宣称 EV / GTO / equilibrium / 盈利能力；不预测强度。
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

CASH_P18_CAPABILITY_PLAN_VERSION = "v3i-p18-capability-verification-plan-v1"

#: 逐项审计的缺口（来自 V3I-P14 / P15 / P16 / P17 的登记）
GAP_IDS: tuple[str, ...] = (
    "G1_accept_not_cross_scorer_verified",
    "G2_conditioning_routing_collapses_accept",
    "G2b_conditioning_changes_decision_but_not_to_accept",
    "G5_decision_quality_not_provable_model_internal",
    "G6_river_terminal_never_reaches_assembly",
    "G7_preflop_no_real_equity_in_frozen_expansion",
    "G8_postflop_no_legitimate_resolution",
)

#: 缺口闭合路径的分类（要求 5）
CLOSURE_CATEGORIES: tuple[str, ...] = (
    "ARCHITECTURE_BOUND_FROZEN",       # 只能改冻结契约才能闭合 ⇒ 冻结期内保持 OPEN
    "REQUIRES_EXTERNAL_EVIDENCE",      # 需要外部证据（RC-2 / Solver / 真人）
    "REQUIRES_EXTERNAL_COVERAGE_BENCHMARK",  # 覆盖率高低的**判据**本身来自外部
    "SETTLED_NOT_A_DEFECT",            # 已查明不是缺陷 ⇒ 无需闭合（问题已关闭）
)

#: 证据等级（与冻结 loop 的 ALLOWED/REJECTED 口径一致）
EVIDENCE_CLASSES: tuple[str, ...] = (
    "MODEL_INTERNAL",
    "EXTERNAL_VALIDATION",
    "REAL_OPPONENT_DATA",
    "OBSERVED_FREQUENCY",
    "SOLVER_EQUILIBRIUM",
    "PROFIT_EVIDENCE",
)

#: 外部证据源的状态（要求 5、6）—— **均已确认不可得，不得重复寻找**
EXTERNAL_SOURCES: dict[str, dict[str, Any]] = {
    "RC2_real_opponent_corpus": {
        "status": "EXHAUSTED",
        "what_it_would_prove": "真实对手行为分布（用于外部有效性验证）",
        "why_unavailable": (
            "IRC Poker Database = REJECTED；11 个候选筛查 = BLOCKED；"
            "C4（PokerAI.org 十亿手牌）= ACCESS_REQUIRED 且 INDEPENDENCE_FAIL；"
            "最终来源搜索 = EXHAUSTED。"),
        "re_searchable": False,
    },
    "solver_equilibrium_reference": {
        "status": "BLOCKED",
        "what_it_would_prove": "策略质量 / 与均衡的距离（用于'更优'类结论）",
        "why_unavailable": (
            "`solver_status = BLOCKED`；无可用求解器依赖；"
            "V3I-P15/P16 的 POC 未产生可用均衡参考。"),
        "re_searchable": False,
    },
    "real_human_validation": {
        "status": "UNAVAILABLE",
        "what_it_would_prove": "对真实人类对手的有效性",
        "why_unavailable": "无合规数据通路；且本程序禁止接真实客户端与自动下注。",
        "re_searchable": False,
    },
}

#: 已确认**不可得**的公开数据源（避免重复寻找，要求 6）
EXHAUSTED_PUBLIC_SOURCES: tuple[dict[str, str], ...] = (
    {"source": "IRC Poker Database",
     "outcome": "RC-2 = REJECTED",
     "reason": "不满足资格/独立性与可用性要求"},
    {"source": "11 个候选公开语料（筛查）",
     "outcome": "RC-2 = BLOCKED",
     "reason": "无一满足'真实对手 + 可取得 + 独立'三条件"},
    {"source": "C4 — PokerAI.org billion-hand database",
     "outcome": "ACCESS_REQUIRED + INDEPENDENCE_FAIL",
     "reason": "需授权访问，且不满足独立性"},
    {"source": "TwoPlusTwo 论坛存档（forumserver/archives1）",
     "outcome": "HTTP 403（不可抓取）",
     "reason": "站点拒绝访问"},
    {"source": "pokerftp.com",
     "outcome": "DNS ENOTFOUND",
     "reason": "域名已不存在"},
    {"source": "Kaggle 数据集页",
     "outcome": "需 JavaScript，无法取用",
     "reason": "非静态可抓取"},
)

#: 最小 CASH 能力验收矩阵的维度（要求 7）
ACCEPTANCE_DIMENSIONS: tuple[str, ...] = (
    "A1_engineering_behavior",
    "A2_decision_coverage_and_provenance",
    "A3_cross_scorer_independence",
    "A4_opponent_conditioned_sensitivity",
    "A5_strategy_quality_vs_equilibrium",
    "A6_real_opponent_external_validity",
    "A7_profitability",
    "A8_river_street_decision_capability",
    "A9_preflop_real_equity_evidence",
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
    "implements_any_unfreeze_item": False,
    "substitutes_internal_for_external_evidence": False,
    "predicts_strength": False,
    "connects_coinpoker_or_auto_betting": False,
    "enters_mtt_pko": False,
}

EVIDENCE_GRADE = "MODEL_INTERNAL"

#: 允许由 `MODEL_INTERNAL` 单独满足的验收维度
INTERNAL_SUFFICIENT_DIMENSIONS: tuple[str, ...] = (
    "A1_engineering_behavior",
    "A2_decision_coverage_and_provenance",
)

#: **必须**外部证据才能满足的验收维度
EXTERNAL_REQUIRED_DIMENSIONS: tuple[str, ...] = (
    "A5_strategy_quality_vs_equilibrium",
    "A6_real_opponent_external_validity",
    "A7_profitability",
)

#: **禁止作出任何宣称**的验收维度（不论证据等级）——本程序的硬约束
FORBIDDEN_DIMENSIONS: tuple[str, ...] = ("A7_profitability",)


def _stub(reason: str) -> dict:
    return {
        "present": False,
        "version": CASH_P18_CAPABILITY_PLAN_VERSION,
        "reason": reason,
        "unknown": [reason],
        "claim_boundary": dict(CLAIM_BOUNDARY),
        "evidence_grade": "NONE",
        "executed": False, "result": None, "wired_to_production": False,
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
# 1. 逐缺口审计（要求 4）
# ---------------------------------------------------------------------------
def audit_gap(*, gap: Any) -> dict:
    """规范化一条缺口审计记录（四问：需要什么 / 已有什么 / 缺什么 / 最小实验）。"""
    if not isinstance(gap, dict):
        return _stub("gap_required")
    gid = gap.get("gap_id")
    if gid not in GAP_IDS:
        return _stub(f"unknown_gap_id:{gid!r}")
    cat = gap.get("closure_category")
    if cat not in CLOSURE_CATEGORIES:
        return _stub(f"unknown_closure_category:{cat!r}")
    required = ("external_evidence_needed", "current_evidence",
                "what_is_missing", "minimal_closure_experiment")
    missing = [k for k in required if not gap.get(k)]
    if missing:
        return _stub("gap_missing_fields:" + ",".join(missing))
    return {
        "present": True,
        "version": CASH_P18_CAPABILITY_PLAN_VERSION,
        "gap_id": gid,
        "title": gap.get("title"),
        "closure_category": cat,
        "external_evidence_needed": gap["external_evidence_needed"],
        "current_evidence": gap["current_evidence"],
        "current_evidence_class": gap.get("current_evidence_class",
                                          EVIDENCE_GRADE),
        "what_is_missing": gap["what_is_missing"],
        "minimal_closure_experiment": gap["minimal_closure_experiment"],
        "closable_without_solver": bool(gap.get("closable_without_solver")),
        "closable_without_rc2": bool(gap.get("closable_without_rc2")),
        "closable_without_human_validation": bool(
            gap.get("closable_without_human_validation")),
        "closable_without_any_external": bool(
            gap.get("closable_without_any_external")),
        "long_term_open": bool(gap.get("long_term_open")),
        "is_a_defect": gap.get("is_a_defect"),
        "requires_unfreeze": bool(gap.get("requires_unfreeze")),
        "blocked_by_freeze": bool(gap.get("blocked_by_freeze")),
        "do_not_search_again": list(gap.get("do_not_search_again") or []),
        "claim_boundary": dict(CLAIM_BOUNDARY),
        "evidence_grade": EVIDENCE_GRADE,
        "unknown": [],
    }


def build_gap_verification_plan(*, gaps: Any) -> dict:
    """汇总七个缺口的能力验证计划。"""
    rows = [audit_gap(gap=g) for g in (gaps if isinstance(gaps, list) else [])]
    ok = [r for r in rows if r.get("present")]
    if not ok:
        return _stub("gaps_required")
    by_cat: dict[str, list[str]] = {c: [] for c in CLOSURE_CATEGORIES}
    for r in ok:
        by_cat[r["closure_category"]].append(r["gap_id"])
    missing = [g for g in GAP_IDS if g not in {r["gap_id"] for r in ok}]
    n_now = sum(1 for r in ok if r["closable_without_any_external"]
                and not r["blocked_by_freeze"]
                and r["closure_category"] != "SETTLED_NOT_A_DEFECT")
    n_unfreeze = sum(1 for r in ok if r["requires_unfreeze"]
                     or r["blocked_by_freeze"])
    n_long = sum(1 for r in ok if r["long_term_open"])
    n_settled = len(by_cat["SETTLED_NOT_A_DEFECT"])
    return {
        "present": True,
        "version": CASH_P18_CAPABILITY_PLAN_VERSION,
        "n_gaps": len(ok),
        "n_gaps_expected": len(GAP_IDS),
        "gaps_not_audited": missing,
        "all_gaps_audited": not missing,
        "per_gap": {r["gap_id"]: r for r in ok},
        "by_closure_category": {k: sorted(v) for k, v in by_cat.items()},
        "n_category": {k: len(v) for k, v in by_cat.items()},
        "n_closable_without_any_external": sum(
            1 for r in ok if r["closable_without_any_external"]),
        "n_closable_now_within_freeze": n_now,
        "n_long_term_open": n_long,
        "n_requires_unfreeze": n_unfreeze,
        "n_settled_not_a_defect": n_settled,
        "headline": (
            "冻结期内**可真正能力闭合**的缺口数 = %d；"
            "需要改冻结契约（本冻结期禁止）才能闭合的 = %d；"
            "必须等待外部证据、长期 OPEN 的 = %d；"
            "已查明**不是缺陷**、问题本身已关闭的 = %d。"
            "**不得**用更多 `MODEL_INTERNAL` 实验把任何一项标成已闭合。"
            % (n_now, n_unfreeze, n_long, n_settled)),
        "claim_boundary": dict(CLAIM_BOUNDARY),
        "evidence_grade": EVIDENCE_GRADE,
        "unknown": [],
    }


# ---------------------------------------------------------------------------
# 2. 外部依赖关系（要求 5、6）
# ---------------------------------------------------------------------------
def external_dependency_graph() -> dict:
    """RC-2 / Solver / 真人验证的依赖关系，以及为何**不可互相替代**。"""
    return {
        "present": True,
        "version": CASH_P18_CAPABILITY_PLAN_VERSION,
        "nodes": {
            "RC2_real_opponent_corpus":
                EXTERNAL_SOURCES["RC2_real_opponent_corpus"],
            "solver_equilibrium_reference":
                EXTERNAL_SOURCES["solver_equilibrium_reference"],
            "real_human_validation":
                EXTERNAL_SOURCES["real_human_validation"],
        },
        "edges": [
            {"from": "RC2_real_opponent_corpus",
             "to": "G5_decision_quality_not_provable_model_internal",
             "kind": "would_provide_observed_opponent_behaviour"},
            {"from": "solver_equilibrium_reference",
             "to": "G5_decision_quality_not_provable_model_internal",
             "kind": "would_provide_equilibrium_distance_reference"},
            {"from": "real_human_validation",
             "to": "G5_decision_quality_not_provable_model_internal",
             "kind": "would_provide_external_validity"},
            {"from": "RC2_real_opponent_corpus",
             "to": "real_human_validation",
             "kind": "is_one_possible_data_path_for"},
        ],
        "why_not_substitutable": [
            "RC-2（已观测频率）与 Solver（均衡参考）是**不同证据等级**："
            "前者说明'对手实际上怎么做'，后者说明'在均衡下应当怎么做'；"
            "前者**不能**推出后者。",
            "Solver 均衡**不能**替代真人有效性：均衡参考不含任何真实人群行为，"
            "无法回答'对这批对手是否有效'。",
            "MODEL_INTERNAL **不能**替代以上任何一项：它是模型内自洽性证据，"
            "不是关于外部世界的证据。",
        ],
        "single_point_of_failure": (
            "三个外部源**全部不可得**且彼此独立 ⇒ G5 在当前条件下"
            "**没有任何可达的闭合路径**，必须长期保持 OPEN。"),
        "avoid_repeat_search": {
            "policy": "本程序**不再**重新搜寻公开语料；已确认不可得的来源见 "
                      "`exhausted_public_sources`。",
            "exhausted_public_sources": [dict(x) for x in
                                         EXHAUSTED_PUBLIC_SOURCES],
        },
        "claim_boundary": dict(CLAIM_BOUNDARY),
        "evidence_grade": EVIDENCE_GRADE,
        "unknown": [],
    }


# ---------------------------------------------------------------------------
# 3. 最小能力验收矩阵（要求 7）
# ---------------------------------------------------------------------------
def acceptance_matrix(*, statuses: Any = None) -> dict:
    """给出**最小 CASH 能力验收矩阵**（规格 + 当前状态）。

    **不虚构结果**：状态必须由调用方按实测证据注入；未注入者记为
    `NO_EVIDENCE`，**不**默认填成通过。
    """
    st = statuses if isinstance(statuses, dict) else {}
    rows: list[dict] = []
    for dim in ACCEPTANCE_DIMENSIONS:
        s = st.get(dim) or {}
        evidence_class = s.get("evidence_class")
        required = s.get("required_evidence_class")
        satisfied = s.get("satisfied")
        rows.append({
            "dimension": dim,
            "criterion": s.get("criterion"),
            "required_evidence_class": required,
            "evidence_class_held": evidence_class,
            "satisfied": bool(satisfied) if satisfied is not None else False,
            "status": s.get("status", "NO_EVIDENCE"),
            "closable_without_solver": s.get("closable_without_solver"),
            "closable_without_rc2": s.get("closable_without_rc2"),
            "falsifier": s.get("falsifier"),
            "evidence_ref": s.get("evidence_ref"),
        })
    by_status: dict[str, int] = {}
    for r in rows:
        by_status[r["status"]] = by_status.get(r["status"], 0) + 1
    return {
        "present": True,
        "version": CASH_P18_CAPABILITY_PLAN_VERSION,
        "n_dimensions": len(rows),
        "rows": rows,
        "by_status": dict(sorted(by_status.items(),
                                 key=lambda kv: (-kv[1], kv[0]))),
        "n_satisfied": sum(1 for r in rows if r["satisfied"]),
        "n_external_required": sum(
            1 for r in rows
            if r["dimension"] in EXTERNAL_REQUIRED_DIMENSIONS),
        "n_external_satisfied": sum(
            1 for r in rows if r["dimension"] in EXTERNAL_REQUIRED_DIMENSIONS
            and r["satisfied"]),
        "internal_only_dimensions": list(INTERNAL_SUFFICIENT_DIMENSIONS),
        "external_required_dimensions": list(EXTERNAL_REQUIRED_DIMENSIONS),
        "note": ("矩阵是**验收规格**：它说明每一项**需要哪一等证据**，"
                 "而不是宣称已达标。未注入实测状态的维度一律记为 "
                 "`NO_EVIDENCE`，**不猜测、不预测强度**。"),
        "claim_boundary": dict(CLAIM_BOUNDARY),
        "evidence_grade": EVIDENCE_GRADE,
        "unknown": [],
    }


def validate_claim(*, criterion: Any, evidence_class: Any,
                   evidence_ref: Any = None) -> dict:
    """**禁止替代守卫**：外部要求的验收项不得用内部证据结清（要求 5）。"""
    if criterion not in ACCEPTANCE_DIMENSIONS:
        return _stub(f"unknown_criterion:{criterion!r}")
    if evidence_class not in EVIDENCE_CLASSES:
        return _stub(f"unknown_evidence_class:{evidence_class!r}")
    needs_external = criterion in EXTERNAL_REQUIRED_DIMENSIONS
    forbidden = criterion in FORBIDDEN_DIMENSIONS
    substitution = needs_external and evidence_class == "MODEL_INTERNAL"
    ok = not forbidden and not substitution
    if forbidden:
        verdict = "REJECTED_FORBIDDEN_CLAIM"
        reason = ("本程序**禁止**作出任何盈利宣称（不论证据等级）；"
                  "该验收维度不设宣称路径。")
    elif substitution:
        verdict = "REJECTED_INTERNAL_SUBSTITUTION"
        reason = ("该验收项要求**外部证据**（外部验证 / 真实对手数据 / "
                  "求解器均衡 / 收益证据 / 已观测频率）；"
                  "以 `MODEL_INTERNAL` 结清属于**用内部自洽替代外部能力证据**，"
                  "本层一律拒绝。")
    else:
        verdict = "ACCEPTED"
        reason = None
    return {
        "present": True,
        "version": CASH_P18_CAPABILITY_PLAN_VERSION,
        "criterion": criterion,
        "evidence_class": evidence_class,
        "evidence_ref": evidence_ref,
        "requires_external_evidence": needs_external,
        "is_forbidden_dimension": forbidden,
        "accepted": ok,
        "verdict": verdict,
        "reason": reason,
        "claim_boundary": dict(CLAIM_BOUNDARY),
        "evidence_grade": EVIDENCE_GRADE,
        "unknown": [],
    }


# ---------------------------------------------------------------------------
# 4. 优先级与可执行性（要求 5、7）
# ---------------------------------------------------------------------------
def prioritise(*, plan: Any) -> dict:
    """按**可执行性**给出优先级：现在可做 / 冻结期内不可做 / 长期 OPEN。"""
    if not isinstance(plan, dict) or not plan.get("present"):
        return _stub("plan_required")
    per = plan["per_gap"]
    now: list[str] = []
    frozen_blocked: list[str] = []
    long_term: list[str] = []
    settled: list[str] = []
    for gid, r in sorted(per.items()):
        if r["closure_category"] == "SETTLED_NOT_A_DEFECT":
            settled.append(gid)
        elif r["requires_unfreeze"] or r["blocked_by_freeze"]:
            frozen_blocked.append(gid)
        elif r["long_term_open"]:
            long_term.append(gid)
        else:
            now.append(gid)
    return {
        "present": True,
        "version": CASH_P18_CAPABILITY_PLAN_VERSION,
        "closable_now_without_external": now,
        "blocked_by_freeze_needs_unfreeze": frozen_blocked,
        "long_term_open_pending_external": long_term,
        "settled_already_closed_as_questions": settled,
        "n_closable_now": len(now),
        "n_blocked_by_freeze": len(frozen_blocked),
        "n_long_term_open": len(long_term),
        "n_settled": len(settled),
        "statement": (
            "冻结期内**没有**可以真正『能力闭合』的缺口："
            "可做的只有①把已查清的问题标为 SETTLED、②把需要外部证据的登记为"
            "长期 OPEN、③把需要改冻结契约的登记为 FROZEN-BLOCKED。"
            "**不得**用更多 `MODEL_INTERNAL` 实验把任何一项标成已闭合。"),
        "claim_boundary": dict(CLAIM_BOUNDARY),
        "evidence_grade": EVIDENCE_GRADE,
        "unknown": [],
    }


def p18_contract() -> dict:
    return {
        "version": CASH_P18_CAPABILITY_PLAN_VERSION,
        "stage": "V3I-P18",
        "purpose": "冻结后的**能力验证计划与可执行性审计**（不堆机制）",
        "frozen_baseline": True,
        "adds_strategy_mechanism": False,
        "modifies_frozen_contract": False,
        "implements_any_unfreeze_item": False,
        "modifies_chain_state_belief_population_conditioning_search_ranking_"
        "assembly": False,
        "modifies_p13_feasible_domain": False,
        "relaxes_leaf_coverage_threshold": False,
        "modifies_budget_or_strategy_core_or_cfi_or_river_terminal": False,
        "substitutes_internal_for_external_evidence": False,
        "predicts_strength": False,
        "fabricates_results": False,
        "connects_coinpoker_or_auto_betting": False,
        "enters_mtt_pko": False,
        "reads_only": True,
        "gap_ids": list(GAP_IDS),
        "closure_categories": list(CLOSURE_CATEGORIES),
        "evidence_classes": list(EVIDENCE_CLASSES),
        "acceptance_dimensions": list(ACCEPTANCE_DIMENSIONS),
        "imports_project_layers": False,
        "claim_boundary": dict(CLAIM_BOUNDARY),
        "evidence_grade": EVIDENCE_GRADE,
    }
