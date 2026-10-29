"""V3I-P17 — CASH Decision Core 的**架构收口审计**层（只读、只 import 标准库）。

授权范围（要求 2、3、9）
------------------------
* **不实施** preflop 解冻；**不修改** `search_tree` / `ranking` / search budget /
  0.8 门槛（本阶段仅做冻结前的**最终反事实记录**）；
* **不启用** override / 投票 / 降门槛 / 强行统一 ranking 与 search；
* 不接 CoinPoker、不自动下注、不进入 MTT/PKO；
* 本层只做**分类、核对与清单**：链路上游一律由调用方**显式注入**。

本层回答三个问题
----------------
1. 链路是否达到**工程冻结**标准（`build_freeze_checklist` / `module_freeze_status`）；
2. 冻结后**剩余的能力验证缺口**是什么（`open_capability_gaps`）；
3. `MODEL_INTERNAL` **能证明什么、不能证明什么**（`model_internal_provable`）。

冻结的语义边界（要求 7）见 `FINAL_DECISION_CLASSES`：**完整多源 ACCEPT** 与
**ranking-only ACCEPT** 必须分开计数，**不得**把后者包装成前者。
"""
from __future__ import annotations

import ast
import hashlib
import json
from typing import Any

CASH_P17_CLOSURE_AUDIT_VERSION = "v3i-p17-architecture-closure-audit-v1"

#: 主链路（用户点名的 7 段顺序）+ 支撑段
CHAIN_STAGES: tuple[tuple[str, str], ...] = (
    ("State", "cash_state_integrity"),
    ("State", "cash_integrity_context"),
    ("Environment", "cash_decision_environment"),
    ("Belief", "cash_strategy_dynamic_belief"),
    ("Belief", "cash_strategy_likelihood_model"),
    ("Belief", "cash_strategy_belief_update"),
    ("Population", "cash_opponent_model"),
    ("Population", "cash_range_model"),
    ("ActionTree", "cash_action_tree"),
    ("ActionPolicy", "cash_strategy_action_policy"),
    ("Ranking", "cash_strategy_action_ranking"),
    ("Search", "cash_strategy_search_tree"),
    ("Search", "cash_strategy_search_eval"),
    ("Assembly", "cash_strategy_decision_assembly"),
    ("Loop", "cash_decision_loop"),
    ("Conditioning", "cash_opponent_conditioned_bridge"),
    ("Conditioning", "cash_conditioned_search_wiring"),
    ("Conditioning", "cash_opponent_conditioned_loop"),
    ("FeasibleDomain", "cash_feasible_domain_wiring"),
    ("BeliefConditioned", "cash_strategy_belief_conditioned_eval"),
    ("Multistreet", "cash_multistreet"),
    ("Solver", "cash_solver_interface"),
)

#: 主链路顺序（仅用于陈述，不含支撑段）
PRIMARY_FLOW: tuple[str, ...] = ("State", "Belief", "Population", "Conditioning",
                                 "Search", "Ranking", "Assembly")

#: 冻结条件清单（要求 6）—— **最小**条件集
FREEZE_CHECKLIST: tuple[str, ...] = (
    "F1_chain_completeness",
    "F2_determinism",
    "F3_fail_closed",
    "F4_claim_boundary_declared_no_positive_claims",
    "F5_evidence_grade_internal_only",
    "F6_no_privilege_escalation",
    "F7_production_isolation",
    "F8_decision_semantics_closed",
    "F9_capability_gaps_declared",
    "F10_no_unfinished_markers",
)

#: 模块冻结状态
MODULE_STATUSES: tuple[str, ...] = (
    "ENGINEERING_FROZEN",       # 决定论 + fail-closed + 边界声明 + 独立验证层
    "CAPABILITY_GAP_OPEN",      # 工程可用，但能力**未经外部证据验证**
    "EXTERNALLY_BLOCKED",       # 依赖外部资源（如 solver），当前不可用
)

#: 最终决策语义边界（要求 7）—— 五类，**不得合并**
FINAL_DECISION_CLASSES: tuple[str, ...] = (
    "FULL_MULTISOURCE_ACCEPT",      # 装配产出 preferred 且 ≥2 个可比 tier 一致
    "RANKING_ONLY_ACCEPT",          # 产出 preferred 但只有 1 个可比 tier（较弱）
    "TRUE_DISAGREEMENT_ABSTAIN",    # ≥2 个可比 tier 在同一域内不一致
    "EVIDENCE_INSUFFICIENT",        # 无法形成可比多源视图 / 无 scorer / fail-closed
    "RIVER_BLOCKED",                # 冻结终局：搜索前即阻断
)

#: 真分歧的两个子类（诚实区分"域不匹配"与"同域不一致"）
DISAGREEMENT_KINDS: tuple[str, ...] = (
    "same_domain_disagreement",
    "structural_domain_mismatch",
)

#: 链路内**禁止**出现的未完成标记
UNFINISHED_MARKERS: tuple[str, ...] = (
    "TODO", "FIXME", "XXX", "HACK", "NotImplementedError",
)

#: `MODEL_INTERNAL` 可证明 / 不可证明（要求 8）
PROVABLE_WITH_MODEL_INTERNAL: tuple[str, ...] = (
    "链路可运行性与阶段完整性（8 个必需阶段可注入、真实跑到装配层）",
    "决定论（同输入 → 同输出，含逐运行决策指纹）",
    "fail-closed 行为（非 CASH / 缺失输入 / 越界证据等级 → 拒绝）",
    "决策覆盖率的**分类计数**（ACCEPT / ABSTAIN / BLOCKED / 各语义子类）",
    "决策**出处**（注额来自 search 实际展开的候选、是否消费条件化权益）",
    "跨街/跨注额/跨 board/跨 population 的**稳定性与敏感性**",
    "根因定位与最小解冻方案的**反事实后果**",
    "冻结边界的完整性（哈希、导入图、门槛、budget 未变）",
)

NOT_PROVABLE_WITH_MODEL_INTERNAL: tuple[str, ...] = (
    "决策是否**更优**（无 EV / GTO / 均衡参考）",
    "是否**盈利**（无收益证据、无真实对局）",
    "对**真实人类对手**是否有效（无 RC-2 外部语料、无带外信任锚）",
    "任何**胜率 / 强度 / 优势**结论",
    "多源一致的**独立性强度**（当两层使用同一权益时一致性恒真）",
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
    "implements_any_unfreeze_item": False,
    "adds_strategy_mechanism": False,
    "modifies_frozen_contract": False,
    "modifies_budget_depth_or_cap": False,
    "relaxes_leaf_coverage_threshold": False,
    "enables_override": False,
    "uses_voting": False,
    "forces_ranking_search_unification": False,
}

EVIDENCE_GRADE = "MODEL_INTERNAL"


def _stub(reason: str) -> dict:
    return {
        "present": False,
        "version": CASH_P17_CLOSURE_AUDIT_VERSION,
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
# 1. CLAIM_BOUNDARY 审计（支持 AnnAssign）
# ---------------------------------------------------------------------------
def extract_claim_boundary(*, source: Any) -> dict:
    """从模块源码里取出 `CLAIM_BOUNDARY` 字面量并判定是否全 False。"""
    if not isinstance(source, str) or not source.strip():
        return _stub("source_text_required")
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return _stub(f"source_unparsable:{type(exc).__name__}")
    found = None
    for node in tree.body:
        targets = []
        if isinstance(node, ast.Assign):
            targets = node.targets
            value = node.value
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
            value = node.value
        else:
            continue
        for t in targets:
            if isinstance(t, ast.Name) and t.id == "CLAIM_BOUNDARY":
                try:
                    found = ast.literal_eval(value)
                except Exception:                          # noqa: BLE001
                    return _stub("claim_boundary_not_literal")
    if not isinstance(found, dict):
        return _stub("claim_boundary_not_found")
    non_false = sorted(k for k, v in found.items() if v is not False)
    return {
        "present": True,
        "version": CASH_P17_CLOSURE_AUDIT_VERSION,
        "n_flags": len(found),
        "all_false": not non_false,
        "non_false_flags": non_false,
        "flags": sorted(found),
        "claim_boundary": dict(CLAIM_BOUNDARY),
        "evidence_grade": EVIDENCE_GRADE,
        "unknown": [],
    }


def audit_claim_boundaries(*, module_sources: Any) -> dict:
    """批量审计一组模块的 `CLAIM_BOUNDARY`（只统计**声明了的**）。"""
    if not isinstance(module_sources, dict) or not module_sources:
        return _stub("module_sources_required")
    per: dict[str, Any] = {}
    offenders: list[str] = []
    undeclared: list[str] = []
    for mod, src in sorted(module_sources.items()):
        r = extract_claim_boundary(source=src)
        per[str(mod)] = {
            "present": r.get("present"),
            "n_flags": r.get("n_flags"),
            "all_false": r.get("all_false"),
            "non_false_flags": r.get("non_false_flags"),
            "reason": r.get("reason"),
        }
        if not r.get("present"):
            undeclared.append(str(mod))
        elif not r.get("all_false"):
            offenders.append(str(mod))
    declared = [m for m, v in per.items() if v["present"]]
    return {
        "present": True,
        "version": CASH_P17_CLOSURE_AUDIT_VERSION,
        "n_modules": len(per),
        "n_declared": len(declared),
        "n_undeclared": len(undeclared),
        "declared_modules": sorted(declared),
        "undeclared_modules": sorted(undeclared),
        "per_module": per,
        "modules_with_positive_claim": offenders,
        "all_declared_are_all_false": not offenders,
        "every_module_declares_boundary": not undeclared,
        "claim_boundary": dict(CLAIM_BOUNDARY),
        "evidence_grade": EVIDENCE_GRADE,
        "unknown": [],
    }


def audit_boundary_coverage(*, module_sources: Any,
                            decision_surface_modules: Any = None) -> dict:
    """把"边界声明"拆成**可分别判定**的三项（不把文档缺口说成正确性缺口）。

    * `declared_all_false`      —— 已声明的 `CLAIM_BOUNDARY` 是否**全部**为 False；
    * `decision_surface_declared` —— **决策面**（loop 输出 + 装配层）是否声明了
      全 False 的边界（这是"输出不被误读为 EV/GTO"的**实质要求**）；
    * `every_module_declares`   —— 每个链路模块是否都在源码内声明了边界
      （**文档性**要求；V3G 期原语层未采用该约定）。
    """
    if not isinstance(module_sources, dict) or not module_sources:
        return _stub("module_sources_required")
    base = audit_claim_boundaries(module_sources=module_sources)
    surf = list(decision_surface_modules or [])
    surf_state = {}
    for m in surf:
        r = extract_claim_boundary(source=module_sources.get(m))
        surf_state[m] = {"declared": bool(r.get("present")),
                         "all_false": r.get("all_false")}
    surface_ok = bool(surf) and all(v["declared"] and v["all_false"]
                                    for v in surf_state.values())
    return {
        "present": True,
        "version": CASH_P17_CLOSURE_AUDIT_VERSION,
        "n_modules": base["n_modules"],
        "declared_all_false": base["all_declared_are_all_false"],
        "modules_with_positive_claim": base["modules_with_positive_claim"],
        "decision_surface_modules": surf,
        "decision_surface_state": surf_state,
        "decision_surface_declared": surface_ok,
        "every_module_declares": base["every_module_declares_boundary"],
        "undeclared_modules": base["undeclared_modules"],
        "n_undeclared": base["n_undeclared"],
        "documentation_gap": {
            "kind": "in_source_boundary_documentation",
            "modules": base["undeclared_modules"],
            "n": base["n_undeclared"],
            "why_not_a_correctness_gap": (
                "这些模块是 V3G 期**上游原语**（range / opponent / action-tree / "
                "multistreet / solver 状态 / integrity），它们**不产出决策**，"
                "也**没有**任何正向宣称（沉默 ≠ 正向宣称）；"
                "而**决策面**（`cash_decision_loop` 输出与 "
                "`cash_strategy_decision_assembly`）已声明全 False 边界，"
                "并由循环把 `evidence_level = MODEL_INTERNAL` 写进决策记录。"),
            "why_it_cannot_be_closed_inside_the_freeze": (
                "这 5 个模块由 **V3G-11 冻结契约逐哈希守护**；"
                "给它们补声明等于修改冻结层，必须走独立的解冻授权 —— "
                "因此本阶段**只登记、不修改**。"),
            "recommended_action": (
                "作为**冻结后**的文档类改进项登记（非阻塞）："
                "若未来获得解冻授权，可统一补齐 `CLAIM_BOUNDARY`，"
                "但**不得**因此顺带改动任何行为。"),
        },
        "claim_boundary": dict(CLAIM_BOUNDARY),
        "evidence_grade": EVIDENCE_GRADE,
        "unknown": [],
    }


# ---------------------------------------------------------------------------
# 2. 链路清单与模块状态（要求 4、5）
# ---------------------------------------------------------------------------
def audit_chain_inventory(*, modules: Any, module_sources: Any = None) -> dict:
    """核对链路模块齐备性、哈希登记与未完成标记。"""
    if not isinstance(modules, dict) or not modules:
        return _stub("modules_required")
    srcs = module_sources if isinstance(module_sources, dict) else {}
    missing: list[str] = []
    unhashed: list[str] = []
    unfinished: dict[str, list[str]] = {}
    by_stage: dict[str, list[str]] = {}
    for stage, mod in CHAIN_STAGES:
        info = modules.get(mod)
        by_stage.setdefault(stage, []).append(mod)
        if not isinstance(info, dict):
            missing.append(mod)
            continue
        if not info.get("sha256_prefix"):
            unhashed.append(mod)
        src = srcs.get(mod)
        if isinstance(src, str):
            hits = [m for m in UNFINISHED_MARKERS if m in src]
            if hits:
                unfinished[mod] = hits
    stages_present = sorted(
        s for s, mods in by_stage.items()
        if all(mod not in missing for mod in mods))
    return {
        "present": True,
        "version": CASH_P17_CLOSURE_AUDIT_VERSION,
        "primary_flow": list(PRIMARY_FLOW),
        "n_stages": len({s for s, _ in CHAIN_STAGES}),
        "n_modules": len(CHAIN_STAGES),
        "modules_by_stage": by_stage,
        "stages_present": stages_present,
        "missing_modules": missing,
        "modules_without_hash": unhashed,
        "modules_with_unfinished_markers": unfinished,
        "chain_complete": not missing,
        "all_hashes_registered": not unhashed,
        "no_unfinished_markers": not unfinished,
        "claim_boundary": dict(CLAIM_BOUNDARY),
        "evidence_grade": EVIDENCE_GRADE,
        "unknown": [],
    }


def module_freeze_status(*, inventory: Any, verification_layers: Any = None,
                         externally_blocked: Any = None) -> dict:
    """逐模块给出冻结状态（要求 5）。"""
    if not isinstance(inventory, dict) or not inventory.get("present"):
        return _stub("inventory_required")
    ver = verification_layers if isinstance(verification_layers, dict) else {}
    blocked = set(externally_blocked or [])
    per: dict[str, str] = {}
    for stage, mod in CHAIN_STAGES:
        if mod in blocked:
            per[mod] = "EXTERNALLY_BLOCKED"
            continue
        verified = mod in ver
        per[mod] = "ENGINEERING_FROZEN" if verified else "CAPABILITY_GAP_OPEN"
    counts: dict[str, int] = {k: 0 for k in MODULE_STATUSES}
    for v in per.values():
        counts[v] = counts.get(v, 0) + 1
    return {
        "present": True,
        "version": CASH_P17_CLOSURE_AUDIT_VERSION,
        "per_module": per,
        "counts": counts,
        "verified_modules": sorted(ver),
        "externally_blocked_modules": sorted(blocked),
        "statement": (
            "「ENGINEERING_FROZEN」= 决定论、fail-closed、边界声明齐备，"
            "**且**有专门的只读验证层覆盖；"
            "「CAPABILITY_GAP_OPEN」= 工程可用但能力未经外部证据验证；"
            "「EXTERNALLY_BLOCKED」= 依赖外部资源（如 solver）当前不可用。"),
        "claim_boundary": dict(CLAIM_BOUNDARY),
        "evidence_grade": EVIDENCE_GRADE,
        "unknown": [],
    }


# ---------------------------------------------------------------------------
# 3. 最终决策语义边界（要求 7）
# ---------------------------------------------------------------------------
def classify_final_semantics(*, record: Any) -> dict:
    """把一次运行归入**五类最终语义**之一（只读）。"""
    if not isinstance(record, dict):
        return _stub("record_not_a_dict")
    preferred = record.get("preferred")
    abstained = record.get("abstained") is True
    disagreements = [d for d in (record.get("disagreements") or [])
                     if isinstance(d, dict)]
    n_align = record.get("n_scorers_alignable")
    n_align = int(n_align) if isinstance(n_align, int) else 0
    reason = str(record.get("reason") or "")
    trace = record.get("stage_trace") or []
    ap_absent = any(isinstance(t, dict) and t.get("stage") == "action_policy"
                    and t.get("present") is False for t in trace)

    cls = None
    sub = None
    if record.get("present") is False or reason.startswith(
            "stage_failed:action_policy") or ap_absent:
        if reason.startswith("stage_failed:action_policy") or ap_absent:
            cls = "RIVER_BLOCKED"
        else:
            cls = "EVIDENCE_INSUFFICIENT"
    elif abstained:
        cls = "TRUE_DISAGREEMENT_ABSTAIN"
        keys = []
        for d in disagreements:
            keys.append(d.get("key_a"))
            keys.append(d.get("key_b"))
        keys = [k for k in keys if k is not None]
        tree = [(c.get("action"), c.get("amount_bb"))
                for c in (record.get("tree_root_children") or [])
                if isinstance(c, dict)]

        def _in(k):
            if not isinstance(k, (list, tuple)) or len(k) != 2:
                return False
            return any(str(k[0]) == str(a) and (
                k[1] is None or b is None
                or abs(float(k[1]) - float(b)) <= 1e-6) for a, b in tree)
        # **两侧** key 都要在展开集内，才算"同域分歧"
        outside = any(not _in(k) for k in keys) if keys else False
        sub = ("structural_domain_mismatch" if outside
               else "same_domain_disagreement")
    elif preferred not in (None, {}):
        cls = ("FULL_MULTISOURCE_ACCEPT" if n_align >= 2
               else "RANKING_ONLY_ACCEPT")
    else:
        cls = "EVIDENCE_INSUFFICIENT"
    return {
        "present": True,
        "version": CASH_P17_CLOSURE_AUDIT_VERSION,
        "outcome_class": cls,
        "disagreement_kind": sub,
        "n_scorers_alignable": n_align,
        "is_full_multisource": cls == "FULL_MULTISOURCE_ACCEPT",
        "is_ranking_only": cls == "RANKING_ONLY_ACCEPT",
        "selected_by_tier": (preferred or {}).get("selected_by_tier")
        if isinstance(preferred, dict) else None,
        "must_not_be_conflated": (
            "`RANKING_ONLY_ACCEPT` **不是** `FULL_MULTISOURCE_ACCEPT`："
            "前者只有单一可比评分源（search 层因覆盖率闸门不可用），"
            "**没有**跨评分源一致性背书。"),
        "claim_boundary": dict(CLAIM_BOUNDARY),
        "evidence_grade": EVIDENCE_GRADE,
        "unknown": [],
    }


def final_semantics_summary(*, records: Any, mode: Any = None) -> dict:
    """汇总五类最终语义的分布（可按 mode 限定）。"""
    recs = [r for r in (records if isinstance(records, list) else [])
            if isinstance(r, dict)]
    if isinstance(mode, str) and mode:
        recs = [r for r in recs if r.get("mode") == mode]
    if not recs:
        return _stub("records_required")
    rows = [classify_final_semantics(record=r) for r in recs]
    rows = [r for r in rows if r.get("present")]
    counts = {k: 0 for k in FINAL_DECISION_CLASSES}
    for r in rows:
        counts[r["outcome_class"]] = counts.get(r["outcome_class"], 0) + 1
    dis_kinds: dict[str, int] = {}
    for r in rows:
        if r["outcome_class"] == "TRUE_DISAGREEMENT_ABSTAIN":
            k = str(r["disagreement_kind"])
            dis_kinds[k] = dis_kinds.get(k, 0) + 1
    n = len(rows)
    return {
        "present": True,
        "version": CASH_P17_CLOSURE_AUDIT_VERSION,
        "mode": mode,
        "n_runs": n,
        "class_counts": counts,
        "class_fraction": {k: (round(v / n, 6) if n else None)
                           for k, v in counts.items()},
        "disagreement_kind_counts": dict(sorted(
            dis_kinds.items(), key=lambda kv: (-kv[1], kv[0]))),
        "accept_total": counts["FULL_MULTISOURCE_ACCEPT"]
        + counts["RANKING_ONLY_ACCEPT"],
        "full_multisource_share_of_accepts": (
            round(counts["FULL_MULTISOURCE_ACCEPT"] /
                  (counts["FULL_MULTISOURCE_ACCEPT"]
                   + counts["RANKING_ONLY_ACCEPT"]), 6)
            if (counts["FULL_MULTISOURCE_ACCEPT"]
                + counts["RANKING_ONLY_ACCEPT"]) else None),
        "all_classes_reachable": all(counts[k] > 0
                                     for k in FINAL_DECISION_CLASSES
                                     if k != "EVIDENCE_INSUFFICIENT"),
        "classes_with_measured_instances": sorted(
            k for k, v in counts.items() if v > 0),
        "claim_boundary": dict(CLAIM_BOUNDARY),
        "evidence_grade": EVIDENCE_GRADE,
        "unknown": [],
    }


# ---------------------------------------------------------------------------
# 4. 冻结条件清单与判定（要求 6、11）
# ---------------------------------------------------------------------------
def build_freeze_checklist(*, evidence: Any) -> dict:
    """把实测证据折算成**最小冻结条件清单**的逐项通过/不通过。"""
    if not isinstance(evidence, dict):
        return _stub("evidence_required")
    items: list[dict] = []

    def _add(cid, ok, detail):
        items.append({"id": cid, "passed": bool(ok), "detail": detail})

    _add("F1_chain_completeness",
         bool(evidence.get("chain_complete"))
         and bool(evidence.get("n_required_stages_ok")),
         {"chain_complete": evidence.get("chain_complete"),
          "n_required_stages": evidence.get("n_required_stages"),
          "full_chain_run_ok": evidence.get("full_chain_run_ok")})
    _add("F2_determinism", bool(evidence.get("deterministic")),
         {"deterministic": evidence.get("deterministic"),
          "n_fingerprints_compared": evidence.get("n_fingerprints_compared")})
    _add("F3_fail_closed", bool(evidence.get("fail_closed_ok")),
         evidence.get("fail_closed_detail"))
    _add("F4_claim_boundary_declared_no_positive_claims",
         bool(evidence.get("claim_boundaries_all_false"))
         and bool(evidence.get("decision_surface_declared")),
         {"n_modules": evidence.get("n_claim_boundary_modules"),
          "n_declared": evidence.get("n_boundary_declared"),
          "n_undeclared": evidence.get("n_boundary_undeclared"),
          "undeclared_modules": evidence.get("boundary_undeclared_modules"),
          "modules_with_positive_claim":
              evidence.get("claim_boundary_offenders"),
          "decision_surface_declared":
              evidence.get("decision_surface_declared"),
          "decision_surface_modules": evidence.get("decision_surface_modules"),
          "specification_note": (
              "本条件要求：① 已声明的 `CLAIM_BOUNDARY` 全为 False；"
              "② **决策面**（loop 输出 + 装配层）声明全 False 边界；"
              "③ 链路内无任何正向宣称。"
              "「每个上游原语模块都在源码内声明边界」是**文档性**要求，"
              "单独登记为**非阻塞**项 N1（见 documented_gaps）。")})
    _add("F5_evidence_grade_internal_only",
         bool(evidence.get("evidence_level_internal_only"))
         and bool(evidence.get("no_external_validation_claims")),
         {"allowed_levels": evidence.get("allowed_evidence_levels"),
          "rejected_levels": evidence.get("rejected_evidence_levels")})
    _add("F6_no_privilege_escalation",
         bool(evidence.get("override_never_used"))
         and float(evidence.get("leaf_coverage_threshold") or 0) == 0.8
         and bool(evidence.get("budget_unchanged")),
         {"override_used_anywhere": evidence.get("override_never_used") is False,
          "leaf_coverage_threshold": evidence.get("leaf_coverage_threshold"),
          "budget": evidence.get("budget")})
    _add("F7_production_isolation",
         bool(evidence.get("production_graph_unchanged"))
         and bool(evidence.get("unwired_layers_still_unwired")),
         {"production_graph": evidence.get("production_graph"),
          "unwired_layers": evidence.get("unwired_layers")})
    _add("F8_decision_semantics_closed",
         bool(evidence.get("semantics_classes_reachable"))
         and bool(evidence.get("evidence_insufficient_reachable")),
         {"classes_with_instances": evidence.get("classes_with_instances"),
          "evidence_insufficient_detail":
              evidence.get("evidence_insufficient_detail")})
    _add("F9_capability_gaps_declared",
         bool(evidence.get("capability_gaps"))
         and bool(evidence.get("gaps_are_explicit")),
         {"n_gaps": len(evidence.get("capability_gaps") or []),
          "gap_codes": evidence.get("capability_gaps")})
    _add("F10_no_unfinished_markers",
         bool(evidence.get("no_unfinished_markers")),
         {"modules_with_markers":
              evidence.get("modules_with_unfinished_markers")})

    passed = [i["id"] for i in items if i["passed"]]
    failed = [i["id"] for i in items if not i["passed"]]
    documented = evidence.get("documented_gaps") or []
    return {
        "present": True,
        "version": CASH_P17_CLOSURE_AUDIT_VERSION,
        "n_conditions": len(items),
        "items": items,
        "passed": passed,
        "failed": failed,
        "all_passed": not failed,
        "documented_gaps": documented,
        "n_documented_gaps": len(documented) if isinstance(documented, list)
        else 0,
        "documented_gaps_are_non_blocking": True,
        "principle": ("冻结条件只覆盖**工程可冻结性**（完整性/决定论/fail-closed/"
                      "边界声明/隔离/语义闭合），**不**覆盖决策质量 —— "
                      "后者是登记在案的能力验证缺口，不因冻结而消失。"
                      "已登记但**无法在冻结内闭合**的文档类缺口单列为 N 项，"
                      "不计入 blocker，也不被隐藏。"),
        "claim_boundary": dict(CLAIM_BOUNDARY),
        "evidence_grade": EVIDENCE_GRADE,
        "unknown": [],
    }


def open_capability_gaps(*, sources: Any) -> dict:
    """汇总仍**未闭合**的能力验证缺口（要求 5、8）。"""
    src = sources if isinstance(sources, dict) else {}
    gaps = []
    for code, detail in sorted(src.items()):
        gaps.append({"code": code, "detail": detail})
    return {
        "present": True,
        "version": CASH_P17_CLOSURE_AUDIT_VERSION,
        "n_open_gaps": len(gaps),
        "open_gaps": gaps,
        "note": ("这些缺口**不阻塞工程冻结**：冻结冻结的是**工程行为**"
                 "（决定论/fail-closed/边界/隔离），"
                 "而能力验证需要**外部证据**（RC-2 语料 / solver 均衡参考 / 真人验证），"
                 "当前均不可得。"),
        "claim_boundary": dict(CLAIM_BOUNDARY),
        "evidence_grade": EVIDENCE_GRADE,
        "unknown": [],
    }


def model_internal_provable() -> dict:
    """`MODEL_INTERNAL` 能证明什么、不能证明什么（要求 8）。"""
    return {
        "present": True,
        "version": CASH_P17_CLOSURE_AUDIT_VERSION,
        "evidence_grade": EVIDENCE_GRADE,
        "provable": list(PROVABLE_WITH_MODEL_INTERNAL),
        "not_provable": list(NOT_PROVABLE_WITH_MODEL_INTERNAL),
        "must_not_be_read_as": ["EV", "GTO", "equilibrium", "solver output",
                                "profit evidence", "win rate",
                                "real-human validation", "strategy strength"],
        "claim_boundary": dict(CLAIM_BOUNDARY),
        "unknown": [],
    }


def freeze_decision(*, checklist: Any, gaps: Any = None,
                    blockers: Any = None) -> dict:
    """给出冻结裁定（要求 11）。"""
    if not isinstance(checklist, dict) or not checklist.get("present"):
        return _stub("checklist_required")
    blk = [b for b in (blockers or []) if b]
    failed = list(checklist.get("failed") or [])
    documented = list(checklist.get("documented_gaps") or [])
    open_gaps = (gaps or {}).get("open_gaps") if isinstance(gaps, dict) else []
    ok = checklist.get("all_passed") and not blk
    return {
        "present": True,
        "version": CASH_P17_CLOSURE_AUDIT_VERSION,
        "decision": ("CASH_CORE_FREEZE_CANDIDATE" if ok else "NOT_READY"),
        "conditions_total": checklist.get("n_conditions"),
        "conditions_passed": len(checklist.get("passed") or []),
        "conditions_failed": failed,
        "blockers": blk,
        "documented_non_blocking_gaps": documented,
        "n_documented_non_blocking_gaps": len(documented),
        "open_capability_gaps_not_blocking": [g.get("code") for g in open_gaps],
        "n_open_capability_gaps": len(open_gaps),
        "recommendation": (
            "建议**冻结**当前 CASH Decision Core 的工程行为："
            "满足最小冻结条件清单，且剩余问题全部是**需要外部证据**的能力验证缺口"
            "或**无法在冻结内闭合**的文档类缺口（已登记、不隐藏）；"
            "继续增加机制不会闭合它们，只会扩大未验证面。"
            if ok else
            "**不建议**冻结：仍有未通过的最小冻结条件或明确 blocker（见上）。"),
        "what_freezing_means": (
            "冻结 = 停止新增策略机制、停止修改链路契约；"
            "能力验证缺口保持登记状态，等待外部证据（RC-2 / solver / 真人验证）。"),
        "what_freezing_does_not_mean": (
            "冻结**不**表示决策质量已被证明；它只表示工程行为已稳定、"
            "边界已声明、fail-closed 已实测。"),
        "claim_boundary": dict(CLAIM_BOUNDARY),
        "evidence_grade": EVIDENCE_GRADE,
        "unknown": [],
    }


def p17_contract() -> dict:
    return {
        "version": CASH_P17_CLOSURE_AUDIT_VERSION,
        "stage": "V3I-P17",
        "purpose": "CASH Decision Core 的**架构收口审计**与冻结条件判定",
        "implements_preflop_unfreeze": False,
        "modifies_search_tree_or_ranking": False,
        "modifies_search_budget": False,
        "relaxes_leaf_coverage_threshold": False,
        "enables_override": False,
        "uses_voting": False,
        "forces_ranking_search_unification": False,
        "connects_coinpoker_or_auto_betting": False,
        "enters_mtt_pko": False,
        "adds_strategy_mechanism": False,
        "reads_only": True,
        "primary_flow": list(PRIMARY_FLOW),
        "freeze_checklist": list(FREEZE_CHECKLIST),
        "final_decision_classes": list(FINAL_DECISION_CLASSES),
        "module_statuses": list(MODULE_STATUSES),
        "unfinished_markers": list(UNFINISHED_MARKERS),
        "imports_project_layers": False,
        "claim_boundary": dict(CLAIM_BOUNDARY),
        "evidence_grade": EVIDENCE_GRADE,
    }
