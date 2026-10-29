"""V3H-15 — CASH 外部信任锚接口契约（只读接入前检查；**不签发、不验证、不执行**）。

**本层要解决的问题**
--------------------
V3H-14 已确认 5 项外部阻塞项（精确解算器与均衡参考、真实执行记录、
B1 独立性测量、真实盈亏证据、带外信任锚），并明确：本项目**不**签发信任锚、
**不**把 SHA-256 内容指纹当作真实性证明。

但那 5 项阻塞项**分别**要求"外部证据"到**什么程度**才算满足，
此前**没有**任何层把这一点写成**机器可读的接口契约**。本层补上这一环：
把"外部信任锚"拆成**六个彼此独立**的断言维度，定义最小 schema、
fail-closed 状态机，并明确**本项目能做到**与**做不到**的边界。

**六个维度彼此独立（关键）**
----------------------------
```
A1 EXTERNAL_RESOURCE_EXISTS        外部资源**存在**
A2 SOURCE_IDENTITY_TRUSTED         来源**身份可信**
A3 RECORD_COMPLETENESS             执行记录**完整**
A4 RECORD_LINKAGE                  记录与本项目决策点**可关联**
A5 RECORD_INTEGRITY                记录**未被篡改**
A6 RECORD_ACTUALLY_OCCURRED        记录**真实发生**
```

**满足任何一项都不得推定其余任何一项。** 例如：
`A5`（内容未被篡改）**不**蕴含 `A6`（真实发生）——
完全可以存在一条内容自洽、指纹正确、但**从未真实发生**的记录。
`A1`（资源存在）**不**蕴含 `A2`（身份可信）。

**SHA-256 的作用边界（必须如实标注）**
--------------------------------------
`SHA-256` 只能表达**内容指纹**；它**不能单独证明**：
真实性、认证、来源、真实发生。它最多支持 `A5` 的**一个分量**，
且需与 `A4` 的关联信息一起才有意义。本层**拒绝**任何
`digest_only` / `digest_as_proof` 形式的断言。

**本项目的能力边界（硬约束）**
------------------------------
* 本层**只**定义 schema / 接口 / 状态机 / fail-closed 校验；
* **不**接入网络、钱包、真实资金、客户端、外部进程、Solver 或真实执行；
* **不**签发、**不**伪造、**不**代填、**不**自行验证真实性；
* `signature_verified` 恒为 `False`、`signature_verification_performed`
  恒为 `False`——本层**不实现**密码学验证，也**不**假装已验签；
* 因此即使六类断言**齐备**，`readiness` 仍为 `False`：
  状态机只给出**结构性**结论，真实性仍需**带外**完成；
* **不**产生任何收益 / GTO / equilibrium / exploitability /
  外部复现 / 执行结论；`result` 恒为 `None`；
* CASH-only，fail closed；**不 import 任何 V3G / V3H 层**；无 IO；无随机。
"""
from __future__ import annotations

from typing import Any

CASH_STRATEGY_TRUST_ANCHOR_CONTRACT_VERSION = (
    "v3h15-cash-strategy-trust-anchor-contract-v1")

#: 六个**独立**断言维度（顺序固定，用于复现与计数）
ASSERTION_CLASSES: tuple[str, ...] = (
    "A1_EXTERNAL_RESOURCE_EXISTS",
    "A2_SOURCE_IDENTITY_TRUSTED",
    "A3_RECORD_COMPLETENESS",
    "A4_RECORD_LINKAGE",
    "A5_RECORD_INTEGRITY",
    "A6_RECORD_ACTUALLY_OCCURRED",
)

#: 每一类的语义与该类**允许**的断言方式（**封闭枚举**）
ASSERTION_CLASS_SPECS: tuple[dict[str, Any], ...] = (
    {
        "class": "A1_EXTERNAL_RESOURCE_EXISTS",
        "question": "does the external resource exist at all?",
        "allowed_attestation_types": [
            "external_party_attestation",
            "resource_inventory_attestation",
            "custodian_attestation",
        ],
        "implied_by_other_classes": [],
        "sha256_alone_is_sufficient": False,
        "note": ("存在性**不**蕴含身份可信（A2），也**不**蕴含记录完整（A3）。"),
    },
    {
        "class": "A2_SOURCE_IDENTITY_TRUSTED",
        "question": "is the identity of the producing source trusted?",
        "allowed_attestation_types": [
            "external_party_attestation",
            "identity_attestation",
            "custodian_attestation",
        ],
        "implied_by_other_classes": [],
        "sha256_alone_is_sufficient": False,
        "note": "身份可信**不**蕴含记录真实发生（A6）。",
    },
    {
        "class": "A3_RECORD_COMPLETENESS",
        "question": "is the execution record complete (no missing fields)?",
        "allowed_attestation_types": [
            "external_party_attestation",
            "completeness_attestation",
        ],
        "implied_by_other_classes": [],
        "sha256_alone_is_sufficient": False,
        "note": ("完整性是**结构**属性；**不**蕴含内容真实发生（A6）。"),
    },
    {
        "class": "A4_RECORD_LINKAGE",
        "question": ("is the record linkable to this project's decision "
                     "points (shadow_record_id / state_hash)?") ,
        "allowed_attestation_types": [
            "external_party_attestation",
            "linkage_attestation",
        ],
        "implied_by_other_classes": [],
        "sha256_alone_is_sufficient": False,
        "note": ("可关联**不**蕴含真实性（A6）；关联本身也可能是伪造的。"),
    },
    {
        "class": "A5_RECORD_INTEGRITY",
        "question": "is the record unmodified since capture?",
        "allowed_attestation_types": [
            "external_party_attestation",
            "integrity_attestation",
            "custodian_attestation",
        ],
        "implied_by_other_classes": [],
        "sha256_alone_is_sufficient": False,
        "sha256_role": ("content fingerprint only - one component at most; "
                        "requires external attestation plus linkage"),
        "note": ("**关键**：内容未被篡改**不**蕴含记录真实发生——"
                 "一条指纹正确、内容自洽的记录完全可能从未真实发生。"),
    },
    {
        "class": "A6_RECORD_ACTUALLY_OCCURRED",
        "question": "did the event actually happen?",
        "allowed_attestation_types": [
            "external_party_attestation",
            "occurrence_attestation",
            "custodian_attestation",
            "notarised_occurrence_attestation",
        ],
        "implied_by_other_classes": [],
        "sha256_alone_is_sufficient": False,
        "note": ("**没有任何**内容指纹可以证明此事；必须由带外主体断言，"
                 "且本项目**不**自行判定其真实性。"),
    },
)

#: 断言记录 schema（**精确**字段集；全部必需）
ATTESTATION_FIELDS: tuple[str, ...] = (
    "attestation_id",         # 外部主体分配，稳定唯一
    "assertion_class",        # 必须是 ASSERTION_CLASSES 之一
    "attestation_type",       # 必须是该类 allowed_attestation_types 之一
    "attesting_party",        # **外部**主体标识
    "attesting_party_kind",   # 必须是 external_*（本项目不得签发）
    "subject_ref",            # 被断言对象（资源 / 记录 / 决策点）
    "subject_content_digest", # 可选：内容指纹（**不**构成证明）
    "issued_at_epoch_ms",     # 外部给出的时间戳（本层不校验时钟）
    "signature_ref",          # 指向**外部**签名/验证材料
    "signature_scheme",       # 签名方案名（本层**不**实现验证）
    "revocation_ref",         # 撤销/失效引用（允许为 None）
    "scope_note",             # 外部说明文字
)

#: 记录级拒绝原因（封闭枚举）
ATTESTATION_REFUSAL_REASONS: tuple[str, ...] = (
    "attestation_not_a_dict",
    "missing_required_field",
    "attestation_id_invalid",
    "assertion_class_not_in_closed_enum",
    "attestation_type_invalid_for_class",
    "attesting_party_invalid",
    "attesting_party_kind_not_external",
    "attesting_party_is_this_project",
    "subject_ref_invalid",
    "issued_at_epoch_ms_invalid",
    "signature_ref_missing",
    "signature_scheme_missing",
    "scope_note_invalid",
    "digest_only_attestation",
    "sha256_alone_claimed_sufficient",
    "claims_imply_other_classes",
    "self_issued_attestation",
    "attestation_duplicate",
)

#: 锚状态机状态（封闭枚举）
ANCHOR_STATES: tuple[str, ...] = (
    "ANCHOR_UNSATISFIED",                  # 无任何结构合法断言
    "ANCHOR_STRUCTURE_VALID",              # 有结构合法断言，但六类不齐
    "ANCHOR_CONTENT_CONSISTENT",           # 六类齐备且互不代偿
    "ANCHOR_REFUSED_DIGEST_AS_PROOF",      # 仅以内容指纹充当证明
    "ANCHOR_REFUSED_SELF_ISSUED",          # 由本项目/内部主体签发
    "ANCHOR_REFUSED_CLASS_MERGE",          # 声称一类断言蕴含另一类
)

#: **明确禁止**的信任假设（机器可读；本层与测试共同固定）
FORBIDDEN_TRUST_ASSUMPTIONS: tuple[str, ...] = (
    "digest_as_proof",
    "digest_only_attestation",
    "self_issued_trust_anchor",
    "internal_attester",
    "class_merge",
    "mock_solver",
    "approximated_equilibrium",
    "declared_exploitability",
    "model_internal_as_external",
    "synthetic_execution",
    "simulated_execution",
    "generated_profit",
    "inferred_realised_outcome",
)

#: 本项目**绝不能**做的事（用于审计与测试固定）
PROJECT_CANNOT_DO: tuple[str, ...] = (
    "issue_an_attestation",
    "fabricate_a_trust_anchor",
    "prefill_attestation_fields",
    "verify_authenticity_itself",
    "verify_signatures_itself",
    "treat_sha256_as_authenticity_proof",
    "declare_an_external_resource_satisfied",
    "produce_execution_or_profit_or_statistical_results",
    "access_network_wallet_client_or_external_process",
    "invoke_a_solver",
)

#: 本项目**能**做的（诚实列出，避免夸大或自我否定）
PROJECT_CAN_DO: tuple[str, ...] = (
    "define_the_attestation_schema",
    "validate_attestation_structure_fail_closed",
    "track_the_six_classes_independently",
    "refuse_digest_only_and_self_issued_attestations",
    "expose_a_state_machine_of_structural_progress",
    "report_that_signature_verification_is_not_implemented_here",
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
    "issues_attestations": False,
    "fabricates_trust_anchor": False,
    "performs_signature_verification": False,
    "verifies_authenticity_itself": False,
    "declares_external_resource_satisfied": False,
    "merges_assertion_classes": False,
    "computes_rates_or_statistics": False,
    "adds_scoring_or_ev_formula": False,
    "generates_execution_or_profit_or_statistical_results": False,
    "executes_action": False,
    "auto_betting": False,
    "claims_experiment_ready": False,
    "wired_to_production": False,
}
CLAIM_FLAGS_TOTAL = len(CLAIM_BOUNDARY)

#: 内部/本项目主体标识（出现即拒绝）
INTERNAL_PARTY_MARKERS: tuple[str, ...] = (
    "this_project", "self", "internal", "poker_decision_ai", "v3h", "local",
    "in_repo", "same_project",
)
#: 外部主体类别前缀（必须匹配其一）
EXTERNAL_PARTY_KINDS: tuple[str, ...] = (
    "external_party", "external_custodian", "external_notary",
    "external_auditor",
)


def _is_cash(state: Any) -> bool:
    if not isinstance(state, dict):
        return False
    gt = state.get("game_type")
    return isinstance(gt, str) and gt.strip().lower() == "cash"


def _is_int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _norm_str(v: Any) -> str | None:
    if isinstance(v, str) and v.strip():
        return v.strip()
    return None


def _stub(reason: str) -> dict:
    return {
        "present": False,
        "version": CASH_STRATEGY_TRUST_ANCHOR_CONTRACT_VERSION,
        "reason": reason,
        "state": "ANCHOR_UNSATISFIED",
        "readiness": False,
        "structural_fully_anchored": False,
        "n_attestations": 0,
        "classes_satisfied": [],
        "classes_missing": list(ASSERTION_CLASSES),
        "evidence_level": "UNAVAILABLE",
        "evidence_grade": "NONE",
        "executed": False,
        "result": None,
        "results_available": False,
        "analysis_executed": False,
        "claim_boundary": dict(CLAIM_BOUNDARY),
        "unknown": [reason],
    }


# ══ 1. 契约与 schema（纯声明；无 IO、无结果） ═════════════════════════
def trust_anchor_contract() -> dict:
    """返回**信任锚接口契约**（六类独立断言、禁止假设、能力边界）。"""
    return {
        "success": True,
        "version": CASH_STRATEGY_TRUST_ANCHOR_CONTRACT_VERSION,
        "issuer_of_this_contract": "this_project",
        "issuer_of_attestations": "must_be_external",
        "attestations_issued_by_this_project": False,
        "assertion_classes": list(ASSERTION_CLASSES),
        "n_assertion_classes": len(ASSERTION_CLASSES),
        "class_specs": [dict(s) for s in ASSERTION_CLASS_SPECS],
        "independence_rule": (
            "assertion classes are mutually independent: satisfying any "
            "subset never implies any other class"),
        "no_class_implication_matrix": {
            s["class"]: [] for s in ASSERTION_CLASS_SPECS
        },
        "sha256_role": {
            "expresses": "content_fingerprint_only",
            "can_support": ["A5_RECORD_INTEGRITY (one component at most)"],
            "cannot_prove": ["authenticity", "certification", "provenance",
                             "source_identity", "actual_occurrence",
                             "completeness", "linkage"],
            "sha256_alone_is_sufficient_for_any_class": False,
            "digest_only_attestations_are_refused": True,
        },
        "states": list(ANCHOR_STATES),
        "forbidden_trust_assumptions": list(FORBIDDEN_TRUST_ASSUMPTIONS),
        "project_cannot_do": list(PROJECT_CANNOT_DO),
        "project_can_do": list(PROJECT_CAN_DO),
        "signature_verification": {
            "implemented_here": False,
            "performed_here": False,
            "must_be_done_out_of_band": True,
            "note": ("本层**不**实现、**不**执行、**不**假装密码学验签；"
                     "验签必须带外完成。"),
        },
        "evidence_level": "INTERNAL_DECLARATION",
        "evidence_grade": "NONE",
        "executed": False,
        "result": None,
        "results_available": False,
        "analysis_executed": False,
        "claim_boundary": dict(CLAIM_BOUNDARY),
        "unknown": [],
        "note": ("契约本身是**声明**：定义外部断言需要满足什么；"
                 "**不**声称任何外部资源已存在、可信或真实发生。"),
    }


def attestation_schema() -> dict:
    """返回断言记录 schema（**声明性**；无任何预填值）。"""
    return {
        "success": True,
        "version": CASH_STRATEGY_TRUST_ANCHOR_CONTRACT_VERSION,
        "fields": list(ATTESTATION_FIELDS),
        "n_fields": len(ATTESTATION_FIELDS),
        "required_fields": list(ATTESTATION_FIELDS),
        "field_kinds": {
            "attestation_id": "non_empty_str_unique",
            "assertion_class": "one_of_assertion_classes",
            "attestation_type": "one_of_allowed_types_for_that_class",
            "attesting_party": "non_empty_str_not_internal",
            "attesting_party_kind": "one_of_external_party_kinds",
            "subject_ref": "non_empty_str",
            "subject_content_digest": "str|None (fingerprint only)",
            "issued_at_epoch_ms": "int (validated for type only, not clock)",
            "signature_ref": "non_empty_str_pointing_outside_this_project",
            "signature_scheme": "non_empty_str",
            "revocation_ref": "str|None",
            "scope_note": "non_empty_str",
        },
        "prefilled_values": [],
        "example_provided": False,
        "external_party_kinds": list(EXTERNAL_PARTY_KINDS),
        "internal_party_markers_rejected": list(INTERNAL_PARTY_MARKERS),
        "refusal_reasons": list(ATTESTATION_REFUSAL_REASONS),
        "result": None,
        "executed": False,
        "evidence_level": "INTERNAL_DECLARATION",
        "evidence_grade": "NONE",
        "claim_boundary": dict(CLAIM_BOUNDARY),
        "unknown": [],
    }


# ══ 2. 断言记录校验（结构 + fail-closed；**不**验签、**不**验真） ═════
def _refuse_attestation(att: Any) -> list[str]:
    if not isinstance(att, dict):
        return ["attestation_not_a_dict"]
    issues: list[str] = []
    for f in ATTESTATION_FIELDS:
        if f not in att:
            issues.append(f"missing_required_field:{f}")
    if issues:
        return issues

    if _norm_str(att.get("attestation_id")) is None:
        issues.append("attestation_id_invalid")

    cls = _norm_str(att.get("assertion_class"))
    spec = next((s for s in ASSERTION_CLASS_SPECS if s["class"] == cls), None)
    if spec is None:
        issues.append("assertion_class_not_in_closed_enum")
    else:
        atype = _norm_str(att.get("attestation_type"))
        if atype not in spec["allowed_attestation_types"]:
            issues.append("attestation_type_invalid_for_class")

    party = _norm_str(att.get("attesting_party"))
    if party is None:
        issues.append("attesting_party_invalid")
    elif any(m in party.lower() for m in INTERNAL_PARTY_MARKERS):
        issues.append("attesting_party_is_this_project")
        issues.append("self_issued_attestation")

    kind = _norm_str(att.get("attesting_party_kind"))
    if kind is None or not any(kind.lower().startswith(p) for p in
                               EXTERNAL_PARTY_KINDS):
        issues.append("attesting_party_kind_not_external")
        if "self_issued_attestation" not in issues:
            issues.append("self_issued_attestation")

    if _norm_str(att.get("subject_ref")) is None:
        issues.append("subject_ref_invalid")

    if not _is_int(att.get("issued_at_epoch_ms")):
        issues.append("issued_at_epoch_ms_invalid")

    if _norm_str(att.get("signature_ref")) is None:
        issues.append("signature_ref_missing")
    if _norm_str(att.get("signature_scheme")) is None:
        issues.append("signature_scheme_missing")
    if _norm_str(att.get("scope_note")) is None:
        issues.append("scope_note_invalid")

    # digest-only：只有内容指纹、没有外部断言主体/签名支撑
    if _norm_str(att.get("subject_content_digest")) is not None \
            and _norm_str(att.get("signature_ref")) is None:
        issues.append("digest_only_attestation")

    # 显式声称"仅凭 sha256 即可" → 拒绝
    if att.get("sha256_alone_sufficient") is True:
        issues.append("sha256_alone_claimed_sufficient")

    # 声称一类断言蕴含其他类 → 拒绝（必须逐类独立提供）
    implies = att.get("implies_classes")
    if isinstance(implies, (list, tuple)) and len(implies) > 0:
        issues.append("claims_imply_other_classes")

    return issues


def validate_attestation(state: Any, attestation: Any = None) -> dict:
    """**只做结构校验**：不验签、不验真、不执行、不统计。"""
    if not _is_cash(state):
        return _stub("not_cash_mode_fail_closed")
    if not isinstance(attestation, dict) or not attestation:
        return _stub("attestation_required")

    issues = _refuse_attestation(attestation)
    return {
        "present": True,
        "version": CASH_STRATEGY_TRUST_ANCHOR_CONTRACT_VERSION,
        "mode": "cash",
        "valid": not issues,
        "issues": issues,
        "n_issues": len(issues),
        "assertion_class": (_norm_str(attestation.get("assertion_class"))
                            if not issues else None),
        "signature_verified": False,
        "signature_verification_performed": False,
        "authenticity_verified": False,
        "authenticity_verification_performed": False,
        "digest_treated_as_proof": False,
        "evidence_level": ("EXTERNAL_ATTESTATION_STRUCTURE_VALID" if not issues
                           else "UNAVAILABLE"),
        "evidence_grade": "NONE",
        "executed": False,
        "result": None,
        "results_available": False,
        "analysis_executed": False,
        "claim_boundary": dict(CLAIM_BOUNDARY),
        "unknown": list(issues),
        "note": ("仅**结构**校验：本层不验签、不验真；"
                 "结构合法 ≠ 真实性成立。"),
    }


# ══ 3. 锚状态机（fail-closed；只给**结构性**结论） ════════════════════
def evaluate_anchor_state(state: Any, *, attestations: Any = None) -> dict:
    """在外部断言集合上运行 fail-closed 状态机（**不验证真实性**）。

    Returns
    -------
    dict
        `state` / `classes_satisfied` / `classes_missing` /
        `structural_fully_anchored` / `readiness`（**恒为 `False`**）/
        逐条拒绝原因 / `result = None`。
    """
    if not _is_cash(state):
        return _stub("not_cash_mode_fail_closed")
    if not isinstance(attestations, (list, tuple)):
        return _stub("attestations_required")

    per_attestation: list[dict] = []
    satisfied: set[str] = set()
    refusal_tokens: set[str] = set()
    seen_ids: dict[str, int] = {}
    n_refused = 0

    for i, att in enumerate(attestations):
        issues = _refuse_attestation(att)
        aid = _norm_str(att.get("attestation_id")) \
            if isinstance(att, dict) else None
        if aid is not None:
            if aid in seen_ids:
                issues = issues + ["attestation_duplicate"]
            else:
                seen_ids[aid] = i
        for x in issues:
            refusal_tokens.add(x.split(":", 1)[0])
        ok = not issues
        if not ok:
            n_refused += 1
        else:
            cls = _norm_str(att.get("assertion_class"))
            if cls in ASSERTION_CLASSES:
                satisfied.add(cls)
        per_attestation.append({
            "index": i, "attestation_id": aid,
            "assertion_class": (_norm_str(att.get("assertion_class"))
                                if isinstance(att, dict) else None),
            "structure_valid": ok, "issues": issues,
        })

    classes_satisfied = [c for c in ASSERTION_CLASSES if c in satisfied]
    classes_missing = [c for c in ASSERTION_CLASSES if c not in satisfied]

    # ── 状态判定（拒绝态优先，确保 fail closed）──
    if "digest_only_attestation" in refusal_tokens \
            or "sha256_alone_claimed_sufficient" in refusal_tokens:
        anchor_state = "ANCHOR_REFUSED_DIGEST_AS_PROOF"
    elif "self_issued_attestation" in refusal_tokens \
            or "attesting_party_is_this_project" in refusal_tokens \
            or "attesting_party_kind_not_external" in refusal_tokens:
        anchor_state = "ANCHOR_REFUSED_SELF_ISSUED"
    elif "claims_imply_other_classes" in refusal_tokens:
        anchor_state = "ANCHOR_REFUSED_CLASS_MERGE"
    elif not classes_satisfied:
        anchor_state = "ANCHOR_UNSATISFIED"
    elif classes_missing:
        anchor_state = "ANCHOR_STRUCTURE_VALID"
    else:
        anchor_state = "ANCHOR_CONTENT_CONSISTENT"

    structural_fully_anchored = (anchor_state == "ANCHOR_CONTENT_CONSISTENT")
    n_valid = len(attestations) - n_refused

    return {
        "present": True,
        "version": CASH_STRATEGY_TRUST_ANCHOR_CONTRACT_VERSION,
        "mode": "cash",
        "state": anchor_state,
        "states": list(ANCHOR_STATES),
        "structural_fully_anchored": structural_fully_anchored,
        "readiness": False,
        "readiness_reason": (
            "signature and authenticity verification are NOT implemented or "
            "performed by this project; structural completeness is not "
            "authenticity and must be completed out of band"),
        "n_attestations": len(attestations),
        "n_structure_valid": n_valid,
        "n_refused": n_refused,
        "classes_satisfied": classes_satisfied,
        "classes_missing": classes_missing,
        "class_satisfaction_counts": {
            c: (1 if c in satisfied else 0) for c in ASSERTION_CLASSES},
        "per_attestation": per_attestation,
        "refusal_tokens": sorted(refusal_tokens),
        "forbidden_trust_assumptions": list(FORBIDDEN_TRUST_ASSUMPTIONS),
        "assumptions_used": [],
        "digest_used_as_proof": False,
        "attestations_issued_by_this_project": False,
        "signature_verified": False,
        "signature_verification_performed": False,
        "authenticity_verified": False,
        "authenticity_verification_performed": False,
        "external_resources_declared_satisfied": [],
        "dependency_manifest_consumed": False,
        "v3h14_verdict_not_recomputed": True,
        "result": None,
        "results_available": False,
        "analysis_executed": False,
        "executed": False,
        "evidence_level": ("EXTERNAL_ATTESTATION_STRUCTURE_VALID"
                           if n_valid else "UNAVAILABLE"),
        "evidence_grade": "NONE",
        "uncertainty": {
            "measured": False,
            "note": ("未执行任何动作、未运行任何分析 → **无**统计不确定性；"
                     "此处只记录**结构性**进度。"),
            "unverified_by_this_layer": ["signature", "authenticity",
                                         "source_trust", "actual_occurrence"],
        },
        "claim_boundary": dict(CLAIM_BOUNDARY),
        "unknown": sorted(refusal_tokens),
        "note": ("状态机只给**结构性**结论：结构齐备**不等于**真实性成立；"
                 "`readiness` **恒为 False**。"),
    }


# ══ 4. 接入前检查清单（只读；不消费 V3H-14 判定） ═════════════════════
def pre_intake_checklist(state: Any, *, anchor_state: Any = None) -> dict:
    """列出"接入真实记录前"必须由**外部**完成的检查项及其当前状态。

    **本层不判定**任何外部资源已满足，也**不重算** V3H-14 的结论。
    """
    if not _is_cash(state):
        return _stub("not_cash_mode_fail_closed")
    if not isinstance(anchor_state, dict) or \
            anchor_state.get("present") is not True:
        return _stub("anchor_state_required")

    satisfied = set(anchor_state.get("classes_satisfied") or [])
    items = [{
        "id": f"CHK-{c.split('_')[0]}",
        "assertion_class": c,
        "must_be_done_by": "external_party",
        "done_by_this_project": False,
        "satisfied_by_external_attestation": c in satisfied,
        "blocking": True,
    } for c in ASSERTION_CLASSES]
    n_done = sum(1 for it in items
                 if it["satisfied_by_external_attestation"])

    return {
        "present": True,
        "version": CASH_STRATEGY_TRUST_ANCHOR_CONTRACT_VERSION,
        "mode": "cash",
        "n_checks": len(items),
        "n_satisfied": n_done,
        "n_outstanding": len(items) - n_done,
        "items": items,
        "anchor_state": anchor_state.get("state"),
        "all_checks_satisfied": n_done == len(items),
        "ready_for_real_intake": False,
        "ready_for_real_intake_reason": (
            "even with all structural checks satisfied, signature and "
            "authenticity verification must be completed out of band; "
            "this project never declares an external resource satisfied"),
        "external_resources_declared_satisfied_by_this_layer": [],
        "network_wallet_client_external_process_or_solver_used": False,
        "result": None,
        "results_available": False,
        "analysis_executed": False,
        "executed": False,
        "evidence_level": "INTERNAL_DECLARATION",
        "evidence_grade": "NONE",
        "claim_boundary": dict(CLAIM_BOUNDARY),
        "unknown": ([] if n_done == len(items)
                    else [f"outstanding:{c}" for c in ASSERTION_CLASSES
                          if c not in satisfied]),
        "note": ("清单是**只读**接入前检查：全部为 `blocking`，"
                 "`ready_for_real_intake` **恒为 False**。"),
    }
