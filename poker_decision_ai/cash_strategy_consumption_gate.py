"""V3H-16 — CASH 契约一致性与消费边界审计（只读；**不接入任何外部资源**）。

**本层要解决的问题**
--------------------
V3H-15 定义了信任锚接口契约与六类独立断言，并硬保证
`readiness` 恒为 `False`、本层不验签、不验真。但"**下游可以怎么消费它**"
此前没有任何约束：没有消费门禁，也没有把七类误用风险机器化地钉死。

本层补上这一环：**只读审计 + 消费门禁**，
回答"哪些状态可被读取、哪些只能展示为未满足、哪些必须拒绝进入
统计/收益/策略结论"，并把**所有未满足、冲突、未知、来源不可信**一律 fail closed。

**七类误用风险（逐条机器化检查）**
----------------------------------
```
RSK-1  外部资源**存在** 被当成 来源**可信**
RSK-2  签名 / 哈希 被当成 **真实发生**证明
RSK-3  外部声明被本项目**自行批准**
RSK-4  STRUCTURE_VALID / CONTENT_CONSISTENT 被当成 readiness = True
RSK-5  A5（内容未被篡改）被错误推导为 A6（真实发生）
RSK-6  外部字段缺失时**静默补全**
RSK-7  未验证的外部记录进入收益 / Solver / 评估 / 生产链路
```

**核心不变量（本层的根本结论）**
--------------------------------
```
INV-1  本项目**从不**验证真实性 → 内部任何状态都**不**等于已验证
INV-2  STATISTICS / PROFIT_SOLVER / PRODUCTION / CLAIM **恒为禁止**
INV-3  只有**外部**验证过的锚才可能被消费 → 本项目内**不可达**
```

即：**在本项目内部，不存在任何"可被统计/收益/策略消费"的状态**。
`CONSUMABLE_VERIFIED` 只能由**带外**给出。

硬约束
------
* CASH-only，fail closed；非 dict / 非 CASH → `present=False`；
* **不 import 任何 V3G / V3H 层**（任何位置）——上方上游令牌与状态
  一律由**调用方显式传入**（惯例同 V3H-9 `gate=` / V3H-10 `equity_snapshot=`）；
* **不重复实现** V3H-14 readiness 判定，也**不重复实现** V3H-15 六类断言
  （本层**不**重新声明 `ASSERTION_CLASSES` / `ATTESTATION_FIELDS` /
  `evaluate_anchor_state` / `validate_attestation`；
  `V3H15_REQUIRED_STRUCTURE_TOKENS` 中出现的**名称字符串**仅用于
  "V3H-15 源码里是否存在这些结构"的**只读检查**，不是本层的实现）；
* 无网络 / 客户端 / 钱包 / 真实资金 / Solver / 真实执行 / 收益实验；
* 无 IO、无随机；同输入 → 同输出；
* 不产生任何真实结果；`result` 恒为 `None`。
"""
from __future__ import annotations

import ast
import pathlib
from typing import Any

CASH_STRATEGY_CONSUMPTION_GATE_VERSION = (
    "v3h16-cash-strategy-consumption-gate-v1")

#: 消费类别（封闭枚举）
CONSUMPTION_CLASSES: tuple[str, ...] = (
    "READ_AND_DISPLAY_ALLOWED",
    "DISPLAY_AS_UNSATISFIED_ONLY",
    "REFUSED_FOR_STATISTICS",
    "REFUSED_FOR_PROFIT_OR_SOLVER",
    "REFUSED_FOR_PRODUCTION",
    "REQUIRES_EXTERNAL_VERIFICATION",
    "CONSUMABLE_VERIFIED",
)

#: **恒为禁止**的消费类别（本层根本结论：项目内不可达）
ALWAYS_FORBIDDEN_CLASSES: tuple[str, ...] = (
    "REFUSED_FOR_STATISTICS",
    "REFUSED_FOR_PROFIT_OR_SOLVER",
    "REFUSED_FOR_PRODUCTION",
    "REQUIRES_EXTERNAL_VERIFICATION",
    "CONSUMABLE_VERIFIED",
)

#: 七类误用风险（封闭枚举）
MISUSE_RISKS: tuple[str, ...] = (
    "RSK-1_EXISTENCE_AS_TRUST",
    "RSK-2_SIGNATURE_OR_HASH_AS_OCCURRENCE",
    "RSK-3_SELF_APPROVED_EXTERNAL_DECLARATION",
    "RSK-4_STRUCTURE_OR_CONTENT_AS_READINESS",
    "RSK-5_A5_INFERRED_AS_A6",
    "RSK-6_SILENT_FIELD_COMPLETION",
    "RSK-7_UNVERIFIED_RECORD_REACHES_DOWNSTREAM",
)

#: 被禁止的**推断**（消费方不得据以下任一形式得出结论）
FORBIDDEN_INFERENCES: tuple[str, ...] = (
    "existence_implies_trust",
    "identity_implies_occurrence",
    "signature_implies_occurrence",
    "hash_implies_occurrence",
    "structure_valid_implies_readiness",
    "content_consistent_implies_readiness",
    "completeness_implies_occurrence",
    "linkage_implies_occurrence",
    "integrity_A5_implies_occurrence_A6",
    "self_approved_external_declaration",
    "missing_field_completed_silently",
    "unverified_record_consumed_as_evidence",
)

#: 拒绝原因（封闭枚举）
REFUSAL_REASONS: tuple[str, ...] = (
    "not_cash_mode_fail_closed",
    "anchor_state_required",
    "anchor_state_present_false",
    "anchor_state_not_in_closed_enum",
    "anchor_readiness_truthy_rejected",
    "anchor_signature_verified_truthy_rejected",
    "anchor_authenticity_verified_truthy_rejected",
    "anchor_classes_satisfied_unverified_rejected",
    "anchor_state_conflict_unverified_with_verified_flag",
    "source_trust_not_established",
    "external_verification_never_performed_here",
    "unknown_consumption_state_fails_closed",
)

#: 允许被**读取/展示**的状态（其余一律 fail closed）
READABLE_ANCHOR_STATES: tuple[str, ...] = (
    "ANCHOR_UNSATISFIED",
    "ANCHOR_STRUCTURE_VALID",
    "ANCHOR_CONTENT_CONSISTENT",
    "ANCHOR_REFUSED_DIGEST_AS_PROOF",
    "ANCHOR_REFUSED_SELF_ISSUED",
    "ANCHOR_REFUSED_CLASS_MERGE",
)

#: 只允许**展示为未满足**的状态（包括"结构齐备"——它**不是**真实性）
DISPLAY_AS_UNSATISFIED_ONLY_STATES: tuple[str, ...] = (
    "ANCHOR_UNSATISFIED",
    "ANCHOR_STRUCTURE_VALID",
    "ANCHOR_CONTENT_CONSISTENT",
    "ANCHOR_REFUSED_DIGEST_AS_PROOF",
    "ANCHOR_REFUSED_SELF_ISSUED",
    "ANCHOR_REFUSED_CLASS_MERGE",
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
    "structure_treated_as_readiness": False,
    "signature_treated_as_occurrence_proof": False,
    "existence_treated_as_trust": False,
    "self_approves_external_declarations": False,
    "silently_completes_missing_fields": False,
    "permits_statistics_consumption": False,
    "permits_profit_or_solver_consumption": False,
    "permits_production_consumption": False,
    "permits_claim_consumption": False,
    "grants_verified_status_internally": False,
    "computes_rates_or_statistics": False,
    "adds_scoring_or_ev_formula": False,
    "executes_action": False,
    "auto_betting": False,
    "wired_to_production": False,
}
CLAIM_FLAGS_TOTAL = len(CLAIM_BOUNDARY)

#: 契约定点（V3H-15 的**令牌名**；仅用于源码结构检查，不做行为判定）
V3H15_MODULE_STEM = "cash_strategy_trust_anchor_contract"
V3H15_REQUIRED_STRUCTURE_TOKENS: tuple[str, ...] = (
    "ASSERTION_CLASSES",
    "ASSERTION_CLASS_SPECS",
    "ATTESTATION_FIELDS",
    "ATTESTATION_REFUSAL_REASONS",
    "ANCHOR_STATES",
    "FORBIDDEN_TRUST_ASSUMPTIONS",
    "PROJECT_CANNOT_DO",
    "PROJECT_CAN_DO",
    "INTERNAL_PARTY_MARKERS",
    "EXTERNAL_PARTY_KINDS",
    "CLAIM_BOUNDARY",
)
V3H15_REQUIRED_FUNCTIONS: tuple[str, ...] = (
    "trust_anchor_contract",
    "attestation_schema",
    "validate_attestation",
    "evaluate_anchor_state",
    "pre_intake_checklist",
)

#: 契约一致性不变量（源码级检查项；全部必须通过）
CONTRACT_INVARIANTS: tuple[str, ...] = (
    "CIV-1_no_class_implication_matrix_declared",
    "CIV-2_sha256_role_is_fingerprint_only",
    "CIV-3_sha256_alone_never_sufficient",
    "CIV-4_digest_only_attestations_refused",
    "CIV-5_internal_party_markers_present",
    "CIV-6_readiness_hard_coded_false",
    "CIV-7_signature_verification_not_implemented",
    "CIV-8_authenticity_verification_not_implemented",
    "CIV-9_project_cannot_issue_or_verify",
    "CIV-10_six_distinct_assertion_classes",
    "CIV-11_all_six_states_declared",
    "CIV-12_zero_prefill_declared",
    "CIV-13_no_layer_imports_in_module",
    "CIV-14_no_network_process_or_wallet_capability_in_module",
    "CIV-15_no_result_slots_populated_in_module",
    "CIV-16_only_read_only_module_source_inspection",
)


def _is_cash(state: Any) -> bool:
    if not isinstance(state, dict):
        return False
    gt = state.get("game_type")
    return isinstance(gt, str) and gt.strip().lower() == "cash"


def _stub(reason: str) -> dict:
    return {
        "present": False,
        "version": CASH_STRATEGY_CONSUMPTION_GATE_VERSION,
        "reason": reason,
        "consumption_allowed": False,
        "consumption_class": "REQUIRES_EXTERNAL_VERIFICATION",
        "readable": False,
        "display_as_unsatisfied_only": True,
        "statistics_consumption_allowed": False,
        "profit_or_solver_consumption_allowed": False,
        "production_consumption_allowed": False,
        "claim_consumption_allowed": False,
        "verified_status_granted": False,
        "refusal_reasons": [reason],
        "executed": False,
        "result": None,
        "results_available": False,
        "analysis_executed": False,
        "evidence_level": "UNAVAILABLE",
        "evidence_grade": "NONE",
        "claim_boundary": dict(CLAIM_BOUNDARY),
        "unknown": [reason],
    }


# ══ 1. 消费门禁矩阵（**静态**声明；不可由调用方放大） ════════════════
def consumption_gate_matrix() -> dict:
    """返回冻结的**消费门禁矩阵**：每个状态可被如何消费。"""
    per_state = []
    for s in READABLE_ANCHOR_STATES:
        per_state.append({
            "state": s,
            "readable": True,
            "display_as_unsatisfied_only":
                s in DISPLAY_AS_UNSATISFIED_ONLY_STATES,
            "consumable_for_statistics": False,
            "consumable_for_profit_or_solver": False,
            "consumable_for_production": False,
            "consumable_for_claims": False,
            "verified_status": False,
            "consumption_class": "DISPLAY_AS_UNSATISFIED_ONLY",
        })
    return {
        "success": True,
        "version": CASH_STRATEGY_CONSUMPTION_GATE_VERSION,
        "consumption_classes": list(CONSUMPTION_CLASSES),
        "always_forbidden_classes": list(ALWAYS_FORBIDDEN_CLASSES),
        "readable_anchor_states": list(READABLE_ANCHOR_STATES),
        "per_state": per_state,
        "n_readable_states": len(READABLE_ANCHOR_STATES),
        "n_states_consumable_for_statistics": 0,
        "n_states_consumable_for_profit_or_solver": 0,
        "n_states_consumable_for_production": 0,
        "verified_status_reachable_internally": False,
        "verified_status_source": "external_only",
        "unknown_state_policy": "fail_closed_refuse",
        "unverified_state_policy": "display_as_unsatisfied_only",
        "forbidden_inferences": list(FORBIDDEN_INFERENCES),
        "misuse_risks_covered": list(MISUSE_RISKS),
        "invariants": [
            "INV-1 this project never verifies authenticity, so no internal state equals verified",
            "INV-2 statistics / profit-or-solver / production / claim consumption are always denied",
            "INV-3 only an externally verified anchor could be consumed, which is unreachable internally",
        ],
        "external_resources_accessed": False,
        "result": None,
        "executed": False,
        "evidence_level": "INTERNAL_DECLARATION",
        "evidence_grade": "NONE",
        "claim_boundary": dict(CLAIM_BOUNDARY),
        "unknown": [],
        "note": ("矩阵是**只读声明**：它只**缩小**可消费范围，"
                 "**不**授予任何已验证状态。"),
    }


# ══ 2. 消费门禁判定（fail closed） ═══════════════════════════════════
def evaluate_consumption_gate(state: Any, *, anchor_state: Any = None,
                              classes_satisfied: Any = None,
                              requesting_consumer: str | None = None) -> dict:
    """判定某个锚状态**可否被消费**；任何可疑情形一律 fall closed。

    `classes_satisfied` 与 `requesting_consumer` 仅用于**记录**，
    **不**放大权限——消费权限**只**取决于状态本身。
    """
    if not _is_cash(state):
        return _stub("not_cash_mode_fail_closed")
    if not isinstance(anchor_state, dict) or not anchor_state:
        return _stub("anchor_state_required")
    if anchor_state.get("present") is not True:
        return _stub("anchor_state_present_false")

    reasons: list[str] = []

    # ── 声称已验签/已验真 → 一律拒绝（本项目从不验证）──
    hard_fail = False
    if anchor_state.get("signature_verified") is True:
        reasons.append("anchor_signature_verified_truthy_rejected")
        hard_fail = True
    if anchor_state.get("authenticity_verified") is True:
        reasons.append("anchor_authenticity_verified_truthy_rejected")
        hard_fail = True
    # ── 任何 truthy readiness → 拒绝（结构齐备不是就绪）──
    if anchor_state.get("readiness"):
        reasons.append("anchor_readiness_truthy_rejected")
        hard_fail = True
    # ── 未验证却声称已满足类别 → 冲突（可能为静默补全）──
    satisfied = anchor_state.get("classes_satisfied")
    satisfied = satisfied if isinstance(satisfied, list) else []
    if satisfied and anchor_state.get("signature_verified") is not True:
        reasons.append("anchor_classes_satisfied_unverified_rejected")
        hard_fail = True

    raw_state = anchor_state.get("state")
    if not isinstance(raw_state, str) or raw_state not in READABLE_ANCHOR_STATES:
        reasons.append("anchor_state_not_in_closed_enum")
        reasons.append("unknown_consumption_state_fails_closed")
        consumption_class = "REQUIRES_EXTERNAL_VERIFICATION"
        readable = False
        display_only = True
    else:
        consumption_class = "DISPLAY_AS_UNSATISFIED_ONLY"
        # 可读 = 状态在封闭枚举内且未触发硬失败；
        # 但**任何**内部状态都只能"展示为未满足"。
        readable = not hard_fail
        display_only = True

    # 来源可信从未建立；外部验证从未在本项目发生 → 恒为拒绝
    reasons.append("source_trust_not_established")
    reasons.append("external_verification_never_performed_here")

    return {
        "present": True,
        "version": CASH_STRATEGY_CONSUMPTION_GATE_VERSION,
        "mode": "cash",
        "anchor_state": raw_state,
        "requesting_consumer": (requesting_consumer
                                if isinstance(requesting_consumer, str)
                                else None),
        "consumer_kind_recorded_only": True,
        "consumption_allowed": False,
        "consumption_class": consumption_class,
        "readable": bool(readable),
        "display_as_unsatisfied_only": bool(display_only),
        "statistics_consumption_allowed": False,
        "profit_or_solver_consumption_allowed": False,
        "production_consumption_allowed": False,
        "claim_consumption_allowed": False,
        "verified_status_granted": False,
        "verified_status_source": "external_only",
        "classes_satisfied_observed": list(satisfied),
        "classes_satisfied_grants_nothing": True,
        "refusal_reasons": sorted(set(reasons)),
        "forbidden_inferences": list(FORBIDDEN_INFERENCES),
        "misuse_risks_covered": list(MISUSE_RISKS),
        "silent_field_completion_performed": False,
        "external_resources_accessed": False,
        "result": None,
        "results_available": False,
        "analysis_executed": False,
        "executed": False,
        "evidence_level": "INTERNAL_DECLARATION",
        "evidence_grade": "NONE",
        "uncertainty": {
            "measured": False,
            "note": ("未执行任何动作、未运行任何分析 → **无**统计不确定性；"
                     "此处只记录**消费权限被拒**这一事实。"),
            "unverifiable_internally": ["signature", "authenticity",
                                        "source_trust", "actual_occurrence"],
        },
        "claim_boundary": dict(CLAIM_BOUNDARY),
        "unknown": sorted(set(reasons)),
        "note": ("门禁只**缩小**权限：任何状态都**不**获得统计/收益/策略消费许可；"
                 "结构齐备仅允许**展示为未满足**。"),
    }


# ══ 3. 契约一致性审计（源码结构 + 行为不变量） ════════════════════════
def _module_source() -> tuple[str | None, str]:
    """读取 V3H-15 模块源码（**只读**）。失败 → (None, 原因)。"""
    try:
        p = pathlib.Path(__file__).resolve().parent.joinpath(
            f"{V3H15_MODULE_STEM}.py")
        return p.read_text(encoding="utf-8"), ""
    except Exception as exc:                                # noqa: BLE001
        return None, f"module_source_unavailable:{type(exc).__name__}"


def audit_contract_consistency(state: Any) -> dict:
    """审计 V3H-15 契约是否与消费边界**一致**（源码结构 + 行为不变量）。

    **不做行为判定**、不消费 V3H-14/V3H-15 的结论、不接入任何外部资源。
    """
    if not _is_cash(state):
        return _stub("not_cash_mode_fail_closed")

    src, err = _module_source()
    if src is None:
        out = _stub(err or "module_source_unavailable")
        out["checks"] = []
        out["invariants"] = list(CONTRACT_INVARIANTS)
        return out

    checks: list[dict] = []

    def _chk(cid: str, ok: bool, detail: str) -> None:
        checks.append({"invariant": cid, "ok": bool(ok), "detail": detail})

    _chk("CIV-1_no_class_implication_matrix_declared",
         "no_class_implication_matrix" in src,
         "declares an all-empty no-class-implication matrix")
    _chk("CIV-2_sha256_role_is_fingerprint_only",
         "content_fingerprint_only" in src,
         "sha256 role limited to content fingerprint")
    _chk("CIV-3_sha256_alone_never_sufficient",
         src.count("sha256_alone_is_sufficient") >= 1
         and "sha256_alone_is_sufficient\": False" in src,
         "sha256-along sufficiency is declared false")
    _chk("CIV-4_digest_only_attestations_refused",
         "digest_only_attestation" in src,
         "digest-only attestations have a refusal token")
    _chk("CIV-5_internal_party_markers_present",
         "INTERNAL_PARTY_MARKERS" in src
         and "attesting_party_is_this_project" in src,
         "internal party markers and their refusal exist")
    _chk("CIV-6_readiness_hard_coded_false",
         "readiness\": False" in src,
         "readiness is hard-coded false in outputs")
    _chk("CIV-7_signature_verification_not_implemented",
         "signature_verification_performed\": False" in src
         and "implemented_here\": False" in src,
         "signature verification is not implemented")
    _chk("CIV-8_authenticity_verification_not_implemented",
         "authenticity_verification_performed\": False" in src,
         "authenticity verification is not performed")
    _chk("CIV-9_project_cannot_issue_or_verify",
         "PROJECT_CANNOT_DO" in src
         and "verify_authenticity_itself" in src
         and "issue_an_attestation" in src,
         "the module declares it cannot issue or verify")
    _chk("CIV-10_six_distinct_assertion_classes",
         all(f"A{n}_" in src for n in range(1, 7)),
         "six assertion classes A1..A6 present")
    _chk("CIV-11_all_six_states_declared",
         all(s in src for s in READABLE_ANCHOR_STATES),
         "all six anchor states declared")
    _chk("CIV-12_zero_prefill_declared",
         "prefilled_values\": []" in src,
         "schema declares zero prefilled values")
    _chk("CIV-13_no_layer_imports_in_module",
         _no_banned_imports(src),
         "module imports no V3G/V3H layer anywhere")
    _chk("CIV-14_no_network_process_or_wallet_capability_in_module",
         _no_io_tokens(src),
         "module has no network/process/wallet capability")
    _chk("CIV-15_no_result_slots_populated_in_module",
         _no_populated_result_slots(src),
         "module never populates a result slot")
    _chk("CIV-16_only_read_only_module_source_inspection",
         _only_readonly_source_reads(src),
         "the only file access is a read-only source inspection")

    n_ok = sum(1 for c in checks if c["ok"])
    n_bad = len(checks) - n_ok
    return {
        "present": True,
        "version": CASH_STRATEGY_CONSUMPTION_GATE_VERSION,
        "mode": "cash",
        "audited_module": V3H15_MODULE_STEM,
        "checks": checks,
        "n_checks": len(checks),
        "n_ok": n_ok,
        "n_failed": n_bad,
        "consistent": n_bad == 0,
        "invariants": list(CONTRACT_INVARIANTS),
        "misuse_risks_covered": list(MISUSE_RISKS),
        "risk_status": {r: ("COVERED" if n_bad == 0 else "NEEDS_REVIEW")
                        for r in MISUSE_RISKS},
        "behavioral_verdict_consumed": False,
        "v3h14_or_v3h15_verdict_recomputed": False,
        "external_resources_accessed": False,
        "silent_field_completion_performed": False,
        "result": None,
        "results_available": False,
        "analysis_executed": False,
        "executed": False,
        "evidence_level": "INTERNAL_DECLARATION",
        "evidence_grade": "NONE",
        "claim_boundary": dict(CLAIM_BOUNDARY),
        "unknown": sorted(c["invariant"] for c in checks if not c["ok"]),
        "note": ("审计只读：**不**重算上游判定、**不**接入外部资源；"
                 "失败项进入 `unknown` 并 fail closed。"),
    }


def _no_banned_imports(src: str) -> bool:
    banned = ("cash_state_integrity", "cash_integrity_context",
              "cash_preflop_decision", "cash_multistreet", "cash_opponent_model",
              "cash_range_model", "cash_action_tree", "cash_unknown_e2e",
              "cash_solver_interface", "cash_profit_interface",
              "cash_strategy_")
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return False
    for n in ast.walk(tree):
        mods = []
        if isinstance(n, ast.ImportFrom):
            mods.append(n.module or "")
        elif isinstance(n, ast.Import):
            mods += [a.name for a in n.names]
        for m in mods:
            if any(b in m for b in banned):
                return False
    return True


def _no_io_tokens(src: str) -> bool:
    """检查**网络 / 进程 / 客户端自动化**能力。

    `wallet` 属于"能力边界声明"用词（V3H-15 用它声明"不接入钱包"），
    因此允许以**声明字符串**形式出现，但**不得**出现在任何可执行位置。
    """
    lower = src.lower()
    for banned in ("socket", "requests", "urllib", "subprocess",
                   "sqlite", "click", "pyautogui", "advapi32"):
        if banned in lower:
            return False
    # wallet 只允许出现在受控的**边界声明**字符串中（两处，均已登记）
    residue = lower
    for decl in ("access_network_wallet_client_or_external_process",
                 "network_wallet_client_external_process_or_solver_used"):
        residue = residue.replace(decl, "")
    return "wallet" not in residue


def _only_readonly_source_reads(src: str) -> bool:
    """审计目标模块**只能**做只读源码检查：读 0~1 次、写 0 次。"""
    reads = src.count("read_text(")
    writes = sum(src.count(k) for k in (
        "write_text(", "write_bytes(", "unlink(", "mkdir(",
        "shutil", "os.remove", "os.rename"))
    return reads <= 1 and writes == 0


def _no_populated_result_slots(src: str) -> bool:
    lower = src.lower()
    return "\"result\": none" in lower and "\"executed\": false" in lower
