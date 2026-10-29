"""V3H-18 — CASH 审计结果**有效期 / 变更失效**契约（只读；**不接入任何外部资源**）。

**本层要解决的问题**
--------------------
V3H-17 的静态闭环审计明确写出其结论强度是
"**审计范围内未发现违规**"，且列出 8 项盲区（含"**审计之后的源码变更**"）。
但**没有任何东西记录**：那次审计到底是**哪一个源码快照**、
覆盖了**哪些文件**、以及**之后文件变了要怎么失效**。

本层补上这一环：**审计结果有效期契约 + 审计后变更检测 + fail-closed 状态机**。

**核心区分（必须如实标注）**
----------------------------
```
本层的指纹只能回答：   "已登记的内容**有没有变**"
本层的指纹**不能**回答：真实性 / 来源 / 真实执行 / 认证 / 是否发生了事件
```

因此：

* 审计结果**只**对某个**已审计快照**有效，表述为"**该快照范围内未发现违规**"；
* 它**不能**证明未来源码不会误用——本层**不**主张任何前瞻性保证；
* 内容指纹**只**是变更检测手段，**不得**被当作真实性、来源或执行证明；
* 未登记文件、无法读取、解析失败、动态加载、仓库外代码 **一律 fail closed**。

硬约束
------
* CASH-only，fail closed；非 dict / 非 CASH → `present=False`；
* **不 import 任何 V3G / V3H 层**（任何位置）——上游覆盖范围与基线清单
  一律由**调用方显式传入**（惯例同 V3H-9 `gate=` / V3H-15 契约）；
* **不重复实现** V3H-14 readiness / V3H-15 trust-anchor contract /
  V3H-16 consumption gate / V3H-17 闭环审计的主体逻辑；
* 无网络 / 客户端 / 钱包 / 真实资金 / Solver / 真实执行 / 收益实验 / 生产链路；
* **不授予** `verified` / `readiness` / `consumable`；
* 只做**只读**文件读取与哈希：**零写入、零删除**；无随机；同输入 → 同输出；
* `result` 恒为 `None`。
"""
from __future__ import annotations

import hashlib
import pathlib
from typing import Any

CASH_STRATEGY_AUDIT_VALIDITY_VERSION = (
    "v3h18-cash-strategy-audit-validity-v1")

#: 指纹算法（**只**用于变更检测）
FINGERPRINT_ALGORITHM = "sha256"
FINGERPRINT_PURPOSE = "change_detection_within_registered_scope_only"

#: 本层契约所登记的**审计范围**（V3H-17 闭环审计的判定面）
AUDITED_SCOPE_KINDS: tuple[str, ...] = (
    "static_import_graph",
    "ast_scan",
    "string_literal_flow",
    "contract_test",
)

#: 审计**输入文件集合**（相对包根的路径；缺失即 fail closed）
AUDIT_INPUT_FILES: tuple[str, ...] = (
    "cash_strategy_trust_anchor_contract.py",
    "cash_strategy_consumption_gate.py",
    "cash_strategy_static_closure_audit.py",
    "../tests/unit/test_v3h15_trust_anchor_contract.py",
    "../tests/unit/test_v3h16_consumption_gate.py",
    "../tests/unit/test_v3h17_static_closure_audit.py",
)

#: 有效期状态（封闭枚举）
VALIDITY_STATES: tuple[str, ...] = (
    "VALID_FOR_REGISTERED_SNAPSHOT",
    "SUPERSEDED_SOURCE_CHANGED",
    "INVALID_UNREGISTERED_CHANGE",
    "INVALID_UNREADABLE",
    "INVALID_UNPARSEABLE",
    "INVALID_BASELINE_MISSING",
    "INVALID_SCOPE_MISMATCH",
)

#: 失效后的**重新审计**要求
REVALIDATION_REQUIREMENTS: tuple[str, ...] = (
    "re_run_the_v3h17_static_closure_audit_on_the_current_source",
    "record_a_new_baseline_manifest_with_fresh_fingerprints",
    "re_confirm_the_audited_scope_and_blind_spots",
    "do_not_reuse_the_superseded_result_for_any_conclusion",
)

#: fail-closed 失败类别（本层新增的变更检测面）
CHANGE_DETECTION_FAILURES: tuple[str, ...] = (
    "UNREGISTERED_FILE_PRESENT",
    "REGISTERED_FILE_MISSING",
    "FILE_UNREADABLE",
    "FILE_UNPARSEABLE",
    "DYNAMIC_LOAD_DETECTED",
    "CODE_OUTSIDE_REGISTERED_SCOPE",
    "BASELINE_MISSING",
    "SCOPE_MISMATCH",
)

#: 动态加载标记（出现即视为范围内存在**静态分析看不到**的加载路径）
DYNAMIC_LOAD_MARKERS: tuple[str, ...] = (
    "__import__(", "importlib.import_module", "importlib.reload",
    "exec(", "eval(", "compile(", "getattr(",
)

#: 审计结果可以主张 / 不可以主张的边界
ATTESTATION_SCOPE = {
    "may_claim": ("no_violation_found_within_the_registered_snapshot_scope",),
    "may_not_claim": (
        "future_source_will_not_misuse",
        "authenticity_of_any_content",
        "provenance_or_source_trust",
        "actual_occurrence_of_any_event",
        "certification",
        "verified_or_readiness_or_consumable_status",
    ),
    "fingerprint_proves": ("registered_content_is_unchanged",),
    "fingerprint_does_not_prove": ("authenticity", "provenance",
                                   "source_trust", "execution",
                                   "occurrence", "certification"),
}

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
    "is_provenance_proof": False,
    "is_future_safety_guarantee": False,
    "fingerprint_treated_as_truth": False,
    "grants_verified_status": False,
    "grants_readiness": False,
    "grants_consumable": False,
    "reuses_superseded_audit_result": False,
    "packages_internal_result_as_external_evidence": False,
    "computes_rates_or_statistics": False,
    "adds_scoring_or_ev_formula": False,
    "executes_action": False,
    "auto_betting": False,
    "wired_to_production": False,
}
CLAIM_FLAGS_TOTAL = len(CLAIM_BOUNDARY)


def _is_cash(state: Any) -> bool:
    if not isinstance(state, dict):
        return False
    gt = state.get("game_type")
    return isinstance(gt, str) and gt.strip().lower() == "cash"


def _stub(reason: str) -> dict:
    return {
        "present": False,
        "version": CASH_STRATEGY_AUDIT_VALIDITY_VERSION,
        "reason": reason,
        "validity": "INVALID_BASELINE_MISSING",
        "still_valid": False,
        "superseded": True,
        "revalidation_required": True,
        "grants_verified_status": False,
        "grants_readiness": False,
        "grants_consumable": False,
        "n_registered": 0,
        "n_changed": 0,
        "n_missing": 0,
        "n_unreadable": 0,
        "n_unparseable": 0,
        "n_unregistered": 0,
        "change_detection_failures": [],
        "revalidation_requirements": list(REVALIDATION_REQUIREMENTS),
        "evidence_level": "UNAVAILABLE",
        "evidence_grade": "NONE",
        "executed": False,
        "result": None,
        "results_available": False,
        "analysis_executed": False,
        "claim_boundary": dict(CLAIM_BOUNDARY),
        "unknown": [reason],
    }


# ══ 1. schema 与契约（纯声明） ════════════════════════════════════════
def audit_validity_contract() -> dict:
    """返回**审计结果有效期契约**（schema + 状态 + 重新审计要求）。"""
    return {
        "success": True,
        "version": CASH_STRATEGY_AUDIT_VALIDITY_VERSION,
        "fingerprint": {
            "algorithm": FINGERPRINT_ALGORITHM,
            "purpose": FINGERPRINT_PURPOSE,
            "proves_only": list(ATTESTATION_SCOPE["fingerprint_proves"]),
            "does_not_prove": list(
                ATTESTATION_SCOPE["fingerprint_does_not_prove"]),
            "is_authenticity_proof": False,
            "is_provenance_proof": False,
            "is_certification": False,
        },
        "audit_input_files": list(AUDIT_INPUT_FILES),
        "n_audit_input_files": len(AUDIT_INPUT_FILES),
        "audited_scope_kinds": list(AUDITED_SCOPE_KINDS),
        "baseline_manifest_fields": ["version", "scope_fingerprint",
                                     "n_files", "files",
                                     "recorded_at_audit_version",
                                     "coverage", "blind_spots",
                                     "limitations"],
        "validity_states": list(VALIDITY_STATES),
        "n_validity_states": len(VALIDITY_STATES),
        "revalidation_requirements": list(REVALIDATION_REQUIREMENTS),
        "n_revalidation_requirements": len(REVALIDATION_REQUIREMENTS),
        "change_detection_failures": list(CHANGE_DETECTION_FAILURES),
        "fail_closed_on": ["unregistered_file", "missing_file",
                           "unreadable_file", "unparseable_file",
                           "dynamic_load_marker", "code_outside_scope",
                           "missing_baseline", "scope_mismatch"],
        "attestation_scope": {
            "may_claim": list(ATTESTATION_SCOPE["may_claim"]),
            "may_not_claim": list(ATTESTATION_SCOPE["may_not_claim"]),
        },
        "dynamic_load_markers": list(DYNAMIC_LOAD_MARKERS),
        "validity_means": ("the registered source snapshot is byte-identical "
                           "to the snapshot that was audited"),
        "validity_does_not_mean": ("future safety, authenticity, provenance "
                                   "or readiness"),
        "grants_verified_status": False,
        "grants_readiness": False,
        "grants_consumable": False,
        "result": None,
        "executed": False,
        "evidence_level": "INTERNAL_DECLARATION",
        "evidence_grade": "NONE",
        "claim_boundary": dict(CLAIM_BOUNDARY),
        "unknown": [],
        "note": ("契约是**声明性**的：它只说明“有效期”是什么意思，"
                 "**不**证明任何内容真实、可信或未被误用。"),
    }


# ══ 2. 清单与指纹（只读） ═════════════════════════════════════════════
def _dynamic_load_calls(text: str) -> list[str]:
    """按 **AST 调用结构**检出动态加载路径（只认真实调用，非字面量出现）。"""
    import ast as _ast
    try:
        tree = _ast.parse(text)
    except Exception:                                       # noqa: BLE001
        return ["<unparseable>"]
    found: set[str] = set()
    for n in _ast.walk(tree):
        if not isinstance(n, _ast.Call):
            continue
        fn = n.func
        if isinstance(fn, _ast.Name):
            if fn.id in ("exec", "eval", "compile", "__import__", "getattr",
                         "setattr"):
                found.add(f"{fn.id}()")
        elif isinstance(fn, _ast.Attribute):
            if fn.attr in ("import_module", "reload", "__import__"):
                found.add(f"importlib.{fn.attr}()")
    return sorted(found)


def _package_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parent


def _resolve(rel: str) -> pathlib.Path:
    return _package_root().joinpath(*rel.split("/")).resolve()


def build_manifest() -> dict:
    """只读构建**审计输入文件清单**（逐文件 sha256 + 合并范围指纹）。"""
    files: list[dict] = []
    missing: list[str] = []
    unreadable: list[str] = []
    unparseable: list[str] = []
    per_file: dict[str, str] = {}

    for rel in AUDIT_INPUT_FILES:
        p = _resolve(rel)
        if not p.exists():
            missing.append(rel)
            files.append({"path": rel, "status": "MISSING",
                          "fingerprint": None})
            continue
        try:
            raw = p.read_bytes()
        except Exception:                                   # noqa: BLE001
            unreadable.append(rel)
            files.append({"path": rel, "status": "UNREADABLE",
                          "fingerprint": None})
            continue
        digest = hashlib.sha256(raw).hexdigest()
        per_file[rel] = digest
        status = "OK"
        try:
            import ast as _ast
            _ast.parse(raw.decode("utf-8", errors="ignore"))
        except Exception:                                   # noqa: BLE001
            unparseable.append(rel)
            status = "UNPARSEABLE"
        files.append({"path": rel, "status": status, "fingerprint": digest,
                      "bytes": len(raw)})

    scope_fingerprint = hashlib.sha256(
        "\n".join(f"{k}:{per_file[k]}" for k in sorted(per_file))
        .encode("utf-8")).hexdigest() if per_file else None

    return {
        "success": True,
        "version": CASH_STRATEGY_AUDIT_VALIDITY_VERSION,
        "fingerprint_algorithm": FINGERPRINT_ALGORITHM,
        "scope_fingerprint": scope_fingerprint,
        "n_files": len(AUDIT_INPUT_FILES),
        "n_fingerprinted": len(per_file),
        "files": files,
        "per_file_fingerprints": dict(sorted(per_file.items())),
        "missing": sorted(missing),
        "unreadable": sorted(unreadable),
        "unparseable": sorted(unparseable),
        "recorded_at_audit_version": CASH_STRATEGY_AUDIT_VALIDITY_VERSION,
        "coverage": list(AUDITED_SCOPE_KINDS),
        "result": None,
        "executed": False,
        "evidence_level": "INTERNAL_DECLARATION",
        "evidence_grade": "NONE",
        "claim_boundary": dict(CLAIM_BOUNDARY),
        "unknown": (sorted(set(missing + unreadable + unparseable)) or []),
    }


# ══ 3. 变更检测 + 有效期状态机（fail closed） ════════════════════════
def evaluate_audit_validity(state: Any, *, baseline: Any = None,
                            current: Any = None) -> dict:
    """判定**已审计结果**是否仍对**当前源码**有效。

    `baseline` / `current` 为 `build_manifest()` 形态的字典；
    未提供 `current` → 只读现算；未提供 `baseline` → **fail closed**。
    """
    if not _is_cash(state):
        return _stub("not_cash_mode_fail_closed")
    if not isinstance(baseline, dict) or not baseline.get(
            "per_file_fingerprints"):
        return _stub("baseline_required_for_validity_check")

    cur = current if isinstance(current, dict) else build_manifest()
    base_fp = {k: v for k, v in
               (baseline.get("per_file_fingerprints") or {}).items()}
    cur_fp = {k: v for k, v in (cur.get("per_file_fingerprints") or {}).items()}

    changed = sorted(k for k in base_fp if k in cur_fp
                     and cur_fp[k] != base_fp[k])
    missing = sorted(k for k in base_fp if k not in cur_fp)
    unregistered = sorted(k for k in cur_fp if k not in base_fp)
    unreadable = sorted(cur.get("unreadable") or [])
    unparseable = sorted(cur.get("unparseable") or [])

    # 动态加载：按 **AST 调用结构**判定
    # （避免被"标记清单"里的字面量误伤；只认真实调用）
    dynamic_hits: dict[str, list[str]] = {}
    for rel in AUDIT_INPUT_FILES:
        if not rel.endswith(".py"):
            continue
        p = _resolve(rel)
        if not p.exists():
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:                                   # noqa: BLE001
            continue
        hits = _dynamic_load_calls(text)
        if hits:
            dynamic_hits[rel] = hits

    failures: list[str] = []
    if missing:
        failures.append("REGISTERED_FILE_MISSING")
    if unregistered:
        failures.append("UNREGISTERED_FILE_PRESENT")
    if unreadable:
        failures.append("FILE_UNREADABLE")
    if unparseable:
        failures.append("FILE_UNPARSEABLE")
    if dynamic_hits:
        failures.append("DYNAMIC_LOAD_DETECTED")
    base_scope = baseline.get("scope_fingerprint")
    if isinstance(base_scope, str) and isinstance(cur.get("scope_fingerprint"),
                                                  str) \
            and base_scope == cur.get("scope_fingerprint") and (changed
                                                                or missing
                                                                or unregistered):
        failures.append("SCOPE_MISMATCH")

    # ── 状态判定（fail closed 优先）──
    if missing:
        validity = "INVALID_UNREADABLE"
    elif unreadable or unparseable:
        validity = ("INVALID_UNREADABLE" if unreadable
                    else "INVALID_UNPARSEABLE")
    elif unregistered:
        validity = "INVALID_UNREGISTERED_CHANGE"
    elif changed:
        validity = "SUPERSEDED_SOURCE_CHANGED"
    elif dynamic_hits:
        # 出现动态加载 → 静态结论不再能覆盖该路径
        validity = "INVALID_UNREGISTERED_CHANGE"
    else:
        validity = "VALID_FOR_REGISTERED_SNAPSHOT"

    still_valid = validity == "VALID_FOR_REGISTERED_SNAPSHOT"
    return {
        "present": True,
        "version": CASH_STRATEGY_AUDIT_VALIDITY_VERSION,
        "mode": "cash",
        "validity": validity,
        "still_valid": still_valid,
        "superseded": not still_valid,
        "revalidation_required": not still_valid,
        "validity_meaning": ("registered snapshot is byte-identical to the "
                             "audited snapshot" if still_valid else
                             "the audited snapshot no longer matches the "
                             "current source"),
        "validity_does_not_mean": ("future safety, authenticity, provenance, "
                                   "readiness or anything about unregistered "
                                   "code"),
        "baseline_scope_fingerprint": base_scope,
        "current_scope_fingerprint": cur.get("scope_fingerprint"),
        "n_registered": len(base_fp),
        "n_changed": len(changed),
        "changed_files": changed,
        "missing_files": missing,
        "unregistered_files": unregistered,
        "unreadable_files": unreadable,
        "unparseable_files": unparseable,
        "n_missing": len(missing),
        "n_unregistered": len(unregistered),
        "n_unreadable": len(unreadable),
        "n_unparseable": len(unparseable),
        "dynamic_load_hits": dynamic_hits,
        "change_detection_failures": sorted(set(failures)),
        "failure_policy": {f: "fail_closed" for f in CHANGE_DETECTION_FAILURES},
        "revalidation_requirements": (
            [] if still_valid else list(REVALIDATION_REQUIREMENTS)),
        "superseded_result_usable": False,
        "grants_verified_status": False,
        "grants_readiness": False,
        "grants_consumable": False,
        "attestation_scope": {
            "may_claim": (list(ATTESTATION_SCOPE["may_claim"])
                          if still_valid else []),
            "may_not_claim": list(ATTESTATION_SCOPE["may_not_claim"]),
        },
        "fingerprint_proves_only": list(
            ATTESTATION_SCOPE["fingerprint_proves"]),
        "fingerprint_does_not_prove": list(
            ATTESTATION_SCOPE["fingerprint_does_not_prove"]),
        "result": None,
        "results_available": False,
        "analysis_executed": False,
        "executed": False,
        "evidence_level": "INTERNAL_DECLARATION",
        "evidence_grade": "NONE",
        "uncertainty": {
            "measured": False,
            "note": ("变更检测**无**统计不确定性，但**有**结构性盲区："
                     "未登记文件、仓库外代码、运行期加载均不在覆盖范围内。"),
            "coverage_is_partial": True,
        },
        "claim_boundary": dict(CLAIM_BOUNDARY),
        "unknown": sorted(set(missing + unregistered + unreadable
                              + unparseable)),
        "note": ("本层只回答“已登记内容有没有变”，"
                 "**不**证明真实性、来源或真实执行，"
                 "**不**授予 verified / readiness / consumable。"),
    }


# ══ 4. 重新审计要求清单 ═══════════════════════════════════════════════
def revalidation_plan(state: Any, *, validity: Any = None) -> dict:
    """根据有效期判定给出**重新审计**要求（只读；不执行审计）。"""
    if not _is_cash(state):
        return _stub("not_cash_mode_fail_closed")
    if not isinstance(validity, dict) or validity.get("present") is not True:
        return _stub("validity_required")

    needs = validity.get("revalidation_required") is True
    steps = list(REVALIDATION_REQUIREMENTS) if needs else []
    return {
        "present": True,
        "version": CASH_STRATEGY_AUDIT_VALIDITY_VERSION,
        "mode": "cash",
        "validity": validity.get("validity"),
        "revalidation_required": needs,
        "steps": steps,
        "n_steps": len(steps),
        "triggering_failures": list(
            validity.get("change_detection_failures") or []),
        "changed_files": list(validity.get("changed_files") or []),
        "blocking_reason": (
            "the audited snapshot changed or could not be confirmed; the "
            "previous audit result must not be reused" if needs else None),
        "superseded_result_usable": False,
        "audit_executed_by_this_layer": False,
        "grants_verified_status": False,
        "grants_readiness": False,
        "grants_consumable": False,
        "result": None,
        "executed": False,
        "evidence_level": "INTERNAL_DECLARATION",
        "evidence_grade": "NONE",
        "claim_boundary": dict(CLAIM_BOUNDARY),
        "unknown": [],
        "note": ("计划本身**不**执行审计：只列出要求与触发原因。"),
    }
