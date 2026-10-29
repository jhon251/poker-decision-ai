"""V3H-13 — CASH 真实执行记录接入闸门（执行前接口，**不执行**）。

**为什么这是 V3H-13 的最小下一能力（源码级缺口审计结论）**
------------------------------------------------------------
对**既有 15 个** V3H 层做源码探针（排除本阶段新模块）：

```
intake / ingest / collector / envelope / execution_record     0 个文件
hashlib / sha256 / fingerprint / dedup / idempot / tamper     0 个文件
provenance_check / real_execution / external_valid / attest   0 个文件
manifest                                                      0 个文件
batch   1 个文件 —— V3H-11 docstring 内的**探针进度文字**，非实现
seal    1 个文件 —— V3H-9 中的普通用词，非内容封存机制
```

即：**没有任何"真实执行记录"的接入路径**——没有内容指纹、没有去重、
没有篡改检出、没有接入闸门。V3H-12 明确声明
"`real_execution_data` 被**主动拒绝消费**、本层**不建立**任何真实结果入口"，
因此**外部验证链条在"数据接入"这一步是断的**。

本层补上这一步：**只实现执行前接口 / 门禁**，
**不执行**、**不伪造执行环境**、**不生成结果**。

**严格定位（必须如实标注）**
----------------------------
* 本层处理的是**调用方声称来自真实执行**的记录；
* 本层**不能**证明任何记录的**真实性**：`integrity_digest` 只证明
  **内容自洽 / 可去重**，**不**证明来源可信、**不**构成证明、**不**构成认证；
* 真实可信性需要**带外信任锚**（`trust_anchor_verified` 必须由**外部**提供）；
  本层**不**自行判定信任；
* `intake_permitted` 在 Solver `BLOCKED` / 独立 blocker 未解时**恒为 `False`**，
  且此时**任何记录都不会被接纳**（`admitted = 0`）；
* 骨架模板 `execution_record_template()` 的**每一个字段都是 `None` 或
  `UNKNOWN`** —— **不预填任何伪造值**；
* 本层**不计算**任何收益 / EV / 比率 / 统计量；`result` 恒为 `None`。

硬约束
------
* CASH-only，fail closed；非 dict / 非 CASH → `present=False`；
* **不 import 任何 V3G / V3H 层**（任何位置）；不 import 伪随机数模块；
  **无任何 IO**（不读文件、不开网络、不连客户端）；
* 同输入 → 同输出（指纹为**确定性**内容哈希，键序与列表序固定）；
* UNKNOWN / UNAVAILABLE / abstention / disagreement 原样传播并阻断接纳；
* B1 / B3 / B4 / B5 / Solver 仍 `BLOCKED` 时**保持 BLOCKED**，
  `equilibrium` / `gto` / `exploitability` 恒为 `None`（**不猜**）；
* `MODEL_INTERNAL` **不得**被当作外部强度证据：本层只接受
  `evidence_level == "REAL_EXECUTION_DATA"` 的记录，其余一律拒绝；
* 不修改输入对象；不写入任何位置。
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

CASH_STRATEGY_EXECUTION_INTAKE_VERSION = (
    "v3h13-cash-strategy-execution-intake-v1")

#: 允许出现的证据等级（**仅**真实执行数据可被接纳）
ACCEPTED_EVIDENCE_LEVELS: tuple[str, ...] = ("REAL_EXECUTION_DATA",)

#: 已知的、**不可**作为真实执行证据的等级（明确拒绝，便于如实报错）
NON_EXECUTION_EVIDENCE_LEVELS: tuple[str, ...] = (
    "DESIGN_DECLARATION", "MODEL_INTERNAL", "UNAVAILABLE", "NONE",
)

#: 执行记录 schema（**精确**字段集；全部必需）
EXECUTION_RECORD_FIELDS: tuple[str, ...] = (
    "execution_id",         # 调用方分配，稳定唯一
    "shadow_record_id",     # 关联的 V3H-10 决策点
    "state_hash",           # 关联状态指纹
    "execution_env",        # 真实执行环境描述（沙盒/模拟环境会被拒绝）
    "executor_kind",        # who/what performed the action
    "observed_action",      # 实际发生的动作
    "observed_amount_bb",   # 实际金额（可为 None）
    "observed_at_seq",      # 单调序号（**不接受**墙上时钟作为唯一序）
    "card_source",          # 牌面来源（必须声明；合成牌被拒绝）
    "evidence_level",       # 必须为 REAL_EXECUTION_DATA
    "result_provenance",    # 必须为 real_execution
    "realised_outcome",     # 必须由调用方提供（本层不推断）
    "unknown_tokens",       # 未解 UNKNOWN（原样保留）
    "integrity_digest",     # 内容指纹（本层用确定性哈希校验）
)

#: 记录级拒绝原因（封闭枚举）
RECORD_REFUSAL_REASONS: tuple[str, ...] = (
    "record_not_a_dict",
    "missing_required_field",
    "execution_id_invalid",
    "shadow_record_id_invalid",
    "state_hash_invalid",
    "execution_env_invalid",
    "execution_env_synthetic",
    "executor_kind_invalid",
    "observed_action_invalid",
    "observed_amount_not_number_or_none",
    "observed_at_seq_invalid",
    "card_source_invalid",
    "card_source_synthetic",
    "evidence_level_not_real_execution_data",
    "result_provenance_not_real_execution",
    "realised_outcome_missing",
    "unknown_tokens_present",
    "unknown_tokens_malformed",
    "integrity_digest_missing",
    "integrity_digest_mismatch",
    "realised_outcome_invents_result",
)

#: 批次级拒绝原因（封闭枚举）
BATCH_REFUSAL_REASONS: tuple[str, ...] = (
    "preregistration_required",
    "solver_blocked_and_no_equilibrium_reference",
    "independent_blockers_unresolved",
    "environment_facts_incomplete",
    "records_not_supplied",
    "trust_anchor_not_verified",
    "minimum_rows_not_reached",
    "no_record_admitted",
)

#: 系统级结果前置条件（与 V3H-12 对齐；缺一即不接纳）
INTAKE_PRECONDITIONS: tuple[str, ...] = (
    "preregistration_supplied_and_present",
    "solver_available_with_equilibrium",
    "independent_blockers_all_resolved",
    "environment_facts_all_true",
    "trust_anchor_verified_by_caller",
    "records_at_or_above_declared_minimum",
    "zero_refused_records",
    "zero_duplicate_records",
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
    "verifies_source_trust": False,
    "computes_rates_or_statistics": False,
    "adds_scoring_or_ev_formula": False,
    "fabricates_execution_environment": False,
    "executes_action": False,
    "auto_betting": False,
    "claims_experiment_ready": False,
    "wired_to_production": False,
}
CLAIM_FLAGS_TOTAL = len(CLAIM_BOUNDARY)

DEFAULT_MIN_RECORDS = 1200

#: 明确的"合成/模拟"环境与牌源标记（拒绝，不伪造真实环境）
SYNTHETIC_MARKERS: tuple[str, ...] = (
    "synthetic", "simulated", "mock", "stub", "fake", "dummy",
    "test_only", "fixture", "dry_run", "paper",
)
#: 明确的真实执行标记
REAL_EXECUTION_MARKERS: tuple[str, ...] = ("live", "real", "production")


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


def _lower(v: Any) -> str:
    s = _norm_str(v)
    return s.lower() if s else ""


def _stub(reason: str) -> dict:
    return {
        "present": False,
        "version": CASH_STRATEGY_EXECUTION_INTAKE_VERSION,
        "reason": reason,
        "evidence_level": "UNAVAILABLE",
        "evidence_grade": "NONE",
        "executed": False,
        "intake_permitted": False,
        "n_admitted": 0,
        "admitted": 0,
        "result": None,
        "results_available": False,
        "analysis_executed": False,
        "claim_boundary": dict(CLAIM_BOUNDARY),
        "unknown": [reason],
    }


def _digest_stub(reason: str, issues: list[str]) -> dict:
    d = _stub(reason)
    d["digest"] = None
    d["is_authenticity_proof"] = False
    d["is_certification"] = False
    d["verifies_source_trust"] = False
    d["requires_out_of_band_trust_anchor"] = True
    d["unknown"] = list(issues)
    return d


# ══ 1. 内容指纹（确定性；**不**构成真实性证明） ══════════════════════
def canonical_payload(record: Any) -> tuple[str | None, list[str]]:
    """返回 (canonical_json, issues)；`integrity_digest` 字段不参与计算。"""
    if not isinstance(record, dict):
        return None, ["record_not_a_dict"]
    issues: list[str] = []
    if "integrity_digest" not in record:
        issues.append("integrity_digest_missing")
    payload = {k: v for k, v in record.items() if k != "integrity_digest"}
    try:
        text = json.dumps(payload, sort_keys=True, ensure_ascii=False,
                          separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError):
        return None, issues + ["payload_not_canonicalisable"]
    return text, issues


def compute_integrity_digest(record: Any) -> dict:
    """计算记录的内容指纹（sha256，确定性）。

    **边界**：指纹只证明**内容自洽 / 可用于去重**；
    **不**证明来源可信，**不**构成真实性证明或认证。
    """
    text, issues = canonical_payload(record)
    if text is None:
        return _digest_stub(
            issues[0] if issues else "payload_not_canonicalisable", issues)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return {
        "present": True,
        "version": CASH_STRATEGY_EXECUTION_INTAKE_VERSION,
        "algorithm": "sha256",
        "scope": "canonical_record_content_excluding_integrity_digest",
        "digest": digest,
        "digest_prefix": digest[:16],
        "canonical_bytes": len(text.encode("utf-8")),
        "is_authenticity_proof": False,
        "is_certification": False,
        "verifies_source_trust": False,
        "requires_out_of_band_trust_anchor": True,
        "note": ("内容指纹：仅证明内容自洽与可用于去重；"
                 "**不**证明来源可信、**不**构成证明或认证。"),
        "executed": False,
        "claim_boundary": dict(CLAIM_BOUNDARY),
        "unknown": [],
    }


# ══ 2. 骨架模板（**不预填任何伪造值**） ══════════════════════════════
def execution_record_template() -> dict:
    """返回执行记录骨架：**每个字段都是 None / UNKNOWN**。

    本函数**不**伪造执行环境、**不**预填动作、**不**预填结果、
    **不**预生成任何指纹。
    """
    template = {f: None for f in EXECUTION_RECORD_FIELDS}
    template["unknown_tokens"] = ["UNKNOWN"]
    return {
        "success": True,
        "version": CASH_STRATEGY_EXECUTION_INTAKE_VERSION,
        "fields": list(EXECUTION_RECORD_FIELDS),
        "n_fields": len(EXECUTION_RECORD_FIELDS),
        "template": template,
        "prefilled_values": [],
        "fabricated_environment": False,
        "note": ("骨架**不含**任何预填值；所有内容必须由**真实执行**方提供。"
                 "`integrity_digest` 需在内容填好后由调用方用 "
                 "`compute_integrity_digest()` 生成。"),
        "accepted_evidence_levels": list(ACCEPTED_EVIDENCE_LEVELS),
        "synthetic_markers_rejected": list(SYNTHETIC_MARKERS),
        "executed": False,
        "claim_boundary": dict(CLAIM_BOUNDARY),
        "unknown": [],
    }


# ══ 3. 记录级校验 ════════════════════════════════════════════════════
def _refuse_record(rec: Any) -> tuple[list[str], str | None]:
    """返回 (issues, record_fingerprint_or_None)。"""
    if not isinstance(rec, dict):
        return ["record_not_a_dict"], None

    issues: list[str] = []
    for f in EXECUTION_RECORD_FIELDS:
        if f not in rec:
            issues.append(f"missing_required_field:{f}")
    if issues:
        return issues, None

    for f in ("execution_id", "shadow_record_id", "state_hash",
              "executor_kind", "observed_action", "card_source",
              "result_provenance", "execution_env"):
        if _norm_str(rec.get(f)) is None:
            issues.append({
                "execution_id": "execution_id_invalid",
                "shadow_record_id": "shadow_record_id_invalid",
                "state_hash": "state_hash_invalid",
                "executor_kind": "executor_kind_invalid",
                "observed_action": "observed_action_invalid",
                "card_source": "card_source_invalid",
                "result_provenance": "result_provenance_not_real_execution",
                "execution_env": "execution_env_invalid",
            }[f])

    env = _lower(rec.get("execution_env"))
    if env and any(m in env for m in SYNTHETIC_MARKERS):
        issues.append("execution_env_synthetic")
    cs = _lower(rec.get("card_source"))
    if cs and any(m in cs for m in SYNTHETIC_MARKERS):
        issues.append("card_source_synthetic")

    amt = rec.get("observed_amount_bb")
    if amt is not None and (isinstance(amt, bool)
                            or not isinstance(amt, (int, float))):
        issues.append("observed_amount_not_number_or_none")

    if not _is_int(rec.get("observed_at_seq")):
        issues.append("observed_at_seq_invalid")

    if rec.get("evidence_level") not in ACCEPTED_EVIDENCE_LEVELS:
        issues.append("evidence_level_not_real_execution_data")

    if _lower(rec.get("result_provenance")) != "real_execution":
        issues.append("result_provenance_not_real_execution")

    ro = rec.get("realised_outcome")
    if ro is None:
        issues.append("realised_outcome_missing")
    elif isinstance(ro, dict) and ro.get("source") == "model_internal":
        issues.append("realised_outcome_invents_result")

    tok = rec.get("unknown_tokens")
    if not isinstance(tok, list) or any(not isinstance(t, str) for t in tok):
        issues.append("unknown_tokens_malformed")
    elif tok:
        issues.append("unknown_tokens_present")

    expected = compute_integrity_digest(rec)
    given = rec.get("integrity_digest")
    if _norm_str(given) is None:
        issues.append("integrity_digest_missing")
    elif not expected.get("present"):
        issues.append("integrity_digest_mismatch")
    elif str(given) != expected["digest"]:
        issues.append("integrity_digest_mismatch")

    fp = hashlib.sha256(
        json.dumps({"execution_id": rec.get("execution_id"),
                    "shadow_record_id": rec.get("shadow_record_id"),
                    "observed_at_seq": rec.get("observed_at_seq")},
                   sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return issues, fp


def validate_execution_record(state: Any, record: Any = None) -> dict:
    """只做**记录级**校验与指纹计算；不接纳、不执行、不统计。"""
    if not _is_cash(state):
        return _stub("not_cash_mode_fail_closed")
    if not isinstance(record, dict) or not record:
        return _stub("record_required")

    issues, fp = _refuse_record(record)
    digest = compute_integrity_digest(record)
    return {
        "present": True,
        "version": CASH_STRATEGY_EXECUTION_INTAKE_VERSION,
        "mode": "cash",
        "valid": not issues,
        "issues": issues,
        "n_issues": len(issues),
        "record_fingerprint": fp,
        "integrity_digest": digest.get("digest"),
        "digest_recomputed_matches": (
            digest.get("present") is True
            and str(record.get("integrity_digest")) == digest.get("digest")),
        "is_authenticity_proof": False,
        "requires_out_of_band_trust_anchor": True,
        "executed": False,
        "result": None,
        "results_available": False,
        "analysis_executed": False,
        "evidence_level": "REAL_EXECUTION_DATA" if not issues else "UNAVAILABLE",
        "claim_boundary": dict(CLAIM_BOUNDARY),
        "unknown": list(issues),
        "note": "仅记录级校验；**不**接纳、**不**执行、**不**统计。",
    }


# ══ 4. 接入闸门（执行前接口） ════════════════════════════════════════
def intake_execution_records(state: Any, *, records: Any = None,
                             preregistration: Any = None,
                             solver: Any = None,
                             environment: Any = None,
                             trust_anchor_verified: bool = False,
                             min_records: int = DEFAULT_MIN_RECORDS) -> dict:
    """**执行前**接入闸门：校验并决定是否可接纳真实执行记录。

    **本函数不执行任何动作**：即使全部条件满足，也只返回
    `intake_permitted=True` 与接纳清单，**不**产生结果、**不**写盘、
    **不**调用任何求解器。
    """
    if not _is_cash(state):
        return _stub("not_cash_mode_fail_closed")
    if not _is_int(min_records) or min_records < 1:
        return _stub("min_records_invalid")
    if not isinstance(preregistration, dict) or \
            preregistration.get("present") is not True:
        return _stub("preregistration_required")

    # ── 系统级前置条件（逐项判定，缺一不接纳）──
    met: list[str] = []
    unmet: list[str] = []
    batch_reasons: list[str] = []

    def _mark(i: int, ok: bool) -> None:
        (met if ok else unmet).append(INTAKE_PRECONDITIONS[i])

    _mark(0, True)                                   # preregistration 已校验

    solver_ok = (isinstance(solver, dict)
                 and solver.get("solver_status") == "AVAILABLE"
                 and solver.get("available") is True
                 and solver.get("equilibrium") is True)
    _mark(1, solver_ok)
    if not solver_ok:
        batch_reasons.append("solver_blocked_and_no_equilibrium_reference")

    blockers = (preregistration.get("blockers_from_upstream")
                if isinstance(preregistration.get("blockers_from_upstream"),
                              list) else None)
    if blockers is None:
        pb = preregistration.get("blockers")
        blockers = (sorted(k for k, v in pb.items()
                           if isinstance(v, dict)
                           and v.get("status") == "BLOCKED")
                    if isinstance(pb, dict) else [])
    blockers_resolved = not blockers
    _mark(2, blockers_resolved)
    if not blockers_resolved:
        batch_reasons.append("independent_blockers_unresolved")

    env_ok = isinstance(environment, dict) and bool(environment) and all(
        v is True for v in environment.values())
    _mark(3, env_ok)
    if not env_ok:
        batch_reasons.append("environment_facts_incomplete")

    anchor_ok = trust_anchor_verified is True
    _mark(4, anchor_ok)
    if not anchor_ok:
        batch_reasons.append("trust_anchor_not_verified")

    if not isinstance(records, (list, tuple)):
        _mark(5, False)
        _mark(6, False)
        _mark(7, False)
        batch_reasons.append("records_not_supplied")
        rows: list[dict] = []
        n_refused = n_dup = 0
        admitted: list[dict] = []
    else:
        n_ok = len(records) >= int(min_records)
        _mark(5, n_ok)
        if not n_ok:
            batch_reasons.append("minimum_rows_not_reached")

        seen: dict[str, int] = {}
        admitted = []
        rows = []
        n_refused = 0
        n_dup = 0
        for i, rec in enumerate(records):
            issues, fp = _refuse_record(rec)
            dup = fp is not None and fp in seen
            if dup:
                n_dup += 1
                issues = issues + ["duplicate_execution_record"]
            if fp is not None:
                seen.setdefault(fp, i)
            ok = not issues
            if not ok:
                n_refused += 1
            rows.append({"index": i, "fingerprint": fp, "admitted": ok,
                         "duplicate": dup, "issues": issues})
            if ok:
                admitted.append({"index": i, "fingerprint": fp,
                                 "execution_id": rec.get("execution_id"),
                                 "shadow_record_id": rec.get("shadow_record_id")})

        _mark(6, n_refused == 0)
        if n_refused:
            batch_reasons.append("zero_refused_records")
        _mark(7, n_dup == 0)
        if n_dup:
            batch_reasons.append("zero_duplicate_records")

    intake_permitted = not unmet         # Solver BLOCKED ⇒ 恒为 False
    n_records = len(records) if isinstance(records, (list, tuple)) else 0
    if not admitted:
        batch_reasons.append("no_record_admitted")

    return {
        "present": True,
        "version": CASH_STRATEGY_EXECUTION_INTAKE_VERSION,
        "mode": "cash",
        "intake_permitted": intake_permitted,
        "admitted": len(admitted) if intake_permitted else 0,
        "n_admitted": len(admitted) if intake_permitted else 0,
        "n_candidates": len(admitted),
        "n_records_supplied": n_records,
        "n_refused": n_refused,
        "n_duplicate": n_dup,
        "admitted_records": admitted if intake_permitted else [],
        "candidate_records": admitted,
        "row_checks": rows,
        "record_refusal_reasons": sorted(
            {i.split(":", 1)[0] for r in rows for i in r["issues"]}),
        "batch_refusal_reasons": sorted(set(batch_reasons)),
        "preconditions": {
            "met": met,
            "unmet": unmet,
            "all": list(INTAKE_PRECONDITIONS),
            "n_met": len(met),
            "n_unmet": len(unmet),
            "would_admit_if": list(INTAKE_PRECONDITIONS),
        },
        "trust_anchor": {
            "verified": anchor_ok,
            "verified_by": "caller" if anchor_ok else None,
            "verified_by_this_layer": False,
            "note": ("信任锚**必须**由带外提供；本层**不**自行判定来源可信，"
                     "内容指纹**不**构成真实性证明。"),
        },
        "solver": {
            "solver_status": (solver.get("solver_status")
                              if isinstance(solver, dict) else "UNKNOWN"),
            "available": bool(solver.get("available"))
            if isinstance(solver, dict) else False,
            "discovery_state": (solver.get("discovery_state")
                                if isinstance(solver, dict) else None),
            "equilibrium": None,
            "gto": None,
            "exploitability": None,
            "blocked": not solver_ok,
            "solver_invoked": False,
            "note": "求解器不可用 / 未提供均衡参考 → 相关量记 None（**不猜**）。",
        },
        "blockers_from_preregistration": list(blockers),
        "blockers_measured_by_this_layer": False,
        "provenance": {
            "evidence_level": ("REAL_EXECUTION_DATA" if intake_permitted
                               else "UNAVAILABLE"),
            "evidence_grade": "NONE",
            "source_components": [
                {"stage": "V3H-12", "role": "preregistration / design surface",
                 "version": preregistration.get("version"),
                 "consumed_as": "gate_reference"},
                {"stage": "V3H-13", "role": "pre-execution intake gate",
                 "version": CASH_STRATEGY_EXECUTION_INTAKE_VERSION},
            ],
            "execution_performed_by_this_layer": False,
            "fabricated_execution_environment": False,
            "wrote_any_file": False,
            "opened_any_client": False,
            "real_money_moved": False,
            "digest_is_authenticity_proof": False,
            "frozen_multistreet_or_sample_runouts_reused": False,
            "future_cards_generated": False,
            "runout_sampling_performed": False,
        },
        "result": None,
        "results_available": False,
        "analysis_executed": False,
        "executed": False,
        "evidence_level": ("REAL_EXECUTION_DATA" if intake_permitted
                           else "UNAVAILABLE"),
        "evidence_grade": "NONE",
        "uncertainty": {
            "measured": False,
            "note": ("未执行任何动作、未运行任何分析 → **无**统计不确定性可报告；"
                     "仅记录**未决条件**。"),
            "unresolved_preconditions": unmet,
            "unresolved_blockers": list(blockers),
        },
        "claim_boundary": dict(CLAIM_BOUNDARY),
        "unknown": sorted(set(batch_reasons)),
        "note": ("本层是**执行前接口 / 门禁**：不执行、不写盘、不连客户端、"
                 "不伪造环境、不产生结果。"),
    }


# ══ 5. 接纳清单导出（纯内存；不写盘） ═══════════════════════════════
def summarize_intake(state: Any, *, intake: Any = None) -> dict:
    """对一次接入结果做**描述性**汇总（只计数；不写盘、不统计）。"""
    if not _is_cash(state):
        return _stub("not_cash_mode_fail_closed")
    if not isinstance(intake, dict) or intake.get("present") is not True:
        return _stub("intake_required")

    rows = intake.get("row_checks")
    rows = rows if isinstance(rows, list) else []
    # 批次闸门未放行时**任何**行都不得计为已接纳（保持与 intake 口径一致）
    gate_open = intake.get("intake_permitted") is True
    reasons: dict[str, int] = {}
    admitted = refused = dup = 0
    for r in rows:
        if not isinstance(r, dict):
            continue
        if r.get("admitted") is True and gate_open:
            admitted += 1
        else:
            refused += 1
        if r.get("duplicate") is True:
            dup += 1
        for i in (r.get("issues") or []):
            tok = str(i).split(":", 1)[0]
            reasons[tok] = reasons.get(tok, 0) + 1

    return {
        "present": True,
        "version": CASH_STRATEGY_EXECUTION_INTAKE_VERSION,
        "mode": "cash",
        "counts": {
            "n_rows": len(rows),
            "n_admitted": admitted,
            "n_refused": refused,
            "n_duplicate": dup,
            "refusal_reasons": dict(sorted(reasons.items())),
            "batch_refusal_reasons": sorted(
                set(intake.get("batch_refusal_reasons") or [])),
        },
        "counts_only": True,
        "rates_forbidden": True,
        "rate": None,
        "pct": None,
        "rate_computed": False,
        "rate_refusal_reason": ("禁止把接入计数折算为比率 / 百分比："
                                "接入率会被误读为数据质量或收益指标。"),
        "intake_permitted": intake.get("intake_permitted") is True,
        "executed": False,
        "result": None,
        "results_available": False,
        "analysis_executed": False,
        "evidence_level": "UNAVAILABLE",
        "evidence_grade": "NONE",
        "claim_boundary": dict(CLAIM_BOUNDARY),
        "unknown": [],
        "note": "只做描述性计数；不写盘、不统计、不推断结果。",
    }
