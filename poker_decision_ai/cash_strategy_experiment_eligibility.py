"""V3H-9 — CASH Decision-to-Experiment Eligibility Audit（决策→实验资格审计）。

**为什么这是 V3H-9 的最小且必要能力（审计结论）**
--------------------------------------------------
现状（实测）：

| 已有 | 缺口 |
|---|---|
| V3G-10 `cash_profit_interface.evaluate_entry_gate` 定义**实验层**何时可进入（5 条判据），但**对决策一无所知**（实测其源码不引用任何 V3H 层） | **没有任何层把"某个具体决策"与"实验进入条件"对齐** |
| V3H-5/6/7/8 产出排序 / 条件化 / 搜索 / 对账建议，并记录 disagreement 与 UNKNOWN | 决策带不带得动"执行环境资格"没有留痕；也答不出"这条建议当前卡在哪一条" |

因此本层是**唯一缺失的最小环节**：把 V3H-8 的建议摘要与 V3G-10 的入口闸门对接，
逐决策记录 `blocked / missing / blocking_trace`，且**永不执行**。

**不引入新评分公式**：本层**不含任何评分/EV/equity** 计算；
闸门判据**完全复用** V3G-10 的 `evaluate_entry_gate()`（唯一口径，不重写五条判据）。

数据接口（生产 import 图零新增）
--------------------------------
**不在模块顶层 import 任何 V3G 层**，也**不做函数内延迟 import**——
V3G-11 边界测试按 AST 扫描**包内全部模块**（只豁免层自身文件），
任何形式的 import 都会被视为"未授权接线"。

因此：

* **闸门结果由调用方通过 `gate=` 传入**
  （由 `cash_profit_interface.evaluate_entry_gate(environment=...)` 产出）；
* 未传入 → **fail closed**：`eligible=False`、`status="BLOCKED"`、
  `entry_gate_unavailable` 记入 `unknown`，**绝不**自行放行或重实现判据；
* 本层仍**声明**期望的键集（`ENVIRONMENT_FACT_KEYS` / `GATE_MISSING_KEYS`），
  并由测试**解析上游源码**逐一校验（防静默漂移）。

**上游既有不对称（如实记录，不"修正"冻结面）**：V3G-10 的**输入**键是
`venue_and_compliance_authorized`，而其**输出** `missing` 键是
`venue_and_compliance_authorization`。两者不同名，本层精确建模两侧。

fail-closed 规则
----------------
以下任一情况 → 该决策 `eligible=false`：
1. 未提供闸门结果（或闸门不可用）；
2. `decision.present=false` 或未提供决策；
3. `decision.abstained=true`（对账弃权 / disagreement）；
4. `decision.level != "MODEL_INTERNAL"`（UNKNOWN / UNAVAILABLE）；
5. 闸门 `allowed=False`（**当前项目环境下恒为真**）；
6. 决策带未消解 `unknown` 或 `disagreements`。

硬约束
------
* CASH-only，fail closed；非 dict / 非 CASH → `present=False`；
* 不 import `random`；无 IO；同输入 → 同输出；
* 不执行任何实验；不产生收益/EV/equilibrium/exploitability 字段；
* `MODEL_INTERNAL` 仅描述**证据等级**，**不是**真实对手证据，也**不是**真实盈利证据。
"""
from __future__ import annotations

from typing import Any

CASH_STRATEGY_EXPERIMENT_ELIGIBILITY_VERSION = (
    "v3h9-cash-strategy-experiment-eligibility-v1")

#: V3G-10 闸门**读取**的环境事实键（调用方构造 `environment=` 时使用）。
#: 必须与其 `env.get(...)` 键逐一相同（测试解析上游源码强制）。
ENVIRONMENT_FACT_KEYS: tuple[str, ...] = (
    "executable_environment",
    "randomized_assignment",
    "venue_and_compliance_authorized",
    "design_sealed",
    "sample_size_meets_mde",
)

#: V3G-10 闸门**输出**的 `missing` 键（与其输入键**并不同名**——上游既有不对称）。
#: 必须与上游 `need` 字典键逐一相同（测试解析上游源码强制）。
GATE_MISSING_KEYS: tuple[str, ...] = (
    "executable_environment",
    "randomized_assignment",
    "venue_and_compliance_authorization",   # 注意：输入键是 ..._authorized
    "design_sealed",
    "sample_size_meets_mde",
)

#: 上游不对称的机器可读记录
UPSTREAM_KEY_ASYMMETRY: dict[str, str] = {
    "input_key": "venue_and_compliance_authorized",
    "output_missing_key": "venue_and_compliance_authorization",
    "source": "frozen V3G-10 cash_profit_interface.evaluate_entry_gate",
    "handling": "modelled exactly on both sides; upstream NOT modified",
}

CLAIM_BOUNDARY: dict[str, bool] = {
    "is_gto": False,
    "is_equilibrium": False,
    "is_solver_output": False,
    "is_profit_evidence": False,
    "is_real_opponent_evidence": False,
    "computes_scoring_or_ev": False,
    "reimplements_entry_gate": False,
    "executes_experiment": False,
    "imports_any_v3g_layer": False,
    "wired_to_production": False,
}


def _is_cash(state: Any) -> bool:
    if not isinstance(state, dict):
        return False
    gt = state.get("game_type")
    return isinstance(gt, str) and gt.strip().lower() == "cash"


def _stub(reason: str) -> dict:
    return {
        "present": False,
        "version": CASH_STRATEGY_EXPERIMENT_ELIGIBILITY_VERSION,
        "reason": reason,
        "evidence_grade": "NONE",
        "unknown": [reason],
        "eligible": False,
        "status": "BLOCKED",
        "result_status": "EXECUTION_ENVIRONMENT_REQUIRED",
        "executed": False,
        "profit_identifiable": False,
        "audit": [],
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }


def _normalize_gate(gate: dict) -> tuple[dict, list[str], list[str]]:
    """规整调用方传入的闸门结果（**不重实现判据**，只做形状校验）。

    返回 `(normalized, supplied_keys, rejected)`。任何非布尔/缺失的判据
    一律**视为未满足**并记录，**不静默补全、不猜测**。
    """
    supplied: list[str] = []
    rejected: list[str] = []
    allowed = gate.get("allowed")
    if not isinstance(allowed, bool):
        rejected.append(f"allowed_not_boolean:{type(allowed).__name__}")
        allowed = False
    missing_raw = gate.get("missing")
    missing: list[str] = []
    if isinstance(missing_raw, list):
        for m in missing_raw:
            if isinstance(m, str) and m.strip():
                missing.append(m.strip())
            else:
                rejected.append(f"missing_entry_invalid:{m!r}")
    else:
        rejected.append(f"missing_not_a_list:{type(missing_raw).__name__}")
    # 一致性自检：allowed=True 但 missing 非空 → 冲突，按未满足处理
    if allowed and missing:
        rejected.append("gate_inconsistent:allowed_true_with_missing")
        allowed = False
    return {"allowed": allowed, "missing": missing,
            "reason": gate.get("reason"), "criteria": gate.get("criteria")}, \
        supplied, rejected


def audit_experiment_eligibility(state: Any, *, decision: dict | None = None,
                                 gate: dict | None = None,
                                 solver: dict | None = None) -> dict:
    """逐决策记录"能否进入真实收益实验"（**只审计，绝不执行**）。

    Parameters
    ----------
    state : CASH state（非 CASH → present=False）
    decision : **必填**，V3H-8 `cash_strategy_decision_assembly.assemble_decision` 输出
    gate : **必填**，V3G-10 `cash_profit_interface.evaluate_entry_gate` 的输出；
        未提供 → fail closed（`entry_gate_unavailable`），**不自行放行**
    solver : 可选，**只读转述**的 solver 状态
    """
    if not _is_cash(state):
        return _stub("not_cash_mode_fail_closed")
    if not isinstance(decision, dict):
        return _stub("decision_required")

    unknown: list[str] = []
    blockers: list[dict] = []

    if not isinstance(gate, dict):
        gate_norm = {"allowed": False, "missing": list(GATE_MISSING_KEYS),
                     "reason": "entry_gate_not_supplied",
                     "criteria": None}
        rejected: list[str] = []
        unknown.append("entry_gate_unavailable")
    else:
        gate_norm, _, rejected = _normalize_gate(gate)
        for r in rejected:
            unknown.append(f"gate_shape_rejected:{r}")

    dec = decision
    if not dec.get("present"):
        blockers.append({"blocker": "decision_absent",
                         "detail": dec.get("reason")})
    if dec.get("abstained") is True:
        blockers.append({"blocker": "decision_abstained",
                         "detail": dec.get("abstention_reason")})
    if dec.get("present") and dec.get("level") != "MODEL_INTERNAL":
        blockers.append({"blocker": "decision_level_not_model_internal",
                         "detail": dec.get("level")})
    if dec.get("disagreements"):
        blockers.append({"blocker": "unresolved_disagreements",
                         "detail": len(dec["disagreements"])})
    dec_unknown = list(dec.get("unknown") or [])
    if dec_unknown:
        blockers.append({"blocker": "decision_reports_unknown",
                         "detail": dec_unknown[:8]})
        for u in dec_unknown:
            if u not in unknown:
                unknown.append(u)

    for m in gate_norm["missing"]:
        blockers.append({"blocker": "entry_gate_criterion_missing", "detail": m})

    eligible = (not blockers) and bool(gate_norm["allowed"])

    preferred = dec.get("preferred")
    audit_entry = {
        "decision_available": bool(dec.get("present")),
        "decision_level": dec.get("level"),
        "decision_abstained": bool(dec.get("abstained")),
        "evidence_grade": dec.get("evidence_grade"),
        "selected_by_tier": (preferred or {}).get("selected_by_tier"),
        "candidate_key": (preferred or {}).get("candidate_key"),
        "score_source": (preferred or {}).get("score_source"),
        "leaf_coverage": (preferred or {}).get("leaf_coverage"),
        "n_disagreements": len(dec.get("disagreements") or []),
        "n_unknown": len(dec_unknown),
        "gate_supplied": isinstance(gate, dict),
        "gate_allowed": bool(gate_norm["allowed"]),
        "gate_missing": gate_norm["missing"],
        "blockers": blockers,
        "eligible_for_real_profit_experiment": eligible,
        "executed": False,
        "is_gto": False,
        "is_equilibrium": False,
    }

    if isinstance(solver, dict):
        solver_label = solver.get("solver_status", "UNKNOWN")
        solver_blocked = solver_label != "VALID_RESULT"
        passthrough = False
    else:
        solver_label = "UNKNOWN"
        solver_blocked = True
        passthrough = True
    if passthrough:
        unknown.append("solver_status_not_supplied")

    return {
        "present": True,
        "version": CASH_STRATEGY_EXPERIMENT_ELIGIBILITY_VERSION,
        "mode": "cash",
        "street": dec.get("street"),
        "eligible": eligible,
        "status": ("ELIGIBLE" if eligible else "BLOCKED"),
        "result_status": ("EXECUTION_ENVIRONMENT_REQUIRED"
                          if not eligible else "ELIGIBLE_NOT_EXECUTED"),
        "why_blocked": ("; ".join(
            f"{b['blocker']}({b.get('detail')})" for b in blockers)
            if blockers else ""),
        "entry_gate": {
            "allowed": bool(gate_norm["allowed"]),
            "missing": gate_norm["missing"],
            "reason": gate_norm["reason"],
            "criteria": gate_norm["criteria"],
            "source": "V3G-10 cash_profit_interface.evaluate_entry_gate",
            "supplied_by_caller": isinstance(gate, dict),
            "reimplemented_here": False,
        },
        "upstream_key_asymmetry": dict(UPSTREAM_KEY_ASYMMETRY),
        "audit": [audit_entry],
        "executed": False,
        "profit_identifiable": False,
        "n_blockers": len(blockers),
        "unknown": unknown,
        "level": ("MODEL_INTERNAL" if dec.get("present")
                  and dec.get("level") == "MODEL_INTERNAL" else "UNKNOWN"),
        "evidence_grade": ("MODEL_INTERNAL" if dec.get("present") else "NONE"),
        "solver": {"solver_status": solver_label, "blocked": bool(solver_blocked),
                   "read_only_passthrough": passthrough},
        "claim_boundary": dict(CLAIM_BOUNDARY),
        "note": ("本层**只审计资格**，不执行实验、不产生收益结论；"
                 "`MODEL_INTERNAL` 仅描述证据等级，"
                 "**不是**真实对手证据，也**不是**真实盈利证据。"),
    }
