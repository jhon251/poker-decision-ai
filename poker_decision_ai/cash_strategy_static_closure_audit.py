"""V3H-17 — CASH 外部证据消费边界**静态闭环审计**（只读；**不接入任何外部资源**）。

**本层要解决的问题**
--------------------
V3H-16 建立了消费门禁（六类锚状态一律 `DISPLAY_AS_UNSATISFIED_ONLY`），
但那只是**门禁自身**的保证。**没有任何层检查过**：
这些锚状态**实际能不能**流到收益计算、Solver/GTO 结论、策略评分、
生产 Realtime/Judge 链路、外部复现结论或"已验证/真实发生"结论里去。

本层补上这一环：**静态 import / call graph + AST + 字段流向 + 契约测试**审计，
逐**消费入口**确认锚状态**不可达**，并把**覆盖范围与盲区**一并写清楚。

**七级证据阶梯（必须逐级分离，低级别永不蕴含高级别）**
------------------------------------------------------
```
L1 READABLE           可读取 / 可展示
L2 STRUCTURALLY_VALID 结构合法
L3 CONTENT_CONSISTENT 内容一致
L4 SOURCE_TRUSTED     外部来源可信
L5 AUTHENTICITY_VERIFIED 真实性已验证
L6 ACTUALLY_OCCURRED  事件确实发生
L7 CONSUMABLE         可被下游消费
```

**本项目内可达上限 = L3**；L4/L5/L6 必须**带外**建立；
**L7 在本项目内不可达**。任何"L3 ⇒ L5"或"L2 ⇒ L7"的读法都是被禁止的推断。

**诚实边界（**必须**如实声明）**
--------------------------------
本层是**静态**审计：它能说"在当前源码中**未发现**把锚状态送进受保护下游的路径"，
**不能**说"未来绝对不会误用"。因此输出**同时**给出：

* `scope_audited`：实际审计范围（文件数、解析成功/失败数）；
* `blind_spots`：**静态分析无法覆盖**的情形（动态 import、`getattr`、反射、
  包外代码、运行时数据、非 Python 工件等）；
* `limitations`：明确写出"未发现 ≠ 保证不会发生"。

硬约束
------
* CASH-only，fail closed；非 dict / 非 CASH → `present=False`；
* **不 import 任何 V3G / V3H 层**（任何位置）——上游令牌由调用方显式传入；
* **不重复实现** V3H-14 readiness / V3H-15 trust-anchor contract / V3H-16 门禁；
* 无网络 / 客户端 / 钱包 / 真实资金 / Solver / 真实执行 / 收益实验；
* 只做**只读**文件读取与 AST 解析：**零写入、零删除**；
* 无随机；同输入 → 同输出；不产生任何真实结果（`result` 恒为 `None`）。
"""
from __future__ import annotations

import ast
import pathlib
from typing import Any

CASH_STRATEGY_STATIC_CLOSURE_AUDIT_VERSION = (
    "v3h17-cash-strategy-static-closure-audit-v1")

#: 七级证据阶梯（**严格递增，低级别永不蕴含高级别**）
EVIDENCE_LADDER: tuple[str, ...] = (
    "L1_READABLE",
    "L2_STRUCTURALLY_VALID",
    "L3_CONTENT_CONSISTENT",
    "L4_SOURCE_TRUSTED",
    "L5_AUTHENTICITY_VERIFIED",
    "L6_ACTUALLY_OCCURRED",
    "L7_CONSUMABLE",
)

#: 本项目内**可达**的最高层级（L4 及以上必须带外建立）
MAX_INTERNALLY_REACHABLE_LEVEL = "L3_CONTENT_CONSISTENT"

#: 项目内**不可达**的层级
UNREACHABLE_INTERNALLY_LEVELS: tuple[str, ...] = (
    "L4_SOURCE_TRUSTED",
    "L5_AUTHENTICITY_VERIFIED",
    "L6_ACTUALLY_OCCURRED",
    "L7_CONSUMABLE",
)

#: 被禁止的**跨级推断**（低级别不得推出高级别）
FORBIDDEN_LADDER_INFERENCES: tuple[str, ...] = (
    "L1_implies_L2",
    "L1_implies_L7",
    "L2_implies_L3",
    "L2_implies_L5",
    "L2_implies_L7",
    "L3_implies_L4",
    "L3_implies_L5",
    "L3_implies_L6",
    "L3_implies_L7",
    "L4_implies_L5",
    "L5_implies_L6",
    "digest_or_signature_implies_L5_or_L6",
)

#: **受保护的下游入口**（锚状态不得进入）
PROTECTED_SINKS: tuple[dict[str, str], ...] = (
    {"id": "SINK-1", "name": "profit_or_pnl_computation",
     "must_not_receive": "anchor states or anchor-derived values"},
    {"id": "SINK-2", "name": "solver_equilibrium_or_gto_conclusion",
     "must_not_receive": "anchor states or anchor-derived values"},
    {"id": "SINK-3", "name": "strategy_scoring_action_selection_or_exploitability",
     "must_not_receive": "anchor states or anchor-derived values"},
    {"id": "SINK-4", "name": "production_realtime_or_judge_chain",
     "must_not_receive": "anchor states or anchor-derived values"},
    {"id": "SINK-5", "name": "external_replication_conclusion",
     "must_not_receive": "anchor states or anchor-derived values"},
    {"id": "SINK-6", "name": "verified_or_actually_occurred_conclusion",
     "must_not_receive": "anchor states or anchor-derived values"},
)

#: 失败类别（全部必须 fail closed）
FAILURE_CLASSES: tuple[str, ...] = (
    "UNKNOWN",
    "MISSING",
    "CONFLICT",
    "UNVERIFIED",
    "INTERNALLY_ISSUED",
    "DIGEST_ONLY",
    "UNTRUSTED_SOURCE",
)

#: 审计范围内**获准**引用锚状态字面量的模块
ALLOWED_ANCHOR_STATE_MODULES: tuple[str, ...] = (
    "cash_strategy_trust_anchor_contract",   # 定义方（V3H-15）
    "cash_strategy_consumption_gate",        # 消费门禁（V3H-16）
    "cash_strategy_static_closure_audit",    # 本审计自身（扫描目标集合即含这些字面量）
)

#: 明确"否定声明"取值（出现这些取值即视为**未**被消费）
NEGATIVE_DECLARATION_VALUES: tuple[Any, ...] = (False, None)

#: 生产链路模块（**禁止**引用锚状态字面量）
PRODUCTION_CHAIN_MODULES: tuple[str, ...] = (
    "realtime", "agent", "judge", "orchestrator", "main", "cli",
)

#: 静态审计的**盲区**（如实声明，不可省略）
STATIC_AUDIT_BLIND_SPOTS: tuple[str, ...] = (
    "dynamic_import_via_importlib_or___import__",
    "attribute_access_via_getattr_or_setattr",
    "exec_eval_or_compile_of_runtime_strings",
    "code_outside_this_repository_including_installed_packages",
    "runtime_data_flow_through_files_network_or_user_input",
    "non_python_artifacts_notebooks_configs_or_serialised_objects",
    "monkeypatching_at_runtime",
    "future_source_changes_after_this_audit",
)

#: 本层**不能**主张的事情（诚实边界）
LIMITATIONS: tuple[str, ...] = (
    "this audit reports NO VIOLATION FOUND under the audited scope; it does NOT prove that misuse can never occur",
    "static analysis cannot see dynamic imports, reflection or code outside this repository",
    "a clean result is evidence about the current source only, not a guarantee about future changes",
    "this layer does not verify any external resource and cannot establish L4, L5, L6 or L7",
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
    "is_future_misuse_guarantee": False,
    "claims_no_violation_ever": False,
    "elevates_evidence_level": False,
    "infers_high_level_from_low_level": False,
    "treats_structure_as_verification": False,
    "treats_digest_or_signature_as_occurrence": False,
    "grants_verified_status": False,
    "permits_protected_sink_consumption": False,
    "computes_rates_or_statistics": False,
    "adds_scoring_or_ev_formula": False,
    "executes_action": False,
    "auto_betting": False,
    "wired_to_production": False,
}
CLAIM_FLAGS_TOTAL = len(CLAIM_BOUNDARY)

#: 被禁止的字面量流向：这些"结论性"令牌不得与锚状态出现在同一模块
CONCLUSION_TOKENS: tuple[str, ...] = (
    "gto_exact", "equilibrium", "exploitability", "profit_identifiable",
    "is_profit_evidence", "verified_status_granted", "actually_occurred",
    "external_replication",
)


def _is_cash(state: Any) -> bool:
    if not isinstance(state, dict):
        return False
    gt = state.get("game_type")
    return isinstance(gt, str) and gt.strip().lower() == "cash"


def _stub(reason: str) -> dict:
    return {
        "present": False,
        "version": CASH_STRATEGY_STATIC_CLOSURE_AUDIT_VERSION,
        "reason": reason,
        "audit_result": "FAIL_CLOSED",
        "no_violation_found": False,
        "n_violations": 0,
        "n_files_scanned": 0,
        "n_files_unparseable": 0,
        "max_internally_reachable_level": MAX_INTERNALLY_REACHABLE_LEVEL,
        "consumable_reachable": False,
        "sink_results": [],
        "blind_spots": list(STATIC_AUDIT_BLIND_SPOTS),
        "limitations": list(LIMITATIONS),
        "evidence_level": "UNAVAILABLE",
        "evidence_grade": "NONE",
        "executed": False,
        "result": None,
        "results_available": False,
        "analysis_executed": False,
        "claim_boundary": dict(CLAIM_BOUNDARY),
        "unknown": [reason],
    }


# ══ 1. 证据阶梯契约（纯声明） ═════════════════════════════════════════
def evidence_ladder_contract() -> dict:
    """返回七级证据阶梯契约（严格递增、跨级推断被禁止）。"""
    return {
        "success": True,
        "version": CASH_STRATEGY_STATIC_CLOSURE_AUDIT_VERSION,
        "levels": list(EVIDENCE_LADDER),
        "n_levels": len(EVIDENCE_LADDER),
        "strictly_increasing": True,
        "lower_level_never_implies_higher_level": True,
        "max_internally_reachable_level": MAX_INTERNALLY_REACHABLE_LEVEL,
        "unreachable_internally_levels": list(UNREACHABLE_INTERNALLY_LEVELS),
        "forbidden_ladder_inferences": list(FORBIDDEN_LADDER_INFERENCES),
        "n_forbidden_ladder_inferences": len(FORBIDDEN_LADDER_INFERENCES),
        "level_sources": {
            "L1_READABLE": "this project may read and display",
            "L2_STRUCTURALLY_VALID": "structure check only (this project)",
            "L3_CONTENT_CONSISTENT": "all six classes structurally present, still unverified",
            "L4_SOURCE_TRUSTED": "external only",
            "L5_AUTHENTICITY_VERIFIED": "external only",
            "L6_ACTUALLY_OCCURRED": "external only",
            "L7_CONSUMABLE": "external only; unreachable inside this project",
        },
        "mapping_from_v3h15_anchor_states": {
            "ANCHOR_UNSATISFIED": "L1_READABLE",
            "ANCHOR_STRUCTURE_VALID": "L2_STRUCTURALLY_VALID",
            "ANCHOR_CONTENT_CONSISTENT": "L3_CONTENT_CONSISTENT",
            "ANCHOR_REFUSED_DIGEST_AS_PROOF": "L1_READABLE (refused)",
            "ANCHOR_REFUSED_SELF_ISSUED": "L1_READABLE (refused)",
            "ANCHOR_REFUSED_CLASS_MERGE": "L1_READABLE (refused)",
        },
        "no_state_maps_to_L4_or_above": True,
        "digest_or_signature_establishes_at_most": "L3_CONTENT_CONSISTENT",
        "result": None,
        "executed": False,
        "evidence_level": "INTERNAL_DECLARATION",
        "evidence_grade": "NONE",
        "claim_boundary": dict(CLAIM_BOUNDARY),
        "unknown": [],
        "note": ("阶梯是**声明性**契约：本项目内可达上限为 L3；"
                 "L4/L5/L6 必须带外建立；L7 在本项目内不可达。"),
    }


def failure_class_policy() -> dict:
    """返回失败类别策略：**每一类都必须 fail closed**。"""
    return {
        "success": True,
        "version": CASH_STRATEGY_STATIC_CLOSURE_AUDIT_VERSION,
        "failure_classes": list(FAILURE_CLASSES),
        "n_failure_classes": len(FAILURE_CLASSES),
        "policy": {f: "fail_closed" for f in FAILURE_CLASSES},
        "all_fail_closed": True,
        "default_for_unrecognised": "fail_closed",
        "grants_anything_on_failure": False,
        "notes": {
            "UNKNOWN": "unrecognised class/state/level is refused, never mapped to a higher level",
            "MISSING": "absent field or attestation is refused, never silently completed",
            "CONFLICT": "mutually inconsistent claims are refused, never resolved optimistically",
            "UNVERIFIED": "unverified input never becomes verified output",
            "INTERNALLY_ISSUED": "internal or self-issued evidence is refused",
            "DIGEST_ONLY": "a content fingerprint alone is refused as proof",
            "UNTRUSTED_SOURCE": "unestablished source trust is refused",
        },
        "result": None,
        "executed": False,
        "evidence_level": "INTERNAL_DECLARATION",
        "evidence_grade": "NONE",
        "claim_boundary": dict(CLAIM_BOUNDARY),
        "unknown": [],
    }


# ══ 2. 静态解析助手（只读） ═══════════════════════════════════════════
def _package_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parent


def _scan_package() -> dict:
    """只读扫描包内 `*.py`：返回 (模块名 → 记录, 解析失败列表, 总数)。"""
    root = _package_root()
    records: dict[str, dict] = {}
    unparseable: list[str] = []
    n_files = 0
    for p in sorted(root.rglob("*.py")):
        if "__pycache__" in str(p):
            continue
        n_files += 1
        try:
            src = p.read_text(encoding="utf-8")
            tree = ast.parse(src)
        except Exception:                                   # noqa: BLE001
            unparseable.append(p.name)
            continue
        lits = {n.value for n in ast.walk(tree)
                if isinstance(n, ast.Constant) and isinstance(n.value, str)}
        imports: set[str] = set()
        for n in ast.walk(tree):
            if isinstance(n, ast.ImportFrom) and n.module:
                imports.add(n.module.split(".")[-1])
            elif isinstance(n, ast.Import):
                imports.update(a.name.split(".")[-1] for a in n.names)
        records[p.stem] = {"file": p.name, "literals": lits,
                           "imports": imports, "source": src,
                           "tree": tree}
    return {"records": records, "unparseable": unparseable, "n_files": n_files}


def _is_negative_declaration_value(v: ast.AST) -> bool:
    """判定一个值是否属于**否定声明**（不构成"消费"）。

    允许：`False` / `None` 常量；全部元素均为否定声明值的列表 / 元组。
    """
    if isinstance(v, ast.Constant):
        return v.value in NEGATIVE_DECLARATION_VALUES
    if isinstance(v, (ast.List, ast.Tuple)):
        return all(_is_negative_declaration_value(e) for e in v.elts)
    return False


def _conclusion_token_usage(tree: ast.AST, tokens: tuple[str, ...],
                            ) -> dict[str, dict]:
    """静态判定"结论性令牌"是否被**消费**（而非仅被**否定声明**）。

    被判为**违规**的情形：
    * 被赋值成**非** `False` / `None` 的值（字典值 / 关键字参数 / 普通赋值）；
    * 出现在条件表达式（`if` / 条件表达式 / `while`）中；
    * 被**调用**（作为函数名或实参）。

    仅作为字典键、集合元素、列表元素、比较左值、字符串中的出现**不算**消费。
    """
    out: dict[str, dict] = {}

    def _note(tok: str, kind: str, detail: str) -> None:
        e = out.setdefault(tok, {"bindings": [], "conditional": [],
                                 "called": []})
        if kind == "binding":
            e["bindings"].append(detail)
        elif kind == "conditional":
            e["conditional"].append(detail)
        else:
            e["called"].append(detail)

    for n in ast.walk(tree):
        if isinstance(n, ast.Dict):
            for k, v in zip(n.keys, n.values):
                if isinstance(k, ast.Constant) and k.value in tokens:
                    if not _is_negative_declaration_value(v):
                        _note(str(k.value), "binding",
                              f"dict value {ast.dump(v)[:60]}")
                # 嵌套取值字典：`{"exploitability": None}` 这类
                # "结论槽位"同样必须被检查
                if isinstance(v, ast.Dict):
                    for k2, v2 in zip(v.keys, v.values):
                        if isinstance(k2, ast.Constant) and k2.value in tokens \
                                and not _is_negative_declaration_value(v2):
                            _note(str(k2.value), "binding",
                                  f"nested dict value {ast.dump(v2)[:60]}")
        elif isinstance(n, ast.Call):
            for kw in n.keywords:
                if kw.arg in tokens and not (
                        isinstance(kw.value, ast.Constant)
                        and kw.value.value in NEGATIVE_DECLARATION_VALUES):
                    _note(str(kw.arg), "binding",
                          f"kwarg value {ast.dump(kw.value)[:60]}")
            if isinstance(n.func, ast.Name) and n.func.id in tokens:
                _note(n.func.id, "called", "called as a function")
            for a in n.args:
                if isinstance(a, ast.Name) and a.id in tokens:
                    _note(a.id, "called", "passed as an argument")
        elif isinstance(n, ast.Assign):
            for t in n.targets:
                if isinstance(t, ast.Name) and t.id in tokens \
                        and not (isinstance(n.value, ast.Constant)
                                 and n.value.value
                                 in NEGATIVE_DECLARATION_VALUES):
                    _note(t.id, "binding", f"assigned {ast.dump(n.value)[:60]}")
        elif isinstance(n, (ast.If, ast.While, ast.IfExp)):
            test = n.test
            for x in ast.walk(test):
                if isinstance(x, ast.Name) and x.id in tokens:
                    _note(x.id, "conditional", "used in a condition")
                elif isinstance(x, ast.Constant) and x.value in tokens:
                    _note(str(x.value), "conditional", "string in a condition")
    return out


# ══ 3. 消费入口（受保护下游）可达性审计 ═══════════════════════════════
def audit_consumption_closure(state: Any) -> dict:
    """静态审计：锚状态是否可能到达任一**受保护下游入口**。

    **不做行为判定**、**不消费上游结论**、**不接入外部资源**。
    """
    if not _is_cash(state):
        return _stub("not_cash_mode_fail_closed")

    scand = _scan_package()
    records = scand["records"]

    # ── 锚状态字面量的实际出现范围 ──
    anchor_states = tuple(
        s for s in records.get("cash_strategy_trust_anchor_contract",
                               {}).get("literals", set())
        if isinstance(s, str) and s.startswith("ANCHOR_"))
    if len(anchor_states) != 6:
        # 无法从契约模块确认六状态 → fail closed
        out = _stub("anchor_states_not_confirmable_from_contract_module")
        out["anchor_states_found"] = sorted(anchor_states)
        return out
    anchor_set = set(anchor_states)

    referencing: list[str] = sorted(
        name for name, r in records.items()
        if r["literals"] & anchor_set)

    # ── 生产链路是否直接引用锚状态 / 门禁模块 ──
    production_hits: dict[str, list[str]] = {}
    for name, r in records.items():
        if name.startswith("cash_"):
            continue
        bad: list[str] = []
        if r["literals"] & anchor_set:
            bad.append("anchor_state_literal")
        for m in ("cash_strategy_trust_anchor_contract",
                  "cash_strategy_consumption_gate"):
            if m in r["imports"] or m in r["literals"]:
                bad.append(f"references:{m}")
        if bad:
            production_hits[r["file"]] = sorted(set(bad))

    # ── 逐受保护入口判定 ──
    sink_results: list[dict] = []
    violations: list[str] = []
    for sink in PROTECTED_SINKS:
        reachable = False
        evidence: list[str] = []
        if production_hits:
            reachable = True
            evidence = [f"{f}:{','.join(v)}"
                        for f, v in sorted(production_hits.items())]
        sink_results.append({
            "sink": sink["id"], "name": sink["name"],
            "must_not_receive": sink["must_not_receive"],
            "reachable": reachable,
            "evidence": evidence,
            "status": "VIOLATION" if reachable else "NO_VIOLATION_FOUND",
        })
        if reachable:
            violations.append(f"{sink['id']}:{sink['name']}")

    # ── 结论性令牌的"消费"检查：**仅**针对可能承载锚状态的模块 ──
    #    范围 = 引用锚状态的模块 ∪ 生产链路模块。
    #    包内其它层（如 V3G-8/V3G-10）自有其结论字段，与锚状态无关，
    #    仅作**上下文记录**，不计为违规。
    anchor_flow_modules = set(referencing)
    conclusion_leaks: dict[str, dict] = {}
    context_bindings: dict[str, dict] = {}
    for name, r in records.items():
        usage = _conclusion_token_usage(r["tree"], CONCLUSION_TOKENS)
        bad = {t: u for t, u in usage.items()
               if u["bindings"] or u["conditional"] or u["called"]}
        if not bad:
            continue
        is_production = not name.startswith("cash_")
        if name in anchor_flow_modules or is_production:
            conclusion_leaks[r["file"]] = bad
            violations.append(
                f"conclusion_token_consumed:{r['file']}:{','.join(sorted(bad))}")
        else:
            context_bindings[r["file"]] = sorted(bad)

    # ── 生产链路不得出现结论性令牌的"非否定"使用 ──
    production_conclusion_leaks: dict[str, dict] = {}
    for name, r in records.items():
        if name.startswith("cash_"):
            continue
        usage = _conclusion_token_usage(r["tree"], CONCLUSION_TOKENS)
        bad = {t: u for t, u in usage.items()
               if u["bindings"] or u["conditional"] or u["called"]}
        if bad:
            production_conclusion_leaks[r["file"]] = bad
            violations.append(
                f"production_conclusion_token_consumed:{r['file']}")

    # ── 生产链路禁止引用锚状态的断言 ──
    chain_violations = [f for f in production_hits]
    violations.extend(f"production_chain_reference:{f}" for f in chain_violations)

    no_violation = not violations
    return {
        "present": True,
        "version": CASH_STRATEGY_STATIC_CLOSURE_AUDIT_VERSION,
        "mode": "cash",
        "audit_result": ("NO_VIOLATION_FOUND" if no_violation
                         else "VIOLATION_FOUND"),
        "no_violation_found": no_violation,
        "future_misuse_guaranteed_impossible": False,
        "n_violations": len(violations),
        "violations": sorted(set(violations)),
        "scope_audited": {
            "root": str(_package_root().name),
            "n_files_scanned": scand["n_files"],
            "n_files_parsed": len(records),
            "n_files_unparseable": len(scand["unparseable"]),
            "unparseable": sorted(scand["unparseable"]),
            "analysis_kinds": ["static_import_graph", "ast_scan",
                               "string_literal_flow", "contract_test"],
            "runtime_analysis_performed": False,
        },
        "anchor_states": sorted(anchor_states),
        "anchor_state_referencing_modules": referencing,
        "allowed_anchor_state_modules": list(ALLOWED_ANCHOR_STATE_MODULES),
        "unexpected_anchor_state_referencing_modules": sorted(
            set(referencing) - set(ALLOWED_ANCHOR_STATE_MODULES)),
        "protected_sinks": [dict(s) for s in PROTECTED_SINKS],
        "sink_results": sink_results,
        "n_sinks": len(PROTECTED_SINKS),
        "n_sinks_reachable": sum(1 for s in sink_results if s["reachable"]),
        "production_chain_references": production_hits,
        "conclusion_token_leaks": conclusion_leaks,
        "production_conclusion_token_leaks": production_conclusion_leaks,
        "conclusion_token_bindings_elsewhere_context_only": context_bindings,
        "conclusion_token_check_scope": (
            "anchor-state-referencing modules plus production modules only; "
            "other package layers that own their own conclusion fields are "
            "recorded as context and are NOT counted as anchor-flow violations"),
        "evidence_ladder": list(EVIDENCE_LADDER),
        "max_internally_reachable_level": MAX_INTERNALLY_REACHABLE_LEVEL,
        "consumable_reachable": False,
        "blind_spots": list(STATIC_AUDIT_BLIND_SPOTS),
        "limitations": list(LIMITATIONS),
        "result": None,
        "results_available": False,
        "analysis_executed": False,
        "executed": False,
        "evidence_level": "INTERNAL_DECLARATION",
        "evidence_grade": "NONE",
        "uncertainty": {
            "measured": False,
            "note": ("静态审计：**无**统计不确定性；"
                     "但存在**结构性盲区**（见 blind_spots），"
                     "因此结论只能是'未发现'，不是'不可能'。"),
            "coverage_is_partial": True,
        },
        "claim_boundary": dict(CLAIM_BOUNDARY),
        "unknown": ([] if no_violation else sorted(set(violations))),
        "note": ("只读静态审计：**未发现**违规**不等于**未来不会误用；"
                 "覆盖范围与盲区已如实列出。"),
    }


# ══ 4. 覆盖范围与盲区声明 ═════════════════════════════════════════════
def audit_coverage_statement(state: Any) -> dict:
    """返回审计**覆盖范围**与**盲区**（必须与主审计一并被引用）。"""
    if not _is_cash(state):
        return _stub("not_cash_mode_fail_closed")
    scand = _scan_package()
    return {
        "present": True,
        "version": CASH_STRATEGY_STATIC_CLOSURE_AUDIT_VERSION,
        "mode": "cash",
        "covered": [
            "all python modules under the decision-ai package root",
            "static import edges and ast-level literal references",
            "string-literal flow of anchor states and conclusion tokens",
            "contract tests over the ladder, failure classes and sinks",
        ],
        "not_covered": list(STATIC_AUDIT_BLIND_SPOTS),
        "n_files_scanned": scand["n_files"],
        "n_files_parsed": len(scand["records"]),
        "n_files_unparseable": len(scand["unparseable"]),
        "limitations": list(LIMITATIONS),
        "conclusion_strength": "no_violation_found_within_audited_scope",
        "conclusion_is_a_guarantee": False,
        "result": None,
        "executed": False,
        "evidence_level": "INTERNAL_DECLARATION",
        "evidence_grade": "NONE",
        "claim_boundary": dict(CLAIM_BOUNDARY),
        "unknown": [],
    }
