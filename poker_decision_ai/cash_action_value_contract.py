"""V3I-G1 Phase 1 — CASH **Action Value Contract**（唯一收益记账定义，纯函数）。

定位
----
把"一个动作值多少钱"这件事从**多层各自的隐含约定**收敛为**一份显式契约**：

```
V(a) = p_fold(a) · fold_value(a) + p_continue(a) · showdown_value(a)

fold_value(a)     = 1.0                                   （赢下底池；自身投入收回）
showdown_value(a) = eq(a) · (pot₀ + 2a) / pot₀ − a / pot₀  （相对**自身投入**的净值）
a = 0（check）    ⇒ 对手不可能对 check 弃牌 ⇒ 无 fold 分支，V = eq
```

三条**定义性**性质（不是拟合，也不是启发式）：

1. **零自由参数**：所有系数都是定义（`1.0` / `2a` / `a`），没有可调权重；
2. **严格推广旧口径**：令 `p_continue ≡ 1` 且 `eq(a) = mean_eq`，则
   `V(a) ≡ eq·(pot₀+2a)/pot₀ − a/pot₀` = 旧 `search_runout_pot_share_v1` **逐位相同**；
   因此新值是旧公式的**严格推广**（新增 fold 项与加权），旧口径不删除、并列保留；
3. **权重与权益必须来自测量**：`p_fold/p_continue` 与 `eq(a)` 必须携带来源与
   定义性指纹；缺失/来源不符 ⇒ **UNKNOWN（fail closed）**，**禁止任何默认值**。

硬约束
------
* CASH-only；不 import 任何 V3G 层；不 import `random`；同输入 → 同输出；
* **不改** gate / tiers / override / on_disagreement / ranking / loop 状态语义；
* 不声称 GTO / equilibrium / exploitability / 盈利 / 真人优势；
* 叶预算被违反时**抑制动作价值**（legacy 分数照旧并列报告）。

探测器（`detect_degeneracy`）只做**定义性**一致性判定（不是质量判定）：

* L1 门槛越高 ⇒ 续跟类别越少/概率不增（`continuing set = {classes : eq ≥ threshold}` 的直接推论）；
* L2 **同一门槛** ⇒ 同一续跟集合 ⇒ `p_continue`/`n_continuation_classes`/`eq` 必须相同；
* L3 每条记录的定义性指纹必须与其自身基（population/threshold/类别数/权益）一致；
* 另：恒定权益、权重缺失或和不为 1、fold-only 证据基、指纹不符、
  无动作间分化、叶预算违反、候选集不一致等**结构性**退化。
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

CASH_ACTION_VALUE_CONTRACT_VERSION = "v3i-g1-phase1-action-value-contract-v1"

#: 唯一记账契约（显式、可复算、零自由参数）
CONTRACT: dict[str, Any] = {
    "name": "action_sensitive_net_pot_share_v1",
    "unit": "pot_share_of_initial_pot",
    "free_parameters": 0,
    "value_definition": "p_fold(a)*fold_value(a) + p_continue(a)*showdown_value(a)",
    "fold_value_definition": "1.0 (pot won net of own investment: (pot0 + a - a)/pot0)",
    "showdown_value_definition": "eq(a)*(pot0 + 2a)/pot0 - a/pot0 (net of own investment)",
    "zero_investment_rule": ("a == 0 => the opponent cannot fold to a check; "
                             "no fold branch; V = eq"),
    "weights_source": ("measured p_fold/p_continue per candidate; no default, "
                       "no fitting"),
    "equity_basis_required": "equity_vs_continuing_measured",
    "missing_input_policy": ("UNKNOWN (fail closed); the legacy score is reported "
                             "separately and is never substituted"),
    "degeneration_property": ("p_continue == 1 with eq == mean_eq reproduces the "
                              "legacy search_runout_pot_share_v1 exactly"),
    "budget_violation_policy": ("leaf budget exceeded => action value suppressed "
                                "(UNKNOWN) while the legacy score stays reported"),
    "candidate_set_policy": ("values are comparable only within one candidate set; "
                             "cross-tier set mismatch is reported separately and is "
                             "NOT an evidence disagreement"),
    "uses_randomness": False,
    "is_gto": False,
    "is_equilibrium": False,
    "is_final_decision": False,
    "claims_quality": False,
}

CLAIM_BOUNDARY: dict[str, bool] = {
    "is_gto": False,
    "is_equilibrium": False,
    "is_solver_output": False,
    "is_profit_evidence": False,
    "computes_exploitability": False,
    "claims_real_opponent_range": False,
    "claims_quality_or_strength": False,
    "modifies_gate_tiers_or_override": False,
    "enables_voting": False,
    "wired_to_production": False,
}

#: 定义性等值容差（**不是**可调参数；仅吸收 6 位舍入）
TOL = 1e-06

#: 记录中必须携带的定义性字段
RECORD_FIELDS: tuple[str, ...] = (
    "action", "amount_bb", "equity_vs_continuing", "p_continue", "p_fold",
    "threshold", "n_continuation_classes", "population", "equity_basis",
    "basis_fingerprint",
)

_AGGRESSIVE = {"bet", "raise", "allin", "all_in"}


def _is_num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _r6(v: Any) -> Any:
    return round(float(v), 6) if _is_num(v) else v


def candidate_key(action: Any, amount_bb: Any) -> str:
    """候选身份键（动作 + 注额；与 V3I-P8/P13 口径一致）。"""
    amt = _r6(amount_bb if _is_num(amount_bb) else 0.0)
    return f"{action}|{amt}"


def basis_fingerprint(record: dict) -> str:
    """记录基的**定义性**指纹：population + threshold + 类别数 + 权益。

    仅由"该候选自身的续跟测量基"决定，与值无关；用于发现
    "权益被置换而其测量基未变"这类不一致（L3）。
    """
    basis = {
        "population": record.get("population"),
        "threshold": _r6(record.get("threshold")),
        "n_continuation_classes": record.get("n_continuation_classes"),
        "equity_vs_continuing": _r6(record.get("equity_vs_continuing")),
        "p_continue": _r6(record.get("p_continue")),
    }
    text = json.dumps(basis, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _validate_record(rec: Any) -> tuple[dict | None, list[str]]:
    """校验单条测量记录（fail closed：任何缺失/来源不符都不放行）。"""
    unknown: list[str] = []
    if not isinstance(rec, dict):
        return None, ["record_not_a_dict"]
    for f in RECORD_FIELDS:
        if f not in rec:
            unknown.append(f"record_field_missing:{f}")
    if unknown:
        return None, unknown
    if rec.get("equity_basis") != CONTRACT["equity_basis_required"]:
        unknown.append(
            f"equity_basis_not_measured:{rec.get('equity_basis')!r}")
    for f in ("equity_vs_continuing", "p_continue", "p_fold", "amount_bb"):
        if not _is_num(rec.get(f)):
            unknown.append(f"record_field_not_numeric:{f}")
    # 门槛只对**有投入**的候选有意义（check 无门槛：对手不可能对 check 弃牌）
    if _is_num(rec.get("amount_bb")) and float(rec["amount_bb"]) > 0.0 \
            and not _is_num(rec.get("threshold")):
        unknown.append("record_field_not_numeric:threshold")
    if unknown:
        return None, unknown
    p_c = float(rec["p_continue"])
    p_f = float(rec["p_fold"])
    eq = float(rec["equity_vs_continuing"])
    if p_c < -TOL or p_c > 1.0 + TOL or p_f < -TOL or p_f > 1.0 + TOL:
        unknown.append("probability_out_of_range")
    if abs((p_c + p_f) - 1.0) > 1e-04:
        unknown.append(f"weights_do_not_sum_to_one:{round(p_c + p_f, 6)}")
    if eq < -TOL or eq > 1.0 + TOL:
        unknown.append("equity_out_of_range")
    if not isinstance(rec.get("n_continuation_classes"), int) \
            or isinstance(rec.get("n_continuation_classes"), bool):
        unknown.append("n_continuation_classes_not_int")
    if unknown:
        return None, unknown
    want_fp = basis_fingerprint(rec)
    if rec.get("basis_fingerprint") != want_fp:
        unknown.append(
            f"basis_fingerprint_mismatch:{rec.get('basis_fingerprint')}!={want_fp}")
    if unknown:
        return None, unknown
    return dict(rec), []


def candidate_action_value(*, action: Any, amount_bb: Any, pot_bb: Any,
                           record: dict) -> dict:
    """单候选动作价值（净 pot-share）。输入不完整 ⇒ status=UNKNOWN。"""
    base: dict[str, Any] = {
        "action": action,
        "amount_bb": _r6(amount_bb),
        "contract": CONTRACT["name"],
        "unit": CONTRACT["unit"],
        "status": "UNKNOWN",
        "value": None,
        "unknown": [],
        "terms": {},
        "provenance": {
            "contract_version": CASH_ACTION_VALUE_CONTRACT_VERSION,
            "free_parameters": 0,
        },
        "is_gto": False,
        "is_equilibrium": False,
    }
    if not _is_num(pot_bb) or float(pot_bb) <= 0:
        base["unknown"] = ["pot_nonpositive"]
        return base
    rec, unknown = _validate_record(record)
    if rec is None:
        base["unknown"] = unknown
        return base

    pot0 = float(pot_bb)
    a = float(rec["amount_bb"]) if _is_num(rec.get("amount_bb")) else 0.0
    eq = float(rec["equity_vs_continuing"])
    p_c = float(rec["p_continue"])
    p_f = float(rec["p_fold"])
    act = str(action or "").strip().lower()

    if act in _AGGRESSIVE and _is_num(amount_bb):
        a = max(a, 0.0)
    if a <= 0.0:
        # 零投入：对手不可能对 check 弃牌 ⇒ 无 fold 分支（定义，不是假设）
        value = eq * (pot0 + 0.0) / pot0 - 0.0 / pot0
        base.update({
            "status": "OK",
            "value": round(value, 6),
            "terms": {
                "fold_applicable": False,
                "fold_probability": 0.0,
                "fold_value": None,
                "fold_term": 0.0,
                "showdown_probability": 1.0,
                "showdown_value": round(value, 6),
                "showdown_term": round(value, 6),
                "showdown_pot_bb": round(pot0, 6),
                "hero_investment_bb": 0.0,
                "terminal_values_distinct": 1,
            },
            "provenance": {
                "contract_version": CASH_ACTION_VALUE_CONTRACT_VERSION,
                "free_parameters": 0,
                "zero_investment_rule": CONTRACT["zero_investment_rule"],
                "population": rec.get("population"),
                "threshold": _r6(rec.get("threshold")),
                "n_continuation_classes": rec["n_continuation_classes"],
                "equity_basis": rec["equity_basis"],
                "basis_fingerprint": rec["basis_fingerprint"],
            },
            "measured": {"equity_vs_continuing": _r6(eq),
                         "p_continue": _r6(p_c), "p_fold": _r6(p_f)},
        })
        return base

    fold_value = 1.0
    showdown_pot = pot0 + 2.0 * a
    showdown_value = eq * showdown_pot / pot0 - a / pot0
    value = p_f * fold_value + p_c * showdown_value
    base.update({
        "status": "OK",
        "value": round(value, 6),
        "terms": {
            "fold_applicable": True,
            "fold_probability": _r6(p_f),
            "fold_value": round(fold_value, 6),
            "fold_term": round(p_f * fold_value, 6),
            "showdown_probability": _r6(p_c),
            "showdown_value": round(showdown_value, 6),
            "showdown_term": round(p_c * showdown_value, 6),
            "showdown_pot_bb": round(showdown_pot, 6),
            "hero_investment_bb": _r6(a),
            "terminal_values_distinct": 2 if (p_f > TOL and p_c > TOL) else 1,
        },
        "provenance": {
            "contract_version": CASH_ACTION_VALUE_CONTRACT_VERSION,
            "free_parameters": 0,
            "population": rec.get("population"),
            "threshold": _r6(rec.get("threshold")),
            "n_continuation_classes": rec["n_continuation_classes"],
            "equity_basis": rec["equity_basis"],
            "basis_fingerprint": rec["basis_fingerprint"],
        },
        "measured": {"equity_vs_continuing": _r6(eq),
                     "p_continue": _r6(p_c), "p_fold": _r6(p_f)},
    })
    return base


def aggregate_action_values(*, pot_bb: Any, candidates: Any,
                            records: Any) -> dict:
    """按候选集聚合成动作价值表（**唯一的聚合实现**）。

    Parameters
    ----------
    pot_bb : 初始底池（单位基准 pot₀）
    candidates : [{action, amount_bb}, ...]（**当前候选集**，顺序被保留）
    records : {候选键 或 (action, amount) 元组: 测量记录}

    Returns
    -------
    dict（含 `values` / `ranked` / `preferred` / `differentiation` /
    `unknown` / `claim_boundary`）；任何输入不可用 ⇒ 对应候选 `UNKNOWN`，
    **不设默认权重**。
    """
    out: dict[str, Any] = {
        "present": False,
        "version": CASH_ACTION_VALUE_CONTRACT_VERSION,
        "contract": dict(CONTRACT),
        "unit": CONTRACT["unit"],
        "free_parameters": 0,
        "pot_bb": _r6(pot_bb),
        "values": {},
        "ranked": [],
        "preferred": None,
        "check_baseline": None,
        "differentiation": {},
        "unknown": [],
        "claim_boundary": dict(CLAIM_BOUNDARY),
        "final_decision_made": False,
    }
    if not _is_num(pot_bb) or float(pot_bb) <= 0:
        out["unknown"] = ["pot_nonpositive"]
        out["reason"] = "pot_nonpositive"
        return out

    rec_map: dict[str, dict] = {}
    if isinstance(records, dict):
        for k, v in records.items():
            rec_map[str(k)] = v
    elif isinstance(records, list):
        for v in records:
            if isinstance(v, dict):
                rec_map[candidate_key(v.get("action"), v.get("amount_bb"))] = v
    else:
        out["unknown"] = ["records_required"]
        out["reason"] = "records_required"
        return out

    cand_list = [c for c in (candidates or []) if isinstance(c, dict)]
    if not cand_list:
        out["unknown"] = ["candidates_required"]
        out["reason"] = "candidates_required"
        return out

    values: dict[str, Any] = {}
    unknown: list[str] = []
    rejections: list[dict] = []
    for c in cand_list:
        act = c.get("action")
        amt = c.get("amount_bb")
        key = candidate_key(act, amt)
        rec = rec_map.get(key)
        if rec is None:
            # 也接受 (action, amount) 元组键
            rec = rec_map.get(f"({act}, {_r6(amt)})")
        if rec is None:
            values[key] = {
                "action": act, "amount_bb": _r6(amt),
                "contract": CONTRACT["name"], "unit": CONTRACT["unit"],
                "status": "UNKNOWN", "value": None, "gain_vs_check": None,
                "unknown": ["response_weights_not_supplied_for_candidate"],
                "terms": {}, "provenance": {"free_parameters": 0},
                "is_gto": False, "is_equilibrium": False,
            }
            unknown.append(f"response_weights_not_supplied:{key}")
            rejections.append({"key": key, "action": act,
                               "amount_bb": _r6(amt),
                               "reasons": ["response_weights_not_supplied"]})
            continue
        row = candidate_action_value(action=act, amount_bb=amt,
                                     pot_bb=pot_bb, record=rec)
        values[key] = row
        if row["status"] != "OK":
            unknown.extend(f"{u}:{key}" for u in (row.get("unknown") or []))
            rejections.append({"key": key, "action": act, "amount_bb": _r6(amt),
                               "reasons": list(row.get("unknown") or [])})

    check_key = None
    for k, v in values.items():
        if str(v.get("action") or "").strip().lower() == "check":
            check_key = k
            break
    check_value = (values[check_key]["value"] if check_key
                   and values[check_key].get("status") == "OK" else None)

    for k, v in values.items():
        if v.get("status") == "OK" and _is_num(v.get("value")):
            v["gain_vs_check"] = (round(float(v["value"]) - float(check_value), 6)
                                  if _is_num(check_value) else None)
    scored = [dict(v, key=k) for k, v in values.items()
              if v.get("status") == "OK" and _is_num(v.get("value"))]
    scored.sort(key=lambda r: (-float(r["value"]), str(r["action"]),
                               float(r["amount_bb"] or 0.0)))
    for i, row in enumerate(scored, 1):
        row["rank_by_action_value"] = i

    eq_vals = sorted({_r6(v["measured"]["equity_vs_continuing"])
                      for v in values.values() if v.get("status") == "OK"})
    w_vals = sorted({(v["measured"]["p_continue"], v["measured"]["p_fold"])
                     for v in values.values() if v.get("status") == "OK"})
    val_vals = sorted({float(v["value"]) for v in values.values()
                       if v.get("status") == "OK"})
    src = []
    if len(eq_vals) > 1:
        src.append("equity_vs_continuing")
    if len(w_vals) > 1:
        src.append("fold_continue_weights")
    if len(val_vals) > 1 and len(eq_vals) <= 1 and len(w_vals) <= 1:
        src.append("investment_terms_only")

    out.update({
        "present": True,
        "values": values,
        "rejections": rejections,
        "ranked": [{"key": r["key"], "action": r["action"],
                    "amount_bb": r["amount_bb"], "value": r["value"],
                    "gain_vs_check": r.get("gain_vs_check"),
                    "p_fold": r["measured"]["p_fold"],
                    "p_continue": r["measured"]["p_continue"],
                    "equity_vs_continuing":
                        r["measured"]["equity_vs_continuing"],
                    "rank_by_action_value": r["rank_by_action_value"]}
                   for r in scored],
        "preferred": ({k: scored[0][k] for k in
                       ("key", "action", "amount_bb", "value",
                        "gain_vs_check", "rank_by_action_value")}
                      if scored else None),
        "check_baseline": ({"key": check_key, "value": check_value}
                           if check_key else None),
        "n_candidates": len(cand_list),
        "n_valued": len(scored),
        "differentiation": {
            "n_distinct_values": len(val_vals),
            "min_value": val_vals[0] if val_vals else None,
            "max_value": val_vals[-1] if val_vals else None,
            "spread": (round(val_vals[-1] - val_vals[0], 6) if val_vals else None),
            "n_distinct_equity_inputs": len(eq_vals),
            "n_distinct_weight_pairs": len(w_vals),
            "source": "+".join(src) if src else "none",
            "differentiates_actions": len(val_vals) > 1,
        },
        "unknown": sorted(set(unknown)),
    })
    return out


def candidate_set_scope(*, tiers: Any) -> dict:
    """候选集范围报告：跨层**不一致必须单独报告**，不得计入证据 disagreement。"""
    per_tier: dict[str, list[str]] = {}
    if isinstance(tiers, dict):
        for name, keys in tiers.items():
            per_tier[str(name)] = sorted({str(k) for k in (keys or [])})
    keysets = [set(v) for v in per_tier.values()]
    common = sorted(set.intersection(*keysets)) if keysets else []
    only_in = {name: sorted(set(keys) - set(common))
               for name, keys in per_tier.items()}
    mismatch = bool(keysets) and any(set(v) != set(common)
                                     for v in per_tier.values())
    return {
        "per_tier": per_tier,
        "common_candidate_keys": common,
        "only_in_tier": only_in,
        "candidate_set_mismatch": mismatch,
        "n_tiers": len(per_tier),
        "reporting_policy": CONTRACT["candidate_set_policy"],
        "counts_as_evidence_disagreement": False,
        "note": ("候选集不一致是**范围**问题（可比性），不是证据冲突；"
                 "本报告不参与、也不得被读作 disagreement 或 ACCEPT 依据"),
    }


def _same_threshold_groups(values: dict) -> dict:
    groups: dict[str, list[str]] = {}
    for key, v in values.items():
        if v.get("status") != "OK":
            continue
        thr = _r6((v.get("provenance") or {}).get("threshold"))
        groups.setdefault(str(thr), []).append(key)
    return groups


def detect_degeneracy(*, artifact: Any) -> dict:
    """结构性退化探测（只做定义性判定；**不是**质量判定）。

    `artifact` = `evaluate_search` 输出（或含 `action_value` 块的字典）。
    """
    flags: dict[str, bool] = {}
    labels: list[str] = []
    av = artifact.get("action_value") if isinstance(artifact, dict) else None
    if not isinstance(av, dict):
        return {"present": False, "reason": "action_value_block_required",
                "flags": {}, "labels": [], "verdict": "UNKNOWN",
                "claim_boundary": dict(CLAIM_BOUNDARY)}
    values = av.get("values") or {}
    ok = {k: v for k, v in values.items() if v.get("status") == "OK"}
    rejected = artifact.get("action_value_rejections") or []

    # L3：指纹不符/证据基不符 ⇒ 生产侧已拒绝（记录在 rejections）
    fp_mismatch = any("basis_fingerprint_mismatch" in str(r)
                      for r in rejected)
    fold_only = any("equity_basis_not_measured" in str(r)
                    or "legacy_all_leaves" in str(r) for r in rejected)

    eq_vals = sorted({_r6(v["measured"]["equity_vs_continuing"])
                      for v in ok.values()})
    w_pairs = sorted({(_r6(v["measured"]["p_continue"]),
                       _r6(v["measured"]["p_fold"])) for v in ok.values()})
    val_vals = sorted({float(v["value"]) for v in ok.values()})

    # L1：门槛 ↑ ⇒ p_continue 与类别数不得增（**只对有投入的候选**；
    #     check 的 p_continue=0 是"无门槛/无弃牌"的定义，不参与该单调律）
    l1 = True
    ordered = []
    for v in ok.values():
        prov = v.get("provenance") or {}
        thr = _r6(prov.get("threshold"))
        n = prov.get("n_continuation_classes")
        amt = _r6(v.get("amount_bb"))
        if _is_num(thr) and isinstance(n, int) and _is_num(amt) and float(amt) > 0:
            ordered.append((float(thr), _r6(v["measured"]["p_continue"]), n))
    ordered.sort(key=lambda t: t[0])
    for (t0, p0, n0), (t1, p1, n1) in zip(ordered, ordered[1:]):
        if t1 > t0 + TOL and (p1 > p0 + 1e-04 or n1 > n0):
            l1 = False

    # L2：同一门槛 ⇒ 同一续跟集合 ⇒ 三量必须相同
    l2 = True
    for thr, keys in _same_threshold_groups(values).items():
        if len(keys) < 2:
            continue
        sig = {(values[k]["measured"]["p_continue"],
                values[k]["measured"]["p_fold"],
                _r6(values[k]["measured"]["equity_vs_continuing"]),
                (values[k]["provenance"] or {}).get("n_continuation_classes"))
               for k in keys}
        if len(sig) > 1:
            l2 = False

    flags["constant_equity_input"] = len(eq_vals) == 1 and len(ok) > 1
    flags["weights_missing"] = any(
        "response_weights_not_supplied" in str(u) for u in (av.get("unknown") or []))
    flags["weights_indistinct"] = len(w_pairs) == 1 and len(ok) > 1
    flags["weights_not_monotone_in_threshold"] = not l1
    flags["same_threshold_group_inconsistent"] = not l2
    flags["basis_fingerprint_mismatch"] = fp_mismatch
    flags["fold_only_equity_basis_rejected"] = fold_only
    flags["budget_violated"] = bool(artifact.get("action_value_budget_violation"))
    flags["candidate_set_mismatch"] = bool(
        (artifact.get("candidate_set_scope") or {}).get("candidate_set_mismatch"))
    flags["no_action_differentiation"] = bool(
        ok and len(val_vals) <= 1
        and (len(w_pairs) > 1 or len(eq_vals) > 1))
    flags["single_terminal_value_only"] = all(
        (v.get("terms") or {}).get("terminal_values_distinct", 0) < 2
        for v in ok.values()) if ok else False
    # ── V3I-G1 Phase 2：统一路径一致性 ──
    uc = artifact.get("unified_consistency") if isinstance(artifact, dict) else None
    flags["unified_tier_value_mismatch"] = bool(
        isinstance(uc, dict) and uc.get("mismatched_keys"))
    flags["unified_value_unavailable"] = bool(
        isinstance(artifact, dict)
        and artifact.get("value_mode") == "unified"
        and not ok)

    for k, v in flags.items():
        if v:
            labels.append(k)
    return {
        "present": True,
        "flags": flags,
        "labels": sorted(labels),
        "verdict": ("DEGENERATE" if any(
            flags[k] for k in ("weights_missing", "weights_not_monotone_in_threshold",
                               "same_threshold_group_inconsistent",
                               "basis_fingerprint_mismatch",
                               "fold_only_equity_basis_rejected",
                               "unified_tier_value_mismatch",
                               "no_action_differentiation")) else "CONSISTENT"),
        "differentiation_source": (av.get("differentiation") or {}).get("source"),
        "note": ("探测器只判定定义性一致性（L1/L2/L3）与结构性退化，"
                 "**不**判定策略质量；任何结论均为 MODEL_INTERNAL"),
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }


def contract() -> dict:
    """契约自述（供审计读取）。"""
    return {
        "version": CASH_ACTION_VALUE_CONTRACT_VERSION,
        "contract": dict(CONTRACT),
        "record_fields": list(RECORD_FIELDS),
        "tol_is_not_a_parameter": TOL,
        "detector_laws": {
            "L1": "higher threshold => p_continue/n_classes non-increasing",
            "L2": "identical threshold => identical (p_continue, p_fold, equity, n_classes)",
            "L3": "declared basis fingerprint must match its own basis",
        },
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }


# ══════════════════════════════════════════════════════════════════════
# V3I-G1 Phase 2 — **统一动作价值路径**（Search / Ranking 同域、同记账、同输入）
# ══════════════════════════════════════════════════════════════════════
VALUE_MODES: tuple[str, ...] = ("legacy", "unified")

#: Phase 2 统一模式策略（纯定义，非可调参数）
UNIFIED_MODE_POLICY: dict[str, Any] = {
    "value_mode": "unified",
    "domain": ("两层只在**共同候选域**（common_keys）上比较；域外候选"
               "**必须报告**（out_of_domain），不得隐藏"),
    "accounting": ("两层消费**同一份**统一动作价值表（由本契约一次性计算），"
                   "因此 pot/stack 后果、fold/continue 权重、终局价值逐位一致"),
    "legacy_preserved": ("旧 v3 `search_score` 与 ranking `pot_share_arithmetic_v1` "
                         "**逐位并列保留**（legacy 排序亦保留）"),
    "mismatch_reporting": ("候选集不一致只作**范围**报告，"
                           "counts_as_evidence_disagreement=False"),
    "fail_closed": ("统一价值缺失/不一致 ⇒ 该层无候选（preferred=None）⇒ "
                    "装配报 <tier>_top_candidate_unavailable ⇒ 决策 UNKNOWN/ABSTAIN；"
                    "**禁止**回落到 legacy 排序当作决策"),
    "no_voting_no_override": True,
    "gate_changed": False,
    "external_data_used": False,
    "free_parameters": 0,
}


def shared_action_domain(*, search_keys: Any, ranking_keys: Any) -> dict:
    """共同候选域（**范围**报告；与证据 disagreement 严格分开）。"""
    per = {
        "search": sorted({str(k) for k in (search_keys or [])}),
        "ranking": sorted({str(k) for k in (ranking_keys or [])}),
    }
    keysets = {name: set(v) for name, v in per.items()}
    common = sorted(set.intersection(*keysets.values())) if keysets else []
    only = {name: sorted(keys - set(common)) for name, keys in keysets.items()}
    mismatch = bool(only.get("search") or only.get("ranking"))
    return {
        "value_mode": "unified",
        "per_tier_keys": per,
        "common_keys": common,
        "only_in_tier": only,
        "n_common": len(common),
        "candidate_set_mismatch": mismatch,
        "counts_as_evidence_disagreement": False,
        "out_of_domain_candidates_reported": sorted(
            set(only.get("search", [])) | set(only.get("ranking", []))),
        "domain_policy": UNIFIED_MODE_POLICY["domain"],
        "note": ("候选集不一致是**范围/可比性**问题，不是证据冲突；"
                 "本块不参与也不得被读作 disagreement 或 ACCEPT 依据"),
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }


def unified_action_value_input(*, pot_bb: Any, candidates: Any, records: Any,
                               search_keys: Any = None,
                               ranking_keys: Any = None,
                               consequences: Any = None,
                               population: Any = None,
                               source: str = "unified_contract",
                               budget_violated: bool = False) -> dict:
    """构造**统一动作价值输入**（两层共享的唯一价值表 + 共同域）。

    * 价值表由本契约**一次性**计算（`aggregate_action_values`），两层只消费；
    * 叶预算被违反 ⇒ 价值表为空 + `unavailable_reason`（fail closed）；
    * `consequences`（pot/stack 后果）由调用方从 search tree 的记账字段传入，
      本函数只**附着**并标注来源（不重算、不发明）。
    """
    out: dict[str, Any] = {
        "present": True,
        "version": CASH_ACTION_VALUE_CONTRACT_VERSION,
        "value_mode": "unified",
        "contract": dict(CONTRACT),
        "unit": CONTRACT["unit"],
        "free_parameters": 0,
        "pot_bb": _r6(pot_bb),
        "source": source,
        "population": population,
        "records": records if isinstance(records, list) else [],
        "unified_values": {},
        "n_valued": 0,
        "consequences": {},
        "unavailable_reason": None,
        "policy": dict(UNIFIED_MODE_POLICY),
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }
    if budget_violated:
        out["unavailable_reason"] = "leaf_budget_violated"
    # 叶预算被违反 ⇒ **不计算**价值表（fail closed：两层都不得凭被截断/超限的
    # 叶池给出价值，也不得让任一层回落到 legacy 排序当作决策）
    agg = ({"present": False, "reason": "leaf_budget_violated",
            "values": {}, "ranked": [], "preferred": None, "n_valued": 0,
            "unknown": ["leaf_budget_violated"],
            "differentiation": {"differentiates_actions": False,
                                "source": "budget_violated"}}
           if budget_violated else
           aggregate_action_values(pot_bb=pot_bb, candidates=candidates,
                                   records=records))
    out["aggregate"] = agg
    if not agg.get("present"):
        out["unavailable_reason"] = out["unavailable_reason"] or agg.get(
            "reason", "aggregate_unavailable")
    cons = consequences if isinstance(consequences, dict) else {}
    values: dict[str, Any] = {}
    for key, row in (agg.get("values") or {}).items():
        values[key] = {
            "action": row.get("action"),
            "amount_bb": row.get("amount_bb"),
            "status": row.get("status"),
            "value": row.get("value"),
            "gain_vs_check": row.get("gain_vs_check"),
            "terms": dict(row.get("terms") or {}),
            "measured": dict(row.get("measured") or {}),
            "provenance": dict(row.get("provenance") or {}),
            "consequences": dict(cons.get(key) or {}),
            "unknown": list(row.get("unknown") or []),
        }
    out["unified_values"] = values
    out["n_valued"] = agg.get("n_valued", 0)
    out["consequences"] = {k: dict(cons.get(k) or {}) for k in cons}
    out["domain"] = shared_action_domain(search_keys=search_keys,
                                         ranking_keys=ranking_keys)
    if not out["n_valued"] and not out["unavailable_reason"]:
        out["unavailable_reason"] = "no_candidate_valued"
    out["unknown"] = sorted({u for u in (agg.get("unknown") or [])})
    return out


def unified_mode_contract() -> dict:
    """Phase 2 统一模式自述（审计用）。"""
    return {
        "version": CASH_ACTION_VALUE_CONTRACT_VERSION,
        "value_modes": list(VALUE_MODES),
        "policy": dict(UNIFIED_MODE_POLICY),
        "shared_domain_function": "shared_action_domain",
        "unified_input_function": "unified_action_value_input",
        "tier_consumers": ["cash_strategy_search_eval.evaluate_search",
                           "cash_strategy_action_ranking.rank_actions"],
        "single_accounting_definition": CONTRACT["name"],
        "legacy_preserved_fields": ["search_score (legacy v3)",
                                    "ranked_legacy", "preferred_legacy",
                                    "score_legacy_v3"],
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }
