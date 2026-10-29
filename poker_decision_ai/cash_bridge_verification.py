"""V3I-P5 — P4 bridge 的受控稳定性与因果一致性验证层。

**本层不新增任何策略能力**：不修改 P4 bridge，不修改任何冻结层，
只对"population → opponent belief/range → hero equity → candidate score → ranking"
这条链做**受控验证**、**前置/后置条件检查**与**反事实一致性检查**。

授权要求对应
------------
* 要求 2：受控场景矩阵（5 个 population + UNKNOWN）
* 要求 3：完整因果链验证
* 要求 4：参数反事实测试（改参数 → 结果变；恢复 → 逐字段一致）
* 要求 5：不得因"人口标签"硬编码动作
* 要求 6：不同 board/street/pot/stack 下不得非法或伪适应
* 要求 7：无 opponent evidence 时不得凭空产生 population 差异
* 要求 8：zero-weight / impossible combo / folded hand 不得进入对手范围
* 要求 9：deterministic repeat、UNKNOWN、conflict、fail-closed
* 要求 13：声明式 utility 必须继续标记 `MODEL_INTERNAL`，不得改称 EV

特别地，本层对 P4 已披露的 **6 个历史缺陷**做**回归检查**（见
`audit_p4_regression_patterns` 与 `check_postconditions`）。

边界：本模块只 import 标准库；不 import 任何项目内模块（P4 bridge 由调用方注入）。
"""
from __future__ import annotations

import hashlib
import json
import re
from typing import Any

CASH_P5_VERIFICATION_VERSION = "v3i-p5-bridge-stability-verification-v1"

#: 必须继续存在的证据标记（要求 13）
REQUIRED_EVIDENCE_GRADE = "MODEL_INTERNAL"

#: 被明确禁止的宣称词（不得把声明式 utility 说成 EV / GTO / 均衡）
FORBIDDEN_CLAIM_TERMS: tuple[str, ...] = (
    "expected_value", "is_ev", "is_gto", "is_equilibrium", "is_solver_output",
    "is_profit_evidence", "is_real_human_validation", "is_external_validation",
)

VALID_STREETS: tuple[str, ...] = ("preflop", "flop", "turn", "river")
VALID_BOARD_SIZE: dict[str, int] = {"preflop": 0, "flop": 3, "turn": 4, "river": 5}
VALID_POPULATIONS: tuple[str, ...] = (
    "DECLARED_TIGHT_PASSIVE", "DECLARED_BALANCED", "DECLARED_LOOSE_PASSIVE",
    "DECLARED_AGGRESSIVE", "DECLARED_MANIAC",
)
RANKS = "23456789TJQKA"
SUITS = "shdc"

CLAIM_BOUNDARY: dict[str, bool] = {
    "is_gto": False,
    "is_equilibrium": False,
    "is_solver_output": False,
    "is_real_opponent_data": False,
    "is_real_human_validation": False,
    "is_profit_evidence": False,
    "is_external_validation": False,
    "is_ev": False,
    "wired_to_production": False,
    "adds_strategy_capability": False,
    "modifies_p4_bridge": False,
    "modifies_v3h4_v3h6_v3h8": False,
    "modifies_cfi_lambda_river_budget": False,
}

EVIDENCE_GRADE = "MODEL_INTERNAL"


def _is_num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _stub(reason: str) -> dict:
    return {
        "present": False,
        "version": CASH_P5_VERIFICATION_VERSION,
        "reason": reason,
        "unknown": [reason],
        "claim_boundary": dict(CLAIM_BOUNDARY),
        "evidence_grade": "NONE",
        "executed": False,
        "result": None,
        "wired_to_production": False,
    }


def stable_fingerprint(value: Any) -> str:
    """确定性指纹（对冻结输出中的 tuple 键先规范化）。"""
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
# 1. 场景合法性前置检查（要求 6）
# ---------------------------------------------------------------------------
def _card_ok(c: Any) -> bool:
    return (isinstance(c, str) and len(c) == 2 and c[0] in RANKS
            and c[1] in SUITS)


def check_scenario(*, state: Any) -> dict:
    """前置条件：state 必须自洽（street/board/pot/stack/hero/牌不重复）。"""
    if not isinstance(state, dict):
        return _stub("state_not_a_dict")
    v: list[str] = []

    if state.get("game_type") != "cash":
        v.append("not_cash_mode:bridge_must_fail_closed")

    street = state.get("street")
    if street not in VALID_STREETS:
        v.append(f"invalid_street:{street!r}")
        want = None
    else:
        want = VALID_BOARD_SIZE[street]

    board = state.get("board")
    if not isinstance(board, list):
        v.append("board_not_a_list")
    else:
        for c in board:
            if not _card_ok(c):
                v.append(f"invalid_board_card:{c!r}")
        if want is not None and len(board) != want:
            v.append(f"board_size_mismatch:street={street},len={len(board)},want={want}")
        if len({str(c) for c in board}) != len(board):
            v.append("duplicate_board_card")

    players = state.get("players")
    hero_cards: list = []
    if not isinstance(players, list) or not players:
        v.append("players_missing")
    else:
        found = False
        for p in players:
            if not isinstance(p, dict):
                v.append("player_not_a_dict")
                continue
            is_hero = bool(p.get("is_hero")) or (
                state.get("hero_id") is not None
                and p.get("player_id") == state.get("hero_id"))
            hc = p.get("hole_cards")
            if is_hero:
                found = True
                if not (isinstance(hc, list) and len(hc) == 2
                        and all(_card_ok(c) for c in hc)):
                    v.append("hero_hole_cards_invalid")
                else:
                    hero_cards = [str(c) for c in hc]
            elif hc:
                # 对手手牌必须为空（V3I-P1 语义：对手牌未知）
                v.append(f"opponent_hole_cards_present:{p.get('player_id')!r}")
        if not found:
            v.append("hero_not_found")

    if hero_cards and isinstance(board, list):
        overlap = {c for c in hero_cards} & {str(c) for c in board}
        if overlap:
            v.append(f"hero_board_card_collision:{sorted(overlap)}")

    pot = state.get("pot_bb")
    if not _is_num(pot) or float(pot) < 0:
        v.append(f"pot_bb_invalid:{pot!r}")
    stack = state.get("effective_stack_bb")
    if not _is_num(stack) or float(stack) <= 0:
        v.append(f"effective_stack_bb_invalid:{stack!r}")

    action_history = state.get("action_history")
    if not isinstance(action_history, list):
        v.append("action_history_not_a_list")

    return {
        "present": True,
        "version": CASH_P5_VERIFICATION_VERSION,
        "valid": not v,
        "violations": v,
        "street": street,
        "expected_board_size": want,
        "board_size": len(board) if isinstance(board, list) else None,
        "hero_cards": hero_cards,
        "pot_bb": pot,
        "effective_stack_bb": stack,
        "action_history_length": len(action_history)
        if isinstance(action_history, list) else None,
        "claim_boundary": dict(CLAIM_BOUNDARY),
        "evidence_grade": EVIDENCE_GRADE,
        "executed": False,
        "result": None,
        "wired_to_production": False,
        "unknown": [],
    }


# ---------------------------------------------------------------------------
# 2. bridge 输出后置条件（要求 3、6、8、13 + 6 个历史缺陷）
# ---------------------------------------------------------------------------
def check_postconditions(*, out: Any, state: Any) -> dict:
    """对 P4 bridge 输出做后置条件检查（含 6 个历史缺陷的回归防线）。"""
    if not isinstance(out, dict):
        return _stub("output_not_a_dict")
    if not out.get("present"):
        # fail-closed 也是合法结果；此处只登记
        return {
            "present": True,
            "version": CASH_P5_VERIFICATION_VERSION,
            "applicable": False,
            "reason": out.get("reason"),
            "violations": [],
            "valid": True,
            "claim_boundary": dict(CLAIM_BOUNDARY),
            "evidence_grade": EVIDENCE_GRADE,
            "executed": False,
            "result": None,
            "wired_to_production": False,
            "unknown": [],
        }

    v: list[str] = []
    checks: dict[str, bool] = {}

    pot = out.get("pot_bb")
    baseline = out.get("check_baseline")
    ranked = out.get("ranked")
    prefs = out.get("preferred")

    # ── 缺陷 6：check 基线必须存在且被扣除 ──
    checks["baseline_present"] = _is_num(baseline)
    if not _is_num(baseline):
        v.append("defect6:check_baseline_missing")

    if not isinstance(ranked, list) or not ranked:
        v.append("ranked_missing")
        ranked = []

    zero_risk_scores = []
    for r in ranked:
        if not isinstance(r, dict):
            v.append("ranked_row_not_a_dict")
            continue
        amt = r.get("amount_bb")
        pc = r.get("p_continue")
        pf = r.get("p_fold")
        sc = r.get("score")
        raw = r.get("raw_pot_share")
        he = r.get("hero_equity_vs_continuing")

        if not _is_num(amt) or float(amt) < 0:
            v.append(f"negative_or_invalid_amount:{amt!r}")
        if not _is_num(pc) or not (0.0 <= float(pc) <= 1.0):
            v.append(f"p_continue_out_of_range:{pc!r}")
        # ── 缺陷 3：p_fold 必须恰为 1 − p_continue（check 不得有假弃牌率）──
        if _is_num(pc) and _is_num(pf):
            if abs(float(pf) - (1.0 - float(pc))) > 1e-6:
                v.append(f"defect3:p_fold_mismatch:{pf}!={1.0 - float(pc)}")
        else:
            v.append("defect3:p_fold_missing")
        checks["p_fold_consistent"] = _is_num(pf) and _is_num(pc) and \
            abs(float(pf) - (1.0 - float(pc))) <= 1e-6

        # ── 缺陷 6：score 必须等于 raw_pot_share − check_baseline ──
        if _is_num(sc) and _is_num(raw) and _is_num(baseline):
            if abs(float(sc) - (float(raw) - float(baseline))) > 1e-6:
                v.append("defect6:score_not_baseline_subtracted")
        else:
            v.append("defect6:raw_or_score_missing")

        # ── 缺陷 2：折扣后收益不得出现"重复计算 hero 自己的下注" ──
        #    弃牌项 ≤ pot（不可能超过底池）；被跟注项 ≤ (pot + 2b)
        terms = r.get("score_terms") or {}
        ft = terms.get("fold_term")
        st_ = terms.get("showdown_term")
        if _is_num(ft) and _is_num(pot):
            if float(ft) > float(pot) + 1e-6:
                v.append(f"defect2:fold_term_exceeds_pot:{ft}>{pot}")
        if _is_num(st_) and _is_num(pot) and _is_num(amt):
            cap = (float(pot) + 2.0 * float(amt)) * 1.0
            if float(st_) > cap + 1e-6:
                v.append(f"defect2:showdown_term_exceeds_capped_pot:{st_}>{cap}")
        checks["terms_within_caps"] = not any(
            x.startswith("defect2:") for x in v)

        # ── 缺陷 1：hero 权益必须落在 [0,1]；与 p_continue 方向一致 ──
        if he is not None:
            if not _is_num(he) or not (0.0 <= float(he) <= 1.0):
                v.append(f"defect1:hero_equity_out_of_range:{he!r}")
        if float(amt or 0.0) == 0.0:
            zero_risk_scores.append(float(sc) if _is_num(sc) else None)
            if _is_num(pc) and float(pc) != 0.0:
                v.append("defect3:check_has_nonzero_p_continue")

    # 排名必须严格按 score 降序且 rank 连续
    scores = [float(r["score"]) for r in ranked if _is_num(r.get("score"))]
    if scores != sorted(scores, reverse=True):
        v.append("ranking_not_sorted_by_score")
    ranks = [r.get("rank") for r in ranked]
    if ranks != list(range(1, len(ranked) + 1)):
        v.append(f"rank_not_contiguous:{ranks}")

    # preferred 必须等于 rank 1
    if isinstance(prefs, dict) and ranked:
        if float(prefs.get("score", 0.0)) != float(ranked[0].get("score", -1.0)):
            v.append("preferred_not_rank1")

    # ── 缺陷 5：zero-weight / impossible combo 不得进入有效对手范围 ──
    n_zero = out.get("n_opponent_classes")
    tested_anchor = (out.get("range_anchor") or {}).get("n_classes")
    if _is_num(n_zero) and _is_num(tested_anchor):
        if int(tested_anchor) > int(n_zero):
            v.append("defect5:anchor_tests_more_classes_than_posterior")
    for r in ranked:
        cr = r.get("candidate_required_equity") or {}
        if _is_num(cr.get("amount_bb")) and float(cr["amount_bb"]) < 0:
            v.append("negative_amount_in_required_equity")

    # ── 缺陷 4：门槛必须落在 [0,1] 且由乘性锚 + 溢价构成 ──
    for r in ranked:
        thr = (r.get("continue_threshold") or {}).get("threshold")
        if not _is_num(thr) or not (0.0 <= float(thr) <= 1.0):
            v.append(f"defect4:threshold_out_of_range:{thr!r}")
        if float(r.get("amount_bb") or 0.0) > 0.0:
            # 更大的下注 ⇒ 更高的 pot odds 分量 ⇒ 门槛不得更低
            pass

    # 门槛随下注额单调不减（pot odds 分量随 b 增加）
    pos = [r for r in ranked if float(r.get("amount_bb") or 0.0) > 0.0]
    pos_sorted = sorted(pos, key=lambda r: float(r["amount_bb"]))
    thr_seq = [(float(r["amount_bb"]),
                (r.get("continue_threshold") or {}).get("threshold"))
               for r in pos_sorted]
    numeric = [(a, t) for a, t in thr_seq if _is_num(t)]
    if any(numeric[i][1] > numeric[i + 1][1] + 1e-9
           for i in range(len(numeric) - 1)):
        v.append(f"defect4:threshold_not_monotone_in_bet_size:{numeric}")

    # ── 要求 13：证据标记与宣称边界 ──
    if out.get("evidence_grade") != REQUIRED_EVIDENCE_GRADE:
        v.append(f"evidence_grade_not_model_internal:{out.get('evidence_grade')!r}")
    cb = out.get("claim_boundary") or {}
    for k, val in cb.items():
        if val is not False:
            v.append(f"claim_boundary_not_false:{k}")
    if out.get("executed") is not False:
        v.append("executed_not_false")
    if out.get("result") is not None:
        v.append("result_not_none")
    if out.get("wired_to_production") is not False:
        v.append("wired_not_false")

    # ── 要求 6：不得出现非法数值 ──
    for r in ranked:
        for key in ("score", "raw_pot_share", "check_baseline"):
            val = r.get(key)
            if val is not None and _is_num(val) and (
                    float(val) != float(val) or abs(float(val)) == float("inf")):
                v.append(f"non_finite:{key}")

    return {
        "present": True,
        "version": CASH_P5_VERIFICATION_VERSION,
        "applicable": True,
        "valid": not v,
        "violations": v,
        "n_violations": len(v),
        "checks": checks,
        "n_candidates": len(ranked),
        "evidence_grade": EVIDENCE_GRADE,
        "claim_boundary": dict(CLAIM_BOUNDARY),
        "executed": False,
        "result": None,
        "wired_to_production": False,
        "unknown": [],
    }


# ---------------------------------------------------------------------------
# 3. 静态回归检查（6 个历史缺陷不得重现）
# ---------------------------------------------------------------------------
P4_DEFECT_PATTERNS: dict[str, dict[str, Any]] = {
    "D1_equity_side": {
        "must_contain": ["1.0 - float(he)"],
        "must_not_contain": ["hero_equity >= thr", "eq >= thr\n        if float(eq) >= thr"],
        "description": "门槛比较必须在**对手**一侧（opponent_equity = 1 − hero_equity）",
    },
    "D2_fold_double_count": {
        "must_contain": ["fold_term = p_fold * pot"],
        "must_not_contain": ["fold_term = p_fold * (pot + amount)"],
        "description": "弃牌只赢底池；hero 未被跟注的下注退回，不计入收益",
    },
    "D3_check_fake_fold_rate": {
        "must_contain": ["no_investment_no_fold_equity"],
        "must_not_contain": [],
        "description": "check 无弃牌收益（p_fold 必须为 1）",
    },
    "D4_threshold_anchor": {
        "must_contain": ["anchor_scale"],
        "must_not_contain": ["req + base + anchor", "total = min(max(req + base + anchor"],
        "description": "门槛用 pot odds 的**乘性**范围锚，不得用加性锚",
    },
    "D5_zero_posterior": {
        "must_contain": ["n_classes_zero_weight"],
        "must_not_contain": [],
        "description": "零后验权重类别不得进入有效对手范围",
    },
    "D6_candidate_size_artifact": {
        "must_contain": ["check_baseline"],
        "must_not_contain": [],
        "description": "决策分数必须扣除无风险基线，避免 check 结构性恒优",
    },
}


def audit_p4_regression_patterns(*, source: Any) -> dict:
    """对 P4 bridge 源码做 6 个历史缺陷的回归检查（要求：特别检查）。

    同时检查"不得把声明式 utility 改称 EV / GTO / 均衡"：
    做法是校验 `CLAIM_BOUNDARY` 的**每一项都为 False**，且**没有**把
    `expected_value` / `is_ev` 用作正向标签（仅出现 `claim_boundary` 键名不算违规，
    因为那正是**否认**该宣称的声明）。
    """
    if not isinstance(source, str) or not source.strip():
        return _stub("source_required")
    per: dict[str, Any] = {}
    all_ok = True
    for name, spec in P4_DEFECT_PATTERNS.items():
        missing = [s for s in spec["must_contain"] if s not in source]
        present_forbidden = [s for s in spec["must_not_contain"] if s in source]
        ok = not missing and not present_forbidden
        all_ok = all_ok and ok
        per[name] = {
            "ok": ok,
            "missing_required": missing,
            "forbidden_present": present_forbidden,
            "description": spec["description"],
        }

    # 正向宣称检测：只把"被赋值为 True"或用作 EV 标签视为违规
    positive_claims = [f'"{t}": True' for t in FORBIDDEN_CLAIM_TERMS
                       if f'"{t}": True' in source]
    ev_labelling = [t for t in ("expected_value", "is_ev", "ev_bb", "equity_ev")
                    if f'"{t}"' in source or f"'{t}'" in source]
    # `claim_boundary` 的键是**否认**该宣称的声明（必须全为 False）。
    # 只检查被审源码中 `claim_boundary` 块内的键，不能把 `"present": True`
    # 这类正常键误判为宣称。
    m = re.search(r"CLAIM_BOUNDARY[^=]*=\s*\{(.*?)\}", source, re.S)
    boundary_keys: list[str] = []
    if m:
        boundary_keys = re.findall(r'"([A-Za-z0-9_]+)":\s*(?:True|False)',
                                   m.group(1))
    boundary_true = [k for k in boundary_keys if f'"{k}": True' in m.group(1)] \
        if m else []
    boundary_false = bool(boundary_keys) and not boundary_true

    return {
        "present": True,
        "version": CASH_P5_VERIFICATION_VERSION,
        "all_defects_guarded": all_ok,
        "per_defect": per,
        "positive_external_claims": positive_claims,
        "ev_relabelling_terms": ev_labelling,
        "claim_boundary_keys_found": boundary_keys,
        "claim_boundary_all_false_in_source": boundary_false,
        "claim_boundary_true_keys": boundary_true,
        "external_claim_guard_ok": (not positive_claims and not ev_labelling
                                    and boundary_false),
        "evidence_grade_marker_present":
            f'EVIDENCE_GRADE = "{REQUIRED_EVIDENCE_GRADE}"' in source,
        "claim_boundary": dict(CLAIM_BOUNDARY),
        "evidence_grade": EVIDENCE_GRADE,
        "executed": False,
        "result": None,
        "wired_to_production": False,
        "unknown": [],
    }


# ---------------------------------------------------------------------------
# 4. 因果链验证（要求 3）
# ---------------------------------------------------------------------------
CHAIN_LINKS: tuple[str, ...] = (
    "population", "opponent_range", "hero_equity", "candidate_score", "ranking",
)


def verify_causal_chain(*, evaluations: Any,
                        populations: Any = None) -> dict:
    """验证 population → range → equity → score → ranking 的**可观测**因果链。

    对每个环节给出"是否随 population 变化"以及**数值证据**。
    """
    evals = evaluations if isinstance(evaluations, list) else []
    ok = [e for e in evals if isinstance(e, dict) and e.get("present")]
    if len(ok) < 2:
        out = _stub("at_least_two_present_evaluations_required")
        out["n_evaluations"] = len(evals)
        return out

    by_pop = {}
    for e in ok:
        rows = e.get("ranked") or []
        # 排序必须按 **score 降序**重建（不得沿用输入顺序——否则"排序是否随
        # population 变化"会被输入顺序掩盖）。
        ordered = sorted(
            [r for r in rows if isinstance(r, dict)],
            key=lambda r: (-float(r.get("score") or 0.0),
                           str(r.get("action")),
                           float(r.get("amount_bb") or 0.0)))
        by_pop[str(e.get("population"))] = {
            "tolerance": (e.get("population_tolerance") or {}).get(
                "tolerance_premium"),
            "thresholds": sorted({(r.get("continue_threshold") or {}).get(
                "threshold") for r in rows
                if (r.get("continue_threshold") or {}).get("threshold")
                is not None}),
            "p_continue": sorted({r.get("p_continue") for r in rows
                                  if r.get("p_continue") is not None}),
            "hero_equity": sorted({r.get("hero_equity_vs_continuing") for r in rows
                                   if r.get("hero_equity_vs_continuing")
                                   is not None}),
            "scores": {f"{r.get('action')}|{r.get('sizing_kind')}|"
                       f"{r.get('amount_bb')}": r.get("score") for r in ordered},
            "ranking": [f"{r.get('action')}|{r.get('sizing_kind')}|"
                        f"{r.get('amount_bb')}" for r in ordered],
            "n_opponent_classes": e.get("n_opponent_classes"),
            "n_continuing": sorted({r.get("n_continuation_classes") for r in rows
                                    if r.get("n_continuation_classes")
                                    is not None}),
        }

    def _vary(key: str) -> bool:
        vals = [json.dumps(v[key], sort_keys=True) for v in by_pop.values()]
        return len(set(vals)) > 1

    links = {
        "population": len(by_pop) > 1,
        "opponent_range": _vary("p_continue") or _vary("n_continuing"),
        "hero_equity": _vary("hero_equity"),
        "candidate_score": _vary("scores"),
        "ranking": _vary("ranking"),
    }
    complete = all(links.values())
    return {
        "present": True,
        "version": CASH_P5_VERIFICATION_VERSION,
        "populations": sorted(by_pop),
        "n_populations": len(by_pop),
        "chain_links": links,
        "chain_complete": complete,
        "first_broken_link": next((k for k in CHAIN_LINKS if not links.get(k)),
                                  None),
        "per_population": by_pop,
        "evidence": {
            "opponent_range": "p_continue / n_continuation_classes 随 population 变化",
            "hero_equity": "hero_equity_vs_continuing 随继续范围变化",
            "candidate_score": "同一候选的分数随 population 变化",
            "ranking": "排序随 population 变化",
        },
        "claim_boundary": dict(CLAIM_BOUNDARY),
        "evidence_grade": EVIDENCE_GRADE,
        "executed": False,
        "result": None,
        "wired_to_production": False,
        "unknown": [],
    }


# ---------------------------------------------------------------------------
# 5. 参数反事实（要求 4、5）
# ---------------------------------------------------------------------------
def counterfactual_population_swap(*, baseline_out: Any,
                                   swapped_out: Any,
                                   identical_params_out: Any) -> dict:
    """反事实一致性：只改参数 → 结果变；参数恢复 → **逐字段一致**。

    Parameters
    ----------
    baseline_out : 群体 A 在参数 P_A 下的输出
    swapped_out : 群体 A 在**替换为** P_B 后的输出（应体现 P_B 的影响）
    identical_params_out : 群体 A 在 P_A 下的**再次**运行（应为逐字段一致）
    """
    for name, o in (("baseline_out", baseline_out), ("swapped_out", swapped_out),
                    ("identical_params_out", identical_params_out)):
        if not isinstance(o, dict) or not o.get("present"):
            return _stub(f"{name}_not_present")

    fp_base = stable_fingerprint(baseline_out.get("per_candidate"))
    fp_swap = stable_fingerprint(swapped_out.get("per_candidate"))
    fp_same = stable_fingerprint(identical_params_out.get("per_candidate"))
    tol_base = (baseline_out.get("population_tolerance") or {}).get(
        "tolerance_premium")
    tol_swap = (swapped_out.get("population_tolerance") or {}).get(
        "tolerance_premium")

    changed = fp_base != fp_swap
    deterministic = fp_base == fp_same
    return {
        "present": True,
        "version": CASH_P5_VERIFICATION_VERSION,
        "baseline_fingerprint": fp_base,
        "swapped_fingerprint": fp_swap,
        "identical_params_fingerprint": fp_same,
        "tolerance_baseline": tol_base,
        "tolerance_swapped": tol_swap,
        "counterfactual_changes_result": changed,
        "restore_is_field_identical": deterministic,
        "label_alone_changes_nothing": (
            stable_fingerprint(baseline_out) ==
            stable_fingerprint(identical_params_out)),
        "pass": changed and deterministic,
        "interpretation": (
            "只改 population 参数 → 结果变化；把参数恢复为同一数值 → 结果逐字段一致。"
            "⇒ 差异来自**参数数值**，不是 population 标签。"),
        "claim_boundary": dict(CLAIM_BOUNDARY),
        "evidence_grade": EVIDENCE_GRADE,
        "executed": False,
        "result": None,
        "wired_to_production": False,
        "unknown": [],
    }


def no_evidence_probe(*, per_population_outputs: Any) -> dict:
    """要求 7：无 opponent evidence 时不得凭空产生 population 差异。"""
    outs = per_population_outputs if isinstance(per_population_outputs, dict) \
        else {}
    if not outs:
        return _stub("outputs_required")
    fps = {p: stable_fingerprint(o) for p, o in outs.items()}
    adapted = [p for p, o in outs.items()
               if isinstance(o, dict) and o.get("adaptation_applied")]
    return {
        "present": True,
        "version": CASH_P5_VERIFICATION_VERSION,
        "populations": sorted(outs),
        "fingerprints": fps,
        "n_distinct": len(set(fps.values())),
        "all_identical": len(set(fps.values())) == 1,
        "population_entities_adapting": adapted,
        "pseudo_adaptation_detected": bool(adapted),
        "pass": len(set(fps.values())) == 1 and not adapted,
        "claim_boundary": dict(CLAIM_BOUNDARY),
        "evidence_grade": EVIDENCE_GRADE,
        "executed": False,
        "result": None,
        "wired_to_production": False,
        "unknown": [],
    }


# ---------------------------------------------------------------------------
# 6. 对手范围有效性（要求 8）
# ---------------------------------------------------------------------------
def check_opponent_range_validity(*, out: Any, belief: Any,
                                  player_id: Any = None,
                                  folded_players: Any = None) -> dict:
    """zero-weight / 非法 combo / 已弃牌玩家不得进入有效对手范围。"""
    if not isinstance(out, dict) or not out.get("present"):
        return _stub("present_bridge_output_required")
    if not isinstance(belief, dict) or not belief.get("present"):
        return _stub("present_belief_required")

    entries = belief.get("belief") or {}
    pid = player_id if player_id is not None else out.get("player_id")
    entry = entries.get(str(pid)) if isinstance(entries, dict) else None
    post = ((entry or {}).get("posterior") or {}).get("posterior") or {}
    pos = {k: float(v) for k, v in post.items()
           if _is_num(v) and float(v) > 0.0}
    zero = sorted(k for k, v in post.items()
                  if _is_num(v) and float(v) <= 0.0)

    v: list[str] = []
    if out.get("n_opponent_classes") != len(pos):
        v.append(f"class_count_mismatch:out={out.get('n_opponent_classes')},"
                 f"positive_posterior={len(pos)}")
    anchor_n = (out.get("range_anchor") or {}).get("n_classes")
    if _is_num(anchor_n) and int(anchor_n) != len(pos):
        v.append(f"anchor_class_count_mismatch:{anchor_n}!={len(pos)}")

    folded = set(str(x) for x in (folded_players or []))
    if str(pid) in folded:
        v.append(f"folded_player_used_as_opponent:{pid}")

    # 对手已弃牌/不在牌局 → 不应被选为对手
    return {
        "present": True,
        "version": CASH_P5_VERIFICATION_VERSION,
        "player_id": pid,
        "n_positive_posterior_classes": len(pos),
        "n_zero_weight_classes": len(zero),
        "zero_weight_classes_excluded": True,
        "bridge_reports": out.get("n_opponent_classes"),
        "anchor_reports": anchor_n,
        "valid": not v,
        "violations": v,
        "claim_boundary": dict(CLAIM_BOUNDARY),
        "evidence_grade": EVIDENCE_GRADE,
        "executed": False,
        "result": None,
        "wired_to_production": False,
        "unknown": [],
    }


def verification_contract() -> dict:
    return {
        "version": CASH_P5_VERIFICATION_VERSION,
        "reuses": "v3i-p4-opponent-conditioned-bridge-v1（未修改）",
        "adds_strategy_capability": False,
        "valid_populations": list(VALID_POPULATIONS),
        "valid_streets": list(VALID_STREETS),
        "valid_board_size": dict(VALID_BOARD_SIZE),
        "chain_links": list(CHAIN_LINKS),
        "p4_defects_guarded": sorted(P4_DEFECT_PATTERNS),
        "required_evidence_grade": REQUIRED_EVIDENCE_GRADE,
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }
