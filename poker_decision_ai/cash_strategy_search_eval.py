"""V3H-7 — CASH Search-Aware Candidate Evaluation（搜索聚合，独立评分）。

定位
----
把**搜索树**结果按 root candidate 聚合，产出**可复算**的 search-aware 评分与
覆盖率/UNKNOWN 统计。这是 Layer 4 的"candidate aggregation"。

**独立性声明（授权要求）**：本层**不修改** V3H-5 的排序公式
（`pot_share_arithmetic_v1` 仍只用于即时权益）；本层使用**独立**的
`search_runout_pot_share_v1` 公式：

```
search_score = future_equity * (pot + 2*invest)/pot − invest/pot
```

其中 `future_equity` = 该 root candidate 下**叶子权益的算术均值**（跑完枚举语义）。
两套分数在输出中**各自带 `score_source` 字段**，供后续层显式区分读取。

**不做最终决策**：`final_decision_made=false`；`preferred` 仅在**有可评估叶子**时给出。

leaves 处理（显式区分）
-----------------------
| 类别 | 计入 |
|---|---|
| `evaluable`（有权益） | 进入均值/中位数/评分 |
| `unknown`（权益不可得） | 单独计数，**不**进均值、**不**补默认值 |
| `unavailable`（预算/终局） | 单独计数 |

硬约束
------
* CASH-only，fail closed；
* 不 import 任何 V3G 层（数据接口显式传入）；
* 不 import `random`；同输入 → 同输出；
* 不声称 GTO/equilibrium/exploitability/真实盈利；
* Solver 只读转述，未传入 → `UNKNOWN` 且 `blocked=True`。
"""
from __future__ import annotations

from typing import Any

from .cash_action_value_contract import (
    CLAIM_BOUNDARY as ACTION_VALUE_CLAIM_BOUNDARY,
    CONTRACT as ACTION_VALUE_CONTRACT,
    aggregate_action_values,
    candidate_key,
    candidate_set_scope,
    detect_degeneracy,
)

#: 动作价值单位（与契约一致；契约是唯一记账定义）
ACTION_VALUE_UNIT: str = ACTION_VALUE_CONTRACT["unit"]

CASH_STRATEGY_SEARCH_EVAL_VERSION = "v3h7-cash-strategy-search-eval-v1"

#: 本层评分模型身份（**与 V3H-5 区分**）
SEARCH_SCORE_MODEL: dict[str, Any] = {
    "name": "search_runout_pot_share_v1",
    "unit": "pot_share",
    "score_source": "MODEL_INTERNAL",
    "free_parameters": 0,
    "uses_opponent_fold_equity": False,
    "independent_of_v3h5_ranking_formula": True,
    "uses_randomness": False,
    "is_gto": False,
    "is_equilibrium": False,
    # ── V3I-RC2 SCORE-HYGIENE-1（v3）：评分**输入口径**（公式未变）──────────
    "input_policy": "real_equity_leaves_only",
    "input_policy_legacy": "all_equity_leaves_including_fold_only_arithmetic",
    "input_policy_note": (
        "评分公式与 unknown/覆盖率/tier 规则均未变；仅把 equity 输入由"
        "『全部 equity_present 叶（含 fold-only 恒真算术叶）』改为『真实权益叶』。"
        "无任何真实权益叶时回退到旧输入并显式标记（该情形 leaf_coverage=0 ⇒ "
        "必被 0.8 覆盖率门拒绝，故回退值不影响选择）。"),
    "evidence_hygiene_change": "SCORE-HYGIENE-1",
    "claims_no_quality_gain": True,
}

CLAIM_BOUNDARY: dict[str, bool] = {
    "is_gto": False,
    "is_equilibrium": False,
    "is_solver_output": False,
    "is_profit_evidence": False,
    "computes_exploitability": False,
    "claims_real_opponent_range": False,
    "modifies_v3h5_ranking": False,
    "wired_to_production": False,
}

_AGGRESSIVE = {"bet", "raise", "allin", "all_in"}

#: V3I-G7-1（S2）：权益可计算的**合法街态** board 张数
#: （与冻结 `_leaf_equity` / `cash_strategy_multistreet_eval` 判据一致）
#: 为保持"生产 import 图零新增"，本层**不**从 `cash_strategy_search_tree` import，
#: 而是就地声明同一常量（两处都必须等于 `(0, 3, 4, 5)`）。
VALID_EQUITY_BOARD_SIZES: tuple[int, ...] = (0, 3, 4, 5)

#: V3I-G7-1（S2）：fold-only 算术终局叶来源前缀（恒真假设，**不计真实权益**）
FOLD_ONLY_EQUITY_SOURCE_PREFIX = "arithmetic:"


def _is_fold_only_equity_leaf(leaf: dict) -> bool:
    """fold-only 算术终局叶（`arithmetic:pot_share_on_opponent_fold` 等）。"""
    return str(leaf.get("equity_source") or "").startswith(
        FOLD_ONLY_EQUITY_SOURCE_PREFIX)


def _is_legal_equity_board(board_size: Any) -> bool:
    """board 张数是否在权益白名单内（非法街态中间叶不得计入覆盖率分母）。"""
    return isinstance(board_size, int) and not isinstance(board_size, bool) \
        and board_size in VALID_EQUITY_BOARD_SIZES


def _is_cash(state: Any) -> bool:
    if not isinstance(state, dict):
        return False
    gt = state.get("game_type")
    return isinstance(gt, str) and gt.strip().lower() == "cash"


def _stub(reason: str) -> dict:
    return {
        "present": False,
        "version": CASH_STRATEGY_SEARCH_EVAL_VERSION,
        "reason": reason,
        "evidence_grade": "NONE",
        "unknown": [reason],
        "candidates": [],
        "preferred": None,
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }


def _median(xs: list[float]) -> float | None:
    if not xs:
        return None
    s = sorted(xs)
    n = len(s)
    mid = n // 2
    return round(s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2.0, 6)


def _is_num(v: Any) -> bool:
    """数值判定（排除 bool；与本包其它层口径一致）。"""
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _leaf_root_action(leaf: dict) -> Any:
    """叶子的 root 动作归属（由 `cash_strategy_search_tree` 写在 `path` 中）。"""
    path = leaf.get("path")
    if isinstance(path, dict):
        return path.get("root_action")
    return None


def _leaf_candidate_amount(leaf: dict) -> Any:
    """叶子的 **root 候选投资额**归属（可选）。

    由调用方（如 opponent-conditioned adapter）在物化条件化输入时写入
    `path.root_candidate_amount_bb`。当两个候选**同动作不同注额**时，
    仅靠 `root_action` 无法区分，必须靠本字段；未提供时退回按动作归属。
    """
    path = leaf.get("path")
    if isinstance(path, dict):
        return path.get("root_candidate_amount_bb")
    return None


def _summarize_leaves(tree: dict, root_action: str,
                      root_amount: Any = None) -> dict:
    """把一个 root candidate 下的叶子分成 evaluable / unknown / unavailable。

    **V3I-P8 最小契约修正**：本函数此前**忽略** `root_action` 参数，对
    `tree["leaves"]` 的**全部**叶子做全局池化，导致每个 candidate 拿到同一个
    叶池（实测 3 个 candidate 都报 219 个叶子，而真实归属是 73/146/146）。

    现按叶子自带的归属过滤：
    * 归属键 1：`path.root_action`（由 search tree 写入，始终存在）；
    * 归属键 2：`path.root_candidate_amount_bb`（可选，由调用方物化时写入；
      **同动作不同注额**时必须用它才能区分候选）。

    评分公式与其它逻辑均未改动。

    **V3I-G7-1（S2）证据记账更正**：额外把叶子分成
    * `real_equities`：**真实权益**叶（来源不是 fold-only 算术终局）；
    * `fold_only_leaves`：`arithmetic:*` 的 fold-only 算术终局叶
      （对手弃牌恒 1.0 的恒真假设）——**不得**计入真实权益；
    * `illegal_street_state_leaves`：board 张数不在 {0,3,4,5} 的中间态叶
      ——**不得**计入"本当需要真实权益"的分母；
    * `should_have_equity_leaves`：既非 fold-only、board 又合法的叶
      = **本当需要真实权益**的叶（覆盖率分母）。
    旧定义数字（`evaluable_equities`）**原样保留**，便于审计对照。
    """
    ev: list[float] = []
    ev_real: list[float] = []
    fold_only: list[dict] = []
    illegal_board: list[dict] = []
    should: list[dict] = []
    unknown: list[dict] = []
    unavailable: list[dict] = []
    want = None if root_action is None else str(root_action)
    want_amt = None
    if _is_num(root_amount):
        want_amt = round(float(root_amount), 6)
    for leaf in tree.get("leaves") or []:
        if not isinstance(leaf, dict):
            continue
        if want is not None and str(_leaf_root_action(leaf)) != want:
            continue
        if want_amt is not None:
            leaf_amt = _leaf_candidate_amount(leaf)
            if _is_num(leaf_amt) and round(float(leaf_amt), 6) != want_amt:
                continue
        # ── V3I-G7-1（S2）记账分类 ──
        legal_board = _is_legal_equity_board(leaf.get("board_size"))
        fold_only_leaf = _is_fold_only_equity_leaf(leaf)
        if fold_only_leaf:
            fold_only.append({"leaf_id": leaf.get("leaf_id"),
                              "equity_source": leaf.get("equity_source")})
        elif not legal_board:
            illegal_board.append({"leaf_id": leaf.get("leaf_id"),
                                  "board_size": leaf.get("board_size"),
                                  "future_board_state":
                                      leaf.get("future_board_state")})
        else:
            should.append({"leaf_id": leaf.get("leaf_id"),
                           "board_size": leaf.get("board_size")})
        # ── 原有（旧定义）分类：逐位不变 ──
        if leaf.get("equity_present"):
            eq = (leaf.get("equity") or {}).get("hero_equity")
            if isinstance(eq, (int, float)):
                ev.append(float(eq))
                if not fold_only_leaf:
                    ev_real.append(float(eq))
                continue
        rec = {"leaf_id": leaf.get("leaf_id"),
               "future_board_state": leaf.get("future_board_state"),
               "terminal_reason": leaf.get("terminal_reason"),
               "unknown": leaf.get("unknown")}
        if leaf.get("future_board_state") == "unavailable":
            unavailable.append(rec)
        else:
            unknown.append(rec)
    return {"evaluable_equities": ev, "unknown_leaves": unknown,
            "unavailable_leaves": unavailable,
            # V3I-G7-1（S2）新增
            "real_equities": ev_real,
            "fold_only_leaves": fold_only,
            "illegal_street_state_leaves": illegal_board,
            "should_have_equity_leaves": should}


def evaluate_search(state: Any, *, search_tree: dict | None = None,
                    solver: dict | None = None,
                    action_value_input: dict | None = None) -> dict:
    """按 root candidate 聚合搜索树结果（**不做最终决策**）。

    Parameters
    ----------
    state : CASH state（非 CASH → present=False）
    search_tree : **必填**，`cash_strategy_search_tree.build_search_tree` 输出
    solver : 可选，**只读转述**
    action_value_input : 可选（V3I-G1 Phase 1）。**逐候选的测量量**
        （`p_fold` / `p_continue` / `equity_vs_continuing` / `threshold` /
        类别数 / 总体），由 `cash_conditioned_search_wiring` 从 P4 bridge 物化。

        * 缺省 ⇒ 动作价值 `UNKNOWN`（**不设默认权重**，fail closed）；
        * **绝不影响** legacy `search_score`、排序、覆盖率、tier 规则；
        * 叶预算被违反（`max_leaves_exceeded`）⇒ 动作价值同样 `UNKNOWN`。
    """
    if not _is_cash(state):
        return _stub("not_cash_mode_fail_closed")
    if not isinstance(search_tree, dict):
        return _stub("search_tree_required")
    tr = search_tree
    if not tr.get("present"):
        return _stub(f"search_tree_unavailable:{tr.get('reason', 'unknown')}")

    if isinstance(solver, dict):
        solver_label = solver.get("solver_status", "UNKNOWN")
        solver_blocked = solver_label != "VALID_RESULT"
        passthrough = False
    else:
        solver_label = "UNKNOWN"
        solver_blocked = True
        passthrough = True

    pot = tr.get("root", {}).get("pot_bb")
    per_candidate: list[dict] = []
    for node in (tr.get("root") or {}).get("children") or []:
        if not isinstance(node, dict):
            continue
        action = node.get("action")
        invest = node.get("amount_bb")
        summ = _summarize_leaves(tr, str(action), invest)
        ev = summ["evaluable_equities"]          # 旧口径：含 fold-only 算术叶
        ev_real = summ["real_equities"]          # 真实权益叶（G7-1/S2 已分类）
        mean_eq_legacy = round(sum(ev) / len(ev), 6) if ev else None
        mean_eq_real = round(sum(ev_real) / len(ev_real), 6) if ev_real else None
        med_eq = _median(ev)

        # ── V3I-RC2 SCORE-HYGIENE-1（**仅证据卫生**，非能力/质量变更）────────
        # 下面用于评分的 `mean_eq` **语义已改为『真实权益叶均值』**；
        # 冻结旧口径（含 fold-only 恒真算术叶 `arithmetic:pot_share_on_opponent_fold`，
        # hero_equity=1.0）改名为 `mean_eq_legacy`，仅用于 unknown 判据与并列报告。
        # **评分公式本身逐字未改**（脚本化断言仍成立）。
        # 当且仅当**没有任何真实权益叶**时回退到旧输入，且该回退被显式标记：
        # 此时该候选 `leaf_coverage = 0.0`，会被装配的 0.8 覆盖率门拒绝
        # ⇒ 回退值不可能影响选择结果（无真实证据即不采信）。
        if mean_eq_real is not None:
            mean_eq = mean_eq_real
            score_input_source = "real_equity_leaves_only"
            score_input_fallback = False
        else:
            mean_eq = mean_eq_legacy
            score_input_source = "legacy_all_equity_leaves_fallback_no_real_equity"
            score_input_fallback = True

        score = None
        score_legacy = None
        components: dict[str, Any] = {}
        unknown: list[str] = []
        if mean_eq_legacy is None:
            unknown.append("no_evaluable_leaves")
        elif not isinstance(pot, (int, float)) or pot <= 0:
            unknown.append("pot_unavailable")
        else:
            act = str(action or "").strip().lower()
            if act == "fold":
                score = 0.0
                score_legacy = 0.0
                components["fold_zero_by_definition"] = 0.0
            else:
                if act == "check":
                    inv = 0.0
                elif act == "call":
                    inv = float(invest) if isinstance(invest, (int, float)) else None
                    if inv is None:
                        unknown.append("call_invest_unavailable")
                elif act in _AGGRESSIVE:
                    inv = float(invest) if isinstance(invest, (int, float)) else None
                    if inv is None:
                        unknown.append("aggressive_invest_unavailable")
                else:
                    inv = None
                    unknown.append(f"unknown_action:{action!r}")
                if not unknown and inv is not None:
                    score = (mean_eq * (float(pot) + 2.0 * inv) / float(pot)
                             - inv / float(pot))
                    score_legacy = (mean_eq_legacy * (float(pot) + 2.0 * inv)
                                    / float(pot) - inv / float(pot))
                    components["future_equity"] = mean_eq
                    components["future_equity_legacy_includes_fold_only"] = mean_eq_legacy
                    components["future_equity_real_equity_only"] = mean_eq_real
                    components["score_input_source"] = score_input_source
                    components["score_input_fallback_to_legacy"] = score_input_fallback
                    components["invest_bb"] = round(inv, 6)
                    components["pot_bb"] = pot
                    components["opponent_fold_equity_excluded"] = True

        per_candidate.append({
            "action": action,
            "size_bucket": node.get("size_bucket"),
            "amount_bb": invest,
            "n_leaves_total": len(ev) + len(summ["unknown_leaves"])
            + len(summ["unavailable_leaves"]),
            "n_leaves_evaluable": len(ev),
            "n_leaves_unknown": len(summ["unknown_leaves"]),
            "n_leaves_unavailable": len(summ["unavailable_leaves"]),
            # V3I-G7-1（S2）：覆盖率改为"真实 equity 叶 / 本当需要真实 equity 的叶"。
            # 旧定义（equity_present / 全部叶）**并列保留**，便于审计对照。
            "leaf_coverage": (
                round(len(summ["real_equities"])
                      / len(summ["should_have_equity_leaves"]), 6)
                if summ["should_have_equity_leaves"] else 0.0),
            "leaf_coverage_legacy": round(
                len(ev) / max(len(ev) + len(summ["unknown_leaves"])
                              + len(summ["unavailable_leaves"]), 1), 6),
            "coverage_definition":
                "real_equity_leaves_over_leaves_requiring_real_equity",
            "coverage_legacy_definition":
                "equity_present_leaves_over_all_leaves",
            "n_leaves_real_equity": len(summ["real_equities"]),
            "n_leaves_fold_only_equity": len(summ["fold_only_leaves"]),
            "n_leaves_illegal_street_state":
                len(summ["illegal_street_state_leaves"]),
            "n_leaves_requiring_real_equity":
                len(summ["should_have_equity_leaves"]),
            "future_equity_mean": mean_eq_legacy,
            "future_equity_mean_legacy_includes_fold_only": mean_eq_legacy,
            "future_equity_mean_real_equity_only": mean_eq_real,
            "score_input_source": score_input_source,
            "score_input_fallback_to_legacy": score_input_fallback,
            "score_inputs_include_fold_only_equity": False,
            "score_inputs_include_fold_only_equity_legacy": True,
            "score_inputs_note": (
                "SCORE-HYGIENE-1（v3）：search_score 的 **equity 输入**已由"
                "『全部 equity_present 叶（含 fold-only 恒真算术叶）』改为"
                "『真实权益叶』（评分的 `mean_eq` 语义变更，**公式逐字未改**；"
                "unknown 判据、覆盖率定义、tier 规则均未变）。"
                "`future_equity_mean` 字段保持**旧口径**语义以确保并列可比；"
                "旧输入下的分数并列保留于 `search_score_legacy_includes_fold_only`。"
                "本变更动机是**证据卫生**（剔除冻结基线自认不合规的输入），"
                "**不**主张质量、正确率、一致率、EV、GTO 或盈利的任何提升。"),
            "future_equity_median": med_eq,
            "search_score": (round(score, 6)
                             if isinstance(score, (int, float)) else None),
            "search_score_legacy_includes_fold_only": (
                round(score_legacy, 6)
                if isinstance(score_legacy, (int, float)) else None),
            "score_unit": SEARCH_SCORE_MODEL["unit"],
            "score_source": SEARCH_SCORE_MODEL["score_source"],
            "components": components,
            "unknown": unknown,
            "unknown_leaves": summ["unknown_leaves"][:5],
            "unavailable_leaves": summ["unavailable_leaves"][:5],
            "is_gto": False,
            "is_equilibrium": False,
        })

    scored = [c for c in per_candidate if isinstance(c["search_score"], (int, float))]
    # 确定性排序：score 降序；同分按 leaf_coverage 降序、再按 action 字典序
    ranked = sorted(scored, key=lambda c: (-float(c["search_score"]),
                                           -float(c["leaf_coverage"]),
                                           str(c["action"] or "")))
    for i, c in enumerate(ranked, 1):
        c["search_rank"] = i
    for c in per_candidate:
        c.setdefault("search_rank", None)

    unknown_all: list[str] = list(tr.get("unknown") or [])
    if passthrough:
        unknown_all.append("solver_status_not_supplied")
    for c in per_candidate:
        for u in c.get("unknown") or []:
            if u not in unknown_all:
                unknown_all.append(u)

    preferred = None
    if ranked:
        top = ranked[0]
        preferred = {
            "action": top["action"],
            "size_bucket": top["size_bucket"],
            "amount_bb": top["amount_bb"],
            "search_score": top["search_score"],
            "score_source": top["score_source"],
            "search_rank": 1,
            "leaf_coverage": top["leaf_coverage"],
        }

    # ══ V3I-G1 Phase 1：动作价值（**附加通道**；legacy 一律不变）══════════
    # 旧的 `search_score` 与 `ranked` 已在上面算完并**原样保留**；下面只新增字段。
    av_input_ok = isinstance(action_value_input, dict) and bool(
        action_value_input.get("present"))
    budget_violation = bool(tr.get("max_leaves_exceeded"))
    av_rejections: list[dict] = []
    av_reason: str | None = None
    if not av_input_ok:
        av_reason = "response_weights_not_supplied"
        av_block: dict[str, Any] = {
            "present": False, "status": av_reason, "reason": av_reason,
            "contract": dict(ACTION_VALUE_CONTRACT), "unit": ACTION_VALUE_UNIT,
            "free_parameters": 0, "values": {}, "ranked": [], "preferred": None,
            "check_baseline": None, "n_valued": 0,
            "differentiation": {"differentiates_actions": False, "source": "none"},
            "unknown": [av_reason],
            "claim_boundary": dict(ACTION_VALUE_CLAIM_BOUNDARY),
        }
    elif budget_violation:
        av_reason = "leaf_budget_violated"
        av_block = {
            "present": False, "status": av_reason, "reason": av_reason,
            "contract": dict(ACTION_VALUE_CONTRACT), "unit": ACTION_VALUE_UNIT,
            "free_parameters": 0, "values": {}, "ranked": [], "preferred": None,
            "check_baseline": None, "n_valued": 0,
            "leaf_budget": tr.get("leaf_budget"),
            "differentiation": {"differentiates_actions": False, "source": "none"},
            "unknown": [av_reason],
            "policy": ("叶预算被违反 ⇒ 动作价值抑制为 UNKNOWN（fail closed）；"
                       "legacy search_score 仍并列报告"),
            "claim_boundary": dict(ACTION_VALUE_CLAIM_BOUNDARY),
        }
    else:
        av_block = aggregate_action_values(
            pot_bb=pot,
            candidates=[{"action": c["action"], "amount_bb": c["amount_bb"]}
                        for c in per_candidate],
            records=action_value_input.get("records"))
        av_rejections = list(av_block.get("rejections") or [])
        if not av_block.get("present"):
            av_reason = av_block.get("reason") or "action_value_unavailable"
        elif not av_block.get("n_valued"):
            av_reason = "action_value_unavailable"

    av_values = av_block.get("values") or {}
    for c in per_candidate:
        row = av_values.get(candidate_key(c["action"], c["amount_bb"]))
        if row is None:
            c["action_value_pot_share"] = None
            c["action_value_gain_vs_check"] = None
            c["action_value_status"] = "UNKNOWN"
            c["action_value_unknown"] = list(av_block.get("unknown") or [])
            c["action_value_terms"] = {}
            c["action_value_measured"] = {}
            c["action_value_provenance"] = {"free_parameters": 0}
        else:
            c["action_value_pot_share"] = row.get("value")
            c["action_value_gain_vs_check"] = row.get("gain_vs_check")
            c["action_value_status"] = row.get("status")
            c["action_value_unknown"] = list(row.get("unknown") or [])
            c["action_value_terms"] = dict(row.get("terms") or {})
            c["action_value_measured"] = dict(row.get("measured") or {})
            c["action_value_provenance"] = dict(row.get("provenance") or {})

    scope = (candidate_set_scope(tiers=action_value_input.get("tiers"))
             if (av_input_ok
                 and isinstance((action_value_input or {}).get("tiers"), dict))
             else None)

    # ── V3I-G1 Phase 2：**统一模式**（同域 + 同记账 + 同一价值表）──────────
    # 触发方式：`action_value_input["value_mode"] == "unified"`（**不改签名**）。
    # 统一模式下 `ranked`/`preferred` 由**统一动作价值**决定；legacy 排序并列保留
    # （`ranked_legacy` / `preferred_legacy`）。共享价值表必须与本层**独立重算**
    # 逐位一致，否则该候选 UNKNOWN（fail closed）；域外候选**报告但不参与排序**。
    unified = bool(av_input_ok
                   and str((action_value_input or {}).get("value_mode"))
                   == "unified")
    unified_consistency: dict[str, Any] | None = None
    unified_out_of_domain: list[dict] = []
    unified_unavailable_reason: str | None = None
    ranked_legacy = ranked
    preferred_legacy = preferred
    n_scored_legacy = len(scored)
    if unified:
        shared = (action_value_input or {}).get("unified_values") or {}
        dom = (action_value_input or {}).get("domain") or {}
        common = set(dom.get("common_keys") or list(shared))
        recomputed = av_block.get("values") or {}
        # 本层**自己的**独立重算是否可用（预算违反/无记录 ⇒ 不可用，而不是"不一致"）
        recompute_available = bool(av_block.get("present"))
        matched: list[str] = []
        mismatched: list[dict] = []
        for k, row in shared.items():
            rc = recomputed.get(k)
            if (row.get("status") == "OK" and rc is not None
                    and rc.get("status") == "OK"
                    and row.get("value") == rc.get("value")):
                matched.append(k)
            elif row.get("status") == "OK" and recompute_available:
                mismatched.append({
                    "key": k, "shared_value": row.get("value"),
                    "recomputed_value": (rc or {}).get("value")})
        unified_consistency = {
            "checked": len(shared), "matched_keys": sorted(matched),
            "mismatched_keys": sorted(x["key"] for x in mismatched),
            "details": mismatched,
            "recomputation_available": recompute_available,
            "recomputation_unavailable_reason": (
                None if recompute_available
                else str(av_block.get("reason") or "value_aggregate_unavailable")),
            "policy": ("共享价值表必须与本层**独立重算**逐位一致；"
                       "重算不可用或结果不一致 ⇒ 该候选 UNKNOWN（fail closed）")}
        mismatched_keys = set(unified_consistency["mismatched_keys"])
        ranked_unified: list[dict] = []
        for c in per_candidate:
            k = candidate_key(c["action"], c["amount_bb"])
            row = shared.get(k)
            in_domain = k in common
            reasons: list[str] = []
            if not in_domain:
                reasons.append("outside_common_candidate_domain")
            if row is None:
                reasons.append("shared_value_table_missing")
            elif not recompute_available:
                reasons.append("own_recomputation_unavailable")
            elif k in mismatched_keys:
                reasons.append("unified_value_inconsistent_with_recomputation")
            elif row.get("status") != "OK" or row.get("value") is None:
                reasons.append(str(row.get("status") or "unified_value_unavailable"))
            c["unified_value_status"] = "UNKNOWN" if reasons else "OK"
            c["unified_value"] = None if reasons else row.get("value")
            c["unified_value_terms"] = {} if reasons else dict(row.get("terms") or {})
            c["unified_value_consequences"] = (
                {} if reasons else dict(row.get("consequences") or {}))
            c["unified_value_unknown"] = reasons
            if reasons:
                if not in_domain:
                    unified_out_of_domain.append({
                        "key": k, "action": c["action"],
                        "amount_bb": c["amount_bb"],
                        "reason": "outside_common_candidate_domain",
                        "search_score_legacy_v3": c.get("search_score"),
                        "counts_as_evidence_disagreement": False})
                continue
            row_out = dict(c)
            row_out["search_score_legacy_v3"] = c.get("search_score")
            row_out["search_score"] = row.get("value")
            row_out["score_source"] = "MODEL_INTERNAL"
            row_out["value_mode"] = "unified"
            row_out["action_value_pot_share"] = row.get("value")
            row_out["action_value_gain_vs_check"] = row.get("gain_vs_check")
            row_out["action_value_terms"] = dict(row.get("terms") or {})
            row_out["action_value_measured"] = dict(row.get("measured") or {})
            row_out["action_value_provenance"] = dict(row.get("provenance") or {})
            ranked_unified.append(row_out)
        ranked_unified.sort(key=lambda r: (-float(r["search_score"]),
                                           str(r["action"]),
                                           float(r["amount_bb"] or 0.0)))
        for i, r in enumerate(ranked_unified, 1):
            r["search_rank"] = i
        ranked = ranked_unified
        if ranked:
            top = ranked[0]
            preferred = {"action": top["action"], "size_bucket": top["size_bucket"],
                         "amount_bb": top["amount_bb"],
                         "search_score": top["search_score"],
                         "score_source": top["score_source"],
                         "search_rank": 1,
                         "leaf_coverage": top.get("leaf_coverage"),
                         "value_mode": "unified"}
        else:
            preferred = None
            unified_unavailable_reason = (
                (action_value_input or {}).get("unavailable_reason")
                or ("no_candidate_in_common_domain" if not common
                    else "no_candidate_with_consistent_unified_value"))
            if unified_unavailable_reason not in unknown_all:
                unknown_all.append(unified_unavailable_reason)
        for x in unified_consistency["mismatched_keys"]:
            u = f"unified_value_inconsistent:{x}"
            if u not in unknown_all:
                unknown_all.append(u)

    result = {
        "present": True,
        "version": CASH_STRATEGY_SEARCH_EVAL_VERSION,
        "mode": "cash",
        "street": tr.get("street"),
        "search_score_model": dict(SEARCH_SCORE_MODEL),
        "candidates": per_candidate,
        "n_candidates": len(per_candidate),
        "n_candidates_scored": (len(ranked) if unified else len(scored)),
        "ranked": ranked,
        "preferred": preferred,
        "preferred_note": ("highest search-aware score; NOT a final decision"
                           if preferred else
                           "suppressed: no leaf produced an evaluable equity"),
        "search_budget": tr.get("budget"),
        "search_budget_used": tr.get("budget_used"),
        "search_budget_hits": tr.get("budget_hits"),
        "search_coverage": tr.get("coverage"),
        "belief_provenance": tr.get("belief"),
        "future_board_states_used": tr.get("future_board_states_used"),
        "sampling_used": tr.get("sampling_used"),
        "seed": tr.get("seed"),
        "solver": {"solver_status": solver_label, "blocked": bool(solver_blocked),
                   "read_only_passthrough": passthrough},
        "unknown": unknown_all,
        "level": ("UNKNOWN" if not ranked else "MODEL_INTERNAL"),
        "evidence_grade": "MODEL_INTERNAL",
        "claim_boundary": dict(CLAIM_BOUNDARY),
        "final_decision_made": False,
        # ── V3I-G1 Phase 1（**只新增**；旧字段与旧排序逐位不变）──────────
        "action_value": av_block,
        "action_value_status": ("OK" if av_block.get("n_valued")
                                and av_block.get("present") else "UNKNOWN"),
        "action_value_reason": av_reason,
        "action_value_model": dict(ACTION_VALUE_CONTRACT),
        "action_value_unit": ACTION_VALUE_UNIT,
        "action_value_source": ((action_value_input or {}).get("source")
                                if av_input_ok else None),
        "action_value_population": ((action_value_input or {}).get("population")
                                    if av_input_ok else None),
        "action_value_rejections": av_rejections,
        "action_value_budget_violation": budget_violation,
        "leaf_budget": tr.get("leaf_budget"),
        "max_leaves_exceeded": tr.get("max_leaves_exceeded"),
        "preferred_by_action_value": av_block.get("preferred"),
        "ranked_by_action_value": av_block.get("ranked") or [],
        "candidate_set_scope": scope,
        # ── V3I-G1 Phase 2（只新增；legacy 模式行为逐位不变）──────────────
        "value_mode": ("unified" if unified else "legacy"),
        "unified_consistency": unified_consistency,
        "unified_domain": ((action_value_input or {}).get("domain")
                           if unified else None),
        "unified_out_of_domain": unified_out_of_domain,
        "unified_unavailable_reason": unified_unavailable_reason,
        "ranked_legacy": list(ranked_legacy) if unified else None,
        "preferred_legacy": preferred_legacy if unified else None,
        "n_candidates_scored_legacy": n_scored_legacy if unified else None,
        "legacy_search_score_preserved": True,
        "dual_report": {
            "legacy_field": ("search_score_legacy_v3" if unified
                             else "search_score"),
            "legacy_model": dict(SEARCH_SCORE_MODEL),
            "legacy_model_unchanged": True,
            "action_value_field": "action_value_pot_share",
            "action_value_model": dict(ACTION_VALUE_CONTRACT),
            "value_mode": ("unified" if unified else "legacy"),
            "legacy_ranking_field": "ranked_legacy" if unified else "ranked",
            "note": ("旧口径逐位保留并**并列**报告（统一模式下 `search_score` "
                     "语义为统一动作价值，旧值并列于 `search_score_legacy_v3`）；"
                     "两者都不是最终决策，也不构成质量 / EV / GTO / 盈利证据"),
        },
    }
    result["detector"] = detect_degeneracy(artifact=result)
    return result
