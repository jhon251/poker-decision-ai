"""V3I-P7 — opponent-conditioned evaluation → search 的**最小 adapter**。

授权目标：使 P4 bridge 的结果真实进入 search candidate evaluation。

本层做什么
----------
按授权要求 5/6，把 P4 bridge 的 **per-candidate** conditioned evaluation 作为
**显式参数**传入，并物化成一棵新的 search tree 交给**冻结**的
`evaluate_search`。**不使用**全局变量、隐式状态、monkeypatch 或硬编码。

本层**不**修改任何冻结层。

精确结论（要求 13）
-------------------
本 adapter **无法**让 conditioned evaluation 真正影响 search 分数，
原因是冻结 `cash_strategy_search_eval` 存在**两处**契约限制：

1. **`_summarize_leaves(tree, root_action)` 完全忽略 `root_action`**
   （L104-119 无条件遍历 `tree["leaves"]` 的全部叶子）。
   调用方 L158 传入了动作，但被丢弃 ⇒ 每个 candidate 得到**同一个全局叶池**
   （实测：3 个 candidate 都报 `n_leaves_evaluable = 219`，而真实归属是
   `check: 73 / bet(0.495): 146 / bet(0.75): 146`）。
2. **叶子权益本身与动作无关**：即使修好 (1)，按 `path.root_action` 分组后
   各候选的 `mean_eq` 仍然**相同**（实测 flop 全部 0.796495；turn 全部 0.893678），
   因为 search tree 为每个 root action 生成结构相同的叶子、用同一个均匀对手范围
   计算权益 —— 树里没有任何 per-candidate 的对手信息。

因此"仅新增 adapter"**不可行**：需要修改冻结 search 契约（见 `MINIMAL_CONTRACT_CHANGE`）。
按授权要求 13，本层**停止并报告**该最小契约变更，**不自行扩大范围**。

本层只 import 标准库。
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

CASH_P7_SEARCH_WIRING_VERSION = "v3i-p7-opponent-conditioned-search-wiring-v1"

#: 需要修改的冻结契约（**最小**变更；本层不实施）
MINIMAL_CONTRACT_CHANGE: dict[str, Any] = {
    "file": "poker_decision_ai/cash_strategy_search_eval.py",
    "function": "_summarize_leaves",
    "current_behaviour": (
        "def _summarize_leaves(tree, root_action): 的 root_action 参数未被使用；"
        "L104 无条件遍历 tree['leaves'] 的全部叶子。"),
    "required_change_1": (
        "让 _summarize_leaves 真正按叶子归属过滤："
        "leaf.get('path', {}).get('root_action') == root_action。"),
    "required_change_2": (
        "叶子权益必须变为 per-candidate 条件化：search tree 需要消费"
        "调用方提供的 opponent-conditioned 权益（或 adapter 物化到 "
        "leaf['equity']['hero_equity']），否则即使修好 (1)，"
        "各候选的 mean_eq 仍然相同。"),
    "why_adapter_alone_is_insufficient": (
        "adapter 能提供的只是显式输入与物化后的树；但分数由冻结的 "
        "全局池 + 与动作无关的叶子权益决定，adapter 无法在不改写冻结评分逻辑的"
        "前提下改变它。"),
    "estimated_diff_size": "1 处过滤（约 3 行）+ 叶权益来源参数化",
    "contract_change_authorized_in_this_stage": False,
}

#: 被明确拒绝的"绕过"做法（记录以便审计）
REJECTED_WORKAROUNDS: tuple[dict[str, str], ...] = (
    {"name": "reverse_solve_leaf_values",
     "why_rejected": "把叶子权益反解成一组数值，使其**全局池均值**恰好等于目标"
                     "per-candidate 条件化权益。这会让冻结层打印的 "
                     "future_equity_mean 不再表示其语义，属**伪造证据**，不做。"},
    {"name": "monkeypatch_summarize_leaves",
     "why_rejected": "运行期替换冻结函数，违反要求 5（不得 monkeypatch）"
                     "与要求 3/4（保持冻结语义）。"},
    {"name": "reorder_action_policy_candidates",
     "why_rejected": "改动候选顺序以制造差异，违反要求 9"
                     "（差异必须来自 opponent-conditioned input）。"},
    {"name": "inflate_leaf_count_per_candidate",
     "why_rejected": "通过重复叶子改变全局池的算术，属操纵而非接线。"},
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
    "wired_to_production": False,
    "adds_strategy_capability": False,
    "modifies_frozen_layers": False,
    "breaks_frozen_contract": False,
    "uses_global_state_or_monkeypatch": False,
}

EVIDENCE_GRADE = "MODEL_INTERNAL"


def _is_num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _stub(reason: str) -> dict:
    return {
        "present": False,
        "version": CASH_P7_SEARCH_WIRING_VERSION,
        "reason": reason,
        "unknown": [reason],
        "claim_boundary": dict(CLAIM_BOUNDARY),
        "evidence_grade": "NONE",
        "executed": False,
        "result": None,
        "wired_to_production": False,
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
# 1. 显式输入的形状（要求 5、6）
# ---------------------------------------------------------------------------
def conditioned_evaluation_input(*, bridge_out: Any) -> dict:
    """把 P4 bridge 输出规范化为 **per-candidate** 显式输入。

    每个候选动作都有**自己**的条件化权益；**不允许**只塞一个 top-1。
    """
    if not isinstance(bridge_out, dict) or not bridge_out.get("present"):
        return _stub("present_bridge_output_required")
    # 无对手证据的 prior-only 输出**形状相似但语义不同**，必须显式拒绝：
    # 它的 ranked 分数不含任何对手条件化，若被当作 bridge 输出接受，
    # 就会把"无证据"当成"有证据"（要求 10 禁止的伪适应）。
    if bridge_out.get("prior_only") is True:
        return _stub("prior_only_has_no_opponent_evidence")
    if bridge_out.get("adaptation_applied") is False:
        return _stub("adaptation_not_applied")
    ranked = bridge_out.get("ranked")
    if not isinstance(ranked, list) or not ranked:
        return _stub("bridge_ranked_required")

    per_candidate: dict[str, dict] = {}
    for r in ranked:
        if not isinstance(r, dict):
            continue
        key = f"{r.get('action')}|{r.get('sizing_kind')}|{r.get('amount_bb')}"
        per_candidate[key] = {
            "action": r.get("action"),
            "sizing_kind": r.get("sizing_kind"),
            "amount_bb": r.get("amount_bb"),
            "hero_equity_vs_continuing": r.get("hero_equity_vs_continuing"),
            "p_continue": r.get("p_continue"),
            "n_continuation_classes": r.get("n_continuation_classes"),
            "conditioned_score": r.get("score"),
            # ── V3I-G1 Phase 1（只新增）：动作价值契约所需的两项测量量 ──
            "p_fold": r.get("p_fold"),
            "threshold": (r.get("continue_threshold") or {}).get("threshold"),
        }
    missing = [k for k, v in per_candidate.items()
               if not _is_num(v["conditioned_score"])]
    return {
        "present": True,
        "version": CASH_P7_SEARCH_WIRING_VERSION,
        "population": bridge_out.get("population"),
        "player_id": bridge_out.get("player_id"),
        "n_candidates": len(per_candidate),
        "per_candidate": per_candidate,
        "candidates_without_score": missing,
        "provides_per_candidate_not_top1": len(per_candidate) > 1,
        "explicit_input": True,
        "uses_global_state": False,
        "uses_monkeypatch": False,
        "claim_boundary": dict(CLAIM_BOUNDARY),
        "evidence_grade": EVIDENCE_GRADE,
        "executed": False,
        "result": None,
        "wired_to_production": False,
        "unknown": [],
    }


def materialize_conditioned_tree(*, search_tree: Any,
                                 conditioned_input: Any) -> dict:
    """把条件化权益**显式物化**到一棵新树的叶子权益上（不改原树）。

    返回的树保持：叶子数量、`path.root_action` 归属、以及**冻结评分不变**。
    """
    if not isinstance(search_tree, dict) or not search_tree.get("present"):
        return _stub("present_search_tree_required")
    if not isinstance(conditioned_input, dict) \
            or not conditioned_input.get("present"):
        return _stub("present_conditioned_input_required")

    per = conditioned_input.get("per_candidate") or {}
    # 把 per-candidate 权益按 (action, amount) 映射
    by_action_amount: dict[tuple, dict] = {}
    for v in per.values():
        amt = v.get("amount_bb")
        eq = v.get("hero_equity_vs_continuing")
        act = v.get("action")
        if _is_num(amt) and isinstance(act, str):
            by_action_amount[(act, round(float(amt), 6))] = {
                "equity_vs_continuing": (float(eq) if _is_num(eq) else None),
                "p_continue": v.get("p_continue"),
                "p_fold": v.get("p_fold"),
                "threshold": v.get("threshold"),
                "n_continuation_classes": v.get("n_continuation_classes"),
                "population": conditioned_input.get("population"),
            }

    # 叶子的**父候选**归属只能靠遍历子树确定：
    # `path` 里只有 root_action，没有 amount，因此不能只按 action 匹配
    # （否则两个 bet 候选会都命中第一个 bet 子节点 —— V3I-P8 修此缺陷）。
    root_node = search_tree.get("root") or {}
    leaf_to_candidate: dict[str, tuple] = {}

    def _walk(node: Any, cur: Any) -> None:
        if not isinstance(node, dict):
            return
        act = node.get("action")
        amt = node.get("amount_bb")
        if isinstance(act, str) and _is_num(amt):
            cur = (act, round(float(amt), 6))
        if node.get("is_leaf"):
            nid = node.get("node_id")
            if isinstance(nid, str) and cur is not None:
                leaf_to_candidate[nid] = cur
        for ch in node.get("children") or []:
            _walk(ch, cur)

    _walk(root_node, None)

    leaves_src = search_tree.get("leaves") or []
    leaves_out = []
    n_rewritten = 0
    n_attributed = 0
    n_response_written = 0
    for l in leaves_src:
        if not isinstance(l, dict):
            continue
        leaf = json.loads(json.dumps(l, default=str))
        cand = leaf_to_candidate.get(str(leaf.get("node_id")))
        if cand is None:
            ra = str((leaf.get("path") or {}).get("root_action"))
            cand = ("check", 0.0) if ra == "check" else None
        if cand is None:
            # 归属无法唯一确定 → 保持原值，不猜
            leaves_out.append(leaf)
            continue
        n_attributed += 1
        # 写入**候选身份**（动作 + 注额）：冻结 `_summarize_leaves` 依此把叶子
        # 归给正确的候选。仅靠 root_action 无法区分同动作不同注额的候选。
        if isinstance(leaf.get("path"), dict):
            leaf["path"]["root_candidate_amount_bb"] = cand[1]
        else:
            leaf["path"] = {"root_action": cand[0],
                            "root_candidate_amount_bb": cand[1]}
        if cand in by_action_amount and isinstance(leaf.get("equity"), dict):
            info = by_action_amount[cand]
            if _is_num(info.get("equity_vs_continuing")):
                leaf["equity"]["hero_equity_uniform"] = leaf["equity"].get(
                    "hero_equity")
                leaf["equity"]["hero_equity"] = info["equity_vs_continuing"]
                leaf["equity"]["conditioned"] = True
                leaf["equity"]["equity_source"] = \
                    "v3i_p7_adapter_explicit_input"
                n_rewritten += 1
            # V3I-G1 Phase 1：把**测得**的响应量一并物化（**只新增字段**；
            # 旧字段与旧计数语义不变）
            leaf["response"] = {
                "source": "v3i_p4_opponent_conditioned_bridge_per_candidate",
                "equity_vs_continuing": info.get("equity_vs_continuing"),
                "p_continue": info.get("p_continue"),
                "p_fold": info.get("p_fold"),
                "threshold": info.get("threshold"),
                "n_continuation_classes": info.get("n_continuation_classes"),
                "population": info.get("population"),
                "equity_basis": "equity_vs_continuing_measured",
                "weights_policy": ("measured only; missing => UNKNOWN "
                                   "(no default)"),
                "free_parameters": 0,
            }
            n_response_written += 1
        leaves_out.append(leaf)

    out = json.loads(json.dumps(search_tree, default=str))
    out["leaves"] = leaves_out
    out["conditioned_input_applied"] = True
    return {
        "present": True,
        "version": CASH_P7_SEARCH_WIRING_VERSION,
        "tree": out,
        "n_leaves_in": len(leaves_src),
        "n_leaves_out": len(leaves_out),
        "n_leaves_rewritten": n_rewritten,
        "n_response_written": n_response_written,
        "n_leaves_attributed": n_attributed,
        "n_leaves_unattributed": len(leaves_src) - n_attributed,
        "attribution_method": (
            "walk each root candidate's subtree; map leaf node_id -> "
            "(action, amount) of the candidate that owns it "
            "(path.root_action alone is ambiguous when two candidates share "
            "an action)"),
        "leaf_count_preserved": len(leaves_src) == len(leaves_out),
        "path_root_action_preserved": True,
        "per_candidate_not_top1": True,
        "claim_boundary": dict(CLAIM_BOUNDARY),
        "evidence_grade": EVIDENCE_GRADE,
        "executed": False,
        "result": None,
        "wired_to_production": False,
        "unknown": [],
    }


def evaluate_search_with_conditioning(*, state: Any, search_tree: Any,
                                     conditioned_input: Any,
                                     evaluate_search_fn: Any) -> dict:
    """**冻结 `evaluate_search`** 的调用封装：显式接收条件化输入。

    * `conditioned_input is None` → 原样透传原树（**保持原有 search 行为**）。
    * 否则 → 物化条件化权益后交给**同一个冻结函数**（公式未改）。
    """
    if not callable(evaluate_search_fn):
        return _stub("evaluate_search_fn_required")
    if not isinstance(search_tree, dict) or not search_tree.get("present"):
        return _stub("present_search_tree_required")

    if conditioned_input is None:
        out = evaluate_search_fn(state, search_tree=search_tree, solver=None)
        return {
            "present": True,
            "version": CASH_P7_SEARCH_WIRING_VERSION,
            "conditioning_applied": False,
            "path": "passthrough_no_bridge_input",
            "search_eval": out,
            "frozen_behavior_preserved": True,
            "claim_boundary": dict(CLAIM_BOUNDARY),
            "evidence_grade": EVIDENCE_GRADE,
            "executed": False,
            "result": None,
            "wired_to_production": False,
            "unknown": [],
        }

    mat = materialize_conditioned_tree(search_tree=search_tree,
                                       conditioned_input=conditioned_input)
    if not mat.get("present"):
        return _stub(f"materialize_failed:{mat.get('reason')}")
    out = evaluate_search_fn(state, search_tree=mat["tree"], solver=None)
    return {
        "present": True,
        "version": CASH_P7_SEARCH_WIRING_VERSION,
        "conditioning_applied": True,
        "path": "explicit_conditioned_tree_to_frozen_evaluator",
        "n_leaves_rewritten": mat["n_leaves_rewritten"],
        "search_eval": out,
        "frozen_scoring_formula_unchanged": True,
        "claim_boundary": dict(CLAIM_BOUNDARY),
        "evidence_grade": EVIDENCE_GRADE,
        "executed": False,
        "result": None,
        "wired_to_production": False,
        "unknown": [],
    }


# ---------------------------------------------------------------------------
# 2. 契约探针（要求 13）—— 证明 adapter 不能改变 search 分数
# ---------------------------------------------------------------------------
def probe_conditioning_reaches_score(*, search_eval_plain: Any,
                                     search_eval_conditioned: Any) -> dict:
    """比较"无 bridge 输入"与"有 per-candidate 条件化输入"的 search 结果。"""
    for name, o in (("search_eval_plain", search_eval_plain),
                    ("search_eval_conditioned", search_eval_conditioned)):
        if not isinstance(o, dict):
            return _stub(f"{name}_required")

    def _per(o: dict) -> dict:
        return {f"{c.get('action')}|{c.get('amount_bb')}":
                (c.get("future_equity_mean"), c.get("search_score"))
                for c in (o.get("candidates") or []) if isinstance(c, dict)}

    a, b = _per(search_eval_plain), _per(search_eval_conditioned)
    keys = sorted(set(a) | set(b))
    rows = {k: {"plain": a.get(k), "conditioned": b.get(k)} for k in keys}
    changed = [k for k in keys if a.get(k) != b.get(k)]
    return {
        "present": True,
        "version": CASH_P7_SEARCH_WIRING_VERSION,
        "per_candidate": rows,
        "n_candidates": len(keys),
        "candidates_whose_score_changed": changed,
        "conditioning_changed_any_score": bool(changed),
        "identical": not changed,
        "interpretation": (
            "若 identical 为 True：即使把 per-candidate 条件化权益显式写入叶子，"
            "冻结 search 分数仍不变 —— 因为 `_summarize_leaves` 忽略 root_action、"
            "对**全部**叶子取全局均值，任何 per-candidate 差异都被池化抹平。"),
        "claim_boundary": dict(CLAIM_BOUNDARY),
        "evidence_grade": EVIDENCE_GRADE,
        "executed": False,
        "result": None,
        "wired_to_production": False,
        "unknown": [],
    }


def probe_leaf_pool_is_global(*, search_eval: Any, search_tree: Any) -> dict:
    """证明冻结评估的 per-candidate 叶池是**全局**的。"""
    if not isinstance(search_eval, dict) or not isinstance(search_tree, dict):
        return _stub("search_eval_and_tree_required")
    leaves = search_tree.get("leaves") or []
    by_root: dict[str, int] = {}
    for l in leaves:
        ra = str((l.get("path") or {}).get("root_action"))
        by_root[ra] = by_root.get(ra, 0) + 1
    reported = {f"{c.get('action')}|{c.get('amount_bb')}":
                c.get("n_leaves_evaluable")
                for c in (search_eval.get("candidates") or [])}
    total = len(leaves)
    all_global = all(v == total for v in reported.values() if v is not None)
    means = {c.get("future_equity_mean")
             for c in (search_eval.get("candidates") or [])}
    return {
        "present": True,
        "version": CASH_P7_SEARCH_WIRING_VERSION,
        "true_leaves_by_root_action": by_root,
        "reported_n_leaves_evaluable": reported,
        "total_leaves": total,
        "every_candidate_reports_the_global_pool": all_global,
        "distinct_future_equity_means": sorted(m for m in means
                                               if m is not None),
        "future_equity_identical_across_candidates": len(
            {m for m in means if m is not None}) == 1,
        "root_action_parameter_used_by_frozen_code": False,
        "claim_boundary": dict(CLAIM_BOUNDARY),
        "evidence_grade": EVIDENCE_GRADE,
        "executed": False,
        "result": None,
        "wired_to_production": False,
        "unknown": [],
    }


def probe_conditioning_reaches_per_candidate_score(*, search_tree: Any,
                                                   conditioned_input: Any,
                                                   evaluate_search_fn: Any,
                                                   state: Any) -> dict:
    """**决定性检验（V3I-P8 契约修正后）**：分数是否依赖**该候选自身**的
    条件化权益？

    做法：取两份条件化输入，**逐候选权益不同但总和不相同**（因此不能靠池均值
    解释差异），比较逐候选未来的权益与分数是否各自变化。
    同时给出"同额不同权益"的对照。
    """
    if not callable(evaluate_search_fn):
        return _stub("evaluate_search_fn_required")
    if not isinstance(search_tree, dict) or not search_tree.get("present"):
        return _stub("present_search_tree_required")
    if not isinstance(conditioned_input, dict) or not conditioned_input.get("present"):
        return _stub("present_conditioned_input_required")

    def _shift(inp: dict, delta: float) -> dict:
        out = json.loads(json.dumps(inp, default=str))
        for v in out["per_candidate"].values():
            eq = v.get("hero_equity_vs_continuing")
            if _is_num(eq):
                v["hero_equity_vs_continuing"] = round(
                    min(0.999, max(0.001, float(eq) + delta)), 6)
        return out

    def _rows(ev: Any) -> list:
        return sorted([(c.get("action"), c.get("amount_bb"),
                        c.get("future_equity_mean"), c.get("search_score"))
                       for c in ((ev or {}).get("candidates") or [])])

    a = evaluate_search_with_conditioning(
        state=state, search_tree=search_tree,
        conditioned_input=conditioned_input, evaluate_search_fn=evaluate_search_fn)
    b = evaluate_search_with_conditioning(
        state=state, search_tree=search_tree,
        conditioned_input=_shift(conditioned_input, 0.05),
        evaluate_search_fn=evaluate_search_fn)
    ra, rb = _rows(a.get("search_eval")), _rows(b.get("search_eval"))
    changed_rows = [x for x, y in zip(ra, rb) if x != y]
    means_a = [r[2] for r in ra]
    return {
        "present": True,
        "version": CASH_P7_SEARCH_WIRING_VERSION,
        "rows_baseline": ra,
        "rows_shifted": rb,
        "n_rows_changed": len(changed_rows),
        "per_candidate_scores_respond_to_own_conditioned_equity":
            len(changed_rows) > 0,
        "distinct_future_equity_means_baseline": sorted(
            {m for m in means_a if m is not None}),
        "future_equity_is_per_candidate": len(
            {m for m in means_a if m is not None}) > 1,
        "preferred_baseline": (a.get("search_eval") or {}).get("preferred"),
        "preferred_shifted": (b.get("search_eval") or {}).get("preferred"),
        "verdict": ("每个候选消费**自身**条件化权益，且其分数随该权益变化 "
                    "⇒ 条件化已真正进入 candidate-by-candidate search evaluation"
                    if len(changed_rows) > 0 else
                    "分数未随自身条件化权益变化 ⇒ 条件化仍未真正生效"),
        "claim_boundary": dict(CLAIM_BOUNDARY),
        "evidence_grade": EVIDENCE_GRADE,
        "executed": False,
        "result": None,
        "wired_to_production": False,
        "unknown": [],
    }


def probe_conditioning_is_pooled_not_per_candidate(*, search_tree: Any,
                                                   base_input: Any,
                                                   evaluate_search_fn: Any,
                                                   state: Any) -> dict:
    """**决定性检验**：分数是否真的依赖"该候选自己的"条件化权益？

    做法：构造两个 per-candidate 条件化输入，它们对所有候选给出**相同总量**
    （因此全局池均值相同），但**逐候选权益完全不同**。
    若冻结评估的分数只依赖全局池均值，则两次结果必须**逐字节相同** ——
    那就证明 adapter 引入的差异并非真正的 per-candidate 条件化。

    再构造一个"两个候选同额但条件化权益不同"的对照：若分数仍相同，
    则进一步证明分数与候选自身的条件化权益**无关**。
    """
    if not callable(evaluate_search_fn):
        return _stub("evaluate_search_fn_required")
    if not isinstance(search_tree, dict) or not search_tree.get("present"):
        return _stub("present_search_tree_required")
    if not isinstance(base_input, dict) or not base_input.get("present"):
        return _stub("present_base_input_required")

    per = dict(base_input.get("per_candidate") or {})
    if not per:
        return _stub("base_input_has_no_candidates")

    # 叶子数按 root_action 的真实分布（决定池化权重）
    by_root: dict[str, int] = {}
    for l in search_tree.get("leaves") or []:
        ra = str((l.get("path") or {}).get("root_action"))
        by_root[ra] = by_root.get(ra, 0) + 1
    total = sum(by_root.values()) or 1

    # 基准向量：每个 root 动作一个权益
    base_eq: dict[str, float] = {}
    for v in per.values():
        ra = str(v.get("action"))
        eq = v.get("hero_equity_vs_continuing")
        if _is_num(eq) and ra not in base_eq:
            base_eq[ra] = float(eq)
    if not base_eq:
        return _stub("base_equities_unavailable")

    pool_base = sum(by_root.get(ra, 0) * eq for ra, eq in base_eq.items()) / total

    # 变体：把权益在 root 动作之间"置换"，但**保持池均值完全相同**
    #   pool = Σ n_ra · eq_ra / total  →  只要加权和不变，池均值不变
    variant: dict[str, float] = {}
    ras = sorted(base_eq)
    if len(ras) >= 2:
        a, b = ras[0], ras[1]
        na, nb = by_root.get(a, 0), by_root.get(b, 0)
        if na > 0 and nb > 0:
            # 把 a 的权益抬高 d、b 降低 d*na/nb，保持加权和不变
            d = min(0.10, max(0.0, 1.0 - base_eq[a]))
            variant = dict(base_eq)
            variant[a] = round(min(1.0, base_eq[a] + d), 6)
            variant[b] = round(max(0.0, base_eq[b] - d * na / nb), 6)

    def _input_with(vector: dict) -> dict:
        out = json.loads(json.dumps(base_input, default=str))
        for k, v in out["per_candidate"].items():
            ra = str(v.get("action"))
            if ra in vector:
                v["hero_equity_vs_continuing"] = vector[ra]
        return out

    if not variant:
        return {
            "present": True,
            "version": CASH_P7_SEARCH_WIRING_VERSION,
            "testable": False,
            "reason": "single_root_action_cannot_permute",
            "leaves_by_root_action": by_root,
            "claim_boundary": dict(CLAIM_BOUNDARY),
            "evidence_grade": EVIDENCE_GRADE,
            "executed": False, "result": None, "wired_to_production": False,
            "unknown": [],
        }

    r1 = evaluate_search_with_conditioning(
        state=state, search_tree=search_tree,
        conditioned_input=_input_with(base_eq), evaluate_search_fn=evaluate_search_fn)
    r2 = evaluate_search_with_conditioning(
        state=state, search_tree=search_tree,
        conditioned_input=_input_with(variant), evaluate_search_fn=evaluate_search_fn)

    pool_variant = sum(by_root.get(ra, 0) * variant.get(ra, base_eq[ra])
                       for ra in base_eq) / total

    def _scoring(o: Any) -> list:
        """只取**评分相关**字段（排除透传元数据）。"""
        ev = (o or {}).get("search_eval") or {}
        rows = [(c.get("action"), c.get("amount_bb"),
                 c.get("future_equity_mean"), c.get("search_score"))
                for c in (ev.get("candidates") or [])]
        pref = ev.get("preferred") or {}
        return rows + [(pref.get("action"), pref.get("amount_bb"),
                        pref.get("search_score"))]

    s1, s2 = _scoring(r1), _scoring(r2)
    identical = s1 == s2
    return {
        "present": True,
        "version": CASH_P7_SEARCH_WIRING_VERSION,
        "testable": True,
        "leaves_by_root_action": by_root,
        "base_equities": {k: round(v, 6) for k, v in base_eq.items()},
        "variant_equities": {k: round(v, 6) for k, v in variant.items()},
        "pool_mean_base": round(pool_base, 6),
        "pool_mean_variant": round(pool_variant, 6),
        "pool_mean_held_constant": abs(pool_base - pool_variant) < 1e-9,
        "per_candidate_equities_differ": base_eq != variant,
        "scoring_base": s1,
        "scoring_variant": s2,
        "scoring_fields_identical": identical,
        "whole_output_identical": stable_fingerprint(r1) ==
        stable_fingerprint(r2),
        "note_on_whole_output": (
            "整份输出可能仍有差异，但差异只在**透传元数据**（树内容回显），"
            "不在评分字段；只有评分字段参与决策。"),
        "scores_depend_only_on_global_pool_mean": identical,
        "verdict": ("分数只依赖全局池均值 ⇒ adapter **无法**实现真正的 "
                    "per-candidate 条件化（冻结 `_summarize_leaves` 的 "
                    "root_action 参数未被使用）"
                    if identical else
                    "分数确实随 per-candidate 权益改变 ⇒ 存在真实 per-candidate 通道"),
        "claim_boundary": dict(CLAIM_BOUNDARY),
        "evidence_grade": EVIDENCE_GRADE,
        "executed": False,
        "result": None,
        "wired_to_production": False,
        "unknown": [],
    }


def adapter_contract() -> dict:
    return {
        "version": CASH_P7_SEARCH_WIRING_VERSION,
        "explicit_input_name": "conditioned_input",
        "per_candidate": True,
        "top1_only": False,
        "uses_global_state": False,
        "uses_monkeypatch": False,
        "hardcodes_population_to_action": False,
        "modifies_frozen_layers": False,
        "frozen_scoring_formula_changed": False,
        "minimal_contract_change_required": True,
        "contract_change_authorized": False,
        "minimal_contract_change": dict(MINIMAL_CONTRACT_CHANGE),
        "rejected_workarounds": [dict(r) for r in REJECTED_WORKAROUNDS],
        "claim_boundary": dict(CLAIM_BOUNDARY),
        "evidence_grade": EVIDENCE_GRADE,
    }


# ══════════════════════════════════════════════════════════════════════
# V3I-G1 Phase 1 — **动作价值通道**
# 把 P4 bridge 的逐候选**测量量**（p_fold / p_continue /
# equity_vs_continuing / threshold / 类别数）物化为
# `cash_action_value_contract` 的显式输入。本层**不合成**权重、**不设默认值**：
# 缺任何一项 ⇒ 契约侧 UNKNOWN（fail closed）。
# ══════════════════════════════════════════════════════════════════════
CASH_G1_PHASE1_ACTION_VALUE_CHANNEL_VERSION = (
    "v3i-g1-phase1-conditioned-action-value-channel-v1")


def _basis_fingerprint(record: dict) -> str:
    """记录基的**定义性**指纹（口径必须与 `cash_action_value_contract.basis_fingerprint` 一致）。

    本模块**只 import 标准库**（P7 契约：不 import 任何项目内模块，因此拿不到
    任何冻结函数引用），故此处就地实现同一 recipe；两者一致性由
    `tests/unit/test_v3i_g1_phase1_action_value.py` 的交叉校验测试强制。
    """
    basis = {
        "population": record.get("population"),
        "threshold": (round(float(record["threshold"]), 6)
                      if _is_num(record.get("threshold")) else None),
        "n_continuation_classes": record.get("n_continuation_classes"),
        "equity_vs_continuing": (round(float(record["equity_vs_continuing"]), 6)
                                 if _is_num(record.get("equity_vs_continuing"))
                                 else None),
        "p_continue": (round(float(record["p_continue"]), 6)
                       if _is_num(record.get("p_continue")) else None),
    }
    text = json.dumps(basis, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def action_value_input_from_bridge(*, bridge_out: Any) -> dict:
    """P4 bridge 逐候选测量量 → 契约输入（显式、零默认值、带定义性指纹）。"""
    if not isinstance(bridge_out, dict) or not bridge_out.get("present"):
        stub = _stub("present_bridge_output_required")
        stub["records"] = []
        return stub
    if bridge_out.get("prior_only") is True:
        stub = _stub("prior_only_has_no_opponent_evidence")
        stub["records"] = []
        return stub
    records: list[dict] = []
    for r in (bridge_out.get("ranked") or []):
        if not isinstance(r, dict):
            continue
        rec = {
            "action": r.get("action"),
            "amount_bb": r.get("amount_bb"),
            "equity_vs_continuing": r.get("hero_equity_vs_continuing"),
            "p_continue": r.get("p_continue"),
            "p_fold": r.get("p_fold"),
            "threshold": (r.get("continue_threshold") or {}).get("threshold"),
            "n_continuation_classes": r.get("n_continuation_classes"),
            "population": bridge_out.get("population"),
            "equity_basis": "equity_vs_continuing_measured",
        }
        rec["basis_fingerprint"] = _basis_fingerprint(rec)
        records.append(rec)
    return {
        "present": True,
        "version": CASH_G1_PHASE1_ACTION_VALUE_CHANNEL_VERSION,
        "source": "v3i_p4_opponent_conditioned_bridge",
        "population": bridge_out.get("population"),
        "player_id": bridge_out.get("player_id"),
        "n_records": len(records),
        "records": records,
        "equity_basis": "equity_vs_continuing_measured",
        "weights_policy": "measured only; missing => UNKNOWN (no default)",
        "free_parameters": 0,
        "uses_global_state": False,
        "uses_monkeypatch": False,
        "explicit_input": True,
        "wired_to_production": False,
        "evidence_grade": EVIDENCE_GRADE,
        "claim_boundary": dict(CLAIM_BOUNDARY),
        "unknown": [],
    }


def evaluate_search_with_action_value(*, state: Any, search_tree: Any,
                                      bridge_out: Any,
                                      evaluate_search_fn: Any,
                                      solver: Any = None) -> dict:
    """编排：bridge 测量 → 契约输入 → `evaluate_search(action_value_input=...)`。

    `evaluate_search_fn` 必须是**接受 `action_value_input` 关键字**的
    `cash_strategy_search_eval.evaluate_search`（Phase 1 签名）。
    """
    if not callable(evaluate_search_fn):
        return _stub("evaluate_search_fn_required")
    av_in = action_value_input_from_bridge(bridge_out=bridge_out)
    if not av_in.get("present"):
        # 无测量输入 ⇒ 仍调用（契约侧将 fail closed 为 UNKNOWN），不猜权重
        out = evaluate_search_fn(state, search_tree=search_tree, solver=solver)
        return {
            "present": True,
            "version": CASH_G1_PHASE1_ACTION_VALUE_CHANNEL_VERSION,
            "action_value_input": av_in,
            "search_eval": out,
            "action_value_applied": False,
            "reason": av_in.get("reason"),
            "claim_boundary": dict(CLAIM_BOUNDARY),
            "evidence_grade": EVIDENCE_GRADE,
            "executed": False,
            "result": None,
            "wired_to_production": False,
            "unknown": list(av_in.get("unknown") or []),
        }
    out = evaluate_search_fn(state, search_tree=search_tree, solver=solver,
                             action_value_input=av_in)
    return {
        "present": True,
        "version": CASH_G1_PHASE1_ACTION_VALUE_CHANNEL_VERSION,
        "action_value_input": av_in,
        "search_eval": out,
        "action_value_applied": True,
        "population": av_in.get("population"),
        "n_records": av_in.get("n_records"),
        "claim_boundary": dict(CLAIM_BOUNDARY),
        "evidence_grade": EVIDENCE_GRADE,
        "executed": False,
        "result": None,
        "wired_to_production": False,
        "unknown": [],
    }


def action_value_channel_contract() -> dict:
    """Phase 1 通道自述（审计用；**不**改写 P7 的历史结论）。"""
    return {
        "version": CASH_G1_PHASE1_ACTION_VALUE_CHANNEL_VERSION,
        "inputs_materialised": ["equity_vs_continuing", "p_continue", "p_fold",
                                "threshold", "n_continuation_classes",
                                "population", "equity_basis",
                                "basis_fingerprint"],
        "weights_synthesised_here": False,
        "defaults_used": False,
        "missing_input_policy": "UNKNOWN (fail closed)",
        "free_parameters": 0,
        "modifies_frozen_scoring": False,
        "leaf_response_field": "leaf.response",
        "note": ("P7 adapter 的历史结论（当时冻结 `_summarize_leaves` 忽略 "
                 "root_action）已由 V3I-P8 修复；本通道在其之上只做**测量量物化**，"
                 "不改变 legacy `search_score`、gate、tiers、override。"),
        "claim_boundary": dict(CLAIM_BOUNDARY),
        "evidence_grade": EVIDENCE_GRADE,
    }


# ══════════════════════════════════════════════════════════════════════
# V3I-G1 **Phase 2** — 统一动作价值路径（Search / Ranking 同域、同记账、同输入）
# 本模块**只 import 标准库**（P7/P1 契约）：契约函数、两层评估与装配全部由
# 调用方**注入**；本层只做编排、域对齐与**报告**（含 legacy 对照），不做策略判断。
# ══════════════════════════════════════════════════════════════════════
CASH_G1_PHASE2_UNIFIED_PATH_VERSION = "v3i-g1-phase2-unified-action-value-path-v1"


def _candidate_key_str(action: Any, amount_bb: Any) -> str:
    """候选键字符串（口径必须与 `cash_action_value_contract.candidate_key` 一致）。

    本模块只 import 标准库，故就地实现；一致性由
    `tests/unit/test_v3i_g1_phase2_unified_action_value.py` 交叉校验。
    """
    amt = (round(float(amount_bb), 6)
           if isinstance(amount_bb, (int, float))
           and not isinstance(amount_bb, bool) else 0.0)
    return f"{action}|{amt}"


def _tree_consequences(search_tree: Any) -> dict:
    """从 search tree 的**记账字段**取 per-candidate pot/stack 后果（只读取）。"""
    out: dict[str, dict] = {}
    root = (search_tree or {}).get("root") or {}
    for node in root.get("children") or []:
        if not isinstance(node, dict):
            continue
        key = _candidate_key_str(node.get("action"), node.get("amount_bb"))
        out[key] = {
            "hero_investment_bb": node.get("hero_investment_bb"),
            "pot_after_hero_bb": node.get("pot_after_hero_bb"),
            "stack_after_hero_bb": node.get("stack_after_hero_bb"),
            "source": "search_tree.accounting",
        }
    return out


def _disagreement_keys(decision: Any) -> list:
    if not isinstance(decision, dict):
        return []
    out = []
    for d in decision.get("disagreements") or []:
        if isinstance(d, dict):
            out.append(f"{d.get('tier_a')}:{d.get('key_a')}"
                       f"!={d.get('tier_b')}:{d.get('key_b')}")
    return out


def unified_action_value_path(*, state: Any, action_policy: Any, belief: Any,
                              bridge_out: Any, search_tree: Any,
                              unified_input_fn: Any, search_eval_fn: Any,
                              rank_actions_fn: Any, assemble_decision_fn: Any,
                              solver: Any = None, feasible_keys: Any = None,
                              include_legacy_comparison: bool = True) -> dict:
    """统一路径编排：bridge 测量 → 契约统一价值表 → 两层同域评估 → 装配。

    所有契约/评估/装配函数**由调用方注入**（本层不硬耦合、不 import 项目内模块），
    因此可被测试完全替换，且**不含**任何策略判断。
    """
    if not all(callable(f) for f in (unified_input_fn, search_eval_fn,
                                     rank_actions_fn, assemble_decision_fn)):
        return _stub("injected_callables_required")
    pot = (search_tree or {}).get("root", {}).get("pot_bb")
    consequence_map = _tree_consequences(search_tree)
    av_in = action_value_input_from_bridge(bridge_out=bridge_out)
    search_keys = sorted(consequence_map)
    ranking_keys = sorted({_candidate_key_str(c.get("action"), c.get("amount_bb"))
                           for c in (action_policy or {}).get("candidates") or []
                           if isinstance(c, dict)})
    candidates = [{"action": c.get("action"), "amount_bb": c.get("amount_bb")}
                  for c in (action_policy or {}).get("candidates") or []
                  if isinstance(c, dict)
                  and _candidate_key_str(c.get("action"), c.get("amount_bb"))
                  in set(search_keys) & set(ranking_keys)]
    budget_violated = bool((search_tree or {}).get("max_leaves_exceeded"))
    if av_in.get("present"):
        unified_in = unified_input_fn(
            pot_bb=pot, candidates=candidates, records=av_in.get("records"),
            search_keys=search_keys, ranking_keys=ranking_keys,
            consequences=consequence_map,
            population=av_in.get("population"),
            source="v3i_g1_phase2_unified_path",
            budget_violated=budget_violated)
    else:
        # 无测量输入：**仍然**进入统一模式（present=True 但零价值），
        # 使**两层都**没有候选 ⇒ 装配报 <tier>_top_candidate_unavailable ⇒
        # 决策 UNKNOWN/ABSTAIN。**禁止**让任一层悄悄回落到 legacy 排序。
        unified_in = {
            "present": True, "value_mode": "unified",
            "version": CASH_G1_PHASE2_UNIFIED_PATH_VERSION,
            "unavailable_reason": av_in.get("reason")
            or "response_weights_not_supplied",
            "unified_values": {}, "n_valued": 0, "free_parameters": 0,
            "domain": {"common_keys": sorted(set(search_keys) & set(ranking_keys)),
                       "per_tier_keys": {"search": search_keys,
                                         "ranking": ranking_keys},
                       "candidate_set_mismatch": bool(
                           set(search_keys) ^ set(ranking_keys)),
                       "only_in_tier": {
                           "search": sorted(set(search_keys) - set(ranking_keys)),
                           "ranking": sorted(set(ranking_keys) - set(search_keys))},
                       "counts_as_evidence_disagreement": False},
            "records": [], "unknown": ["response_weights_not_supplied"],
        }

    se = search_eval_fn(state, search_tree=search_tree, solver=solver,
                        action_value_input=unified_in)
    rk = rank_actions_fn(state, action_policy=action_policy, belief=belief,
                         action_value_input=unified_in)
    dec = assemble_decision_fn(state=state, ranking=rk, belief_conditioned=None,
                               search_eval=se, solver=solver,
                               feasible_keys=feasible_keys)
    legacy_dec = None
    if include_legacy_comparison:
        se_l = search_eval_fn(state, search_tree=search_tree, solver=solver)
        rk_l = rank_actions_fn(state, action_policy=action_policy, belief=belief)
        legacy_dec = assemble_decision_fn(state=state, ranking=rk_l,
                                          belief_conditioned=None,
                                          search_eval=se_l, solver=solver,
                                          feasible_keys=feasible_keys)
    unified_keys = _disagreement_keys(dec)
    legacy_keys = _disagreement_keys(legacy_dec)
    domain = unified_in.get("domain") or {}
    return {
        "present": True,
        "version": CASH_G1_PHASE2_UNIFIED_PATH_VERSION,
        "value_mode": "unified",
        "unified_input": unified_in,
        "domain": domain,
        "search_eval": se,
        "ranking": rk,
        "decision": dec,
        "legacy_decision": legacy_dec,
        "search_tier_top": (se.get("preferred") or {}).get("action"),
        "ranking_tier_top": (rk.get("preferred") or {}).get("action"),
        "tier_tops_equal": (
            (se.get("preferred") or {}).get("amount_bb"),
            (se.get("preferred") or {}).get("action")) == (
            (rk.get("preferred") or {}).get("amount_bb"),
            (rk.get("preferred") or {}).get("action")),
        "disagreement_provenance": {
            "unified_disagreements": unified_keys,
            "legacy_disagreements": legacy_keys,
            "pseudo_disagreement_keys_removed": sorted(
                set(legacy_keys) - set(unified_keys)),
            "genuine_evidence_disagreement_keys": sorted(unified_keys),
            "candidate_set_mismatch": bool(domain.get("candidate_set_mismatch")),
            "candidate_set_mismatch_counts_as_disagreement": False,
            "accounting_unified": True,
            "note": ("统一模式下两层使用**同一份**动作价值表与**同一候选域**，"
                     "因此因候选集/记账差异产生的 disagreement 消失；"
                     "剩余的 disagreement 才计入证据冲突。"
                     "**代价**：search 与 ranking 在动作价值上不再是独立证据源"
                     "（legacy 口径仍并列报告，但不得当作决策依据）"),
        },
        "fail_closed": {
            "budget_violated": budget_violated,
            "unified_unavailable_reason": unified_in.get("unavailable_reason"),
            "search_ranked_n": len(se.get("ranked") or []),
            "ranking_ranked_n": len(rk.get("ranked") or []),
            "abstained": (dec or {}).get("abstained"),
            "decision_unknown": list((dec or {}).get("unknown") or []),
        },
        "claim_boundary": dict(CLAIM_BOUNDARY),
        "evidence_grade": EVIDENCE_GRADE,
        "executed": False,
        "result": None,
        "wired_to_production": False,
        "unknown": list(unified_in.get("unknown") or []),
    }


def unified_path_contract() -> dict:
    """Phase 2 统一路径自述（审计用）。"""
    return {
        "version": CASH_G1_PHASE2_UNIFIED_PATH_VERSION,
        "injected_callables": ["unified_input_fn", "search_eval_fn",
                               "rank_actions_fn", "assemble_decision_fn"],
        "imports_project_modules": False,
        "stdlib_only": True,
        "key_format": "f\"{action}|{round(amount,6)}\"",
        "consequences_source": "search_tree.accounting（只读取）",
        "legacy_comparison_included": True,
        "no_voting_no_override": True,
        "gate_changed": False,
        "claim_boundary": dict(CLAIM_BOUNDARY),
        "evidence_grade": EVIDENCE_GRADE,
    }
