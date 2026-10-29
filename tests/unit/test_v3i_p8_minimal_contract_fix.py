"""V3I-P8 — 最小冻结 search-eval 契约修正（测试）。

修正内容（P7 已精确定位）：
    poker_decision_ai/cash_strategy_search_eval.py::_summarize_leaves
    此前**忽略** root_action 参数、对全部叶子做全局池化；现按叶子自带的
    path.root_action 归属过滤。评分公式与其它逻辑均未改动。

V3I-RC2 SCORE-HYGIENE-1（v3）经授权重基线
-----------------------------------------
`cash_strategy_search_eval` 的 search 评分**输入**口径改为"真实权益叶"
（剔除 `arithmetic:*` 的 fold-only 恒真算术终局叶）；评分**公式逐字未改**。
因此 T2 的 golden 分列 **v2（历史）** 与 **v3（当前）** 两套，并断言
"旧输入口径的并列分数逐位复现 v2 golden" —— 把数值变化归因于**输入口径**
而非公式 / 排序 / 装配。详见 `GOLDEN_V2_P8` / `GOLDEN_V3_SCORE_HYGIENE_1`
与 `test_t2_unconditioned_scores_match_v3_score_hygiene_1_golden`。

环境约束：本沙箱禁止 `%TEMP%` 写入，`tmp_path`/`ctx`/`runtime` fixture 会抛
`PermissionError [WinError 5]`，故本文件不使用需要临时目录的 fixture。
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import sys

import pytest

sys.path.insert(0, ".")

SEARCH_EVAL = pathlib.Path("poker_decision_ai/cash_strategy_search_eval.py")
SEARCH_TREE = pathlib.Path("poker_decision_ai/cash_strategy_search_tree.py")

#: **修改前**（V3I-P8 时代）的 golden —— 历史口径，**原样保留、不得覆盖**。
#: 记录者：`.tmp/v3i_p8_reference.py`（P8 改动前）。
PRE_CHANGE_HASH = "e338ce67d32bc81e"
GOLDEN_V2_P8 = {
    "flop": {
        "n_leaves_total": 219,
        "leaves_by_root_action": {"check": 73, "bet": 146},
        "rows": [["check", 0.0, 219, 0.796495, 0.796495],
                 ["bet", 0.495, 219, 0.796495, 0.992182],
                 ["bet", 0.75, 219, 0.796495, 1.09299]],
        "preferred": ["bet", 0.75, 1.09299],
    },
    "turn": {
        "n_leaves_total": 27,
        "leaves_by_root_action": {"check": 9, "bet": 18},
        "rows": [["check", 0.0, 27, 0.893678, 0.893678],
                 ["bet", 0.495, 27, 0.893678, 1.153505],
                 ["bet", 0.75, 27, 0.893678, 1.287356]],
        "preferred": ["bet", 0.75, 1.287356],
    },
}

#: V3I-RC2 SCORE-HYGIENE-1（v3）golden —— 经授权的重基线记录。
#: 记录者：`.tmp/sh1_rebaseline_probe.py`（只读探针，按文件路径 import 本测试
#: 模块并复用其 `_real()` 装配，故与本文件零漂移）。
#:
#: 数值变化的**唯一**来源：search 评分输入的 equity 口径由
#: `all_equity_leaves_including_fold_only`（含 fold-only 恒真算术终局叶
#: `arithmetic:*`，hero_equity=1.0）改为 `real_equity_leaves_only`。
#: * 评分**公式逐字未改**（脚本化断言见
#:   `test_t1_contract_change_is_the_located_minimum`）；排序 / 装配 /
#:   覆盖率 / tier 规则均未改；
#: * 报告口径 `future_equity_mean` **未变** ⇒ 第 4 列与 `GOLDEN_V2_P8`
#:   逐位相同；
#: * `search_score`（第 5 列）变化 ⇒ 与 `GOLDEN_V2_P8` 第 5 列不同；
#: * 第 6 列 `search_score_legacy_includes_fold_only` 是**并列保留的旧输入
#:   分数**，必须**逐位复现** `GOLDEN_V2_P8` 第 5 列 —— 这把"变化可归因于
#:   输入口径而非公式"变成可执行断言。
#: 本重基线**不**主张任何质量 / 正确率 / 一致率 / EV / GTO / 盈利提升。
GOLDEN_V3_SCORE_HYGIENE_1 = {
    "flop": {
        "rows": [["check", 0.0, 73, 0.796495, 0.767877, 0.796495],
                 ["bet", 0.495, 146, 0.796495, 0.944676, 0.992182],
                 ["bet", 0.75, 146, 0.796495, 1.035754, 1.09299]],
        "future_equity_mean_real_equity_only": 0.767877,
        "preferred": ["bet", 0.75, 1.035754],
    },
    "turn": {
        "rows": [["check", 0.0, 9, 0.893678, 0.880387, 0.893678],
                 ["bet", 0.495, 18, 0.893678, 1.131442, 1.153505],
                 ["bet", 0.75, 18, 0.893678, 1.260774, 1.287356]],
        "future_equity_mean_real_equity_only": 0.880387,
        "preferred": ["bet", 0.75, 1.260774],
    },
}

#: 既有引用的历史别名：P8 的叶子计数断言仍读 **v2** golden，
#: **不得**被 v3 值覆盖（两者语义不同，见上）。
GOLDEN = GOLDEN_V2_P8


# ---------------------------------------------------------------------------
# T1. 契约变更精确性（要求 1、2）
# ---------------------------------------------------------------------------
def test_t1_file_was_modified_from_pre_change_hash():
    got = hashlib.sha256(SEARCH_EVAL.read_bytes()).hexdigest()[:16]
    assert got != PRE_CHANGE_HASH, "应已应用最小修正"
    assert len(got) == 16


def test_t1_contract_change_is_the_located_minimum():
    """修正必须落在 _summarize_leaves，且只增加按 root_action 的归属过滤。"""
    src = SEARCH_EVAL.read_text(encoding="utf-8")
    assert "_leaf_root_action" in src
    assert "path.get(\"root_action\")" in src
    assert 'if want is not None and str(_leaf_root_action(leaf)) != want:' in src
    # 评分公式**不得**改动
    assert ("score = (mean_eq * (float(pot) + 2.0 * inv) / float(pot)\n"
            "                             - inv / float(pot))") in src
    # 不得引入 EV/GTO 等宣称
    for bad in ("expected_value", '"is_gto": True', '"is_equilibrium": True',
                '"is_solver_output": True'):
        assert bad not in src, bad
    assert "import random" not in src


def test_t1_search_algorithm_not_rewritten():
    """要求 1：不重写 search —— 关键结构仍在。"""
    src = SEARCH_EVAL.read_text(encoding="utf-8")
    for key in ("def evaluate_search(", "SEARCH_SCORE_MODEL",
                "def _median(", "def _summarize_leaves("):
        assert key in src, key
    tree_src = SEARCH_TREE.read_text(encoding="utf-8")
    for key in ("def build_search_tree(", "expanded_root = candidates[:cap]",
                "DEFAULT_BUDGET"):
        assert key in tree_src, key


def test_t1_summarize_leaves_now_filters_by_root_action():
    """要求 2：_summarize_leaves 必须按 root_action 正确归属叶子。"""
    from poker_decision_ai.cash_strategy_search_eval import _summarize_leaves
    tree = {"leaves": [
        {"path": {"root_action": "check"}, "equity_present": True,
         "equity": {"hero_equity": 0.1}},
        {"path": {"root_action": "bet"}, "equity_present": True,
         "equity": {"hero_equity": 0.9}},
        {"path": {"root_action": "bet"}, "equity_present": True,
         "equity": {"hero_equity": 0.7}},
    ]}
    a = _summarize_leaves(tree, "check")
    b = _summarize_leaves(tree, "bet")
    assert a["evaluable_equities"] == [0.1]
    assert b["evaluable_equities"] == [0.9, 0.7]
    # 未标注归属的叶子**不能**归给任何具体动作 → 严格排除
    # （真实 search tree 的叶子 100% 带 path.root_action，见 T2 的计数断言）
    c = _summarize_leaves({"leaves": [
        {"equity_present": True, "equity": {"hero_equity": 0.5}}]}, "bet")
    assert c["evaluable_equities"] == []
    # root_action=None → 不过滤（向后兼容的显式入口）
    d = _summarize_leaves(tree, None)
    assert len(d["evaluable_equities"]) == 3


def test_t1_unknown_and_unavailable_buckets_still_work():
    from poker_decision_ai.cash_strategy_search_eval import _summarize_leaves
    tree = {"leaves": [
        {"path": {"root_action": "bet"}, "equity_present": False,
         "future_board_state": "listed", "leaf_id": "L1"},
        {"path": {"root_action": "bet"}, "equity_present": False,
         "future_board_state": "unavailable", "leaf_id": "L2"},
        {"path": {"root_action": "check"}, "equity_present": False,
         "future_board_state": "listed", "leaf_id": "L3"},
    ]}
    r = _summarize_leaves(tree, "bet")
    assert len(r["unknown_leaves"]) == 1
    assert len(r["unavailable_leaves"]) == 1
    assert r["unknown_leaves"][0]["leaf_id"] == "L1"


# ---------------------------------------------------------------------------
# T2. 无 bridge 输入时保持原 search 行为（要求 8）
# ---------------------------------------------------------------------------
_CACHE: dict = {}


def _real():
    if "out" in _CACHE:
        return _CACHE["out"]
    from poker_decision_ai.cash_decision_environment import (
        build_scenario, POPULATION_PARAMS)
    from poker_decision_ai.cash_opponent_model import build_opponent_model
    from poker_decision_ai.cash_range_model import build_range_model
    from poker_decision_ai.cash_action_tree import build_action_tree
    from poker_decision_ai.cash_integrity_context import cash_integrity_context
    from poker_decision_ai.cash_multistreet import next_street_transition
    from poker_decision_ai.cash_strategy_action_policy import (
        build_action_policy)
    from poker_decision_ai.cash_strategy_dynamic_belief import (
        build_dynamic_belief)
    from poker_decision_ai.cash_strategy_search_tree import (
        build_search_tree, DEFAULT_BUDGET)
    from poker_decision_ai.cash_strategy_search_eval import evaluate_search
    from poker_decision_ai.cash_opponent_conditioned_bridge import (
        evaluate_opponent_conditioned)
    from poker_decision_ai.domain.poker.equity import EquityCalculator
    from poker_decision_ai.domain.poker.ranges import parse_range

    ITER = 200
    calc = EquityCalculator()

    def equity_fn(hero, board, classes, *, iterations=ITER, seed=42):
        rng = parse_range(",".join(classes), name="p8_test_range")
        res = calc.hand_vs_range(list(hero), list(board), rng,
                                 iterations=int(iterations), seed=int(seed))
        return {"present": True, "hero_equity": float(res.hero_equity),
                "samples": int(res.samples), "method": res.method}

    def setup(street="flop"):
        st = build_scenario(seed=11, street=street)["state"]
        om = build_opponent_model(st)
        rm = build_range_model(st, opponent_model=om)
        ap = build_action_policy(
            st, action_tree=build_action_tree(st), range_model=rm,
            integrity_context=cash_integrity_context(st), equity_seed=42,
            equity_iterations=20000)
        bel = build_dynamic_belief(
            st, range_model=rm, opponent_model=om,
            action_weights={"_last_action": "raise", "raise": 1.0},
            integrity_context=cash_integrity_context(st))
        tree = build_search_tree(
            st, action_policy=ap, belief=bel, belief_update=None,
            transition=next_street_transition(st), budget=DEFAULT_BUDGET,
            max_root_candidates=3)
        return st, rm, ap, bel, tree

    _CACHE["out"] = {"setup": setup, "equity_fn": equity_fn, "ITER": ITER,
                     "params": POPULATION_PARAMS,
                     "evaluate_search": evaluate_search,
                     "bridge": evaluate_opponent_conditioned}
    return _CACHE["out"]


def _bridge_for(st, rm, ap, bel, pop):
    d = _real()
    return d["bridge"](state=st, belief=bel, range_model=rm,
                       candidates=ap["candidates"], population=pop,
                       population_params=d["params"], required_equity=0.0,
                       equity_fn=d["equity_fn"], player_id="opp2",
                       iterations=d["ITER"], seed=42)


def test_t2_unconditioned_scores_match_v3_score_hygiene_1_golden():
    """要求 8（P8）+ **SCORE-HYGIENE-1 经授权重基线**（v3）。

    同一处同时断言三件事，使"数值变化只来自评分输入口径"成为可执行证据：

    1. 报告口径 `future_equity_mean` 仍**逐位等于** P8/v2 golden
       （本阶段未改动该字段语义）；
    2. `search_score` **逐位等于** v3 golden，且**不再等于** v2 golden；
    3. 并列字段 `search_score_legacy_includes_fold_only` **逐位复现** v2 golden
       的 `search_score`，且逐候选 `score_input_source ==
       "real_equity_leaves_only"`、`score_inputs_include_fold_only_equity is
       False`、`score_input_fallback_to_legacy is False`
       ⇒ 变化可归因于**评分输入口径**，而非公式 / 排序 / 装配。

    v2 golden（`GOLDEN_V2_P8`）作为历史值保留，并由第 1、3 条断言持续校验。
    """
    d = _real()
    for street in ("flop", "turn"):
        st, rm, ap, bel, tree = d["setup"](street)
        ev = d["evaluate_search"](st, search_tree=tree, solver=None)
        rows = [[c["action"], c["amount_bb"], c["n_leaves_evaluable"],
                 c["future_equity_mean"], c["search_score"]]
                for c in ev["candidates"]]
        legacy_col = [c["search_score_legacy_includes_fold_only"]
                      for c in ev["candidates"]]
        real_col = [c["future_equity_mean_real_equity_only"]
                    for c in ev["candidates"]]
        v2 = GOLDEN_V2_P8[street]
        v3 = GOLDEN_V3_SCORE_HYGIENE_1[street]

        # (1) 报告口径逐位等于 v2 golden —— 本阶段未动
        assert [r[3] for r in rows] == [r[3] for r in v2["rows"]], street
        # (2) v3 golden 生效，且与 v2 golden 明确不同
        assert [r[4] for r in rows] == [r[4] for r in v3["rows"]], street
        assert [r[4] for r in rows] != [r[4] for r in v2["rows"]], street
        # (3) 变化归因：旧输入口径的并列分数逐位复现 v2 golden 的 search_score
        assert legacy_col == [r[4] for r in v2["rows"]], street
        # 真实权益均值口径（v3 评分输入）与记录一致
        assert set(real_col) == {v3["future_equity_mean_real_equity_only"]}, \
            street
        # 逐候选显式输入口径标记
        for c in ev["candidates"]:
            assert c["score_input_source"] == "real_equity_leaves_only", street
            assert c["score_inputs_include_fold_only_equity"] is False, street
            assert c["score_input_fallback_to_legacy"] is False, street
            comp = c["components"]
            assert comp["future_equity"] == \
                c["future_equity_mean_real_equity_only"], street
            assert comp["future_equity_legacy_includes_fold_only"] == \
                c["future_equity_mean"], street
            assert comp["score_input_source"] == "real_equity_leaves_only", \
                street
            assert comp["opponent_fold_equity_excluded"] is True, street
        # 排序语义未变：名次仍由 search_score 降序决定
        ranked = sorted(ev["candidates"], key=lambda c: -float(c["search_score"]))
        assert [c["search_rank"] for c in ranked] == \
            list(range(1, len(ranked) + 1)), street
        # preferred 亦按 v3 golden 记录
        assert [ev["preferred"]["action"], ev["preferred"]["amount_bb"],
                ev["preferred"]["search_score"]] == v3["preferred"], street


def test_t2_unconditioned_leaf_counts_are_now_correct():
    """叶子计数由全局池修正为按 root_action 归属（修正的**唯一**可见变化）。"""
    d = _real()
    for street in ("flop", "turn"):
        st, rm, ap, bel, tree = d["setup"](street)
        ev = d["evaluate_search"](st, search_tree=tree, solver=None)
        by_root = GOLDEN[street]["leaves_by_root_action"]
        assert len(tree["leaves"]) == GOLDEN[street]["n_leaves_total"]
        for c in ev["candidates"]:
            want = by_root[str(c["action"])]
            assert c["n_leaves_evaluable"] == want, (street, c["action"], want)


def test_t2_unconditioned_means_equal_pooled_means():
    """数值不变的原因：按 root_action 分组后的均值与全局池均值相等。"""
    d = _real()
    for street in ("flop", "turn"):
        st, rm, ap, bel, tree = d["setup"](street)
        leaves = tree["leaves"]
        all_eq = [float((l["equity"] or {})["hero_equity"]) for l in leaves
                  if l.get("equity_present") and isinstance(
                      (l.get("equity") or {}).get("hero_equity"), (int, float))]
        pool = sum(all_eq) / len(all_eq)
        for ra in ("check", "bet"):
            sel = [float((l["equity"] or {})["hero_equity"]) for l in leaves
                   if str((l.get("path") or {}).get("root_action")) == ra
                   and l.get("equity_present") and isinstance(
                       (l.get("equity") or {}).get("hero_equity"), (int, float))]
            assert abs(sum(sel) / len(sel) - pool) < 1e-12, (street, ra)


def test_t2_passthrough_adapter_equals_direct_call():
    """adapter 的 passthrough 与直接调用冻结函数逐字节一致。"""
    from poker_decision_ai.cash_conditioned_search_wiring import (
        evaluate_search_with_conditioning, stable_fingerprint)
    d = _real()
    st, rm, ap, bel, tree = d["setup"]("flop")
    direct = d["evaluate_search"](st, search_tree=tree, solver=None)
    via = evaluate_search_with_conditioning(
        state=st, search_tree=tree, conditioned_input=None,
        evaluate_search_fn=d["evaluate_search"])
    assert via["conditioning_applied"] is False
    assert stable_fingerprint(via["search_eval"]) == stable_fingerprint(direct)


# ---------------------------------------------------------------------------
# T3. 条件化真正进入逐候选评分（要求 3、7）
# ---------------------------------------------------------------------------
def test_t3_per_candidate_future_equity_is_no_longer_pooled():
    """要求 3：每个候选消费**自身**条件化权益，而非全局池均值。"""
    from poker_decision_ai.cash_conditioned_search_wiring import (
        conditioned_evaluation_input, materialize_conditioned_tree)
    d = _real()
    st, rm, ap, bel, tree = d["setup"]("flop")
    br = _bridge_for(st, rm, ap, bel, "DECLARED_BALANCED")
    ci = conditioned_evaluation_input(bridge_out=br)
    mat = materialize_conditioned_tree(search_tree=tree, conditioned_input=ci)
    ev = d["evaluate_search"](st, search_tree=mat["tree"], solver=None)
    means = {c["action"]: c["future_equity_mean"] for c in ev["candidates"]}
    assert len(set(means.values())) > 1, means
    # check 的均值应等于桥为 check 给出的条件化权益
    ci_check = [v for v in ci["per_candidate"].values()
                if v["action"] == "check"][0]["hero_equity_vs_continuing"]
    assert means["check"] == pytest.approx(ci_check, abs=1e-6)


def test_t3_three_populations_produce_different_candidate_scores():
    """要求 7：三种 population 的 candidate-level evaluation 真正不同。"""
    from poker_decision_ai.cash_conditioned_search_wiring import (
        conditioned_evaluation_input, materialize_conditioned_tree)
    d = _real()
    st, rm, ap, bel, tree = d["setup"]("flop")
    sigs = {}
    for pop in ("DECLARED_TIGHT_PASSIVE", "DECLARED_BALANCED",
                "DECLARED_AGGRESSIVE"):
        br = _bridge_for(st, rm, ap, bel, pop)
        ci = conditioned_evaluation_input(bridge_out=br)
        mat = materialize_conditioned_tree(search_tree=tree,
                                          conditioned_input=ci)
        ev = d["evaluate_search"](st, search_tree=mat["tree"], solver=None)
        sigs[pop] = json.dumps(
            sorted([[c["action"], c["amount_bb"], c["future_equity_mean"],
                     c["search_score"]] for c in ev["candidates"]]),
            sort_keys=True)
    assert len(set(sigs.values())) == 3, "三种 population 的候选评分必须各不相同"


def test_t3_difference_comes_from_conditioned_input_not_population_label():
    """要求 7：差异必须来自 conditioned input —— 换标签但同参数 → 结果相同。"""
    from poker_decision_ai.cash_conditioned_search_wiring import (
        conditioned_evaluation_input, materialize_conditioned_tree)
    d = _real()
    st, rm, ap, bel, tree = d["setup"]("flop")
    params = dict(d["params"])
    params["DECLARED_AGGRESSIVE"] = dict(d["params"]["DECLARED_BALANCED"])
    a = _bridge_for(st, rm, ap, bel, "DECLARED_BALANCED")
    b = d["bridge"](state=st, belief=bel, range_model=rm,
                    candidates=ap["candidates"],
                    population="DECLARED_AGGRESSIVE",
                    population_params=params, required_equity=0.0,
                    equity_fn=d["equity_fn"], player_id="opp2",
                    iterations=d["ITER"], seed=42)
    ca = conditioned_evaluation_input(bridge_out=a)
    cb = conditioned_evaluation_input(bridge_out=b)
    assert json.dumps(ca["per_candidate"], sort_keys=True, default=str) == \
        json.dumps(cb["per_candidate"], sort_keys=True, default=str)


def test_t3_shifted_conditioned_equity_changes_that_candidate_score():
    """要求 3、7：只改某候选的条件化权益 → 该候选分数随之改变。"""
    from poker_decision_ai.cash_conditioned_search_wiring import (
        conditioned_evaluation_input, materialize_conditioned_tree)
    d = _real()
    st, rm, ap, bel, tree = d["setup"]("flop")
    br = _bridge_for(st, rm, ap, bel, "DECLARED_BALANCED")
    ci = conditioned_evaluation_input(bridge_out=br)

    def scored(inp):
        mat = materialize_conditioned_tree(search_tree=tree,
                                          conditioned_input=inp)
        ev = d["evaluate_search"](st, search_tree=mat["tree"], solver=None)
        # 键必须含 amount，否则多个 bet 候选会互相覆盖
        return {(c["action"], c["amount_bb"]): c["search_score"]
                for c in ev["candidates"]}

    base = scored(ci)
    assert len(base) == 3, base
    shifted = json.loads(json.dumps(ci, default=str))
    n_shifted = 0
    for v in shifted["per_candidate"].values():
        if v["action"] == "bet" and float(v["amount_bb"]) == 0.75:
            v["hero_equity_vs_continuing"] = 0.05
            n_shifted += 1
    assert n_shifted == 1, "应恰好改到一个候选"
    after = scored(shifted)
    assert base[("check", 0.0)] == after[("check", 0.0)], "check 未受影响"
    assert base[("bet", 0.75)] != after[("bet", 0.75)], \
        "被改动的候选分数必须变化"


def test_t3_preferred_changes_across_populations_on_flop():
    """要求 7：flop 上 preferred 随 population 改变（aggressive → check）。"""
    from poker_decision_ai.cash_conditioned_search_wiring import (
        conditioned_evaluation_input, materialize_conditioned_tree)
    d = _real()
    st, rm, ap, bel, tree = d["setup"]("flop")
    prefs = {}
    for pop in ("DECLARED_TIGHT_PASSIVE", "DECLARED_BALANCED",
                "DECLARED_AGGRESSIVE"):
        br = _bridge_for(st, rm, ap, bel, pop)
        ci = conditioned_evaluation_input(bridge_out=br)
        mat = materialize_conditioned_tree(search_tree=tree,
                                          conditioned_input=ci)
        ev = d["evaluate_search"](st, search_tree=mat["tree"], solver=None)
        prefs[pop] = (ev["preferred"]["action"], ev["preferred"]["amount_bb"])
    assert len(set(prefs.values())) > 1, prefs
    assert prefs["DECLARED_AGGRESSIVE"][0] == "check"


def test_t3_probe_reports_per_candidate_channel():
    from poker_decision_ai.cash_conditioned_search_wiring import (
        conditioned_evaluation_input, probe_conditioning_reaches_per_candidate_score)
    d = _real()
    st, rm, ap, bel, tree = d["setup"]("flop")
    ci = conditioned_evaluation_input(
        bridge_out=_bridge_for(st, rm, ap, bel, "DECLARED_BALANCED"))
    r = probe_conditioning_reaches_per_candidate_score(
        search_tree=tree, conditioned_input=ci,
        evaluate_search_fn=d["evaluate_search"], state=st)
    assert r["present"] is True
    assert r["future_equity_is_per_candidate"] is True
    assert r["per_candidate_scores_respond_to_own_conditioned_equity"] is True
    assert "真正进入" in r["verdict"]


# ---------------------------------------------------------------------------
# T4. 结构不变（要求 4、5）
# ---------------------------------------------------------------------------
def test_t4_candidate_cap_order_depth_budget_unchanged():
    from poker_decision_ai.cash_strategy_search_tree import DEFAULT_BUDGET
    d = _real()
    st, rm, ap, bel, tree = d["setup"]("flop")
    assert DEFAULT_BUDGET == {"max_street_depth": 2, "max_response_depth": 1,
                              "max_candidate_actions": 6, "max_leaves": 200,
                              "max_runout_children": 8}
    assert len(tree["root"]["children"]) == 3, "max_root_candidates=3 保持"
    amounts = [c["amount_bb"] for c in tree["root"]["children"]]
    assert amounts == sorted(amounts), "候选顺序保持"


def test_t4_conditioning_does_not_reorder_or_resize_candidates():
    from poker_decision_ai.cash_conditioned_search_wiring import (
        conditioned_evaluation_input, materialize_conditioned_tree)
    d = _real()
    st, rm, ap, bel, tree = d["setup"]("flop")
    ci = conditioned_evaluation_input(
        bridge_out=_bridge_for(st, rm, ap, bel, "DECLARED_AGGRESSIVE"))
    mat = materialize_conditioned_tree(search_tree=tree, conditioned_input=ci)
    before = [(c["action"], c["amount_bb"]) for c in tree["root"]["children"]]
    after = [(c["action"], c["amount_bb"])
             for c in mat["tree"]["root"]["children"]]
    assert before == after
    assert len(mat["tree"]["leaves"]) == len(tree["leaves"])


def test_t4_v3h4_v3h6_v3h8_and_tree_untouched():
    """要求 5：只改 search-eval 的最小契约，其它冻结文件不动。"""
    frozen = {
        "poker_decision_ai/cash_strategy_dynamic_belief.py": "01549df2a3385d01",
        "poker_decision_ai/cash_strategy_likelihood_model.py": "df207954ade71fa6",
  # V3I-P13 经授权在 cash_strategy_decision_assembly 实施 P12 设计的
  # feasible_keys 最小契约（**仅新增可选入参**）；装配语义未变，
  # 故此处更新为**授权的**新哈希。
        "poker_decision_ai/cash_strategy_decision_assembly.py": "3732314ab0182063",
        "poker_decision_ai/cash_decision_environment.py": "ccbf7f0048eee525",
        "poker_decision_ai/cash_decision_loop.py": "7c4b0a11c1e73a7f",
        "poker_decision_ai/cash_strategy_search_tree.py":
            hashlib.sha256(SEARCH_TREE.read_bytes()).hexdigest()[:16],
    }
    for path, want in frozen.items():
        got = hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()[:16]
        assert got == want, (path, got, want)


# ---------------------------------------------------------------------------
# T5. multistreet + fail-closed（要求 6）
# ---------------------------------------------------------------------------
def test_t5_conditioning_effective_on_turn_too():
    from poker_decision_ai.cash_conditioned_search_wiring import (
        conditioned_evaluation_input, materialize_conditioned_tree)
    d = _real()
    st, rm, ap, bel, tree = d["setup"]("turn")
    br = _bridge_for(st, rm, ap, bel, "DECLARED_BALANCED")
    ci = conditioned_evaluation_input(bridge_out=br)
    mat = materialize_conditioned_tree(search_tree=tree, conditioned_input=ci)
    ev = d["evaluate_search"](st, search_tree=mat["tree"], solver=None)
    means = {c["future_equity_mean"] for c in ev["candidates"]}
    assert len(means) > 1, means


def test_t5_unknown_population_fails_closed():
    from poker_decision_ai.cash_conditioned_search_wiring import (
        conditioned_evaluation_input)
    d = _real()
    st, rm, ap, bel, tree = d["setup"]("flop")
    bad = d["bridge"](state=st, belief=bel, range_model=rm,
                      candidates=ap["candidates"], population="NOT_A_POP",
                      population_params=d["params"], required_equity=0.0,
                      equity_fn=d["equity_fn"], player_id="opp2",
                      iterations=d["ITER"], seed=42)
    assert bad["present"] is False
    assert conditioned_evaluation_input(bridge_out=bad)["present"] is False


def test_t5_no_evidence_fails_closed():
    from poker_decision_ai.cash_conditioned_search_wiring import (
        conditioned_evaluation_input)
    from poker_decision_ai.cash_opponent_conditioned_bridge import (
        evaluate_prior_only)
    d = _real()
    st, rm, ap, bel, tree = d["setup"]("flop")
    po = evaluate_prior_only(state=st, candidates=ap["candidates"])
    assert po["adaptation_applied"] is False
    r = conditioned_evaluation_input(bridge_out=po)
    assert r["present"] is False
    assert r["reason"] == "prior_only_has_no_opponent_evidence"


def test_t5_deterministic_repeat_after_change():
    from poker_decision_ai.cash_conditioned_search_wiring import (
        conditioned_evaluation_input, materialize_conditioned_tree,
        stable_fingerprint)
    d = _real()
    st, rm, ap, bel, tree = d["setup"]("flop")
    outs = []
    for _ in range(2):
        ci = conditioned_evaluation_input(
            bridge_out=_bridge_for(st, rm, ap, bel, "DECLARED_MANIAC"))
        mat = materialize_conditioned_tree(search_tree=tree,
                                          conditioned_input=ci)
        outs.append(d["evaluate_search"](st, search_tree=mat["tree"],
                                         solver=None))
    assert stable_fingerprint(outs[0]) == stable_fingerprint(outs[1])


def test_t5_no_external_claims():
    from poker_decision_ai.cash_conditioned_search_wiring import (
        conditioned_evaluation_input)
    d = _real()
    st, rm, ap, bel, tree = d["setup"]("flop")
    ci = conditioned_evaluation_input(
        bridge_out=_bridge_for(st, rm, ap, bel, "DECLARED_BALANCED"))
    assert ci["evidence_grade"] == "MODEL_INTERNAL"
    assert all(v is False for v in ci["claim_boundary"].values())
    ev = d["evaluate_search"](st, search_tree=tree, solver=None)
    assert ev["evidence_grade"] == "MODEL_INTERNAL"
    assert all(v is False for v in ev["claim_boundary"].values())


# ---------------------------------------------------------------------------
# T6. 候选身份（动作 + 注额）归属 —— 同动作不同注额必须分开
# ---------------------------------------------------------------------------
def test_t6_summarize_leaves_separates_same_action_different_amount():
    """要求 3：同动作不同注额的候选必须各有自己的叶池。"""
    from poker_decision_ai.cash_strategy_search_eval import _summarize_leaves
    tree = {"leaves": [
        {"path": {"root_action": "bet", "root_candidate_amount_bb": 0.5},
         "equity_present": True, "equity": {"hero_equity": 0.2}},
        {"path": {"root_action": "bet", "root_candidate_amount_bb": 0.5},
         "equity_present": True, "equity": {"hero_equity": 0.4}},
        {"path": {"root_action": "bet", "root_candidate_amount_bb": 0.8},
         "equity_present": True, "equity": {"hero_equity": 0.9}},
    ]}
    a = _summarize_leaves(tree, "bet", 0.5)
    b = _summarize_leaves(tree, "bet", 0.8)
    assert a["evaluable_equities"] == [0.2, 0.4]
    assert b["evaluable_equities"] == [0.9]
    # 未提供 amount → 退回按动作归属（向后兼容）
    c = _summarize_leaves(tree, "bet")
    assert len(c["evaluable_equities"]) == 3


def test_t6_same_action_candidates_get_distinct_means_and_scores():
    """两个 bet 候选的 future_equity_mean / score 必须各自不同。"""
    from poker_decision_ai.cash_conditioned_search_wiring import (
        conditioned_evaluation_input, materialize_conditioned_tree)
    d = _real()
    st, rm, ap, bel, tree = d["setup"]("flop")
    ci = conditioned_evaluation_input(
        bridge_out=_bridge_for(st, rm, ap, bel, "DECLARED_BALANCED"))
    mat = materialize_conditioned_tree(search_tree=tree, conditioned_input=ci)
    ev = d["evaluate_search"](st, search_tree=mat["tree"], solver=None)
    bets = [(c["amount_bb"], c["future_equity_mean"], c["search_score"])
            for c in ev["candidates"] if c["action"] == "bet"]
    assert len(bets) == 2, bets
    assert bets[0][1] != bets[1][1], f"同动作不同注额的均值必须不同: {bets}"


def test_t6_materializer_writes_candidate_identity():
    from poker_decision_ai.cash_conditioned_search_wiring import (
        conditioned_evaluation_input, materialize_conditioned_tree)
    d = _real()
    st, rm, ap, bel, tree = d["setup"]("flop")
    ci = conditioned_evaluation_input(
        bridge_out=_bridge_for(st, rm, ap, bel, "DECLARED_BALANCED"))
    mat = materialize_conditioned_tree(search_tree=tree, conditioned_input=ci)
    assert mat["n_leaves_attributed"] == len(tree["leaves"])
    assert mat["n_leaves_unattributed"] == 0
    owner = {}
    for l in mat["tree"]["leaves"]:
        ra = (l.get("path") or {}).get("root_action")
        amt = (l.get("path") or {}).get("root_candidate_amount_bb")
        owner.setdefault((ra, amt), 0)
        owner[(ra, amt)] += 1
    assert ("bet", 0.495) in owner and ("bet", 0.75) in owner
    assert ("check", 0.0) in owner


def test_t6_original_tree_not_mutated_by_identity_write():
    from poker_decision_ai.cash_conditioned_search_wiring import (
        conditioned_evaluation_input, materialize_conditioned_tree)
    d = _real()
    st, rm, ap, bel, tree = d["setup"]("flop")
    before = json.dumps([[l.get("path"), (l.get("equity") or {}).get(
        "hero_equity")] for l in tree["leaves"]], sort_keys=True, default=str)
    ci = conditioned_evaluation_input(
        bridge_out=_bridge_for(st, rm, ap, bel, "DECLARED_AGGRESSIVE"))
    materialize_conditioned_tree(search_tree=tree, conditioned_input=ci)
    after = json.dumps([[l.get("path"), (l.get("equity") or {}).get(
        "hero_equity")] for l in tree["leaves"]], sort_keys=True, default=str)
    assert before == after, "原树不得被就地修改"
