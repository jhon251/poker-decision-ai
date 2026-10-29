# -*- coding: utf-8 -*-
"""V3H-10 测试 — CASH Shadow Evaluation（建议 ↔ 已观测 hero 动作 对账留痕）。

覆盖：CASH-only fail closed / 非 dict / 缺决策 fail closed /
建议缺失（弃权·分歧·无首选）→ UNAVAILABLE 不猜 / 观测缺失 → UNKNOWN 不猜 /
aligned·different 分类与金额容差 / **描述性**（不判优、不称盈利）/
反事实快照**必须由调用方注入**（未注入 fail closed，不新增公式）/
MODEL_INTERNAL vs NONE 证据等级 / unknown·弃权·分歧原样传播 /
零评分·零 EV·零执行 / 定输入复现 / 真实全栈端到端 /
AST·import 隔离（零 V3G import、非 stdlib 零 import）/ 生产 import 图零新增 /
frozen·sealed drift / Solver BLOCKED 不翻转。
"""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PKG = ROOT / "poker_decision_ai"

from poker_decision_ai.cash_strategy_shadow_evaluation import (  # noqa: E402
    CASH_STRATEGY_SHADOW_EVALUATION_VERSION, CLAIM_BOUNDARY,
    DEFAULT_AMOUNT_TOLERANCE_BB, DIVERGENCE_CLASSES, evaluate_shadow_decision,
    make_shadow_observation)

# 测试侧允许 import V3G/V3H 层（生产侧不允许，由 AST 测试强制）
from poker_decision_ai.cash_action_tree import build_action_tree  # noqa: E402
from poker_decision_ai.cash_range_model import build_range_model  # noqa: E402
from poker_decision_ai.cash_solver_interface import solver_status  # noqa: E402
from poker_decision_ai.cash_strategy_action_policy import (  # noqa: E402
    build_action_policy)
from poker_decision_ai.cash_strategy_action_ranking import rank_actions  # noqa: E402
from poker_decision_ai.cash_strategy_belief_conditioned_eval import (  # noqa: E402
    evaluate_belief_conditioned)
from poker_decision_ai.cash_strategy_decision_assembly import (  # noqa: E402
    assemble_decision)
from poker_decision_ai.cash_strategy_dynamic_belief import (  # noqa: E402
    build_dynamic_belief)
from poker_decision_ai.cash_strategy_experiment_eligibility import (  # noqa: E402
    audit_experiment_eligibility)
from poker_decision_ai.cash_strategy_multistreet_eval import (  # noqa: E402
    conditional_absolute_equity)
from poker_decision_ai.cash_strategy_search_eval import evaluate_search  # noqa: E402
from poker_decision_ai.cash_strategy_search_tree import build_search_tree  # noqa: E402

NEW_MODULE = "cash_strategy_shadow_evaluation"
MODULE_PATH = PKG / f"{NEW_MODULE}.py"

FROZEN = {
    "poker_decision_ai/domain/poker/strategy_core.py": "378a68671ff09e3f",
    "poker_decision_ai/domain/poker/models.py": "6775c06050319a85",
    "poker_decision_ai/domain/poker/conditional_model.py": "686c9f58b71d9408",
    "poker_decision_ai/agents/judge/agent.py": "239436fae7299c47",
    "poker_decision_ai/realtime.py": "3400a20ff857880d",
    "poker_decision_ai/research_unfreeze/pi_ev_contract.py": "52e6267db49a2e11",
    "poker_decision_ai/research_unfreeze/unfreeze_core.py": "17236d4ed8070d01",
}

V3G_LAYERS = ("cash_state_integrity", "cash_integrity_context",
              "cash_preflop_decision", "cash_multistreet", "cash_opponent_model",
              "cash_range_model", "cash_action_tree", "cash_unknown_e2e",
              "cash_solver_interface", "cash_profit_interface")
V3H_LAYERS = tuple(f.stem for f in PKG.glob("cash_strategy_*.py"))

BUDGET = {"max_street_depth": 2, "max_response_depth": 1,
          "max_candidate_actions": 3, "max_leaves": 60, "max_runout_children": 2}


def _sha(rel: str) -> str:
    p = ROOT / rel
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else "MISSING"


def cash_state(n_players: int = 3, street: str = "flop", board=None, **over) -> dict:
    pos = {2: ("SB", "BB"), 3: ("SB", "BB", "BTN"),
           4: ("SB", "BB", "MP", "BTN"),
           5: ("SB", "BB", "MP", "CO", "BTN"),
           6: ("SB", "BB", "UTG", "MP", "CO", "BTN")}[n_players]
    hero_seat = n_players
    players, vi = [], 1
    for seat in range(1, n_players + 1):
        if seat == hero_seat:
            players.append({"player_id": "hero", "seat": seat,
                            "position": pos[seat - 1], "is_hero": True,
                            "status": "active", "hole_cards": ["As", "Kd"],
                            "stack": 100.0})
        else:
            players.append({"player_id": f"v{vi}", "seat": seat,
                            "position": pos[seat - 1], "status": "active",
                            "stack": 100.0, "hole_cards": []})
            vi += 1
    st = {
        "game_type": "cash", "street": street,
        "board": ["Qs", "7d", "2c"] if board is None else board,
        "pot": 10.0, "to_call": 0.0, "bb": 1.0, "sb": 0.5,
        "hero_id": "hero", "button_seat": 2, "table_size": n_players,
        "synthetic_test_only": True, "players": players, "action_history": [],
    }
    st.update(over)
    return st


def good_decision(**over) -> dict:
    """合成一个"对账成功"的 V3H-8 决策摘要。"""
    d = {"present": True, "level": "MODEL_INTERNAL", "abstained": False,
         "disagreements": [], "unknown": [], "street": "flop",
         "evidence_grade": "MODEL_INTERNAL",
         "preferred": {"action": "check", "candidate_key": ["check", 0.0],
                       "selected_by_tier": "search",
                       "score_source": "MODEL_INTERNAL", "leaf_coverage": 1.0}}
    d.update(over)
    return d


def real_stack(state, **over):
    """真实全栈（V3H-4→8），返回决策与旁路工件。"""
    ap = build_action_policy(state, action_tree=build_action_tree(state))
    bl = build_dynamic_belief(state, range_model=build_range_model(state))
    rk = rank_actions(state, action_policy=ap, belief=bl)
    bc = evaluate_belief_conditioned(state, action_policy=ap, ranking=rk)
    tr = build_search_tree(state, action_policy=ap, belief=bl, budget=BUDGET,
                           max_root_candidates=3)
    se = evaluate_search(state, search_tree=tr)
    dec = assemble_decision(state, ranking=rk, belief_conditioned=bc,
                            search_eval=se, solver=solver_status(state), **over)
    return dec, {"action_policy": ap, "belief": bl, "ranking": rk,
                 "belief_conditioned": bc}


# ── 1. CASH-only fail closed ──────────────────────────────────────────
@pytest.mark.parametrize("mode", ["mtt", "pko", "ko", "mystery", "spin",
                                 "MTT", "PKO", "", "   ", None])
def test_non_cash_fail_closed(mode):
    r = evaluate_shadow_decision(cash_state(game_type=mode),
                                 decision=good_decision(),
                                 observed=make_shadow_observation("check", 0.0))
    assert r["present"] is False
    assert r["reason"] == "not_cash_mode_fail_closed"
    assert r["divergence"] == "unavailable"
    assert r["record"] is None
    assert r["executed"] is False
    assert r["evidence_grade"] == "NONE"


@pytest.mark.parametrize("bad", [None, [], "cash", 42, 3.5, True])
def test_non_dict_state_fail_closed(bad):
    r = evaluate_shadow_decision(bad, decision=good_decision())
    assert r["present"] is False
    assert r["reason"] == "not_cash_mode_fail_closed"
    assert r["executed"] is False


@pytest.mark.parametrize("bad", [None, [], "decision", 42])
def test_missing_decision_fails_closed(bad):
    r = evaluate_shadow_decision(cash_state(), decision=bad)
    assert r["present"] is False
    assert r["reason"] == "decision_required"
    assert r["record"] is None
    assert r["executed"] is False


def test_claim_boundary_all_false_and_present_on_stub():
    for v in CLAIM_BOUNDARY.values():
        assert v is False
    r = evaluate_shadow_decision(cash_state(), decision=None)
    assert all(v is False for v in r["claim_boundary"].values())


# ── 2. 建议侧缺失 → UNAVAILABLE，不猜 ────────────────────────────────
def test_abstained_decision_yields_unavailable():
    d = good_decision(abstained=True, preferred=None,
                      unknown=["scorer_disagreement"])
    r = evaluate_shadow_decision(cash_state(), decision=d,
                                 observed=make_shadow_observation("check", 0.0))
    assert r["present"] is True
    assert r["divergence"] == "unavailable"
    assert r["available"] is False
    assert r["level"] == "UNKNOWN"
    assert r["record"]["recommended"] is None
    assert any("recommendation_unavailable" in u for u in r["unknown"])
    assert "decision_abstained" in r["unknown"]          # 原样传播
    assert "scorer_disagreement" in r["unknown"]         # 原样传播


def test_decision_absent_present_false_yields_unavailable():
    d = good_decision(present=False, preferred=None, reason="no_ranking")
    r = evaluate_shadow_decision(cash_state(), decision=d,
                                 observed=make_shadow_observation("call", 1.0))
    assert r["divergence"] == "unavailable"
    assert r["available"] is False
    assert any("no_ranking" in u for u in r["unknown"])


def test_decision_without_preferred_yields_unavailable():
    d = good_decision(preferred=None)
    r = evaluate_shadow_decision(cash_state(), decision=d,
                                 observed=make_shadow_observation("check", 0.0))
    assert r["divergence"] == "unavailable"
    assert r["record"]["recommended"] is None


def test_invalid_recommendation_action_is_unknown_not_guessed():
    d = good_decision(preferred={"action": "   ", "amount_bb": 1.0})
    r = evaluate_shadow_decision(cash_state(), decision=d,
                                 observed=make_shadow_observation("check", 0.0))
    assert r["divergence"] == "unavailable"
    assert "recommendation_action_invalid" in r["unknown"]


# ── 3. 观测侧缺失 → UNKNOWN，不猜 ────────────────────────────────────
def test_observed_absent_is_unknown_and_not_guessed():
    r = evaluate_shadow_decision(cash_state(), decision=good_decision())
    assert r["divergence"] == "unavailable"
    assert r["record"]["observed"] is None
    assert "hero_observed_action_absent" in r["unknown"]
    assert r["available"] is False
    assert r["evidence_grade"] == "NONE"
    assert r["record"]["evidence_grade"] == "NONE"


def test_observed_read_from_state_field():
    st = cash_state(hero_observed_action={"action": "Check", "amount_bb": 0.0,
                                          "source": "hand_log"})
    d = good_decision(preferred={"action": "check", "amount_bb": 0.0,
                                 "selected_by_tier": "search",
                                 "score_source": "MODEL_INTERNAL"})
    r = evaluate_shadow_decision(st, decision=d)
    assert r["divergence"] == "aligned"
    assert r["record"]["observed"]["action"] == "check"   # 规范化
    assert r["record"]["observed"]["source"] == "hand_log"
    assert r["available"] is True
    assert r["evidence_grade"] == "MODEL_INTERNAL"


def test_observed_read_from_action_history():
    st = cash_state(action_history=[{"player_id": "v1", "action": "call"},
                                    {"player_id": "hero", "action": "RAISE",
                                     "amount_bb": 6.0}])
    d = good_decision(preferred={"action": "raise", "amount_bb": 6.0,
                                 "selected_by_tier": "search",
                                 "score_source": "MODEL_INTERNAL"})
    r = evaluate_shadow_decision(st, decision=d)
    assert r["record"]["observed"]["action"] == "raise"
    assert r["divergence"] == "aligned"
    assert r["record"]["observed"]["source"] == "state.action_history(hero)"


def test_observed_state_dict_without_action_is_unknown():
    st = cash_state(hero_observed_action={"amount_bb": 3.0})
    r = evaluate_shadow_decision(st, decision=good_decision())
    assert r["divergence"] == "unavailable"
    assert "hero_observed_action_missing_action" in r["unknown"]


def test_make_shadow_observation_invalid():
    assert make_shadow_observation(None)["present"] is False
    assert make_shadow_observation("  ")["present"] is False
    assert "observed_action_invalid" in make_shadow_observation(7)["unknown"]
    ok = make_shadow_observation("Bet", 2.5, source="log")
    assert ok == {"present": True, "action": "bet", "amount_bb": 2.5,
                  "source": "log"}


def test_make_shadow_observation_amount_type_guarded():
    assert make_shadow_observation("bet", True)["amount_bb"] is None
    assert make_shadow_observation("bet", "3")["amount_bb"] is None


# ── 4. 分歧分类（**描述性**）─────────────────────────────────────────
def test_aligned_same_action_and_amount():
    d = good_decision(preferred={"action": "bet", "amount_bb": 3.0,
                                 "selected_by_tier": "ranking",
                                 "score_source": "MODEL_INTERNAL"})
    r = evaluate_shadow_decision(cash_state(), decision=d,
                                 observed=make_shadow_observation("bet", 3.0))
    assert r["divergence"] == "aligned"
    assert r["record"]["divergence_is_descriptive_only"] is True
    assert r["record"]["preference_judgement_made"] is False
    assert r["record"]["executed"] is False
    assert r["record"]["is_profit_evidence"] is False
    assert r["record"]["is_causal_claim"] is False


def test_different_action_is_descriptive_not_judgement():
    d = good_decision(preferred={"action": "bet", "amount_bb": 3.0,
                                 "selected_by_tier": "search",
                                 "score_source": "MODEL_INTERNAL"})
    r = evaluate_shadow_decision(cash_state(), decision=d,
                                 observed=make_shadow_observation("fold", None))
    assert r["divergence"] == "different"
    assert r["record"]["preference_judgement_made"] is False
    # 不得出现任何"更优/盈利"判定字段（精确键名匹配，避免误伤 is_profit_evidence）
    rec_keys = set(r) | set(r["record"]) | set(r["claim_boundary"])
    for banned in ("better", "worse", "profit", "expected_value", "ev_delta",
                   "regret", "advantage", "superior"):
        assert not any(k == banned or k.startswith(banned + "_")
                       or k.endswith("_" + banned) for k in rec_keys), banned
    assert r["record"]["is_profit_evidence"] is False
    assert r["record"]["is_causal_claim"] is False


def test_same_action_different_amount_tolerance_boundary():
    d = good_decision(preferred={"action": "bet", "amount_bb": 3.0,
                                 "selected_by_tier": "search",
                                 "score_source": "MODEL_INTERNAL"})
    inside = evaluate_shadow_decision(
        cash_state(), decision=d,
        observed=make_shadow_observation("bet", 3.0 + 5e-7))
    assert inside["divergence"] == "aligned"
    outside = evaluate_shadow_decision(
        cash_state(), decision=d,
        observed=make_shadow_observation("bet", 3.0 + 2e-6))
    assert outside["divergence"] == "different"


def test_amount_tolerance_is_explicit_and_recorded():
    assert DEFAULT_AMOUNT_TOLERANCE_BB == 1e-6
    d = good_decision(preferred={"action": "bet", "amount_bb": 3.0,
                                 "selected_by_tier": "search",
                                 "score_source": "MODEL_INTERNAL"})
    r = evaluate_shadow_decision(cash_state(), decision=d,
                                 observed=make_shadow_observation("bet", 3.01),
                                 amount_tolerance_bb=0.05)
    assert r["divergence"] == "aligned"
    assert r["record"]["amount_tolerance_bb"] == 0.05
    assert r["uncertainty"]["amount_tolerance_bb"] == 0.05


def test_missing_amount_on_one_side_requires_both_missing():
    d = good_decision(preferred={"action": "check", "amount_bb": 1.0,
                                 "selected_by_tier": "search",
                                 "score_source": "MODEL_INTERNAL"})
    r = evaluate_shadow_decision(cash_state(), decision=d,
                                 observed=make_shadow_observation("check", None))
    assert r["divergence"] == "different"     # 数量不可比 → 不当作 aligned
    d2 = good_decision(preferred={"action": "check", "amount_bb": None,
                                  "selected_by_tier": "search",
                                  "score_source": "MODEL_INTERNAL"})
    r2 = evaluate_shadow_decision(cash_state(), decision=d2,
                                  observed=make_shadow_observation("check", None))
    assert r2["divergence"] == "aligned"


def test_divergence_classes_closed_enum():
    assert DIVERGENCE_CLASSES == ("aligned", "different", "unavailable")
    d = good_decision(preferred={"action": "bet", "amount_bb": 3.0})
    for obs in (make_shadow_observation("bet", 3.0),
                make_shadow_observation("fold", None), None):
        r = evaluate_shadow_decision(cash_state(), decision=d, observed=obs)
        assert r["divergence"] in DIVERGENCE_CLASSES
        assert r["divergence_classes"] == list(DIVERGENCE_CLASSES)


# ── 5. 反事实快照：必须调用方注入、fail closed、不新增公式 ────────────
def test_snapshot_fail_closed_when_primitive_not_injected():
    d = good_decision(preferred={"action": "check", "amount_bb": 0.0,
                                 "selected_by_tier": "search",
                                 "score_source": "MODEL_INTERNAL"})
    r = evaluate_shadow_decision(
        cash_state(), decision=d,
        observed=make_shadow_observation("check", 0.0))
    snap = r["record"]["shadow_snapshot"]
    assert snap["present"] is False
    assert "equity_snapshot_not_supplied" in snap["unknown"]
    assert snap["adds_new_formula"] is False
    assert r["uncertainty"]["snapshot_present"] is False
    # 快照缺失**不**污染 divergence 判定（两者独立）
    assert r["divergence"] == "aligned"


def test_snapshot_uses_injected_primitive_and_is_flagged_not_profit():
    r = evaluate_shadow_decision(
        cash_state(), decision=good_decision(),
        observed=make_shadow_observation("bet", 2.0),
        equity_snapshot=conditional_absolute_equity)
    snap = r["record"]["shadow_snapshot"]
    assert snap["present"] is True
    assert snap["adds_new_formula"] is False
    assert 0.0 <= snap["conditional_absolute_equity"] <= 1.0
    assert "不是" in snap["note"] and "EV" in snap["note"]
    assert r["uncertainty"]["snapshot_method"] == snap["method"]
    assert r["uncertainty"]["snapshot_samples"] == snap["samples"]
    assert snap["source"].startswith("domain.poker.equity")


def test_snapshot_accepts_synthetic_injected_callable():
    calls = []

    def fake(hole, board):
        calls.append((list(hole), list(board)))
        return {"present": True, "hero_equity": 0.5, "samples": 10,
                "method": "synthetic", "source": "test_stub"}

    r = evaluate_shadow_decision(cash_state(), decision=good_decision(),
                                 observed=make_shadow_observation("bet", 2.0),
                                 equity_snapshot=fake)
    assert calls and calls[0][1] == ["Qs", "7d", "2c"]
    assert r["record"]["shadow_snapshot"]["conditional_absolute_equity"] == 0.5


def test_snapshot_propagates_upstream_failure_and_exception():
    r = evaluate_shadow_decision(
        cash_state(), decision=good_decision(),
        observed=make_shadow_observation("bet", 2.0),
        equity_snapshot=lambda h, b: {"present": False,
                                      "unknown": ["insufficient_cards"]})
    assert r["record"]["shadow_snapshot"]["present"] is False
    assert "insufficient_cards" in r["record"]["shadow_snapshot"]["unknown"]

    def boom(h, b):
        raise RuntimeError("x")

    r2 = evaluate_shadow_decision(
        cash_state(), decision=good_decision(),
        observed=make_shadow_observation("bet", 2.0), equity_snapshot=boom)
    assert r2["record"]["shadow_snapshot"]["present"] is False
    assert any("equity_snapshot_error" in u
               for u in r2["record"]["shadow_snapshot"]["unknown"])

    r3 = evaluate_shadow_decision(
        cash_state(), decision=good_decision(),
        observed=make_shadow_observation("bet", 2.0),
        equity_snapshot=lambda h, b: "bad")
    assert r3["record"]["shadow_snapshot"]["present"] is False


@pytest.mark.parametrize("board,why", [("nope", "board_missing"),
                                       ([1, 2, 3], "board_malformed"),
                                       (["Qs", "7d"], "board_size_invalid")])
def test_snapshot_board_fail_closed(board, why):
    r = evaluate_shadow_decision(
        cash_state(board=board), decision=good_decision(),
        observed=make_shadow_observation("check", 0.0),
        equity_snapshot=conditional_absolute_equity)
    snap = r["record"]["shadow_snapshot"]
    assert snap["present"] is False
    assert why in snap["unknown"]


def test_snapshot_missing_hero_cards_fail_closed():
    st = cash_state()
    st["players"][-1]["hole_cards"] = []
    r = evaluate_shadow_decision(st, decision=good_decision(),
                                 observed=make_shadow_observation("check", 0.0),
                                 equity_snapshot=conditional_absolute_equity)
    assert r["record"]["shadow_snapshot"]["present"] is False
    assert "hero_hole_cards_missing" in r["record"]["shadow_snapshot"]["unknown"]


# ── 6. 旁路工件转述（候选尺寸 / 合法集 / 资格）───────────────────────
def test_action_policy_supplied_records_sizes_and_legality():
    st = cash_state()
    ap = build_action_policy(st, action_tree=build_action_tree(st))
    r = evaluate_shadow_decision(st, decision=good_decision(),
                                 action_policy=ap,
                                 observed=make_shadow_observation("check", 0.0))
    rec = r["record"]
    assert r["record"]["legal_actions"] == [str(a) for a in ap["legal_actions"]]
    assert rec["candidate_sizes_from_action_policy"] == {"check": 0.0,
                                                         "bet": pytest.approx(3.3)}


def test_action_policy_absent_is_unknown_not_defaulted():
    r = evaluate_shadow_decision(cash_state(), decision=good_decision(),
                                 observed=make_shadow_observation("check", 0.0))
    assert "action_policy_not_supplied" in r["unknown"]
    assert r["record"]["legal_actions"] == []
    assert r["record"]["candidate_sizes_from_action_policy"] == {}


def test_eligibility_trace_transcribed_not_recomputed():
    st = cash_state()
    ap = build_action_policy(st, action_tree=build_action_tree(st))
    el = audit_experiment_eligibility(st, decision=good_decision(),
                                      solver=solver_status(st))
    r = evaluate_shadow_decision(st, decision=good_decision(),
                                 action_policy=ap, eligibility=el,
                                 observed=make_shadow_observation("check", 0.0))
    tr = r["record"]["eligibility_trace"]
    assert tr is not None
    assert tr["source"] == "V3H-9 audit_experiment_eligibility"
    assert tr["eligible"] == el["eligible"]
    assert tr["status"] == el["status"]
    assert tr["n_blockers"] == el["n_blockers"]


def test_eligibility_absent_is_unknown():
    r = evaluate_shadow_decision(cash_state(), decision=good_decision(),
                                 observed=make_shadow_observation("check", 0.0))
    assert r["record"]["eligibility_trace"] is None
    assert "eligibility_not_supplied" in r["unknown"]


def test_disagreements_are_counted_and_propagated():
    d = good_decision(abstained=True, preferred=None,
                      disagreements=[{"scorer": "ranking"},
                                     {"scorer": "search"}])
    r = evaluate_shadow_decision(cash_state(), decision=d,
                                 observed=make_shadow_observation("check", 0.0))
    assert "decision_disagreements:2" in r["unknown"]
    assert r["divergence"] == "unavailable"


def test_unknown_list_is_deduplicated_and_ordered():
    d = good_decision(abstained=True, preferred=None,
                      unknown=["a", "b", "a"])
    r = evaluate_shadow_decision(cash_state(), decision=d)
    assert r["unknown"].count("a") == 1
    assert r["unknown"].index("a") < r["unknown"].index("b")


# ── 7. 真实全栈端到端 ────────────────────────────────────────────────
def test_end_to_end_real_stack_aligned():
    st = cash_state()
    dec, arts = real_stack(st)
    assert dec["present"] is True, dec.get("reason")
    pref = dec["preferred"]
    r = evaluate_shadow_decision(
        st, decision=dec, action_policy=arts["action_policy"],
        eligibility=audit_experiment_eligibility(st, decision=dec,
                                                solver=solver_status(st)),
        observed=make_shadow_observation(pref["action"], pref.get("amount_bb")),
        equity_snapshot=conditional_absolute_equity)
    assert r["present"] is True
    assert r["divergence"] == "aligned"
    assert r["evidence_grade"] == "MODEL_INTERNAL"
    assert r["record"]["recommended"]["action"] == pref["action"]
    assert r["record"]["recommended"]["selected_by_tier"] == \
        pref.get("selected_by_tier")
    assert r["record"]["shadow_snapshot"]["present"] is True
    assert r["executed"] is False
    assert r["record"]["is_profit_evidence"] is False
    assert r["unknown"] == []                                # 无污染
    assert r["record"]["divergence"] == "aligned"


def test_end_to_end_real_stack_divergent_observation():
    st = cash_state()
    dec, arts = real_stack(st)
    other = "fold" if dec["preferred"]["action"] != "fold" else "check"
    r = evaluate_shadow_decision(st, decision=dec,
                                 action_policy=arts["action_policy"],
                                 observed=make_shadow_observation(other, None),
                                 equity_snapshot=conditional_absolute_equity)
    assert r["divergence"] == "different"
    assert r["record"]["preference_judgement_made"] is False
    assert r["executed"] is False


def test_end_to_end_river_terminal_still_fail_closed():
    """冻结语义（V3G-7）：river 为终局 → 上游弃权、无建议，
    本层**如实**记为 UNAVAILABLE，**不**伪造建议、**不**绕过。"""
    st = cash_state(street="river", board=["Qs", "7d", "2c", "8h", "3s"])
    dec, arts = real_stack(st)
    assert dec["present"] is False and dec["abstained"] is True
    assert dec["preferred"] is None
    assert dec["reason"] == "no_scorer_supplied"
    r = evaluate_shadow_decision(st, decision=dec,
                                 action_policy=arts["action_policy"],
                                 observed=make_shadow_observation("check", 0.0))
    assert r["present"] is True
    assert r["divergence"] == "unavailable"
    assert r["available"] is False
    assert r["level"] == "UNKNOWN"
    assert r["evidence_grade"] == "NONE"
    assert r["record"]["recommended"] is None
    assert r["record"]["observed"]["action"] == "check"
    assert "decision_abstained" in r["unknown"]
    assert "no_scorer_supplied" in r["unknown"]
    assert r["executed"] is False
    assert r["record"]["executed"] is False


# ── 8. 无执行 / 无评分 / 无 EV / 定输入复现 ──────────────────────────
def test_no_execution_or_betting_path_anywhere():
    src = MODULE_PATH.read_text(encoding="utf-8")
    for banned in ("click", "pyautogui", "keyboard.", "subprocess", "socket",
                   "requests", "urllib", "open(", "input(", "screenshot"):
        assert banned not in src, banned


def test_no_new_scoring_or_ev_formula():
    """本层不得引入新公式：数值算式应仅限键对齐/容差比较。"""
    src = MODULE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(src)
    numeric_ops = []
    for n in ast.walk(tree):
        if isinstance(n, ast.BinOp) and isinstance(n.op, (ast.Mult, ast.Div,
                                                          ast.Pow, ast.Sub)):
            numeric_ops.append(type(n.op).__name__)
    # 仅允许容差差值 |a-b|
    assert set(numeric_ops) <= {"Sub"}, numeric_ops
    for banned in ("compute_ev", "ev_delta", "expected_value", "pot_odds",
                   "bluff_ratio", "exploitability"):
        assert banned not in src, banned
    # `is_equilibrium` 只允许作为 claim_boundary 的**否定声明**出现（两处定义）
    assert src.count("is_equilibrium") == 2
    assert src.count('"is_equilibrium": False') == 2


def test_deterministic_same_input_same_output():
    st = cash_state(hero_observed_action={"action": "bet", "amount_bb": 2.0})
    d = good_decision(preferred={"action": "bet", "amount_bb": 2.0,
                                 "selected_by_tier": "search",
                                 "score_source": "MODEL_INTERNAL"})
    outs = [evaluate_shadow_decision(st, decision=d) for _ in range(5)]
    assert all(o == outs[0] for o in outs)
    outs2 = [evaluate_shadow_decision(st, decision=d,
                                      observed=make_shadow_observation("fold"))
             for _ in range(3)]
    assert all(o == outs2[0] for o in outs2)


def test_state_not_mutated():
    st = cash_state(hero_observed_action={"action": "check", "amount_bb": 0.0})
    before = json.dumps(st, sort_keys=True, ensure_ascii=False)
    evaluate_shadow_decision(st, decision=good_decision(),
                            action_policy=None,
                            observed=make_shadow_observation("bet", 2.0),
                            equity_snapshot=conditional_absolute_equity)
    assert json.dumps(st, sort_keys=True, ensure_ascii=False) == before


# ── 9. AST · import 隔离 ─────────────────────────────────────────────
def test_declares_version_constant():
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    consts = {n.targets[0].id for n in tree.body
              if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)}
    assert any(c.endswith("_VERSION") for c in consts)
    assert CASH_STRATEGY_SHADOW_EVALUATION_VERSION.startswith("v3h10-")


def test_top_level_imports_are_stdlib_only():
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    mods = [n.module for n in tree.body if isinstance(n, ast.ImportFrom)]
    assert mods == ["__future__", "typing"], mods


def test_zero_v3g_imports_anywhere_including_function_bodies():
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    for n in ast.walk(tree):
        if isinstance(n, ast.ImportFrom):
            assert not any(v in (n.module or "") for v in V3G_LAYERS), n.module
        if isinstance(n, ast.Import):
            for a in n.names:
                assert not any(v in a.name for v in V3G_LAYERS), a.name


def test_zero_layer_imports_anywhere():
    """V3G-11 边界语义：本层不 import **任何** V3G/V3H 层（依赖全部参数注入）。"""
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    for n in ast.walk(tree):
        mods = []
        if isinstance(n, ast.ImportFrom):
            mods.append(n.module or "")
        elif isinstance(n, ast.Import):
            mods += [a.name for a in n.names]
        for m in mods:
            assert not any(v in m for v in V3G_LAYERS + V3H_LAYERS), m


def test_no_random_import():
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            assert all(a.name.split(".")[0] != "random" for a in n.names)
        if isinstance(n, ast.ImportFrom):
            assert (n.module or "").split(".")[0] != "random"


def test_no_frozen_multistreet_or_sample_runouts():
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            assert all("cash_multistreet" not in a.name for a in n.names)
        if isinstance(n, ast.Name):
            assert n.id != "sample_runouts"


# ── 10. 生产 import 图零新增 ─────────────────────────────────────────
def _production_importers() -> dict[str, set[str]]:
    found: dict[str, set[str]] = {}
    for p in PKG.rglob("*.py"):
        if p.name.startswith("cash_") or "__pycache__" in str(p):
            continue
        try:
            tree = ast.parse(p.read_text(encoding="utf-8-sig"))
        except SyntaxError:
            continue
        for n in ast.walk(tree):
            mods = []
            if isinstance(n, ast.ImportFrom) and n.module:
                mods.append(n.module.split(".")[-1])
            elif isinstance(n, ast.Import):
                mods += [a.name.split(".")[-1] for a in n.names]
            for m in mods:
                if m in V3G_LAYERS or m in V3H_LAYERS:
                    found.setdefault(m, set()).add(p.name)
    return found


def test_no_new_production_imports():
    found = _production_importers()
    assert set(found) == {"cash_integrity_context", "cash_unknown_e2e"}, found


def test_new_module_not_imported_in_production():
    assert NEW_MODULE not in _production_importers()


def test_all_v3h_layers_remain_unwired():
    found = _production_importers()
    for layer in V3H_LAYERS:
        assert layer not in found, layer


# ── 11. frozen / sealed drift · Solver ───────────────────────────────
@pytest.mark.parametrize("rel,prefix", sorted(FROZEN.items()))
def test_frozen_files_unchanged(rel, prefix):
    assert _sha(rel).startswith(prefix), rel


def test_v3g_manifest_drift_unchanged():
    man = json.loads((ROOT / "docs" / "V3G_11_FREEZE_MANIFEST.json")
                     .read_text(encoding="utf-8"))
    drift = []
    for block in ("frozen_modules", "frozen_tests", "frozen_closeouts",
                  "production_baseline"):
        for rel, h in man[block].items():
            if h == "__SELF__":
                continue
            if _sha(rel) != h:
                drift.append(rel)
    assert drift in ([], ["docs/V3G_11_EVALUATION_FREEZE_CLOSEOUT.md"]), drift


def test_solver_still_blocked():
    st = cash_state()
    s = solver_status(st)
    assert s["solver_status"] == "BLOCKED"
    assert s["available"] is False
    assert s["equilibrium"] is False
    assert s["gto_exact"] is False
    assert s["executed"] is False
    r = evaluate_shadow_decision(st, decision=good_decision(),
                                 observed=make_shadow_observation("check", 0.0))
    assert r["record"]["is_gto"] is False
    assert r["record"]["is_equilibrium"] is False
    assert r["record"]["is_solver_output"] is False
    assert r["claim_boundary"]["is_gto"] is False


def test_non_cash_solver_state_never_upgraded():
    for mode in ("mtt", "pko"):
        r = evaluate_shadow_decision(cash_state(game_type=mode),
                                     decision=good_decision(),
                                     observed=make_shadow_observation("call", 1.0))
        assert r["present"] is False
