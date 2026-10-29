# -*- coding: utf-8 -*-
"""V3H-8 测试 — CASH Decision Assembly（多评分源对账 + 弃权 + UNKNOWN 传播）。

覆盖：真三层对账（真实层输出）/ 一致性采用 / 分歧弃权 / 金额容差 /
覆盖率闸门 / 显式覆盖（含"无理由即忽略"）/ UNKNOWN 原样传播 / 2–6 人 / 四街 /
CASH 与 MTT·PKO 隔离 / Solver BLOCKED 不翻转 / 无 GTO·equilibrium /
不修改任何评分公式 / 定输入复现 / Judge·RealtimeRunner 逐位不变 /
生产 import 图零新增 / frozen·sealed SHA 零漂移。
"""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PKG = ROOT / "poker_decision_ai"

from poker_decision_ai.cash_strategy_decision_assembly import (  # noqa: E402
    ASSEMBLY_POLICY, CLAIM_BOUNDARY, TIERS, assemble_decision)

# 测试侧允许 import V3G/V3H 层（生产侧不允许，由 AST 测试强制）
from poker_decision_ai.cash_action_tree import build_action_tree  # noqa: E402
from poker_decision_ai.cash_range_model import build_range_model  # noqa: E402
from poker_decision_ai.cash_solver_interface import solver_status  # noqa: E402
from poker_decision_ai.cash_strategy_action_policy import (  # noqa: E402
    build_action_policy)
from poker_decision_ai.cash_strategy_action_ranking import rank_actions  # noqa: E402
from poker_decision_ai.cash_strategy_belief_conditioned_eval import (  # noqa: E402
    evaluate_belief_conditioned)
from poker_decision_ai.cash_strategy_dynamic_belief import (  # noqa: E402
    build_dynamic_belief)
from poker_decision_ai.cash_strategy_search_eval import evaluate_search  # noqa: E402
from poker_decision_ai.cash_strategy_search_tree import build_search_tree  # noqa: E402

NEW_MODULES = ("cash_strategy_decision_assembly",)

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
V3H_LAYERS = ("cash_strategy_action_policy", "cash_strategy_dynamic_belief",
              "cash_strategy_policy_evaluator", "cash_strategy_action_ranking",
              "cash_strategy_multistreet_eval", "cash_strategy_belief_eval",
              "cash_strategy_likelihood_model", "cash_strategy_belief_update",
              "cash_strategy_belief_conditioned_eval", "cash_strategy_search_tree",
              "cash_strategy_search_eval", "cash_strategy_decision_assembly")

POSITIONS_6 = ("SB", "BB", "UTG", "MP", "CO", "BTN")
POSITIONS_N = {2: ("SB", "BB"), 3: ("SB", "BB", "BTN"),
               4: ("SB", "BB", "MP", "BTN"), 5: ("SB", "BB", "MP", "CO", "BTN")}
BUDGET = {"max_street_depth": 2, "max_response_depth": 1,
          "max_candidate_actions": 3, "max_leaves": 60, "max_runout_children": 2}


def _sha(rel: str) -> str:
    p = ROOT / rel
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else "MISSING"


def cash_state(n_players: int = 3, street: str = "flop", board=None,
               history=None, **over) -> dict:
    pos = POSITIONS_N.get(n_players, POSITIONS_6)
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
        "synthetic_test_only": True, "players": players,
        "action_history": history or [],
    }
    st.update(over)
    return st


def ap(state):
    return build_action_policy(state, action_tree=build_action_tree(state))


def bl(state):
    return build_dynamic_belief(state, range_model=build_range_model(state))


def scorers(state, budget=None):
    """真实三层评分输出（用于端到端对账测试）。"""
    rk = rank_actions(state, action_policy=ap(state), belief=bl(state))
    bc = evaluate_belief_conditioned(state, action_policy=ap(state), ranking=rk)
    tr = build_search_tree(state, action_policy=ap(state), belief=bl(state),
                           budget=budget or BUDGET, max_root_candidates=3)
    se = evaluate_search(state, search_tree=tr)
    return rk, bc, se


def full(state, **over):
    rk, bc, se = scorers(state)
    return assemble_decision(state, ranking=rk, belief_conditioned=bc,
                             search_eval=se, solver=solver_status(state), **over)


# 合成评分源（用于精确构造分歧/覆盖率场景）
def syn_search(action, amt, cov=1.0):
    return {"present": True, "street": "flop",
            "ranked": [{"action": action, "amount_bb": amt, "search_score": 0.4,
                        "leaf_coverage": cov, "score_source": "MODEL_INTERNAL"}]}


def syn_rank(action, amt):
    return {"present": True, "street": "flop",
            "ranked": [{"action": action, "amount_bb": amt, "score": 0.2,
                        "score_source": "MODEL_INTERNAL"}]}


def syn_bc(action, amt):
    return {"present": True, "street": "flop",
            "ranked": [{"action": action, "amount_bb": amt, "score": 0.2,
                        "score_source": "MODEL_INTERNAL",
                        "conditioning": "prior_only"}]}


# ── 1. CASH-only fail closed ──────────────────────────────────────────
@pytest.mark.parametrize("mode", ["mtt", "pko", "ko", "mystery", "spin",
                                 "MTT", "PKO", "", "   ", None])
def test_non_cash_fail_closed(mode):
    st = cash_state(game_type=mode)
    r = assemble_decision(st, ranking=syn_rank("check", 0.0))
    assert r["present"] is False
    assert r["evidence_grade"] == "NONE"
    assert r["abstained"] is True
    assert r["preferred"] is None


@pytest.mark.parametrize("bad", [None, [], "cash", 42])
def test_non_dict_fail_closed(bad):
    assert assemble_decision(bad)["present"] is False


def test_no_scorer_supplied_fails_closed():
    r = assemble_decision(cash_state())
    assert r["present"] is False
    assert r["reason"] == "no_scorer_supplied"
    assert r["preferred"] is None


def test_scorer_without_present_flag_is_not_supplied():
    r = assemble_decision(cash_state(), ranking={"present": False})
    assert r["present"] is False


# ── 2. 真实三层端到端对账 ─────────────────────────────────────────────
def test_full_stack_assembles_agreed_candidate():
    st = cash_state()
    r = full(st)
    assert r["present"] is True
    assert r["scorers_supplied"] == {"search": True, "belief_conditioned": True,
                                    "ranking": True}
    assert r["abstained"] is False
    assert r["level"] == "MODEL_INTERNAL"
    assert r["preferred"] is not None
    assert r["preferred"]["selected_by_tier"] == "search"
    assert r["preferred"]["candidate_key"] == r["agreed_key"]


def test_all_three_tiers_produce_alignable_tops():
    r = full(cash_state())
    for tier in TIERS:
        top = r["scorer_tops"][tier]
        assert top is not None, tier
        assert top["key"] is not None, tier
        assert top["action"]


def test_preferred_carries_provenance_and_coverage():
    r = full(cash_state())
    p = r["preferred"]
    assert p["score_source"] == "MODEL_INTERNAL"
    assert p["tier_detail"] == "search_runout_pot_share_v1"
    assert isinstance(p["leaf_coverage"], (int, float))
    assert r["uncertainty"]["search_leaf_coverage"] is not None


# ── 3. 一致性采用 / 分歧弃权 ──────────────────────────────────────────
def test_all_agree_selects_highest_priority_tier():
    st = cash_state()
    r = assemble_decision(st, ranking=syn_rank("check", 0.0),
                          belief_conditioned=syn_bc("check", 0.0),
                          search_eval=syn_search("check", 0.0))
    assert r["abstained"] is False
    assert r["preferred"]["selected_by_tier"] == "search"
    assert r["disagreements"] == []


def test_disagreement_abstains_with_unknown_and_no_preferred():
    st = cash_state()
    r = assemble_decision(st, ranking=syn_rank("bet", 3.3),
                          search_eval=syn_search("check", 0.0))
    assert r["abstained"] is True
    assert r["level"] == "UNKNOWN"
    assert r["preferred"] is None
    assert r["disagreements"]
    assert "cross_scorer_disagreement" in r["unknown"]
    assert "no averaging" in r["disagreements"][0]["note"]


def test_disagreement_never_averages_or_picks_silently():
    st = cash_state()
    r = assemble_decision(st, ranking=syn_rank("bet", 3.3),
                          belief_conditioned=syn_bc("fold", None),
                          search_eval=syn_search("check", 0.0))
    assert r["preferred"] is None
    assert r["agreed_key"] is None or r["abstained"] is True
    assert len(r["disagreements"]) >= 1


def test_two_tier_agreement_among_three():
    """search 与 ranking 一致、belief 不同 → 仍弃权（不做多数表决）。"""
    st = cash_state()
    r = assemble_decision(st, ranking=syn_rank("check", 0.0),
                          belief_conditioned=syn_bc("bet", 3.3),
                          search_eval=syn_search("check", 0.0))
    assert r["abstained"] is True
    assert r["preferred"] is None


# ── 4. 金额容差 ───────────────────────────────────────────────────────
def test_amount_within_tolerance_still_agrees():
    st = cash_state()
    r = assemble_decision(st, ranking=syn_rank("bet", 3.3),
                          search_eval=syn_search("bet", 3.3 + 1e-9))
    assert r["abstained"] is False
    assert r["agreed_key"][0] == "bet"


def test_amount_beyond_tolerance_abstains():
    st = cash_state()
    r = assemble_decision(st, ranking=syn_rank("bet", 3.3),
                          search_eval=syn_search("bet", 5.0))
    assert r["abstained"] is True


def test_missing_amount_aligns_on_action_only():
    st = cash_state()
    r = assemble_decision(st, ranking=syn_rank("check", None),
                          search_eval=syn_search("check", None))
    assert r["abstained"] is False
    assert r["agreed_key"] == ["check", None]


# ── 5. 覆盖率闸门 ─────────────────────────────────────────────────────
def test_low_coverage_search_top_is_rejected():
    st = cash_state()
    r = assemble_decision(st, ranking=syn_rank("bet", 3.3),
                          search_eval=syn_search("check", 0.0, cov=0.5))
    assert r["rejections"]
    assert r["rejections"][0]["reason"] == "leaf_coverage_below_minimum"
    # search 被拒后仅剩 ranking → 用它，但必须记录 search 不可用
    assert r["preferred"]["action"] == "bet"
    assert r["preferred"]["selected_by_tier"] == "ranking"
    assert any("search" in u for u in r["unknown"])


def test_coverage_threshold_is_declared_and_configurable():
    assert ASSEMBLY_POLICY["min_leaf_coverage"] == 0.8
    st = cash_state()
    r = assemble_decision(st, ranking=syn_rank("bet", 3.3),
                          search_eval=syn_search("check", 0.0, cov=0.5),
                          policy={"min_leaf_coverage": 0.4})
    assert r["assembly_policy"]["min_leaf_coverage"] == 0.4
    assert r["rejections"] == []
    assert r["abstained"] is True   # 现在 search 可用 → 与 ranking 分歧 → 弃权


def test_scorer_without_numeric_score_is_rejected():
    st = cash_state()
    se = {"present": True, "street": "flop",
          "ranked": [{"action": "check", "amount_bb": 0.0,
                      "search_score": None, "leaf_coverage": 1.0}]}
    r = assemble_decision(st, search_eval=se)
    assert r["rejections"]
    assert r["rejections"][0]["reason"] == "search_score_unavailable"


# ── 6. 显式覆盖（默认关闭；无理由即忽略）──────────────────────────────
def test_override_default_off():
    assert ASSEMBLY_POLICY["override_default"] is False
    st = cash_state()
    r = assemble_decision(st, ranking=syn_rank("bet", 3.3),
                          search_eval=syn_search("check", 0.0))
    assert r["override_used"] is False
    assert r["abstained"] is True


def test_override_requires_justification():
    st = cash_state()
    r = assemble_decision(st, ranking=syn_rank("bet", 3.3),
                          search_eval=syn_search("check", 0.0),
                          allow_higher_tier_override=True,
                          override_justification="   ")
    assert r["override_used"] is False
    assert r["abstained"] is True


def test_override_with_justification_is_recorded():
    st = cash_state()
    r = assemble_decision(st, ranking=syn_rank("bet", 3.3),
                          search_eval=syn_search("check", 0.0),
                          allow_higher_tier_override=True,
                          override_justification="manual review by operator")
    assert r["override_used"] is True
    assert r["abstained"] is False
    assert r["override_justification"] == "manual review by operator"
    assert r["preferred"]["selected_by_tier"] == "search"


# ── 7. UNKNOWN / UNAVAILABLE 原样传播 ─────────────────────────────────
def test_missing_tiers_are_recorded_as_unknown():
    st = cash_state()
    r = assemble_decision(st, ranking=syn_rank("check", 0.0))
    assert "search_not_supplied" in r["unknown"]
    assert "belief_conditioned_not_supplied" in r["unknown"]
    # 只有一层可用 → 无分歧 → 采用它
    assert r["preferred"]["selected_by_tier"] == "ranking"


def test_tier_with_no_alignable_candidate_is_unknown():
    st = cash_state()
    r = assemble_decision(st, ranking={"present": True, "street": "flop",
                                       "ranked": [{"action": None, "amount_bb": None,
                                                   "score": 0.1}]})
    assert "ranking_candidate_key_unalignable" in r["unknown"]
    assert r["preferred"] is None
    assert r["level"] == "UNKNOWN"


def test_empty_ranked_yields_top_unavailable():
    st = cash_state()
    r = assemble_decision(st, ranking={"present": True, "street": "flop",
                                       "ranked": []})
    assert "ranking_top_candidate_unavailable" in r["unknown"]
    assert r["preferred"] is None


def test_unknown_policy_keys_ignored_and_recorded():
    st = cash_state()
    r = assemble_decision(st, ranking=syn_rank("check", 0.0),
                          policy={"bogus": 1})
    assert "unknown_policy_keys_ignored" in r["unknown"]
    assert "bogus" not in r["assembly_policy"]


def test_solver_without_input_is_unknown_but_blocked():
    st = cash_state()
    r = assemble_decision(st, ranking=syn_rank("check", 0.0))
    assert r["solver"]["solver_status"] == "UNKNOWN"
    assert r["solver"]["blocked"] is True
    assert r["solver"]["read_only_passthrough"] is True
    assert "solver_status_not_supplied" in r["unknown"]


# ── 8. 2–6 人 / 四街（真实层）─────────────────────────────────────────
@pytest.mark.parametrize("n", [2, 3, 4, 5, 6])
def test_player_counts_real_stack(n):
    r = full(cash_state(n_players=n))
    assert r["present"] is True
    assert r["level"] in ("MODEL_INTERNAL", "UNKNOWN")


@pytest.mark.parametrize("street,board,to_call", [
    ("preflop", [], 1.0), ("flop", ["Qs", "7d", "2c"], 0.0),
    ("turn", ["Qs", "7d", "2c", "Th"], 0.0)])
def test_streets_real_stack(street, board, to_call):
    st = cash_state(street=street, board=board, to_call=to_call)
    r = full(st)
    assert r["present"] is True
    assert r["street"] == street


def test_river_upstream_terminal_propagates_without_fabrication():
    """river 上上游全部终止 → 本层 fail closed（不伪造建议）。

    事实链（冻结 V3G-7）：river 为 terminal → V3H-4 action policy 无候选 →
    V3H-5/6/7 相应 fail closed → 本层无可用评分源。
    """
    st = cash_state(street="river", board=["Qs", "7d", "2c", "Th", "4s"],
                    to_call=4.0)
    rk, bc, se = scorers(st)
    assert rk.get("present") is False
    r = assemble_decision(st, ranking=rk, belief_conditioned=bc, search_eval=se)
    assert r["present"] is False
    assert r["reason"] == "no_scorer_supplied"
    assert r["preferred"] is None
    assert r["abstained"] is True


def test_partial_supply_on_river_still_reports_unknown_not_guess():
    """即使人为提供一层 river 评分，river 的 UNKNOWN 也必须如实反映。"""
    st = cash_state(street="river", board=["Qs", "7d", "2c", "Th", "4s"],
                    to_call=4.0)
    r = assemble_decision(st, ranking=syn_rank("call", 4.0))
    assert r["present"] is True
    assert "search_not_supplied" in r["unknown"]
    assert "belief_conditioned_not_supplied" in r["unknown"]
    assert r["preferred"]["selected_by_tier"] == "ranking"


# ── 9. Solver BLOCKED 不翻转 ──────────────────────────────────────────
def test_solver_remains_blocked():
    st = cash_state()
    before = solver_status(st).get("solver_status")
    r = full(st)
    after = solver_status(st).get("solver_status")
    assert before == after == "BLOCKED"
    assert r["solver"]["solver_status"] == "BLOCKED"
    assert r["solver"]["blocked"] is True


# ── 10. 不修改任何评分公式 / 无 GTO ───────────────────────────────────
def test_does_not_modify_scorer_formulas():
    assert CLAIM_BOUNDARY["modifies_any_scorer_formula"] is False
    src = (PKG / "cash_strategy_decision_assembly.py").read_text(encoding="utf-8")
    for banned in ("pot_share_arithmetic_v1 =", "search_runout_pot_share_v1 =",
                   "hero_equity *", "* (pot + 2"):
        assert banned not in src, banned


@pytest.mark.parametrize("flag", ["is_gto", "is_equilibrium", "is_solver_output",
                                  "is_profit_evidence", "computes_exploitability",
                                  "final_decision_made", "wired_to_production"])
def test_claim_boundary_flags_are_false(flag):
    assert CLAIM_BOUNDARY[flag] is False


def test_output_contains_no_positive_gto_flags():
    blob = json.dumps(full(cash_state()), default=str).lower()
    for bad in ('"is_gto": true', '"is_equilibrium": true',
                '"equilibrium": true', '"gto_exact": true'):
        assert bad not in blob


def test_no_final_decision_made():
    r = full(cash_state())
    assert r["final_decision_made"] is False
    assert "不是生产决策" in r["note"]


def test_no_frozen_multistreet_or_sample_runouts_reuse():
    tree_src = ast.parse((PKG / "cash_strategy_decision_assembly.py")
                         .read_text(encoding="utf-8"))
    for n in ast.walk(tree_src):
        if isinstance(n, ast.ImportFrom):
            assert "cash_multistreet" not in (n.module or "")
            assert not any(v in (n.module or "") for v in V3G_LAYERS)
        if isinstance(n, ast.Import):
            assert all("cash_multistreet" not in a.name for a in n.names)
        if isinstance(n, ast.Name):
            assert n.id != "sample_runouts"


# ── 11. 定输入确定性复现 ──────────────────────────────────────────────
def test_deterministic_full_stack():
    st = cash_state()
    a = json.dumps(full(st), sort_keys=True, default=str)
    b = json.dumps(full(st), sort_keys=True, default=str)
    assert a == b


def test_deterministic_synthetic_paths():
    st = cash_state()
    a = assemble_decision(st, ranking=syn_rank("bet", 3.3),
                          search_eval=syn_search("check", 0.0))
    b = assemble_decision(st, ranking=syn_rank("bet", 3.3),
                          search_eval=syn_search("check", 0.0))
    assert json.dumps(a, sort_keys=True, default=str) == \
        json.dumps(b, sort_keys=True, default=str)


def test_no_randomness_imported():
    tree_src = ast.parse((PKG / "cash_strategy_decision_assembly.py")
                         .read_text(encoding="utf-8"))
    for n in ast.walk(tree_src):
        if isinstance(n, ast.Import):
            assert all(a.name.split(".")[0] != "random" for a in n.names)
        if isinstance(n, ast.ImportFrom):
            assert (n.module or "").split(".")[0] != "random"


# ── 12. Judge / RealtimeRunner 逐位不变 + 冻结 SHA ────────────────────
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


# ── 13. 生产 import 图零新增 ──────────────────────────────────────────
def _production_importers() -> dict[str, set[str]]:
    found: dict[str, set[str]] = {}
    for p in PKG.rglob("*.py"):
        if p.name.startswith("cash_") or "__pycache__" in str(p):
            continue
        try:
            tree_src = ast.parse(p.read_text(encoding="utf-8-sig"))
        except SyntaxError:
            continue
        for n in ast.walk(tree_src):
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
    assert "cash_strategy_decision_assembly" not in _production_importers()


def test_new_module_has_no_toplevel_v3g_imports():
    tree_src = ast.parse((PKG / "cash_strategy_decision_assembly.py")
                         .read_text(encoding="utf-8"))
    for n in tree_src.body:
        if isinstance(n, ast.ImportFrom):
            assert not any(v in (n.module or "") for v in V3G_LAYERS), n.module


def test_new_module_declares_version_constant():
    tree_src = ast.parse((PKG / "cash_strategy_decision_assembly.py")
                         .read_text(encoding="utf-8"))
    consts = {n.targets[0].id for n in tree_src.body
              if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)}
    assert any(c.endswith("_VERSION") for c in consts)
