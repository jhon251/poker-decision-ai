# -*- coding: utf-8 -*-
"""V3H-6 测试 — CASH 策略智能 Layer 3（离线、零生产接线）。

覆盖授权清单：2–6 人 CASH / 四街 / prior-only / 显式 action likelihood /
多玩家独立 belief / street·action·size bucket 维度 / known-card removal /
hero-opponent 隔离 / 缺失 likelihood / 非法·负数·越界 likelihood / 冲突参数 /
UNKNOWN·EMPTY·缺 board·缺 hole cards·非法 seat / 定输入复现 / CASH 与 MTT·PKO 隔离 /
Solver BLOCKED 不翻转 / 无 GTO·equilibrium·真实 exploitability 声明 /
V3H-5 排序可读取 belief / Judge·RealtimeRunner 逐位不变 / 生产 import 图零新增 /
frozen·sealed SHA 零漂移。
"""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PKG = ROOT / "poker_decision_ai"

from poker_decision_ai.cash_strategy_belief_conditioned_eval import (  # noqa: E402
    CLAIM_BOUNDARY as BCE_BOUNDARY, STATES, UNAVAILABLE_METRICS,
    evaluate_belief_conditioned)
from poker_decision_ai.cash_strategy_belief_update import (  # noqa: E402
    CLAIM_BOUNDARY as BU_BOUNDARY, PATH_LIKELIHOOD_CONDITIONED,
    PATH_PRIOR_ONLY, PRIOR_ONLY_DEFINITION, update_belief_with_likelihood)
from poker_decision_ai.cash_strategy_likelihood_model import (  # noqa: E402
    CLAIM_BOUNDARY as LM_BOUNDARY, EVIDENCE_LEVELS, MAX_MULTIPLIER,
    REQUIRED_FIELDS, SCHEMA_FIELDS, action_likelihood_model)

# 测试侧允许 import V3G 层（生产侧不允许，由 AST 测试强制）
from poker_decision_ai.cash_action_tree import build_action_tree  # noqa: E402
from poker_decision_ai.cash_range_model import build_range_model  # noqa: E402
from poker_decision_ai.cash_solver_interface import solver_status  # noqa: E402
from poker_decision_ai.cash_strategy_action_policy import (  # noqa: E402
    build_action_policy)
from poker_decision_ai.cash_strategy_action_ranking import (  # noqa: E402
    rank_actions)
from poker_decision_ai.cash_strategy_dynamic_belief import (  # noqa: E402
    build_dynamic_belief)

NEW_MODULES = ("cash_strategy_likelihood_model", "cash_strategy_belief_update",
               "cash_strategy_belief_conditioned_eval")

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

POSITIONS_6 = ("SB", "BB", "UTG", "MP", "CO", "BTN")
POSITIONS_N = {2: ("SB", "BB"), 3: ("SB", "BB", "BTN"),
               4: ("SB", "BB", "MP", "BTN"), 5: ("SB", "BB", "MP", "CO", "BTN")}

STREETS = [("preflop", [], 1.0), ("flop", ["Qs", "7d", "2c"], 0.0),
           ("turn", ["Qs", "7d", "2c", "Th"], 0.0),
           ("river", ["Qs", "7d", "2c", "Th", "4s"], 4.0)]


def _sha(rel: str) -> str:
    p = ROOT / rel
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else "MISSING"


def cash_state(n_players: int = 6, street: str = "flop", board=None,
               history=None, **over) -> dict:
    """构造 n 座 CASH state：默认 hero 坐最后一座，其余座位为 v1..v(n-1)。

    这样"对手数 = n-1"与 id 顺序一致，便于逐玩家断言。
    """
    pos = POSITIONS_N.get(n_players, POSITIONS_6)
    hero_seat = n_players
    players = []
    vi = 1
    for seat in range(1, n_players + 1):
        if seat == hero_seat:
            players.append({
                "player_id": "hero", "seat": seat, "position": pos[seat - 1],
                "is_hero": True, "status": "active",
                "hole_cards": ["As", "Kd"], "stack": 100.0})
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


def _param(**over) -> dict:
    item = {"action": "bet", "street": "flop", "size_bucket": "small",
            "multiplier": 1.4, "source": "declared_demo",
            "evidence_level": "DECLARED_MODEL",
            "provenance": {"note": "unit test"}, "uncertainty": {"quantified": False}}
    item.update(over)
    return item


def pol(state, **over):
    return build_action_policy(state, action_tree=build_action_tree(state), **over)


def bel(state, **over):
    return build_dynamic_belief(state, range_model=build_range_model(state), **over)


def rank(state, **over):
    return rank_actions(state, action_policy=pol(state), belief=bel(state), **over)


def upd(state, params=None, **over):
    lm = action_likelihood_model(params)
    return update_belief_with_likelihood(
        state, belief=bel(state), likelihood_model=lm, pot_bb=10.0, **over)


def bce(state, params=None, **over):
    return evaluate_belief_conditioned(
        state, action_policy=pol(state), ranking=rank(state),
        belief_update=upd(state, params), solver=solver_status(state), **over)


HIST = [{"player_id": "v1", "action": "bet", "street": "flop", "amount_bb": 3.0},
        {"player_id": "hero", "action": "call", "street": "flop", "amount_bb": 3.0}]


# ── 1. CASH-only fail closed ──────────────────────────────────────────
@pytest.mark.parametrize("mode", ["mtt", "pko", "ko", "mystery", "spin",
                                  "MTT", "PKO", "", "   ", None])
def test_non_cash_fail_closed(mode):
    st = cash_state(game_type=mode)
    assert update_belief_with_likelihood(st, belief={"present": False})["present"] is False
    assert evaluate_belief_conditioned(st, ranking={"present": False})["present"] is False


@pytest.mark.parametrize("bad", [None, [], "cash", 42])
def test_non_dict_fail_closed(bad):
    assert update_belief_with_likelihood(bad)["present"] is False
    assert evaluate_belief_conditioned(bad)["present"] is False


def test_missing_inputs_fail_closed():
    st = cash_state()
    assert update_belief_with_likelihood(st)["reason"] == "belief_required"
    assert evaluate_belief_conditioned(st)["reason"] == "ranking_required"


# ── 2. likelihood 参数 schema ─────────────────────────────────────────
def test_schema_has_required_fields():
    for f in ("action", "street", "size_bucket", "multiplier", "source",
              "evidence_level", "provenance", "uncertainty", "prior_only", "seed"):
        assert f in SCHEMA_FIELDS, f


def test_required_fields_enforced():
    v = action_likelihood_model([{"action": "bet", "street": "flop",
                                  "multiplier": 1.0}])
    assert v["invalid"]
    assert v["invalid"][0]["reason"] == "missing_required_fields"
    assert set(v["invalid"][0]["missing"]) <= set(REQUIRED_FIELDS)


def test_valid_parameter_carries_full_provenance():
    r = action_likelihood_model([_param()])
    assert r["applied"] is True
    p = r["parameters"][0]
    for f in ("source", "evidence_level", "provenance", "uncertainty",
              "prior_only", "multiplier"):
        assert f in p, f
    assert p["evidence_level"] in EVIDENCE_LEVELS
    assert p["prior_only"] is False


def test_no_parameters_means_prior_only():
    r = action_likelihood_model(None)
    assert r["prior_only"] is True
    assert r["applied"] is False
    assert r["evidence_grade"] == "NONE"
    assert "no_parameters_supplied" in r["unknown"]


# ── 3. 非法 / 负数 / 越界 likelihood ──────────────────────────────────
@pytest.mark.parametrize("value,reason", [
    (-0.5, "multiplier_negative"),
    (-1.0, "multiplier_negative"),
    (MAX_MULTIPLIER + 1, "multiplier_out_of_range"),
    (999.0, "multiplier_out_of_range"),
    (0.0, "zero_multiplier_requires_explicit_flag"),
    ("1.5", "multiplier_not_numeric"),
    (True, "multiplier_not_numeric"),
])
def test_invalid_multipliers_rejected(value, reason):
    r = action_likelihood_model([_param(multiplier=value)])
    assert r["invalid"], (value, reason)
    assert r["invalid"][0]["reason"] == reason
    assert r["applied"] is False
    assert r["prior_only"] is True


def test_zero_multiplier_allowed_only_with_explicit_flag():
    r = action_likelihood_model([_param(multiplier=0.0,
                                        zero_means_impossible=True)])
    assert r["applied"] is True
    assert r["parameters"][0]["multiplier"] == 0.0


def test_unknown_action_or_street_rejected():
    assert action_likelihood_model([_param(action="teleport")])["invalid"]
    assert action_likelihood_model([_param(street="space")])["invalid"]
    assert action_likelihood_model([_param(size_bucket="gigantic")])["invalid"]


def test_bad_evidence_level_rejected():
    r = action_likelihood_model([_param(evidence_level="REAL_PLAYER_DATA")])
    assert r["invalid"]
    assert "unknown_evidence_level" in r["invalid"][0]["reason"]


# ── 4. 缺失不静默补全 / 冲突不静默覆盖 ────────────────────────────────
def test_missing_likelihood_is_not_silently_filled():
    """只有 flop bet 的参数；turn 的 bet 必须 unmatched，不得用默认值。"""
    st = cash_state(history=[
        {"player_id": "v1", "action": "bet", "street": "turn", "amount_bb": 3.0}])
    r = upd(st, [_param(street="flop")])
    e = r["belief"]["v1"]
    assert e["likelihood_applied"] == []
    assert e["likelihood_unmatched"]
    assert e["likelihood_unmatched"][0]["reason"] == "unmatched_no_default_applied"
    assert e["likelihood_total_multiplier"] == 1.0


def test_conflicting_parameters_are_rejected_not_overwritten():
    a = _param(multiplier=1.5)
    b = _param(multiplier=2.5)
    r = action_likelihood_model([a, b])
    assert r["conflicts"]
    assert r["conflicts"][0]["reason"] == "conflicting_multiplier_for_same_key"
    # 冲突键不得进入 parameters（既不覆盖也不择一）
    assert r["parameters"] == []
    assert r["applied"] is False


def test_duplicate_identical_entry_recorded_as_conflict():
    r = action_likelihood_model([_param(), _param()])
    assert r["conflicts"]
    assert r["conflicts"][0]["reason"] == "duplicate_identical_entry"


# ── 5. prior-only 与 likelihood_conditioned 两路 ──────────────────────
def test_prior_only_path_marked_and_definition_present():
    r = upd(cash_state(history=HIST))
    assert r["path"] == PATH_PRIOR_ONLY
    assert r["prior_only"] is True
    assert r["prior_only_definition"] == PRIOR_ONLY_DEFINITION
    assert "does NOT mean" in r["prior_only_definition"]


def test_likelihood_conditioned_path_marked():
    r = upd(cash_state(history=HIST), [_param()])
    assert r["path"] == PATH_LIKELIHOOD_CONDITIONED
    assert r["prior_only"] is False
    e = r["belief"]["v1"]
    assert e["likelihood_conditioned"] is True
    assert e["likelihood_applied"]
    assert e["likelihood_total_multiplier"] == 1.4


def test_paths_are_distinguishable_in_output():
    a = upd(cash_state(history=HIST))
    b = upd(cash_state(history=HIST), [_param()])
    assert a["path"] != b["path"]
    assert a["prior_only"] != b["prior_only"]


# ── 6. 多玩家独立 belief ──────────────────────────────────────────────
@pytest.mark.parametrize("n", [2, 3, 4, 5, 6])
def test_multi_player_independent_belief(n):
    hist = [{"player_id": "v1", "action": "bet", "street": "flop", "amount_bb": 3.0},
            {"player_id": "v2", "action": "fold", "street": "flop"}]
    st = cash_state(n_players=n, history=hist)
    r = upd(st, [_param(size_bucket="small"), _param(action="fold",
                                                     size_bucket="no_investment",
                                                     multiplier=0.6)])
    assert r["n_opponents"] == n - 1
    assert "hero" not in r["belief"]
    # v1 有动作 → 命中 bet 规则（1.4）
    assert r["belief"]["v1"]["likelihood_total_multiplier"] == 1.4
    if n >= 3:
        # v2 有动作 → 命中 fold 规则（0.6），且**独立于 v1**
        assert r["belief"]["v2"]["likelihood_total_multiplier"] == 0.6
    # 无动作的对手：乘数保持 1.0 且观测为空（未被他人动作污染）
    for pid, e in r["belief"].items():
        if pid == "v1" or (pid == "v2" and n >= 3):
            continue
        assert e["likelihood_total_multiplier"] == 1.0, pid
        assert e["observed_actions"] == [], pid
    # 每位对手只看到自己的动作（不串台）
    for pid, e in r["belief"].items():
        assert all(a["player_id"] == pid for a in e["observed_actions"]), pid


# ── 7. 四街 ───────────────────────────────────────────────────────────
@pytest.mark.parametrize("street,board,to_call", STREETS)
def test_streets_belief_update(street, board, to_call):
    st = cash_state(street=street, board=board, to_call=to_call,
                    history=[{"player_id": "v1", "action": "check",
                              "street": street}])
    r = upd(st)
    assert r["present"] is True
    assert r["street"] == street
    assert r["n_opponents"] >= 1


# ── 8. size bucket 维度 ───────────────────────────────────────────────
@pytest.mark.parametrize("amount,expected", [(1.0, "small"), (5.0, "medium"),
                                             (15.0, "large")])
def test_size_bucket_dimension(amount, expected):
    st = cash_state(history=[{"player_id": "v1", "action": "bet",
                              "street": "flop", "amount_bb": amount}])
    r = upd(st, [_param(size_bucket=expected)])
    e = r["belief"]["v1"]
    assert e["likelihood_applied"], (amount, expected)
    assert e["likelihood_applied"][0]["size_bucket"] == expected


def test_size_bucket_unknown_when_geometry_missing():
    st = cash_state(history=[{"player_id": "v1", "action": "bet",
                              "street": "flop"}])
    r = upd(st, [_param(size_bucket="medium")])
    e = r["belief"]["v1"]
    # 无 amount → bucket 未知 → 只能命中不含 bucket 的规则
    assert e["likelihood_applied"] == []
    assert e["likelihood_unmatched"][0]["size_bucket"] is None


# ── 9. known-card removal 与 posterior 不变量 ─────────────────────────
def test_known_card_removal_preserved():
    r = upd(cash_state(history=HIST), [_param()])
    assert set(r["known_cards_considered"]) >= {"Qs", "7d", "2c", "As", "Kd"}


def test_posterior_invariants_no_reentry_and_parseable():
    r = upd(cash_state(history=HIST), [_param()])
    for pid, e in r["belief"].items():
        inv = e["invariants"]
        assert inv["checked"] is True, pid
        assert inv["no_known_card_reentered"] is True, (pid, inv)
        assert inv["all_combos_parseable"] is True, (pid, inv)
        assert inv["impossible_combos_present"] == []


def test_posterior_mass_scales_with_likelihood():
    a = upd(cash_state(history=HIST))
    b = upd(cash_state(history=HIST), [_param(multiplier=2.0)])
    ea, eb = a["belief"]["v1"], b["belief"]["v1"]
    assert ea["mass_before_likelihood"] == eb["mass_before_likelihood"]
    assert eb["mass_after_likelihood"] == round(
        ea["mass_before_likelihood"] * 2.0, 6)


# ── 10. hero / opponent 隔离 ──────────────────────────────────────────
def test_hero_actions_not_in_opponent_likelihood():
    r = upd(cash_state(history=HIST), [_param()])
    assert r["hero_actions_in_opponent_likelihood"] is False
    assert [a["player_id"] for a in r["hero_actions_excluded"]] == ["hero"]
    # hero 的动作不得给 v1 增加乘数
    assert r["belief"]["v1"]["likelihood_total_multiplier"] == 1.4


def test_hero_flag_path_also_excluded():
    st = cash_state(history=[
        {"player_id": "x", "is_hero": True, "action": "bet", "street": "flop",
         "amount_bb": 3.0}])
    r = upd(st, [_param()])
    assert r["hero_actions_in_opponent_likelihood"] is False
    assert all(e["likelihood_total_multiplier"] == 1.0
               for e in r["belief"].values())


# ── 11. UNKNOWN / EMPTY / 缺失输入 ────────────────────────────────────
def test_missing_board_becomes_unknown():
    st = cash_state()
    st.pop("board")
    r = upd(st, [_param()])
    assert any("board" in u for u in r["unknown"])
    assert r["level"] == "UNKNOWN"


def test_missing_hole_cards_belief_still_reports_unknown():
    st = cash_state()
    st["players"][1]["hole_cards"] = []
    r = upd(st)
    assert r["present"] is True
    assert r["known_cards_considered"] or r["unknown"]


def test_invalid_seat_propagates():
    st = cash_state()
    st["players"][0]["seat"] = 99
    r = upd(st)
    assert any("seat" in u or "contiguous" in u for u in r["unknown"])


def test_unknown_action_in_history_is_recorded():
    st = cash_state(history=[{"player_id": "v1", "action": "teleport",
                              "street": "flop"}])
    r = upd(st, [_param()])
    assert r["belief"]["v1"]["likelihood_unmatched"]


# ── 12. 定输入确定性复现 ──────────────────────────────────────────────
@pytest.mark.parametrize("params", [None, "valid"])
def test_deterministic_reproducibility(params):
    p = [_param()] if params == "valid" else None
    st = cash_state(history=HIST)
    a = json.dumps(upd(st, p), sort_keys=True, default=str)
    b = json.dumps(upd(st, p), sort_keys=True, default=str)
    assert a == b


def test_belief_conditioned_eval_deterministic():
    st = cash_state(history=HIST)
    a = json.dumps(bce(st, [_param()]), sort_keys=True, default=str)
    b = json.dumps(bce(st, [_param()]), sort_keys=True, default=str)
    assert a == b


def test_no_randomness_imported_in_new_modules():
    for m in NEW_MODULES:
        tree = ast.parse((PKG / f"{m}.py").read_text(encoding="utf-8"))
        for n in ast.walk(tree):
            if isinstance(n, ast.Import):
                assert all(a.name.split(".")[0] != "random" for a in n.names), m
            if isinstance(n, ast.ImportFrom):
                assert (n.module or "").split(".")[0] != "random", m


# ── 13. V3H-5 排序可读取 belief ───────────────────────────────────────
def test_ranking_can_read_belief():
    r = rank(cash_state(history=HIST))
    assert r["present"] is True
    assert r["n_ranked"] > 0
    assert r["preferred_reason"]["belief_used"] is True


def test_belief_conditioned_eval_exposes_ranking_with_conditioning():
    r = bce(cash_state(history=HIST), [_param()])
    assert r["n_ranked"] > 0
    assert all(row["conditioning"] == "likelihood_conditioned" for row in r["ranked"])
    assert r["conditioning"] == "likelihood_conditioned"


# ── 14. 四态显式区分 ──────────────────────────────────────────────────
def test_states_enum_is_closed():
    assert set(STATES) == {"prior_only", "likelihood_conditioned",
                           "MODEL_INTERNAL", "UNAVAILABLE"}


def test_prior_only_state_reported():
    r = bce(cash_state(history=HIST))
    assert "prior_only" in r["states"]
    assert "likelihood_conditioned" not in r["states"]


def test_likelihood_conditioned_state_reported():
    r = bce(cash_state(history=HIST), [_param()])
    assert "likelihood_conditioned" in r["states"]
    assert "MODEL_INTERNAL" in r["states"]


def test_unavailable_metrics_are_declared_not_computed():
    r = bce(cash_state(history=HIST), [_param()])
    assert set(r["unavailable_metrics"]) == set(UNAVAILABLE_METRICS)
    blob = json.dumps(r, default=str)
    for m in UNAVAILABLE_METRICS:
        assert f'"{m}":' not in blob or m in r["unavailable_metrics"]


def test_state_definitions_are_explicit():
    r = bce(cash_state(history=HIST), [_param()])
    for s in STATES:
        assert s in r["state_definitions"]


# ── 15. Solver BLOCKED 不翻转 ─────────────────────────────────────────
def test_solver_remains_blocked():
    st = cash_state(history=HIST)
    before = solver_status(st).get("solver_status")
    upd(st, [_param()])
    r = bce(st, [_param()])
    after = solver_status(st).get("solver_status")
    assert before == after == "BLOCKED"
    assert r["solver"]["blocked"] is True
    assert r["solver"]["solver_status"] == "BLOCKED"
    assert r["solver"]["read_only_passthrough"] is False


def test_solver_without_input_is_unknown_but_blocked():
    r = evaluate_belief_conditioned(
        cash_state(history=HIST), ranking=rank(cash_state(history=HIST)),
        belief_update=upd(cash_state(history=HIST), [_param()]))
    assert r["solver"]["solver_status"] == "UNKNOWN"
    assert r["solver"]["blocked"] is True
    assert r["solver"]["read_only_passthrough"] is True


# ── 16. 无 GTO / equilibrium / 真实 exploitability 声明 ───────────────
@pytest.mark.parametrize("boundary", [LM_BOUNDARY, BU_BOUNDARY, BCE_BOUNDARY])
def test_claim_boundaries(boundary):
    assert boundary["is_gto"] is False
    assert boundary["is_equilibrium"] is False
    assert boundary["is_solver_output"] is False
    assert boundary["wired_to_production"] is False


def test_no_positive_gto_or_equilibrium_flags():
    blob = json.dumps([action_likelihood_model([_param()]),
                       upd(cash_state(history=HIST), [_param()]),
                       bce(cash_state(history=HIST), [_param()])],
                      default=str).lower()
    for bad in ('"is_gto": true', '"is_equilibrium": true',
                '"equilibrium": true', '"gto_exact": true'):
        assert bad not in blob


def test_no_real_opponent_range_claim():
    r = upd(cash_state(history=HIST), [_param()])
    assert r["claim_boundary"]["is_real_opponent_range"] is False
    assert "不得表述为真实对手范围" in r["note"]
    assert BU_BOUNDARY["is_real_player_frequency"] is False


def test_likelihood_model_denies_real_player_frequency():
    r = action_likelihood_model([_param()])
    assert r["claim_boundary"]["is_real_player_frequency"] is False


# ── 17. Judge / RealtimeRunner 逐位不变 + 冻结 SHA ────────────────────
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


# ── 18. 生产 import 图零新增 ──────────────────────────────────────────
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
                if m in V3G_LAYERS or m in NEW_MODULES:
                    found.setdefault(m, set()).add(p.name)
    return found


def test_no_new_production_imports():
    found = _production_importers()
    assert set(found) == {"cash_integrity_context", "cash_unknown_e2e"}, found


def test_new_modules_not_imported_in_production():
    found = _production_importers()
    for m in NEW_MODULES:
        assert m not in found


def test_new_modules_have_no_toplevel_v3g_imports():
    for m in NEW_MODULES:
        tree = ast.parse((PKG / f"{m}.py").read_text(encoding="utf-8"))
        for n in tree.body:
            if isinstance(n, ast.ImportFrom):
                assert not any(v in (n.module or "") for v in V3G_LAYERS), \
                    (m, n.module)


def test_new_modules_declare_version_constants():
    for m in NEW_MODULES:
        tree = ast.parse((PKG / f"{m}.py").read_text(encoding="utf-8"))
        consts = {n.targets[0].id for n in tree.body
                  if isinstance(n, ast.Assign)
                  and isinstance(n.targets[0], ast.Name)}
        assert any(c.endswith("_VERSION") for c in consts), m
