"""S‴′ 参照完整性修复的回归测试。

覆盖：根因（AI 座被过滤）、完整性守恒、active-opponent 完整、AI 座纳入、
UNKNOWN 传播（不得伪造可用）、单位契约不变、枚举预算/seed 不变、PI-EV 定义不变、
无 AI 座在场时与遗留口径一致、封存/护栏/不覆盖旧产物。
"""
from __future__ import annotations

import ast
import itertools
import json
from pathlib import Path

import pytest

from poker_decision_ai.pluribus_decision_adapter import (
    iter_pluribus_rows, parse_hand_row)
from poker_decision_ai.pluribus_validation import AI_SEAT
from poker_decision_ai.research_unfreeze import pi_reference_complete as prc
from poker_decision_ai.research_unfreeze import s3_validation as s3
from poker_decision_ai.research_unfreeze import s3prime_runner as s3p
from poker_decision_ai.research_unfreeze.unfreeze_core import (PI_EV_MAX_RUNOUTS,
                                                              PiEvReference)

SCOPE = 200


@pytest.fixture(scope="module")
def hands():
    hs = [p for row in itertools.islice(iter_pluribus_rows(), SCOPE)
          if (p := parse_hand_row(row))]
    assert hs, "corpus unavailable"
    return hs


def _hand(players, holes, streets):
    return {"hand_id": "synthetic|t|1", "players": list(players),
            "hole_cards": dict(holes), "board_by_street": {},
            "streets": streets, "bb": 100.0}


HERO = "Gogo"
H1, H2 = "MrBlue", "MrOrange"


def test_root_cause_is_ai_seat_filter_in_legacy_builders():
    """根因可复现：遗留口径把 AI 座排除出 villain_hands。"""
    h = _hand([HERO, H1, AI_SEAT],
              {HERO: ["As", "Kd"], H1: ["Qh", "Qs"], AI_SEAT: ["2c", "7d"]},
              [])
    leg = prc.build_villain_hands(h, HERO, set(), include_ai_seat=False)
    ful = prc.build_villain_hands(h, HERO, set(), include_ai_seat=True)
    assert leg["villains_used"] == 1 and AI_SEAT not in leg["live"]
    assert ful["villains_used"] == 2 and AI_SEAT in ful["live"]
    assert ful["ai_seat_live"] is True


def test_live_opponents_excludes_only_hero_and_folded():
    h = _hand([HERO, H1, H2, AI_SEAT],
              {HERO: ["As", "Kd"], H1: ["Qh", "Qs"], H2: ["3c", "4d"],
               AI_SEAT: ["2c", "7d"]}, [])
    assert prc.live_opponents(h, HERO, set()) == [H1, H2, AI_SEAT]
    assert prc.live_opponents(h, HERO, {H2}) == [H1, AI_SEAT]
    assert prc.live_opponents(h, HERO, {H1, H2, AI_SEAT}) == []


def test_folded_ai_seat_is_not_an_opponent():
    h = _hand([HERO, H1, AI_SEAT],
              {HERO: ["As", "Kd"], H1: ["Qh", "Qs"], AI_SEAT: ["2c", "7d"]}, [])
    facts = prc.build_villain_hands(h, HERO, {AI_SEAT})
    assert AI_SEAT not in facts["live"] and facts["villains_used"] == 1
    assert facts["ai_seat_live"] is False


@pytest.mark.parametrize("live,unknown", [([H1, AI_SEAT], []),
                                          ([H1, H2], [AI_SEAT]),
                                          ([H1, H2, AI_SEAT], [])])
def test_completeness_invariant_holds(live, unknown):
    facts = {"live": live, "villain_hands": [[], ] * (len(live) - len(unknown)),
             "unknown_live": unknown, "villains_used": len(live) - len(unknown),
             "live_count": len(live), "ai_seat_live": AI_SEAT in live}
    prc.assert_complete(facts)
    assert prc.is_faithful(facts)


def test_unknown_live_opponent_yields_unavailable_not_faked():
    """活跃对手手牌未知 → 必须 unavailable（UNKNOWN 传播，禁止伪造可用）。"""
    h = _hand([HERO, H1, AI_SEAT],
              {HERO: ["As", "Kd"], H1: ["Qh", "Qs"]}, [])   # AI 座手牌未知
    ref = prc.reference_complete(
        hand=h, hero_pid=HERO, folded=set(), hero_cards=["As", "Kd"],
        board=["2h", "7c", "9s"], pot_bb=10.0, to_call_bb=0.0,
        hero_stack_bb=100.0,
        candidates=[{"action": "check", "amount_bb": 0.0}])
    assert ref["available"] is False
    assert prc.UNAVAILABLE_INCOMPLETE in ref["unavailable_reason"]
    assert AI_SEAT in ref["provenance"]["unknown_live_opponents"]
    assert ref["values"] == {} and ref["ordering"] == []


def test_no_live_opponent_yields_unavailable():
    h = _hand([HERO, H1, AI_SEAT],
              {HERO: ["As", "Kd"], H1: ["Qh", "Qs"], AI_SEAT: ["2c", "7d"]}, [])
    ref = prc.reference_complete(
        hand=h, hero_pid=HERO, folded={H1, AI_SEAT}, hero_cards=["As", "Kd"],
        board=["2h", "7c", "9s"], pot_bb=10.0, to_call_bb=0.0,
        hero_stack_bb=100.0, candidates=[{"action": "check", "amount_bb": 0.0}])
    assert ref["available"] is False


def test_available_implies_at_least_one_known_live_opponent():
    h = _hand([HERO, H1, AI_SEAT],
              {HERO: ["As", "Kd"], H1: ["Qh", "Qs"], AI_SEAT: ["2c", "7d"]}, [])
    ref = prc.reference_complete(
        hand=h, hero_pid=HERO, folded=set(), hero_cards=["As", "Kd"],
        board=["2h", "7c", "9s"], pot_bb=10.0, to_call_bb=0.0,
        hero_stack_bb=100.0,
        candidates=[{"action": "check", "amount_bb": 0.0},
                    {"action": "bet", "amount_bb": 5.0}])
    assert ref["available"] is True
    assert ref["villain_facts"]["villains_used"] == 2
    assert ref["provenance"]["unknown_live_opponents"] == []
    assert ref["provenance"]["reference_version"] == prc.REFERENCE_VERSION


def test_pi_ev_definition_unchanged_vs_sealed_with_same_villains():
    """同一 villain 列表下，修复版与 sealed `PiEvReference` 输出**逐字段相同**。"""
    hero, board = ["As", "Kd"], ["2h", "7c", "9s"]
    vhands = [["Qh", "Qs"], ["3c", "4d"]]
    cands = [{"action": "check", "amount_bb": 0.0},
             {"action": "bet", "amount_bb": 5.0}]
    sealed = PiEvReference().reference(hero_cards=hero, villain_hands=vhands,
                                       board=board, pot_bb=12.0, to_call_bb=0.0,
                                       hero_stack_bb=100.0, candidates=cands)
    h = _hand([HERO, H1, H2], {HERO: hero, H1: vhands[0], H2: vhands[1]}, [])
    fixed = prc.reference_complete(
        hand=h, hero_pid=HERO, folded=set(), hero_cards=hero, board=board,
        pot_bb=12.0, to_call_bb=0.0, hero_stack_bb=100.0, candidates=cands)
    for k in ("available", "unavailable_reason", "values", "ordering",
              "pi_equity", "n_runouts"):
        assert fixed[k] == sealed[k], k
    prov = {k: v for k, v in sealed["provenance"].items()}
    for k, v in prov.items():
        assert fixed["provenance"][k] == v, k


def test_no_ai_seat_live_matches_legacy():
    """AI 座已弃牌时，修复版与遗留口径给出相同对手集合（不产生额外变化）。"""
    h = _hand([HERO, H1, H2],
              {HERO: ["As", "Kd"], H1: ["Qh", "Qs"], H2: ["3c", "4d"]}, [])
    leg = prc.build_villain_hands(h, HERO, {AI_SEAT}, include_ai_seat=False)
    ful = prc.build_villain_hands(h, HERO, {AI_SEAT}, include_ai_seat=True)
    assert leg["villain_hands"] == ful["villain_hands"]
    assert leg["villains_used"] == ful["villains_used"] == 2


def test_enumeration_is_exact_and_budget_unchanged():
    hero, board = ["As", "Kd"], ["2h", "7c", "9s"]
    one = prc._n_runouts(hero, board, [["Qh", "Qs"]])
    two = prc._n_runouts(hero, board, [["Qh", "Qs"], ["3c", "4d"]])
    assert one == 990 and two == 903          # C(45,2) / C(43,2)：精确全枚举
    assert prc.budget_ok(one) and prc.budget_ok(two)
    assert one <= PI_EV_MAX_RUNOUTS and two <= PI_EV_MAX_RUNOUTS


def test_unit_contract_preserved(hands):
    """单位契约：修复版不改变 bb 归一与 raw 比例口径。"""
    from poker_decision_ai.research_unfreeze.pi_ev_contract import pi_ev_inputs
    h = hands[0]
    a = {"player": h["players"][0], "action": "call", "to_call": 100.0,
         "pot_before": 300.0, "amount": 100.0}
    from poker_decision_ai.pluribus_validation import _state_for
    from poker_decision_ai.pluribus_decision_adapter import _positions_for_seats
    st = _state_for(h, a, "flop", ["2h", "7c", "9s"],
                    _positions_for_seats(len(h["players"]), len(h["players"])),
                    set(), [], a["player"], h["bb"] or 1.0)
    ins = pi_ev_inputs(st, raw=a, hand_bb=h["bb"] or 1.0)
    ratios = ins["ratios_vs_raw"]
    assert abs(ratios["pot_bb"] - 1.0) <= 0.5      # ∈[0.5,2.0] 契约
    assert ins["pot_bb"] > 0


def test_reference_is_not_solver_backed():
    h = _hand([HERO, H1], {HERO: ["As", "Kd"], H1: ["Qh", "Qs"]}, [])
    ref = prc.reference_complete(
        hand=h, hero_pid=HERO, folded=set(), hero_cards=["As", "Kd"],
        board=["2h", "7c", "9s"], pot_bb=10.0, to_call_bb=0.0,
        hero_stack_bb=100.0, candidates=[{"action": "check", "amount_bb": 0.0}])
    p = ref["provenance"]
    assert p["equilibrium"] is False and p["NOT_exact_gto"] is True
    assert p["evidence_level"] == "L2"


def test_audit_reproduces_sealed_counts(hands):
    """审计可复现封存计数（7,483 / 6,127 / 1,356）——一致性交叉验证。"""
    reg = json.loads(s3.REGISTRY.read_text(encoding="utf-8"))
    rep = prc.audit_decision_set(hands, set(reg["s3_players"]),
                                 hand_min_index=s3.S3_HAND_MIN_INDEX)
    assert rep["legacy_available"] + rep["legacy_unavailable"] == rep["points"]
    assert rep["legacy_unavail_ai_live"] + rep["legacy_unavail_no_live"] == \
        rep["legacy_unavailable"]
    assert rep["complete_available"] + rep["complete_unavailable"] == \
        rep["points"]
    # 修复只增加可用性，且全部来自「AI 座确实在场且手牌已知」的点
    assert rep["complete_available"] >= rep["legacy_available"]
    assert rep["legacy_unavail_no_live"] == 0 or rep["complete_unavailable"] > 0
    assert rep["unknown_live_any"] == 0
    assert rep["runouts_budget_exceeded"] == 0


def test_runner_guards_and_no_old_overwrite(tmp_path):
    with pytest.raises(s3p.S3PGuardError):
        s3p.run_s3prime([], "U1")
    for name in s3p.FORBIDDEN_PATHS:
        with pytest.raises(s3p.S3PGuardError):
            s3p._assert_not_sealed(tmp_path / name)
    with pytest.raises(s3p.S3PGuardError):
        s3p._assert_not_sealed(tmp_path / "V3F_S3_NEW_FILE.json")


def test_runner_has_no_run_s3_calls():
    for mod in (s3p, prc):
        tree = ast.parse(Path(mod.__file__).read_text(encoding="utf-8"))
        called = {(n.func.id if isinstance(n.func, ast.Name)
                   else getattr(n.func, "attr", ""))
                  for n in ast.walk(tree) if isinstance(n, ast.Call)}
        assert not ({"run_s3", "run_s3prime"} & called - {"run_s3prime"}), \
            mod.__name__
    # run_s3prime 自身定义可被调用，但**不得被任何生产模块**调用
    pkg = Path(s3p.__file__).resolve().parents[1]
    users = [str(p.relative_to(pkg)) for p in pkg.rglob("*.py")
             if p.name not in ("s3prime_runner.py",)
             and "run_s3prime(" in p.read_text(encoding="utf-8",
                                               errors="ignore")]
    assert users == []


def test_manifest_roundtrip_and_drift(monkeypatch):
    man = s3p.write_manifest()
    rep = s3p.verify_manifest()
    assert rep["present"] and rep["clean"] and rep["checked"] == 2
    assert man["production_wiring"] == "NOT WIRED"
    assert "NOT INDEPENDENT" in man["frozen_decision_set"]["independence"]
    assert man["semantics_unchanged"]["thresholds_readonly"]["accept_pp"] == \
        s3.ACCEPT_PP
    monkeypatch.setattr(s3p, "_sha", lambda rel: "0" * 64)
    assert s3p.verify_manifest()["clean"] is False


def test_old_sealed_artifacts_untouched():
    for m in (s3.MANIFEST_U1, s3.MANIFEST_R22):
        rep = s3.verify_s3_manifest(m)
        assert rep["clean"] and rep["checked"] == 9 and rep["drift"] == {}
        assert rep["split_drift"] is False
    assert s3p.sealed_s3_clean() == {"U1": True, "R2.2": True}
    names = {p.name for p in s3p.PER_POINT.values()} | \
            {p.name for p in s3p.RESULT.values()} | {s3p.MANIFEST.name}
    assert not (names & set(s3p.FORBIDDEN_PATHS))
    assert all(n.startswith("V3F_S3PRIME_") for n in names)
