"""第一次解冻实验测试（研究分支；不接触生产模块）。"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from poker_decision_ai.domain.poker.strategy_core import compute_ev  # noqa: E402
from poker_decision_ai.research_unfreeze import (  # noqa: E402
    PiEvReference,
    SolverReference,
    compute_ev_variant,
)
from poker_decision_ai.research_unfreeze import unfreeze_experiment as ux  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
TIGHT = {"AA": 0.2, "KK": 0.15, "QQ": 0.2, "AKs": 0.2, "AKo": 0.25}


def _state(*, to_call=100.0, pot=200.0, stack=4000.0, hole=("Ah", "Ad"),
           board=("Ks", "9c", "4d"), street="flop"):
    return {"game_type": "cash", "street": street, "pot": pot,
            "to_call": to_call, "bb": 2.0, "board": list(board),
            "players": [
                {"player_id": "Hero", "is_hero": True, "stack": stack,
                 "hole_cards": list(hole), "status": "active",
                 "position": "BTN"},
                {"player_id": "V1", "is_hero": False, "stack": stack,
                 "status": "active", "position": "BB"}]}


def _eq(v=0.694):
    return {"hero_equity": v,
            "outcome": {"win": v, "tie": 0.0, "lose": round(1 - v, 6)}}


# ── IMR：Solver BLOCKED ────────────────────────────────────────────────
def test_solver_unavailable_is_blocked_not_faked():
    s = SolverReference()
    assert s.is_configured() is False
    assert "BLOCKED" in s.unavailable_reason()
    out = s.reference()
    assert out["available"] is False
    assert out["provenance"]["blocked"] is True
    assert out["provenance"]["equilibrium"] is False


# ── IMR：PI-EV 契约 ───────────────────────────────────────────────────
def test_pi_ev_provenance_and_labeling():
    pi = PiEvReference()
    assert pi.is_configured() is True
    ref = pi.reference(hero_cards=["Ah", "Ad"], villain_hands=[["Kc", "Kd"]],
                       board=["Ks", "9c", "4d"], pot_bb=100.0, to_call_bb=50.0,
                       hero_stack_bb=2000.0,
                       candidates=[{"action": "call", "amount_bb": 50.0},
                                   {"action": "fold", "amount_bb": None}])
    assert ref["available"] is True
    prov = ref["provenance"]
    assert prov["reference"] == "perfect-information"
    assert prov["equilibrium"] is False
    assert prov["NOT_exact_gto"] is True
    assert ref["n_runouts"] == 990        # flop → C(45,2)
    assert prov["runouts_enumerated"] == 990


def test_pi_ev_exact_on_river_and_budget_guard(monkeypatch):
    pi = PiEvReference()
    # river：hero 已知获胜（AAA vs KKK）→ PI 权益 = 1.0，runout 数 = 1
    ref = pi.reference(hero_cards=["Ah", "Ad"], villain_hands=[["Kc", "Kd"]],
                       board=["Ks", "9c", "4d", "7s", "Ac"], pot_bb=100.0,
                       to_call_bb=0.0, hero_stack_bb=2000.0,
                       candidates=[{"action": "check", "amount_bb": 0.0}])
    assert ref["available"] and ref["pi_equity"] == 1.0
    assert ref["n_runouts"] == 1
    # 反向对照：villain 获胜的牌面 → 0.0（参照必须真实反映明牌结果）
    ref0 = pi.reference(hero_cards=["Ah", "Ad"], villain_hands=[["Kc", "Kd"]],
                        board=["Ks", "9c", "4d", "7s", "2h"], pot_bb=100.0,
                        to_call_bb=0.0, hero_stack_bb=2000.0,
                        candidates=[{"action": "check", "amount_bb": 0.0}])
    assert ref0["pi_equity"] == 0.0
    # 预算守卫：人为压低预算 → 必须 unavailable（不得近似冒充精确）
    monkeypatch.setattr(ux, "PI_EV_MAX_RUNOUTS", 10, raising=False)
    import poker_decision_ai.research_unfreeze.unfreeze_core as core
    monkeypatch.setattr(core, "PI_EV_MAX_RUNOUTS", 10)
    ref2 = core.PiEvReference().reference(
        hero_cards=["Ah", "Ad"], villain_hands=[["Kc", "Kd"]],
        board=["Ks", "9c", "4d"], pot_bb=100.0, to_call_bb=0.0,
        hero_stack_bb=2000.0, candidates=[{"action": "check",
                                           "amount_bb": 0.0}])
    assert ref2["available"] is False
    assert "budget exceeded" in ref2["unavailable_reason"]


# ── 变体：零变体等价 / U1 / U2 ────────────────────────────────────────
def test_zero_variant_bit_exact_equivalence():
    st = _state()
    rs = {"villains": {"V1": {"distribution": TIGHT}}}
    frozen = compute_ev(st, _eq(), range_src=rs, seed=42)
    var, meta = compute_ev_variant(st, _eq(), rs, None, variant="frozen")
    assert meta["u1_applied"] is False and meta["u2_applied"] is False
    assert var.best_action == frozen.best_action
    for a, b in zip(frozen.candidates, var.candidates):
        assert a.action == b.action
        assert float(a.ev_bb) == float(b.ev_bb)
        assert float(a.utility_bb) == float(b.utility_bb)


def test_u1_lowers_raise_value_and_u3_equals_u1():
    st = _state()
    rs = {"villains": {"V1": {"distribution": TIGHT}}}
    frozen = compute_ev(st, _eq(), range_src=rs, seed=42)
    u1, meta1 = compute_ev_variant(st, _eq(), rs, None, variant="U1")
    u13, _ = compute_ev_variant(st, _eq(), rs, None, variant="U1+U3")
    assert meta1["u1_applied"] is True
    fz = {c.action: float(c.ev_bb) for c in frozen.candidates}
    r1 = {c.action: float(c.ev_bb) for c in u1.candidates}
    r13 = {c.action: float(c.ev_bb) for c in u13.candidates}
    assert any(fz[a] != r1[a] for a in fz if a == "raise"), \
        "U1 必须改变 raise 候选价值"
    for a in fz:
        assert r1[a] == r13[a]            # U3 审核结论：与 U1 同一修正
        if a != "raise":
            assert r1[a] == fz[a]         # 其他动作不受 U1 影响
    for a in fz:
        if a == "raise":
            assert r1[a] <= fz[a] + 1e-12  # 损失增加 → 价值下降


def test_u2_cv_scaling_direction():
    st = _state()
    rs = {"villains": {"V1": {"distribution": TIGHT}}}
    base = compute_ev(st, _eq(), range_src=rs, seed=42)
    has_cv = [c for c in base.candidates if c.cv_bb]
    if not has_cv:
        pytest.skip("no CV candidates in this state")
    lo, _ = compute_ev_variant(st, _eq(), rs, None, variant="U1+U2+U3",
                               lambda_scale=0.5)
    hi, _ = compute_ev_variant(st, _eq(), rs, None, variant="U1+U2+U3",
                               lambda_scale=1.5)
    cv_actions = {c.action for c in has_cv}
    for c_lo in lo.candidates:
        if c_lo.action not in cv_actions:
            continue
        c_hi = next(c for c in hi.candidates if c.action == c_lo.action)
        if c_lo.cv_bb is None:
            continue
        assert float(c_hi.ev_bb) >= float(c_lo.ev_bb)   # λ↑ → CV 影响↑


# ── split / manifest / 指标工具 ────────────────────────────────────────
def test_split_assignment_matches_frozen_registry():
    from poker_decision_ai.pluribus_decision_adapter import (
        iter_pluribus_rows, parse_hand_row)
    reg = ux.load_registry()
    hands = [p for row in iter_pluribus_rows(limit=2000)
             if (p := parse_hand_row(row))]
    hand_split, s_players = ux.assign_split(hands, reg)
    assert isinstance(hand_split, dict) and hand_split
    assert set(s_players) == set(reg["assignment"]["s_players"])
    # 同一规则在全体语料上必须复现 registry 的 assignment 哈希
    all_hands = [p for row in iter_pluribus_rows()
                 if (p := parse_hand_row(row))]
    full_split, _ = ux.assign_split(all_hands, reg)
    import hashlib
    h = hashlib.sha256(json.dumps(sorted(full_split.items()),
                                  ensure_ascii=False).encode("utf-8")
                       ).hexdigest()
    assert h == reg["assignment"]["hand_assignment_sha256"]


def test_metrics_and_bootstrap_sanity():
    points = [
        {"variants": {"frozen": {"best": "call"}, "U1": {"best": "raise"}},
         "pi": {"best": "raise"}, "observed": "call", "legal": True},
        {"variants": {"frozen": {"best": "raise"}, "U1": {"best": "raise"}},
         "pi": {"best": "raise"}, "observed": "raise", "legal": False},
    ]
    assert ux.pi_agreement(points, "frozen")["pi_agreement"] == 0.5
    assert ux.pi_agreement(points, "U1")["pi_agreement"] == 1.0
    b = ux.paired_bootstrap(points, "U1", "frozen", n=200)
    assert b["n"] == 2 and b["delta_pp"] == 50.0
    assert b["ci95_low_pp"] <= b["delta_pp"] <= b["ci95_high_pp"]
    lg = ux.legality(points)
    assert lg == {"legal": 1, "illegal": 1, "legal_share": 0.5}
    assert ux.observed_metrics(points, "frozen")["bet_bias_share"] == 0.5


def test_manifest_sealed_before_s_and_verifiable():
    man = ux.write_manifest(ux.MANIFEST_U, experiment_id="umatrix",
                            config={"variants": list(ux.VARIANTS),
                                    "lambda_scales": list(ux.LAMBDA_SCALES)})
    assert "任何 S 集计算之前" in man["s_peek_declaration"]
    assert man["production_wiring"].startswith("NOT WIRED")
    v = ux.verify_manifest(ux.MANIFEST_U)
    assert v["present"] and v["clean"] is True and v["split_drift"] is False
    assert len(man["rejection_lines"]) >= 7


def test_prior_phases_still_readonly():
    from poker_decision_ai.quality_benchmark import (frozen_hashes,
                                                     verify_frozen)
    r1 = json.loads((ROOT / "docs" / "V3F_R1_QUALITY_BASELINE.json")
                    .read_text(encoding="utf-8"))
    assert verify_frozen(r1["manifest"]["frozen_files"])["match"] is True
    assert frozen_hashes() == r1["manifest"]["frozen_files"]
    for name, ver in (("multistreet_v2_benchmark", None),
                      ("multistreet_v3_benchmark", None)):
        mod = __import__(f"poker_decision_ai.{name}", fromlist=["x"])
        assert mod.verify_manifest()["clean"] is True
