"""S‴ 验证器测试（schema / 分层完整率 / 自洽重算 / 封存 / 阈值冻结 / 运行护栏）。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from poker_decision_ai.research_unfreeze import s3_validation as s3  # noqa: E402


def _row(**over):
    base = {
        "arm": "U1", "hand_id": "h1", "street": "flop", "observed": "call",
        "legal": True, "hero_ip": True, "spr": 5.0, "pot_bb": 8.0,
        "hero_stack_bb": 40.0, "to_call_bb": 2.0,
        "raw_pot_before_chips": 800.0, "raw_to_call_chips": 200.0,
        "hand_bb_chips": 100.0, "state_pot_bb": 8.0, "state_to_call_bb": 2.0,
        "ratio_pot_input_over_raw": 1.0, "ratio_stack_input_over_state": 1.0,
        "frozen_best": "call", "cand_best": "raise", "cand_applicable": True,
        "cand_declined": False, "pi_best": "raise", "pi_equity": 0.55,
        "n_runouts": 990,
    }
    base.update(over)
    return base


def test_thresholds_frozen():
    """验收阈值必须与历史封存值一致（防误改）。"""
    assert (s3.ACCEPT_PP, s3.UNCERTAIN_PP) == (4.0, 2.0)
    assert s3.COVERAGE_MIN == 0.70
    assert s3.VALID_POINTS_MIN == 1200
    assert s3.DECLINE_MAX == 0.40


def test_per_point_schema_validation():
    assert s3.validate_point(_row()) == []
    assert any("missing" in v for v in s3.validate_point(
        {k: v for k, v in _row().items() if k != "spr"}))
    assert any("type" in v for v in s3.validate_point(_row(pi_equity="high")))
    assert any("domain" in v for v in s3.validate_point(
        _row(street="river")))
    assert any("domain" in v for v in s3.validate_point(
        _row(frozen_best="allin")))
    assert any("domain" in v for v in s3.validate_point(_row(pi_equity=1.5)))
    assert any("domain" in v for v in s3.validate_point(_row(n_runouts=0)))


def test_strata_completeness_gate():
    good = [_row(), _row(street="turn", hero_ip=False)]
    assert s3.strata_complete(good)["complete"] is True
    bad = [_row(), _row(hero_ip=None)]
    rep = s3.strata_complete(bad)
    assert rep["complete"] is False and rep["missing"]["hero_ip"] == 1
    bad2 = [_row(spr=None)]
    assert s3.strata_complete(bad2)["complete"] is False
    assert s3.strata_complete([])["complete"] is False


def test_recompute_summary_strata_and_verdicts():
    reg = {"s3_players": ["A", "B"], "scope_rule": "test-coupling"}
    # ACCEPT 级：候选在 60% 点上优于 frozen
    rows = []
    for i in range(1500):
        hit_c = i < 900          # cand 命中 60%
        hit_f = i < 600          # frozen 命中 40%
        rows.append(_row(
            frozen_best="raise" if hit_f else "call",
            cand_best="raise" if hit_c else "call",
            pi_best="raise",
            street="flop" if i % 2 == 0 else "turn",
            hero_ip=(i % 3 == 0), spr=1.0 + (i % 7), pot_bb=3.0 + (i % 25)))
    out = s3.recompute_summary(rows, arm="U1", reg=reg, attempted=1600,
                               unavailable=100)
    assert out["valid_points"] == 1500
    assert out["paired_delta_pp"] == pytest.approx(20.0, abs=0.01)
    assert out["ci95_pp"][0] > 0
    assert out["verdict"] == "ACCEPT"
    assert set(out["strata"]) == {"street", "position", "spr", "pot"}
    assert out["strata"]["position"]["IP"]["n"] > 0
    assert out["strata_completeness"]["complete"] is True
    assert out["schema_violations"] == 0
    # REJECT 级：候选与 frozen 完全相同（Δ = 0）；pot/spr 需保持非退化（IG3）
    rows2 = [_row(frozen_best="call", cand_best="call", pi_best="raise",
                  street="flop" if i % 2 == 0 else "turn",
                  hero_ip=(i % 3 == 0), spr=1.0 + (i % 7),
                  pot_bb=3.0 + (i % 25)) for i in range(1500)]
    out2 = s3.recompute_summary(rows2, arm="U1", reg=reg, attempted=1500,
                                unavailable=0)
    assert out2["paired_delta_pp"] == 0.0
    assert out2["ci95_pp"][0] <= 0 <= out2["ci95_pp"][1]
    assert out2["verdict"] == "REJECT"
    # PARTIAL：valid < 1200
    out3 = s3.recompute_summary(rows[:1000], arm="U1", reg=reg,
                                attempted=1000, unavailable=0)
    assert out3["verdict"] == "PARTIAL"
    assert out3["valid_rule_satisfied"] is False
    # BLOCKED：illegal > 0
    bad = [_row(legal=False) for _ in range(1500)]
    out4 = s3.recompute_summary(bad, arm="U1", reg=reg, attempted=1500,
                                unavailable=0)
    assert out4["verdict"] == "BLOCKED"


def test_registry_players_disjoint_and_run_guard():
    reg = json.loads(s3.REGISTRY.read_text(encoding="utf-8"))
    used = set(s3.USED_PLAYERS)
    assert not (set(reg["s3_players"]) & used), "S‴ 玩家必须与 S/S′/S″ 不相交"
    assert len(reg["s3_players"]) >= 2
    assert "手级上下文" in reg["scope_rule"]
    assert reg["frozen_before_model_run"] is True
    with pytest.raises(RuntimeError) as e:
        s3.run_s3([], arm="U1")
    assert "已禁用" in str(e.value)


def test_manifests_sealed_with_ig1_to_ig10():
    for path, arm in ((s3.MANIFEST_U1, "U1"), (s3.MANIFEST_R22, "R2.2")):
        v = s3.verify_s3_manifest(path)
        assert v["present"] and v["clean"] is True and v["checked"] >= 9
        man = json.loads(path.read_text(encoding="utf-8"))
        assert man["arm"] == arm and man["config"]["baseline"] == "frozen"
        assert set(man["instrument_gates"]) == {
            f"IG{i}_{n}" for i, n in ((1, "raw_ratio"), (2, "anomaly_rate"),
                                      (3, "distribution"), (4, "audit_log"),
                                      (5, "no_approximation"),
                                      (6, "no_dry_run"),
                                      (7, "per_point_schema"),
                                      (8, "strata_completeness"),
                                      (9, "self_consistency"),
                                      (10, "traceability"))}
        assert "不得作为 S‴ 的预期" in man["config"]["no_expectation_from_s2"]
        assert man["production_wiring"].startswith("NOT WIRED")
        assert man["gates"]["partial"].startswith("coverage < 70%")
