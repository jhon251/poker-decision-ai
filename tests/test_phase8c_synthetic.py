"""Phase 8C 测试：spec 冻结 / 校准不可变与结果无关 / 观测白名单 / V3 正交性 /
真实 7C 链 / 检测 cell 规则 / controls / 词汇红线 / rollout 确定性。"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import phase8c_detect as detect  # noqa: E402
import phase8c_ev as ev8c  # noqa: E402
import phase8c_sim as sim  # noqa: E402
import phase8c_spec as spec  # noqa: E402

OUT = spec.OUT_DIR
CAL = json.load(open(OUT / "villains_calibration.json", encoding="utf-8"))
LATENT = json.load(open(OUT / "villains_latent.json", encoding="utf-8"))
DET = json.load(open(OUT / "detection.json", encoding="utf-8"))
EV = json.load(open(OUT / "ev.json", encoding="utf-8"))
PAM = json.load(open("data/uoft/models/phase8/phase8b/pam.json", encoding="utf-8"))


# ---------------- spec 冻结 ----------------
def test_spec_frozen():
    assert spec.TIERS == {"strong": 0.20, "moderate": 0.10, "near": 0.02}
    assert spec.HORIZONS == (50, 100, 200)
    assert spec.H_HISTORY == 200 and spec.E_EVAL == 100
    assert spec.PRACTICAL_BB100 == 1.0
    assert spec.BOOT_N == 1000 and spec.BOOT_SEED == 42
    assert spec.SEED == 42
    assert spec.V3_DONK_MULT == 1.5


# ---------------- 校准：确定性 + 结果无关（反向测试） ----------------
def test_calibration_deterministic(monkeypatch):
    monkeypatch.setattr(spec, "CAL_SIM_HANDS", 400)
    monkeypatch.setattr(spec, "CAL_MAX_ITERS", 4)
    monkeypatch.setattr(spec, "CAL_TOL", 0.05)
    targets = {"vpip": 0.3, "pfr": 0.15, "three_bet": 0.02, "cbet": 0.6,
               "fold_to_3bet": 0.7}
    a = sim.calibrate(PAM, targets, seed=42)
    b = sim.calibrate(PAM, targets, seed=42)
    assert a == b


def test_calibration_outcome_blind_reverse(monkeypatch):
    """修改 validation/eval outcome / detection / EV 结果 → 校准产物必须不变。"""
    monkeypatch.setattr(spec, "CAL_SIM_HANDS", 400)
    monkeypatch.setattr(spec, "CAL_MAX_ITERS", 4)
    monkeypatch.setattr(spec, "CAL_TOL", 0.05)
    targets = {"vpip": 0.3, "pfr": 0.15, "three_bet": 0.02, "cbet": 0.6,
               "fold_to_3bet": 0.7}
    before = sim.calibrate(PAM, targets, seed=42)
    # 篡改 8B experiment.json（calibration 不得读取）
    exp_path = Path("data/uoft/models/phase8/phase8b/experiment.json")
    if exp_path.exists():
        backup = exp_path.read_text(encoding="utf-8")
        mutated = json.loads(backup)
        for arm in mutated["arms"].values():
            arm["delta_bb_100"] = 999.0
        exp_path.write_text(json.dumps(mutated, ensure_ascii=False),
                            encoding="utf-8")
        try:
            after = sim.calibrate(PAM, targets, seed=42)
            assert before == after
        finally:
            exp_path.write_text(backup, encoding="utf-8")


def test_calibration_source_blind():
    """calibrate/sim 源码不读取任何 outcome/accuracy/ΔEV/verdict 产物文件。"""
    for mod in (sim,):
        src = open(Path(mod.__file__), encoding="utf-8").read()
        assert "experiment.json" not in src
        assert "detection.json" not in src
        assert "ev.json" not in src
        assert "delta_bb" not in src
        assert "verdict.json" not in src
    cal_src = open(Path(__file__).parent.parent / "scripts/phase8c_calibrate.py",
                   encoding="utf-8").read()
    assert "experiment.json" not in cal_src
    assert "detection.json" not in cal_src


def test_calibration_artifact_immutable():
    assert CAL["immutable"] is True
    assert CAL["seed"] == 42
    assert len(CAL["hard_asserts"]) >= 6
    assert CAL["priors"]["vpip"]["provenance"] == "hb_fit"
    assert CAL["priors"]["fold_to_3bet"]["provenance"] == "scheme_b"


def test_v0_alignment_within_tolerance():
    r = CAL["cells"]["v0"]["realized"]
    mu = CAL["priors"]
    assert abs(r["vpip"] - mu["vpip"]["mu"]) < 0.05
    assert abs(r["pfr"] - mu["pfr"]["mu"]) < 0.05


# ---------------- 生成：白名单 + latent 隔离 + V3 正交 ----------------
def test_generated_rows_whitelist():
    import glob as _g
    f = sorted(_g.glob(str(OUT / "generated" / "*.json")))[0]
    payload = json.load(open(f, encoding="utf-8"))
    allowed = {"player_id", "ts", "table_type", "vpip", "pfr", "three_bet",
               "faced_3bet", "folded_to_3bet", "cbet_opportunity", "cbet"}
    for r in payload["history_rows"]:
        assert set(r) == allowed


def test_latent_admin_only():
    assert LATENT["admin_only"] is True
    assert "注意" not in LATENT or "admin" in LATENT["note"].lower() or \
        "latent" in LATENT["note"]
    for vid, info in LATENT["villains"].items():
        assert set(info) <= {"cell", "feature", "direction", "tier", "delta",
                             "targets", "realized", "note"}


def test_v3_orthogonality_structural(monkeypatch):
    """V3 donk 乘子不改变 5 特征 suff stats（同 seed 行为测试）。"""
    monkeypatch.setattr(spec, "H_HISTORY", 300)
    pol_a = sim.PlayerPolicy(PAM, CAL["cells"]["v0"]["multipliers"], 1.0)
    pol_b = sim.PlayerPolicy(PAM, CAL["cells"]["v0"]["multipliers"],
                             spec.V3_DONK_MULT)
    pa = {f"S{i}": pol_a for i in range(1, 7)}
    pb = {f"S{i}": pol_a for i in range(1, 7)}
    pb[spec.VILLAIN_SEAT] = pol_b
    ha = sim.simulate_hands(PAM, pa, 300, seed="ortho-a")
    hb = sim.simulate_hands(PAM, pb, 300, seed="ortho-a")
    assert sim.feature_rows(ha, spec.VILLAIN_SEAT) == \
        sim.feature_rows(hb, spec.VILLAIN_SEAT)


# ---------------- 检测：真实 7C 链 ----------------
def test_detect_uses_real_7c_chain():
    src = open(Path(detect.__file__), encoding="utf-8").read()
    assert "from phase7a_decision import decide" in src
    assert "from phase7c_recommend import recommend" in src
    assert "SyntheticRecommendationProvider" not in src
    assert "phase8b_arms" not in src


def test_detection_three_bet_cbet_insufficient():
    assert DET["config"]["three_bet_cbet"]
    # 三条记录全量验证：直接重扫（已由 detect 主流程断言；此处验证产物配置声明）
    assert "insufficient" in DET["config"]["three_bet_cbet"]


def test_detection_cell_flags_and_scoring():
    for cell, d in DET["per_cell"].items():
        assert "cell_flag" in d
        assert "in_p2a_scoring" in d
        if d["feature"] and d["cell_flag"] != "ok":
            assert d["direction_accuracy"] is None     # 排除出评分
        if not d["feature"] or d["cell"] == "v3":
            assert d["in_p2a_scoring"] is True          # baseline 对照进报告


def test_detection_horizons_frozen():
    assert DET["config"]["horizons"] == [50, 100, 200]


# ---------------- EV：controls + 标注 + 中性一致性 ----------------
def test_ev_controls_present():
    for cell, d in EV["per_cell"].items():
        assert "c1_reverse_delta_bb100" in d
        assert "c3_random_delta_bb100" in d
        assert d["ev_label"] == spec.EV_LABEL
    assert EV["ev_label"] == spec.EV_LABEL
    assert EV["verdict"] is None
    assert "controls" in EV["config"]


def test_real7c_provider_mapping_and_controls():
    recs = {"vpip": {"direction": "under", "confidence": 0.9,
                     "deviation_magnitude": 0.15}}
    obs = {"street": "preflop", "hero_position": "BTN", "to_call": 0.0,
           "players": [{"player_id": spec.VILLAIN_SEAT, "status": "active"}]}
    p = ev8c.Real7CProvider(recs, spec.VILLAIN_SEAT, "normal")
    r = p(obs)
    assert r["mapping"]["action_class"] == "raise"
    assert r["mapping"]["sign"] == +1
    assert r["label"] == "real-7c"
    rv = ev8c.Real7CProvider(recs, spec.VILLAIN_SEAT, "reverse")(obs)
    assert rv["mapping"]["sign"] == -1
    rr = ev8c.Real7CProvider(recs, spec.VILLAIN_SEAT, "random", seed=1)(obs)
    assert rr["mapping"]["sign"] in (+1, -1)
    # 目标 villain 不在场 → None
    obs2 = dict(obs, players=[{"player_id": "S9", "status": "active"}])
    assert ev8c.Real7CProvider(recs, spec.VILLAIN_SEAT, "normal")(obs2) is None


def test_rollout8c_mirrors_8b_semantics():
    import phase8b_experiment as exp8b
    hand = {
        "hand_key": "t", "table": "t", "hand_id": "t1",
        "players": [{"pid": "p1", "stack_start": 1000.0, "stack_end": None,
                     "hole_cards": ["Kh", "Kd"]},
                    {"pid": "p2", "stack_start": 1000.0, "stack_end": None,
                     "hole_cards": ["Ah", "Ad"]}],
        "sb": 25.0, "bb": 50.0, "positions": {"p1": "BTN", "p2": "BB"},
        "orbit": ["p1", "p2"],
        "actions": {"preflop": [
            {"street": "preflop", "seq": 1, "player_id": "p1",
             "action": "post", "amount": 25.0},
            {"street": "preflop", "seq": 2, "player_id": "p2",
             "action": "post", "amount": 50.0},
            {"street": "preflop", "seq": 3, "player_id": "p1",
             "action": "call", "amount": 25.0},
            {"street": "preflop", "seq": 4, "player_id": "p2",
             "action": "check"}],
            "flop": [{"street": "flop", "seq": 5, "player_id": "p2",
                      "action": "check"},
                     {"street": "flop", "seq": 6, "player_id": "p1",
                      "action": "bet", "amount": 100.0, "to": 100.0},
                     {"street": "flop", "seq": 7, "player_id": "p2",
                      "action": "call", "amount": 100.0}],
            "turn": [{"street": "turn", "seq": 8, "player_id": "p2",
                      "action": "check"},
                     {"street": "turn", "seq": 9, "player_id": "p1",
                      "action": "bet", "amount": 200.0, "to": 200.0},
                     {"street": "turn", "seq": 10, "player_id": "p2",
                      "action": "call", "amount": 200.0}],
            "river": [{"street": "river", "seq": 11, "player_id": "p2",
                       "action": "check"},
                      {"street": "river", "seq": 12, "player_id": "p1",
                       "action": "bet", "amount": 300.0, "to": 300.0},
                      {"street": "river", "seq": 13, "player_id": "p2",
                       "action": "call", "amount": 300.0}]},
        "flop": ["2c", "7d", "Jh"], "turn": ["3s"], "river": ["5d"],
        "all_in": [],
    }
    dps = exp8b.walk_dps(hand, "p2")
    dp = dps[-1]
    vill = lambda pid, s, p, sp, b, legal: {"call": 1.0}
    n1 = ev8c.rollout8c(hand, "p2", dp, lambda obs: {"call": 1.0}, vill,
                        random.Random("z"), "b")
    n2 = ev8c.rollout8c(hand, "p2", dp, lambda obs: {"call": 1.0}, vill,
                        random.Random("z"), "b")
    assert n1 == n2 == pytest.approx(1000.0)      # 与 8B 同语义：AA 赢


# ---------------- 词汇红线 ----------------
def test_forbidden_vocabulary_8c():
    import re as _re
    for path in (OUT / "detection.json", OUT / "ev.json",
                 OUT / "villains_calibration.json"):
        raw = path.read_text(encoding="utf-8")
        stripped = _re.sub(r'"prohibited": \[[^\]]*\]', '""', raw)
        for bad in ("exploit_utility_proven", "EV proven", "ROI proven",
                    "profitable", "real-world profitability", "winnings"):
            assert bad not in stripped
    for mod in (sim, detect, ev8c):
        src = open(Path(mod.__file__), encoding="utf-8").read()
        assert "exploit_utility_proven" not in src
        assert "phase6g" not in src
        assert "get_profile" not in src
        assert "population_priors" not in src
        assert "half_life" not in src
        assert "gmm" not in src
