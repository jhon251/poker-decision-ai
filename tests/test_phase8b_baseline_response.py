"""Phase 8B 测试：split 隔离 / PAM 白名单与确定性 / 中性一致性硬测试 /
rollout 正确性 / 合成 rec 冻结 / 源码红线 / EV 标注。"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import phase8b_arms as p8arms  # noqa: E402
import phase8b_experiment as p8exp  # noqa: E402
import phase8b_pam as p8pam  # noqa: E402
from phase8a_replay import (  # noqa: E402
    ACTION_MAPPINGS, FROZEN_EPS, FROZEN_K, PRIMARY_ARM, BaselineArm, ExploitArm)

OUT = Path("data/uoft/models/phase8/phase8b")
PAM = json.load(open(OUT / "pam.json", encoding="utf-8"))
SPLITS = json.load(open(OUT / "splits.json", encoding="utf-8"))
EXP = json.load(open(OUT / "experiment.json", encoding="utf-8"))


# ---------------- split（60/20/20，表分层，互斥） ----------------
def test_split_sizes_and_disjoint():
    n = SPLITS["n"]
    assert SPLITS["ratio"] == [0.6, 0.2, 0.2]
    assert SPLITS["seed"] == 42
    total = n["train"] + n["cal"] + n["eval"]
    assert total == SPLITS["total_replayable"] == 8942
    assert abs(n["train"] / total - 0.6) < 0.01
    assert abs(n["cal"] / total - 0.2) < 0.01
    sets = [set(SPLITS["sets"][k]) for k in ("train", "cal", "eval")]
    for i in range(3):
        for j in range(i + 1, 3):
            assert not (sets[i] & sets[j])


def test_split_stratified_by_table():
    tables = {}
    for line in open(OUT / "hands_replayable.jsonl", encoding="utf-8"):
        h = json.loads(line)
        tables[h["hand_key"]] = h["table"]
    for part in ("train", "cal", "eval"):
        part_tables = {tables[hk] for hk in SPLITS["sets"][part]}
        assert len(part_tables) > 80          # 92 表基本全部覆盖（分层）


# ---------------- PAM ----------------
def test_pam_train_only_and_deterministic():
    assert PAM["train_hands"] == SPLITS["n"]["train"]
    train = set(SPLITS["sets"]["train"])
    cal = set(SPLITS["sets"]["cal"]) | set(SPLITS["sets"]["eval"])
    assert not (train & cal)
    # 重新拟合（只读 artifacts）→ 同一产物（确定性）
    hands = []
    for line in open(OUT / "hands_replayable.jsonl", encoding="utf-8"):
        h = json.loads(line)
        if h["hand_key"] in train:
            hands.append(h)
    again = p8pam.fit(hands)
    assert again["action_probs"] == PAM["action_probs"]
    assert again["size_probs"] == PAM["size_probs"]


def test_pam_fallback_recorded_and_ladder():
    fb = PAM["fallback"]
    assert isinstance(fb["street"], int) and fb["street"] >= 0
    # 阶梯语义：cell → street×spot → street（拟合代码常量）
    assert PAM["config"]["fallback_ladder"]
    assert PAM["config"]["pseudo_count"] == 1.0
    assert PAM["config"]["min_cell"] == 50


def test_pam_forbidden_fields_source():
    src = open(Path(p8pam.__file__), encoding="utf-8").read()
    assert "hole_cards" not in src
    assert "stack_end" not in src
    assert "winners" not in src
    assert "showdown" not in src


def test_pam_not_ml():
    for mod in ("sklearn", "torch", "tensorflow", "xgboost"):
        for pymod in (p8pam, p8arms, p8exp):
            src = open(Path(pymod.__file__), encoding="utf-8").read()
            assert mod not in src


def test_pam_policy_cell_lookup():
    pol = p8arms.PAMPolicy(PAM)
    obs = {"street": "preflop", "hero_position": "BTN", "pot": 100.0,
           "to_call": 0.0, "action_history": [],
           "available_actions": ["fold", "call", "raise"]}
    probs = pol(obs)
    assert set(probs) == {"fold", "call", "raise"}
    assert abs(sum(probs.values()) - 1.0) < 1e-9
    assert pol.cell(obs) == "preflop|BTN|unopened|b0"


# ---------------- 中性一致性硬测试（用户要求 6） ----------------
def test_arms_identical_when_signal_off():
    rng_b = random.Random(7)
    rng_e = random.Random(7)
    base = BaselineArm(rng_b, policy=p8arms.PAMPolicy(PAM))
    expl = ExploitArm(rng_e, k=1.0, eps=1.0,
                      recommendation_provider=p8arms.SyntheticRecommendationProvider(
                          "h", mode="off"),
                      policy=p8arms.PAMPolicy(PAM))
    obs = {"street": "preflop", "hero_position": "BTN", "pot": 100.0,
           "to_call": 0.0, "action_history": [],
           "available_actions": ["fold", "call", "raise"]}
    for _ in range(200):
        assert base.decide(obs) == expl.decide(obs)


def test_arms_identical_when_multiplier_neutral():
    """乘子中性（deviation=0 → mult=1）→ 概率逐位一致（用户要求 6 的数学形式）。"""
    base = BaselineArm(random.Random(3), policy=p8arms.PAMPolicy(PAM))
    rec = {"eligible": True, "mapping": {"action_class": "raise", "sign": +1},
           "confidence": 0.9, "deviation_magnitude": 0.0}
    expl = ExploitArm(random.Random(3), k=1.0, eps=1.0,
                      recommendation_provider=lambda obs: rec,
                      policy=p8arms.PAMPolicy(PAM))
    obs = {"street": "preflop", "hero_position": "BTN", "pot": 100.0,
           "to_call": 0.0, "action_history": [],
           "available_actions": ["fold", "call", "raise"]}
    assert base.action_probs(obs) == expl.action_probs(obs)


def test_divergence_only_with_active_signal():
    """信号开启（正偏差）→ 映射行动类概率必然变化；关闭 → 完全一致。"""
    obs = {"street": "preflop", "hero_position": "BTN", "pot": 100.0,
           "to_call": 0.0, "action_history": [],
           "available_actions": ["fold", "call", "raise"]}
    rec = {"eligible": True, "mapping": {"action_class": "raise", "sign": +1},
           "confidence": 0.9, "deviation_magnitude": 0.1}
    base = BaselineArm(random.Random(5), policy=p8arms.PAMPolicy(PAM))
    expl = ExploitArm(random.Random(5), k=1.0, eps=1.0,
                      recommendation_provider=lambda o: rec,
                      policy=p8arms.PAMPolicy(PAM))
    pb, pe = base.action_probs(obs), expl.action_probs(obs)
    assert pe["raise"] > pb["raise"]          # 乘子 1+0.09 生效
    assert pb != pe


def test_synthetic_provider_frozen_and_deterministic():
    p1 = p8arms.SyntheticRecommendationProvider("h1", mode="on")
    p2 = p8arms.SyntheticRecommendationProvider("h1", mode="on")
    obs = {"street": "preflop", "hero_position": "BTN", "pot": 100.0,
           "to_call": 0.0, "action_history": [],
           "available_actions": ["fold", "call", "raise"],
           "hand_id": "h1", "game_type": "cash", "board": [],
           "hero_hole_cards": [], "current_bet": 0.0, "min_raise_to": 0.0,
           "blinds": {"sb": 0.0, "bb": 0.0, "ante": 0.0}, "hero_stack": 100.0,
           "players": []}
    r1, r2 = p1(obs), p2(obs)
    assert r1 == r2 and r1["label"] == "synthetic"
    assert p8arms.SyntheticRecommendationProvider("h", mode="off")(obs) is None


# ---------------- rollout 引擎正确性 ----------------
def _hu_hand():
    return {
        "hand_key": "test-hu", "table": "test", "hand_id": "t1",
        "players": [{"pid": "p1", "stack_start": 1000.0, "stack_end": None,
                     "hole_cards": ["Kh", "Kd"]},       # villain KK
                    {"pid": "p2", "stack_start": 1000.0, "stack_end": None,
                     "hole_cards": ["Ah", "Ad"]}],      # hero AA
        "sb": 25.0, "bb": 50.0,
        "positions": {"p1": "BTN", "p2": "BB"}, "orbit": ["p1", "p2"],
        "actions": {
            "preflop": [
                {"street": "preflop", "seq": 1, "player_id": "p1",
                 "action": "post", "amount": 25.0},
                {"street": "preflop", "seq": 2, "player_id": "p2",
                 "action": "post", "amount": 50.0},
                {"street": "preflop", "seq": 3, "player_id": "p1",
                 "action": "call", "amount": 25.0},
                {"street": "preflop", "seq": 4, "player_id": "p2",
                 "action": "check"}],
            "flop": [
                {"street": "flop", "seq": 5, "player_id": "p2",
                 "action": "check"},
                {"street": "flop", "seq": 6, "player_id": "p1",
                 "action": "bet", "amount": 100.0, "to": 100.0},
                {"street": "flop", "seq": 7, "player_id": "p2",
                 "action": "call", "amount": 100.0}],
            "turn": [
                {"street": "turn", "seq": 8, "player_id": "p2",
                 "action": "check"},
                {"street": "turn", "seq": 9, "player_id": "p1",
                 "action": "bet", "amount": 200.0, "to": 200.0},
                {"street": "turn", "seq": 10, "player_id": "p2",
                 "action": "call", "amount": 200.0}],
            "river": [
                {"street": "river", "seq": 11, "player_id": "p2",
                 "action": "check"},
                {"street": "river", "seq": 12, "player_id": "p1",
                 "action": "bet", "amount": 300.0, "to": 300.0},
                {"street": "river", "seq": 13, "player_id": "p2",
                 "action": "call", "amount": 300.0}]},
        "flop": ["2c", "7d", "Jh"], "turn": ["3s"], "river": ["5d"],
        "all_in": [],
    }


def test_walk_dps_hu():
    hand = _hu_hand()
    dps = p8exp.walk_dps(hand, "p2")
    assert [d["street"] for d in dps] == ["preflop", "flop", "turn", "river"]
    assert dps[-1]["pot"] == pytest.approx(1000.0)   # 100+200+400+300(bet 已计入)


def test_rollout_deterministic_and_evaluator():
    hand = _hu_hand()
    dps = p8exp.walk_dps(hand, "p2")
    dp = dps[-1]                                # river，facing 300
    always_call = lambda obs: {"call": 1.0}
    vill = lambda pam, s, p, sp, b, legal: {"call": 1.0}
    net1 = p8exp.rollout(hand, "p2", dp, always_call, vill, PAM,
                         random.Random("t-rng"))
    net2 = p8exp.rollout(hand, "p2", dp, always_call, vill, PAM,
                         random.Random("t-rng"))
    assert net1 == net2 == pytest.approx(1000.0)   # AA 赢：1300 pot − 300 投入


def test_rollout_fold_net_zero():
    hand = _hu_hand()
    dp = p8exp.walk_dps(hand, "p2")[-1]
    always_fold = lambda obs: {"fold": 1.0}
    vill = lambda pam, s, p, sp, b, legal: {"check": 1.0}
    net = p8exp.rollout(hand, "p2", dp, always_fold, vill, PAM,
                        random.Random("t-rng2"))
    assert net == 0.0


def test_walk_dps_no_decision_when_bb_wins():
    hand = {
        "hand_key": "t2", "table": "test", "hand_id": "t2",
        "players": [{"pid": "p1", "stack_start": 1000.0, "stack_end": None,
                     "hole_cards": []},
                    {"pid": "p2", "stack_start": 1000.0, "stack_end": None,
                     "hole_cards": []}],
        "sb": 25.0, "bb": 50.0,
        "positions": {"p1": "BTN", "p2": "BB"}, "orbit": ["p1", "p2"],
        "actions": {"preflop": [
            {"street": "preflop", "seq": 1, "player_id": "p1",
             "action": "post", "amount": 25.0},
            {"street": "preflop", "seq": 2, "player_id": "p2",
             "action": "post", "amount": 50.0},
            {"street": "preflop", "seq": 3, "player_id": "p1",
             "action": "fold"}],
            "flop": [], "turn": [], "river": []},
        "flop": [], "turn": [], "river": [], "all_in": [],
    }
    assert p8exp.walk_dps(hand, "p2") == []      # BB 无需行动


# ---------------- 实验产物语义 ----------------
def test_experiment_primary_arm_fixed():
    arms = EXP["arms"]
    assert EXP["config"]["primary_arm"] == [1.0, 1.0]
    assert [a for a in arms.values() if a["primary"]][0]["k"] == 1.0
    assert len(arms) == 6
    assert EXP["config"]["no_arm_selection"]


def test_experiment_ev_label_and_no_verdict():
    assert EXP["ev_label"] == "simulated EV under frozen response model"
    assert EXP["verdict"] is None
    import re as _re
    raw = json.dumps(EXP, ensure_ascii=False)
    stripped = _re.sub(r'"prohibited": \[[^\]]*\]', '""', raw)
    for bad in ("exploit_utility_proven", "EV proven", "ROI proven",
                "profitable", "real-world profitability"):
        assert bad not in stripped          # 禁止词只允许出现在显式 prohibited 枚举
    for arm in EXP["arms"].values():
        assert arm["ev_label"] == "simulated EV under frozen response model"


def test_experiment_observed_simulated_counterfactual_separated():
    assert "observed_sanity" in EXP
    assert EXP["observed_sanity"]["label"].startswith("observed")
    assert "simulated" in EXP["ev_label"]
    # counterfactual = paired Δ（arms 的 delta 字段即反事实差）
    for arm in EXP["arms"].values():
        assert "delta_bb_hand" in arm and "ci95_hand_level" in arm


def test_experiment_practical_significance_convention():
    assert EXP["config"]["practical_bb100"] == 1.0
    assert "convention" in EXP["config"]["practical_note"]
    for arm in EXP["arms"].values():
        assert "convention" in arm["practical_significance"]
        assert arm["statistical_significance"] in (
            "pass (CI lower > 0)", "not significant")


def test_experiment_table_block_sensitivity_present():
    for arm in EXP["arms"].values():
        assert "ci95_table_block" in arm


def test_experiment_divergence_neutral_only_via_signal():
    """分歧只能来自 exploit signal（实现层面已保证 off/中性 → 一致；
    实验层面：分歧数 = 0 当且仅当无信号（由 provider off 硬测试覆盖）。"""
    assert "divergent_dp" in EXP["arms"][
        next(iter(EXP["arms"]))]


# ---------------- 源码红线 ----------------
def test_source_redlines():
    import re as _re
    for pymod in (p8pam, p8arms, p8exp):
        src = open(Path(pymod.__file__), encoding="utf-8").read()
        assert "phase6g" not in src
        assert "get_profile" not in src
        assert "population_priors" not in src
        assert "half_life" not in src
        assert "gmm" not in src
        assert "hero_equity" not in src
        assert "solver" not in src.lower()
        # 禁止 P2 词汇：只允许出现在显式 prohibited 枚举（声明 ≠ 声称）
        stripped = _re.sub(r'"prohibited": \[[^\]]*\]', '""', src)
        for bad in ("exploit_utility_proven", "EV proven", "ROI proven",
                    "profitable"):
            assert bad not in stripped


def test_no_file_order_forged_history():
    """8B 三个模块均不按文件名/手号排序构造任何 history（时间顺序伪造 = 硬禁）。"""
    for pymod in (p8pam, p8arms, p8exp):
        src = open(Path(pymod.__file__), encoding="utf-8").read()
        assert "hand_number_in_file" not in src
        assert "rsplit" not in src or "rsplit(\"/\", 1)[0]" in src
