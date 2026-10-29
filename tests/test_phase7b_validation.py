"""Phase 7B 验证测试：泄漏/口径/阈值冻结/safety/verdict 词汇/报告完整性。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import phase7b_validate as p7b  # noqa: E402

RES = Path("data/uoft/models/phase7/phase7b/validation.json")


def _res() -> dict:
    return json.load(open(RES, encoding="utf-8"))


# ---------------- 泄漏（§七） ----------------
def test_train_test_disjoint_hard_failure_on_overlap():
    from phase6f_common import load_test_players
    train, test = load_test_players("data/uoft/splits.json")
    assert len(train) == 3999 and len(test) == 1001
    assert not (train & test)


def test_prior_train_only_provenance():
    cfg = _res()["config"]
    assert "train-only" in cfg["prior_provenance"]
    assert "hb_fit.json" in cfg["prior_provenance"]
    assert "task_a_train_players" in cfg["prior_provenance"]
    assert "方案 B" in cfg["prior_provenance"]
    assert "跳过 1001" in cfg["prior_provenance"]
    assert "disabled" in cfg["cluster"]
    assert "none" in cfg["time_decay"]
    assert cfg["no_fitting"] is not None
    # mu 由冻结 a,b 精确重算（7A 一致性硬断言），已透明记录
    assert "1.5e-5" in cfg["mu_recomputed_note"]


def test_history_future_boundary_on_test_players():
    """每 horizon：history 末 ts <= future 首 ts（runner 内硬断言；此处抽查）。"""
    from phase6f_common import load_ring_sequences, load_test_players
    _, test = load_test_players("data/uoft/splits.json")
    seq = load_ring_sequences("data/uoft/feature_ready/player_event_table.jsonl",
                              player_filter=test)
    checked = 0
    for pid, hands in seq.items():
        if len(hands) < 250:
            continue
        h = sorted(hands, key=lambda x: x["ts"] or "")
        for N in (50, 100, 200):
            hist, fut = h[:N], h[N:N + 50]
            assert hist[-1]["ts"] <= fut[0]["ts"]
        checked += 1
        if checked >= 40:
            break
    assert checked > 0


def test_source_redlines():
    src = open(Path(p7b.__file__), encoding="utf-8").read()
    assert "phase6g" not in src              # 静态画像/6G 禁止
    assert "get_profile" not in src
    assert "population_priors" not in src    # F3：全 pool 先验禁止
    assert "half_life" not in src            # 无时间衰减
    assert "gmm" not in src                  # cluster 不重启
    assert "hero_equity" not in src          # 无 hero/equity 路径
    # verdict 词汇白名单恰为 3 个允许词（禁止词只在声明语境出现，非声称）
    assert set(p7b.VERDICT_ALLOWED) == {"advisor_evidence_validated",
                                        "advisor_utility_unproven",
                                        "insufficient_evidence"}


def test_prohibited_vocabulary_only_as_declaration():
    """禁止词不得作为任何 verdict/结论出现；只能出现在显式 prohibited 枚举中。"""
    res = _res()
    mj = res["main_judgment"]
    assert mj["verdict"] in p7b.VERDICT_ALLOWED
    prohibited = ("exploit_utility_proven", "EV proven", "ROI proven",
                  "profitable", "profitability", "winnings improvement")
    mj_raw = json.dumps(mj, ensure_ascii=False)
    for bad in prohibited:
        assert bad not in mj_raw      # verdict/结论/证据中禁止出现
    # 显式 prohibited 枚举与冻结清单一致（声明 ≠ 声称）
    assert res["config"]["verdict_vocabulary"]["prohibited"] == list(prohibited)
    # 全产物中除该枚举外不得出现禁止词
    stripped = json.dumps(res, ensure_ascii=False).replace(
        json.dumps({"allowed": res["config"]["verdict_vocabulary"]["allowed"],
                    "prohibited": res["config"]["verdict_vocabulary"]["prohibited"]},
                   ensure_ascii=False), "")
    for bad in prohibited:
        assert bad not in stripped


# ---------------- 阈值冻结（§二/§七-13：config echo） ----------------
def test_config_echo_matches_frozen():
    fr = p7b.FROZEN
    cfg = _res()["config"]["frozen_thresholds"]
    assert cfg["horizons"] == [50, 100, 200]
    assert cfg["min_future"] == 50
    assert cfg["tail_sig"] == 0.975
    assert cfg["p7_effect_size"] == 0.02
    assert cfg["persistence_lower"] == 0.5
    assert cfg["utility_gate_min"] == 3
    assert cfg["feature_universe"] == 5
    assert cfg["min_units"] == 30
    assert cfg["min_evaluable_buckets"] == 2
    assert cfg["min_opp"] == {"default": 10, "cbet": 2}
    assert cfg["conf_strong"] == 0.95
    assert cfg["bootstrap"] == {"unit": "player", "paired": True,
                                "n": 1000, "seed": 42}
    assert cfg["buckets"] == [[0.5, 0.7], [0.7, 0.85], [0.85, 0.95], [0.95, 1.0]]
    assert fr == cfg


def test_bootstrap_config():
    cfg = _res()["config"]["frozen_thresholds"]["bootstrap"]
    assert cfg["unit"] == "player"
    assert cfg["n"] == 1000 and cfg["seed"] == 42


# ---------------- Safety contract（§六） ----------------
def test_safety_contract_all_clean():
    s = _res()["safety_contract"]
    assert s["all_clean"] is True
    assert s["violations"] == 0
    assert s["scanned_decisions"] > 0


def test_gate_fail_null_confidence_in_runner():
    """runner 对 status=none 的候选强制 confidence null（AssertionError 语义）。"""
    src = open(Path(p7b.__file__), encoding="utf-8").read()
    assert 'c["confidence"] is None' in src
    assert 'c["strength_recommended"] == 0.0' in src


# ---------------- 主判定与 verdict 词汇（§十/§十一） ----------------
def test_main_judgment_horizon_200():
    assert _res()["main_judgment"]["horizon"] == 200


def test_verdict_in_allowed_vocabulary():
    v = _res()["main_judgment"]["verdict"]
    assert v in ("advisor_evidence_validated", "advisor_utility_unproven",
                 "insufficient_evidence")


def test_verdict_consistency_with_evidence():
    mj = _res()["main_judgment"]
    v = mj["verdict"]
    if v == "advisor_evidence_validated":
        assert mj["g_persist_pass"] is True
        assert mj["g_calib_pass"] is True
        assert len(mj["features_passed"]) >= 3
        assert len(mj["evaluable_features"]) >= 3
    elif v == "insufficient_evidence":
        assert (len(mj["evaluable_features"]) < 3
                or mj["g_calib_insufficient"] is True
                or _res()["pooled_persistence"]["n_players"] < 30)
    else:  # advisor_utility_unproven
        assert not (mj["g_persist_pass"] and mj["g_calib_pass"]
                    and len(mj["features_passed"]) >= 3)


def test_no_negative_result_hidden():
    """全部 5 特征必须出现在 N=200 报告（含 insufficient），逐项标注原因。"""
    pf = _res()["per_horizon"]["200"]["per_feature"]
    for f in ("vpip", "pfr", "three_bet", "cbet", "fold_to_3bet"):
        assert f in pf
        assert "status" in pf[f]
        if pf[f]["status"] == "insufficient_evidence":
            assert pf[f]["reason"]


def test_insufficient_features_are_below_min_units():
    mj = _res()["main_judgment"]
    pf = _res()["per_horizon"]["200"]["per_feature"]
    for f in mj["insufficient_features"]:
        assert pf[f]["n_full_pass"] < 30
        assert pf[f]["status"] == "insufficient_evidence"


def test_learning_curves_present_all_horizons():
    per = _res()["per_horizon"]
    for N in ("50", "100", "200"):
        assert N in per
        assert set(per[N]["per_feature"]) == {"vpip", "pfr", "three_bet",
                                              "cbet", "fold_to_3bet"}


# ---------------- C1 校准口径（§三） ----------------
def test_calibration_uses_paired_empirical_vs_predicted():
    cal = _res()["calibration"]
    assert "buckets" in cal
    for name, b in cal["buckets"].items():
        assert b["range"] in [[0.5, 0.7], [0.7, 0.85], [0.85, 0.95], [0.95, 1.0]]
        if b["evaluable"]:
            assert b["mean_predicted"] is not None   # p̂ 存在（C1 口径）
            assert b["diff_ci95"] is not None
            assert b["passed"] == b["contains_zero"]
        else:
            assert b["low_power"] is True
            assert b["reason"]
    if not cal["insufficient"]:
        assert len(cal["evaluable_buckets"]) >= 2
        assert all(cal["buckets"][n]["passed"] for n in cal["evaluable_buckets"])


def test_calibration_buckets_frozen_not_refit():
    cal = _res()["calibration"]
    ranges = [cal["buckets"][n]["range"] for n in ("b1", "b2", "b3", "b4")]
    assert ranges == [[0.5, 0.7], [0.7, 0.85], [0.85, 0.95], [0.95, 1.0]]


# ---------------- 降级（§八/审计 F） ----------------
def test_degrade_log_only_on_failure():
    v = _res()["main_judgment"]["verdict"]
    log = Path("data/uoft/models/phase7/phase7b/DEGRADE_LOG.json")
    if v == "advisor_evidence_validated":
        assert not log.exists()
    else:
        assert log.exists()
        entries = json.load(open(log, encoding="utf-8"))["entries"]
        assert len(entries) >= 1
        assert all(e["reason"] for e in entries)


# ---------------- 口径细节 ----------------
def test_one_decision_per_player_feature_horizon():
    """runner 内断言每 (player, feature, N) 至多 1 个决策（防伪复制）。"""
    src = open(Path(p7b.__file__), encoding="utf-8").read()
    assert "每玩家每特征至多 1 决策" in src


def test_per_feature_ci_structure():
    pf = _res()["per_horizon"]["200"]["per_feature"]
    for f, rec in pf.items():
        if rec["n_full_pass"] >= 30:
            lo, hi = rec["persistence_ci95"]
            assert lo <= rec["persistence_rate"] <= hi
            assert rec["passed"] == (lo > 0.5)
            assert rec["lower_bound"] == lo
