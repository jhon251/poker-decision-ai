"""Phase 6.5 基准测试：泄漏/口径/判定/cbet 保留/ps horizon/输出 schema。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from phase65_benchmark import (  # noqa: E402
    CBET_MIN_OPP, HORIZONS, LL_PROBES, MIN_FUTURE, _eval_ll_probe,
    _eval_ps)


# ---------------- 泄漏与口径 ----------------
def test_horizons_frozen():
    assert HORIZONS == (50, 100, 200)
    assert MIN_FUTURE == 50


def test_no_static_profile_import():
    """6.5 实现不得导入 6G 接口（static profile 禁止作 prediction input）。"""
    import phase65_benchmark
    src = open(Path(phase65_benchmark.__file__), encoding="utf-8").read()
    assert "phase6g" not in src
    assert "get_profile" not in src


def test_history_future_no_overlap():
    """每 horizon：history 末 ts <= future 首 ts（_eval_ll_probe 内断言）。"""
    from phase6f_common import load_ring_sequences, load_test_players
    train, test = load_test_players("data/uoft/splits.json")
    assert not (train & test)
    seq = load_ring_sequences("data/uoft/feature_ready/player_event_table.jsonl",
                              player_filter=test)
    for pid, hands in seq.items():
        if len(hands) < 150:
            continue
        h = sorted(hands, key=lambda x: x["ts"] or "")
        N = 100
        hist, fut = h[:N], h[N:]
        assert hist[-1]["ts"] <= fut[0]["ts"]


def test_prior_train_only_provenance():
    d = json.load(open("data/uoft/models/phase6/phase65/benchmark.json",
                       encoding="utf-8"))
    cfg = d["config"]
    assert "train-only" in cfg["prior_provenance"]
    assert "hb_fit.json" in cfg["prior_provenance"]
    assert "fold_to_3bet" in cfg["prior_provenance"]
    assert cfg["static_profile_used"] is False
    assert "disabled" in cfg["cluster"]


# ---------------- 判定 ----------------
def test_main_judgment_threshold():
    d = json.load(open("data/uoft/models/phase6/phase65/benchmark.json",
                       encoding="utf-8"))
    mj = d["main_judgment"]
    assert mj["horizon"] == 200
    assert mj["threshold"] == ">=3/6"
    assert mj["utility_proven"] == (mj["improved_count"] >= 3)
    # improved_features 全部来自 6 特征集
    allowed = set(LL_PROBES) | {"position_sensitivity"}
    assert set(mj["improved_features"]) <= allowed


def test_ps_n50_insufficient():
    d = json.load(open("data/uoft/models/phase6/phase65/benchmark.json",
                       encoding="utf-8"))
    ps = d["per_horizon"]["50"]["position_sensitivity"]
    assert ps["status"] == "insufficient_evidence"
    # N=100/200 正常参与（有 metric 或 insufficient_players 之外的结果）
    for N in ("100", "200"):
        p = d["per_horizon"][N]["position_sensitivity"]
        assert p.get("metric") == "rmse" or p.get("status") == "insufficient_players"


def test_cbet_low_power_present_in_count():
    d = json.load(open("data/uoft/models/phase6/phase65/benchmark.json",
                       encoding="utf-8"))
    for N in ("50", "100", "200"):
        cbet = d["per_horizon"][N]["cbet"]
        assert cbet["low_power"] is True, f"N={N} cbet 应为 low_power"
        assert "ll_individual" in cbet and "ll_population" in cbet
        assert cbet["low_power_note"], "low_power 必须显式标注"


def test_learning_curve_three_points():
    d = json.load(open("data/uoft/models/phase6/phase65/benchmark.json",
                       encoding="utf-8"))
    assert set(d["per_horizon"].keys()) == {"50", "100", "200"}


def test_metrics_negative_is_improvement():
    d = json.load(open("data/uoft/models/phase6/phase65/benchmark.json",
                       encoding="utf-8"))
    cfg = d["config"]
    assert cfg["direction"] == "negative=improvement"
    for N in ("50", "100", "200"):
        for f, v in d["per_horizon"][N].items():
            if "diff_ll_vs_pop" in v:
                assert v["improved_vs_pop"] == (v["diff_ll_vs_pop_ci95"][1] < 0)


# ---------------- 合成方向性 ----------------
def test_eval_ll_probe_synthetic_direction():
    """构造合成序列：个体后验应优于 population（有玩家间差异时）。"""
    import random
    from phase6f_common import load_test_players
    train, test = load_test_players("data/uoft/splits.json")
    rng = random.Random(7)
    seq = {}
    for i in range(60):
        pid = f"syn_{i}"
        theta = 0.2 + rng.random() * 0.4          # 玩家真实 vpip
        hands = []
        for t in range(1, 401):
            vpip = rng.random() < theta
            hands.append({"ts": f"2009-07-{(t % 20) + 1:02d}T12:00:00",
                          "player_id": pid, "table_type": "6max",
                          "vpip": vpip, "pfr": False, "three_bet": False,
                          "cbet": False, "cbet_opportunity": False,
                          "faced_3bet": False, "folded_to_3bet": False,
                          "position": "CO", "rfi_opp": False, "rfi_did": False})
        seq[pid] = hands
    prior = {"a": 3.0, "b": 7.0}
    res = _eval_ll_probe(seq, prior, 0.3, 200, "vpip")
    assert res["n_evaluated"] == 60
    assert res["improved_vs_pop"] is True
    assert res["ll_individual"] < res["ll_population"]
