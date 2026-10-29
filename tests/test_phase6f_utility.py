"""Phase 6F 测试：泄漏防护/方向/口径一致性/低功效标记/判定阈值。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from phase6f_common import (  # noqa: E402
    feature_suff, fit_fold_to_3bet_prior, load_hb_fit_priors,
    load_ring_sequences, load_test_players, posterior_mean, split_half)


def _mk_ev(ts, **kw):
    ev = {"ts": ts, "table_type": "6max", "vpip": False, "pfr": False,
          "three_bet": False, "folded_to_3bet": False, "faced_3bet": False,
          "cbet": False, "cbet_opportunity": False,
          "position": "CO", "rfi_opp": False, "rfi_did": False}
    ev.update(kw)
    return ev


# ---------------- split / 口径 ----------------
def test_split_half_time_ordered():
    hands = [_mk_ev(f"2009-07-{d:02d}T12:00:00") for d in range(1, 41)]
    halves = split_half(hands, min_half=10)
    assert halves is not None
    hist, fut = halves
    assert len(hist) == 20 and len(fut) == 20
    assert hist[-1]["ts"] <= fut[0]["ts"]      # as_of: history 全部早于 future
    assert split_half(hands[:15], min_half=10) is None


def test_feature_suff_vpip_pfr():
    hands = [_mk_ev("2009-07-01T12:00:00", vpip=True),
             _mk_ev("2009-07-02T12:00:00", pfr=True),
             _mk_ev("2009-07-03T12:00:00")]
    assert feature_suff(hands, "vpip") == (1, 3)
    assert feature_suff(hands, "pfr") == (1, 3)


def test_feature_suff_cbet_opportunity_gated():
    hands = [_mk_ev("2009-07-01T12:00:00", cbet_opportunity=True, cbet=True),
             _mk_ev("2009-07-02T12:00:00", cbet_opportunity=True, cbet=False),
             _mk_ev("2009-07-03T12:00:00")]
    assert feature_suff(hands, "cbet") == (1, 2)


def test_feature_suff_fold_to_3bet_gated():
    hands = [_mk_ev("2009-07-01T12:00:00", faced_3bet=True, folded_to_3bet=True),
             _mk_ev("2009-07-02T12:00:00", faced_3bet=True, folded_to_3bet=False),
             _mk_ev("2009-07-03T12:00:00")]
    assert feature_suff(hands, "fold_to_3bet") == (1, 2)


def test_posterior_mean_formula():
    prior = {"a": 3.0, "b": 7.0}
    assert posterior_mean(prior, 5, 10) == pytest.approx(8.0 / 20.0)
    assert posterior_mean(prior, 0, 0) == pytest.approx(3.0 / 10.0)


# ---------------- 泄漏防护（硬失败） ----------------
def test_train_test_disjoint():
    train, test = load_test_players("data/uoft/splits.json")
    assert not (train & test)
    assert len(test) == 1001


def test_hb_fit_is_train_only():
    train, test = load_test_players("data/uoft/splits.json")
    priors = load_hb_fit_priors("data/uoft/models/mvp/hb_fit.json", train)
    assert set(priors) == {"vpip", "pfr", "three_bet", "cbet",
                           "rfi_ep", "rfi_lp"}
    # hb_fit.json 本身声明 train-only
    hb = json.load(open("data/uoft/models/mvp/hb_fit.json", encoding="utf-8"))
    assert hb["config"]["fit_on"] == "task_a_train_players"


def test_fold_to_3bet_prior_train_only_no_test_leak():
    """方案 B：fold_to_3bet prior 拟合必须跳过测试玩家并断言。"""
    train, test = load_test_players("data/uoft/splits.json")
    fit = fit_fold_to_3bet_prior("data/uoft/stats/full_pool_suff_stats.jsonl",
                                 train, test)
    assert fit["n_test_players_skipped"] > 0, "测试玩家应被显式跳过"
    assert fit["leakage_assertion"].startswith("passed")
    assert fit["a"] > 0 and fit["b"] > 0
    assert "protection" in fit


def test_train_test_disjoint_hard_failure_on_overlap():
    """构造重叠 splits → load_test_players 必须硬失败（泄漏硬门）。"""
    import tempfile
    import os
    d = json.load(open("data/uoft/splits.json", encoding="utf-8"))
    leak = {"task_a": {"train_players": d["task_a"]["train_players"][:10]
                       + d["task_a"]["test_players"][:1],   # 故意混入 1 名测试玩家
                       "test_players": d["task_a"]["test_players"],
                       "n_train_players": 10, "n_test_players": 1001}}
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False,
                                     encoding="utf-8") as f:
        json.dump(leak, f)
        path = f.name
    try:
        from phase6f_common import load_test_players as ltp
        with pytest.raises(AssertionError):
            ltp(path)
    finally:
        os.unlink(path)


def test_ring_sequences_only_test_players():
    train, test = load_test_players("data/uoft/splits.json")
    seq = load_ring_sequences("data/uoft/feature_ready/player_event_table.jsonl",
                              player_filter=test)
    assert set(seq) <= test
    # 每玩家序列时间升序
    for pid, hands in seq.items():
        ts = [h["ts"] for h in hands]
        assert ts == sorted(ts)


# ---------------- 产物一致性 ----------------
def test_utility_output_schema():
    d = json.load(open("data/uoft/models/phase6/phase6f/utility.json",
                       encoding="utf-8"))
    assert d["config"]["direction"] == "negative=improvement（log-loss/Brier/RMSE 均越低越好）"
    assert "arm_c" in d["config"] and "skipped" in d["config"]["arm_c"]
    for f in ("vpip", "pfr", "three_bet", "cbet", "fold_to_3bet",
              "position_sensitivity"):
        assert f in d["features"]
    assert d["improved_count"] >= 0
    assert d["utility_proven"] in (True, False)


def test_utility_proven_threshold_logic():
    d = json.load(open("data/uoft/models/phase6/phase6f/utility.json",
                       encoding="utf-8"))
    improved = d["improved_count"]
    assert (improved >= 3) == d["utility_proven"]
    assert d["conclusion"] is not None


def test_cbet_low_power_marked_not_dropped():
    """cbet 保留结果且标记 low_power，不得删除。"""
    d = json.load(open("data/uoft/models/phase6/phase6f/utility.json",
                       encoding="utf-8"))
    cbet = d["features"]["cbet"]
    assert "ll_individual" in cbet and "ll_population" in cbet
    assert cbet.get("low_power") is True          # 历史机会中位 <20
    assert cbet["history_opp_median"] < 20
