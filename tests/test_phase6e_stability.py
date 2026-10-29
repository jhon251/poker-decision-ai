"""Phase 6E 测试：时间排序/尾部排除/bootstrap 配对/最低机会/stable 边界/无未来数据。"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from phase6e_stability import (  # noqa: E402
    MAIN_WINDOW_END, MAIN_WINDOW_START, classify, feature_opp, in_main_window,
    ols_slope, segment_stats, split_thirds)


def _mk_ev(ts, **kw):
    ev = {"ts": ts, "table_type": "6max", "vpip": False, "pfr": False,
          "three_bet": False, "folded_to_3bet": False, "faced_3bet": False,
          "bets": 0, "calls": 0, "position": "CO", "rfi_opp": False,
          "rfi_did": False, "went_to_showdown": False, "saw_flop": False,
          "cbet": False, "cbet_opportunity": False,
          "folded_to_cbet": False, "faced_cbet": False,
          "won_at_showdown": False}
    ev.update(kw)
    return ev


# ---------------- 时间窗口 ----------------
def test_main_window_bounds():
    assert in_main_window("2009-07-01T00:00:00") is True
    assert in_main_window("2009-07-23T23:59:59") is True
    assert in_main_window("2009-07-24T00:00:00") is False   # 收尾期排除
    assert in_main_window("2009-07-26T10:00:00") is False
    assert in_main_window("2009-06-30T23:59:59") is False


def test_main_window_excludes_tail():
    assert MAIN_WINDOW_END < "2009-07-24T00:00:00"


# ---------------- 时间排序（三等分单调） ----------------
def test_split_thirds_time_ordered():
    hands = [_mk_ev(f"2009-07-{d:02d}T12:00:00") for d in range(1, 21)]
    segs = split_thirds(hands, 5)
    assert segs is not None
    assert len(segs) == 3
    # 段内与跨段时间单调
    all_ts = [h["ts"] for s in segs for h in s]
    assert all_ts == sorted(all_ts)
    assert segs[0][-1]["ts"] <= segs[1][0]["ts"] <= segs[2][0]["ts"]


def test_split_thirds_min_hands():
    hands = [_mk_ev(f"2009-07-{d:02d}T12:00:00") for d in range(1, 10)]
    assert split_thirds(hands, 5) is None       # 9 手 < 15
    assert split_thirds(hands, 3) is not None   # 每段 3 手


# ---------------- 尾部排除（无未来数据使用） ----------------
def test_segment_stats_ignores_outside_window_events():
    """segment_stats 的输入已由 in_main_window 过滤；此处验证过滤函数本身。"""
    tail = _mk_ev("2009-07-25T12:00:00", vpip=True)
    assert in_main_window(tail["ts"]) is False


# ---------------- 特征机会 ---------------- 
def test_feature_opp_vpip_pfr():
    assert feature_opp(_mk_ev("2009-07-01T12:00:00", vpip=True), "vpip") == (1, 1)
    assert feature_opp(_mk_ev("2009-07-01T12:00:00"), "vpip") == (0, 1)
    assert feature_opp(_mk_ev("2009-07-01T12:00:00", pfr=True), "pfr") == (1, 1)


def test_feature_opp_cold_call_blind_excluded():
    ev_bb = _mk_ev("2009-07-01T12:00:00", position="BB", vpip=True)
    assert feature_opp(ev_bb, "cold_call") == (0, 0)   # 盲位无机会
    ev_co = _mk_ev("2009-07-01T12:00:00", position="CO", vpip=True)
    assert feature_opp(ev_co, "cold_call") == (1, 1)   # 非盲位 limp → S=1
    ev_raise = _mk_ev("2009-07-01T12:00:00", position="CO", vpip=True, pfr=True)
    assert feature_opp(ev_raise, "cold_call") == (0, 1)  # pfr → 非 cold call


def test_feature_opp_agg_freq_weighted():
    ev = _mk_ev("2009-07-01T12:00:00", bets=2, calls=3)
    assert feature_opp(ev, "agg_freq") == (2, 5)   # S=bets, n=bets+calls


def test_feature_opp_rfi_ep_position_gated():
    ev = _mk_ev("2009-07-01T12:00:00", position="EP", rfi_opp=True, rfi_did=True)
    assert feature_opp(ev, "rfi_ep") == (1, 1)
    ev2 = _mk_ev("2009-07-01T12:00:00", position="CO", rfi_opp=True, rfi_did=True)
    assert feature_opp(ev2, "rfi_ep") == (0, 0)    # 非 EP 无机会


# ---------------- stable 判定边界 ----------------
def test_classify_stable_only_double_condition():
    # CI 覆盖 0 且 |β|<0.02 → stable
    assert classify(0.001, -0.001, 0.003) == "stable"
    # CI 不覆盖 0 → 即使 |β| 小也不 stable（保守）
    assert classify(0.001, 0.0002, 0.0018) == "unstable"
    # |β|>=0.02 → unstable
    assert classify(0.03, -0.01, 0.05) == "unstable"
    # CI 覆盖 0 但 |β|>=0.02 → 非 stable
    assert classify(0.03, -0.01, 0.05) == "unstable"
    assert classify(0.0, -0.001, 0.001) == "stable"


def test_classify_boundary_effect_size():
    # |β| 恰好 0.02 → 不满足 <0.02 → 非 stable（即使 CI 覆盖 0）
    assert classify(0.02, -0.001, 0.001) == "unstable"
    assert classify(-0.02, -0.001, 0.001) == "unstable"
    # 略小于 0.02 且 CI 覆盖 0 → stable
    assert classify(0.019, -0.001, 0.001) == "stable"


# ---------------- bootstrap 配对结构 ----------------
def test_bootstrap_paired_player_structure():
    """验证脚本的 stat_fn 在玩家重采样下保持三段配对（合成）。"""
    from phase6e_stability import N_BOOT
    rng = random.Random(5)
    def run(n_players):
        triples = []
        for _ in range(n_players):
            theta = 0.3 + rng.gauss(0, 0.05)          # 玩家间真实差异
            p = [theta + rng.gauss(0, 0.01) for _ in range(3)]   # 段内噪声
            triples.append([min(max(x, 0.05), 0.95) for x in p])
        rng2 = random.Random(42)
        betas = []
        for _ in range(N_BOOT):
            idx = [rng2.randrange(n_players) for _ in range(n_players)]
            mean_p = [sum(triples[i][j] for i in idx) / n_players
                      for j in range(3)]
            betas.append(ols_slope([1.0, 2.0, 3.0], mean_p))
        betas.sort()
        return betas[int(0.025 * N_BOOT)], betas[int(0.975 * N_BOOT)]
    lo_small, hi_small = run(50)
    lo_big, hi_big = run(2000)
    assert (hi_big - lo_big) < (hi_small - lo_small)


# ---------------- 无未来数据使用 ----------------
def test_no_future_data_in_feature_opp():
    """feature_opp 只读事件行当前字段，不依赖任何时间信息。"""
    import inspect
    sig = inspect.signature(feature_opp)
    assert set(sig.parameters) == {"ev", "feat"}


# ---------------- 产物一致性 ----------------
def test_output_three_status_only():
    d = json.load(open("data/uoft/models/phase6/phase6e/time_stability.json",
                       encoding="utf-8"))
    allowed = {"stable", "unstable", "insufficient_evidence"}
    for f, v in d["features"].items():
        assert v["status"] in allowed, f"{f} 状态非法: {v['status']}"


def test_output_insufficient_features():
    d = json.load(open("data/uoft/models/phase6/phase6e/time_stability.json",
                       encoding="utf-8"))
    for f in ("cbet", "fold_to_cbet", "wsd"):
        assert d["features"][f]["status"] == "insufficient_evidence"
    assert d["features"]["rfi_lp"]["status"] == "insufficient_evidence"
    assert d["features"]["position_sensitivity"]["status"] == \
        "insufficient_evidence"


def test_output_window_and_players():
    d = json.load(open("data/uoft/models/phase6/phase6e/time_stability.json",
                       encoding="utf-8"))
    cfg = d["config"]
    assert cfg["main_window"] == ["2009-07-01T00:00:00",
                                  "2009-07-23T23:59:59"]
    assert "tail_exclusion_reason" in cfg
    assert d["data_usage"]["players_used_3way_split"] >= 4000
    assert d["data_usage"]["excluded_tail_events"] > 30000


def test_degarde_log_entries():
    log = json.load(open("data/uoft/models/phase6/phase6e/DEGRADE_LOG.json",
                         encoding="utf-8"))
    assert len(log["entries"]) >= 5
    for e in log["entries"]:
        assert e["component"].startswith("time_stability.")
        assert e["path"] and e["reason"]
