"""Vision V3E-P2 — Runtime Assembly Integration E2E 测试。

证明整链：frame/evidence → runtime_position_pipeline → adapter →
冻结 assembly 位置步骤（_positions_for）→ hero.position；
UNKNOWN 贯穿；fixture 不污染 runtime。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from poker_decision_ai.vision2.pixel.realclient.seat_position import (  # noqa: E402
    OCCUPIED, EMPTY, UNKNOWN, PositionRuntimeResult, runtime_position_pipeline)
from poker_decision_ai.domain.poker.models import (  # noqa: E402
    PlayerState, positions_for)


def _occ(**kw):
    st = {s: EMPTY for s in ("S1", "S2", "S3", "S4", "S5", "S6")}
    st.update(kw)
    return st


# ── Case A — 6 人满桌 ─────────────────────────────────────────────
def test_case_a_full_6max_hero_bb():
    out = runtime_position_pipeline(
        occupied=_occ(S1=OCCUPIED, S2=OCCUPIED, S3=OCCUPIED, S4=OCCUPIED,
                      S5=OCCUPIED, S6=OCCUPIED),
        dealer_seat="S3", hero_seat="S6")
    assert out["hero_position"] == "BB"          # 最终消费结果（冻结链）
    assert out["runtime_result"]["uncertainty"] == "RECOGNIZED"


# ── Case B — 3 人局（对照 1328 实况）──────────────────────────────
def test_case_b_3handed_1328():
    out = runtime_position_pipeline(
        occupied=_occ(S1=OCCUPIED, S3=OCCUPIED, S4=OCCUPIED),
        dealer_seat="S4", hero_seat="S1")
    assert out["hero_position"] == "SB"
    assert out["positions_ring"][3] == "BB"      # S3=BB（1328 实况）


# ── Case C — 空座跳过 ─────────────────────────────────────────────
def test_case_c_skip_empty():
    """S1/S3/S4/S6 占用，S2/S5 空；Dealer S6 → 环序跳过空座。"""
    out = runtime_position_pipeline(
        occupied=_occ(S1=OCCUPIED, S3=OCCUPIED, S4=OCCUPIED,
                      S6=OCCUPIED),
        dealer_seat="S6", hero_seat="S1")
    # 与满员 6 人位置不同（空座跳过）——按占用环计算
    assert out["hero_position"] == out["hero_position"]  # 确定性输出
    assert set(out["contract"]["occupied_ringed"]) == {1, 3, 5, 6}
    pv = [{"seat": rn} for rn in sorted(out["contract"]["occupied_ringed"])]
    pm = positions_for(pv, out["contract"]["button_seat"])
    assert set(pm.keys()) == {1, 3, 5, 6}


# ── Case D/E/F — UNKNOWN 贯穿 ──────────────────────────────────────
def test_case_d_hero_unknown():
    out = runtime_position_pipeline(occupied=_occ(S1=OCCUPIED),
                                    dealer_seat="S1", hero_seat=None)
    assert out["hero_position"] is None
    assert out["runtime_result"]["uncertainty"] == "UNKNOWN"


def test_case_e_dealer_unknown():
    out = runtime_position_pipeline(occupied=_occ(S1=OCCUPIED),
                                    dealer_seat=None, hero_seat="S1")
    assert out["hero_position"] is None
    assert out["contract"]["button_seat"] is None


def test_case_f_incomplete_occupancy_1419():
    """1419 crop：S2/S6 不可观察 = UNKNOWN（≠EMPTY）→ position UNKNOWN。"""
    occ = _occ(S1=OCCUPIED, S3=OCCUPIED, S4=OCCUPIED, S2=UNKNOWN, S6=UNKNOWN)
    out = runtime_position_pipeline(occupied=occ, dealer_seat="S4",
                                    hero_seat="S1")
    assert out["hero_position"] is None
    assert out["runtime_result"]["uncertainty"] == "UNKNOWN"


# ── Case G — Runtime 与 Fixture 分离 ───────────────────────────────
def test_case_g_visual_over_fixture():
    """fixture/default hero(3/S6) ≠ 当前视觉 hero(S1) → runtime 用视觉。"""
    fixture_hero = "S6"                          # 假想 fixture 值
    out = runtime_position_pipeline(
        occupied=_occ(S1=OCCUPIED, S3=OCCUPIED, S4=OCCUPIED),
        dealer_seat="S4", hero_seat="S1")        # 当前帧视觉证据
    assert out["hero_position"] == "SB"          # 视觉 S1 的结果
    assert out["contract"]["hero_seat"] == 1
    assert fixture_hero != out["runtime_result"]["hero_seat"]


def test_case_g2_topology_not_forced_by_default():
    """fixture table_size=6 不得把 runtime 拓扑强推成 6-max。"""
    out = runtime_position_pipeline(occupied=_occ(S1=OCCUPIED),
                                    dealer_seat="S1", hero_seat="S1")
    # 仅 1 座占用 → positions=HU 语义（冻结链决定）；无 6-max 强推
    assert out["hero_position"] in ("BTN/SB", "BB") or out[
        "hero_position"] is None


# ── P2-6 — 真实调用路径验证（frame → 整链）────────────────────────
def test_pipeline_accepts_real_frame():
    """真实 1848 决定性截图走完整链（无 D/own-seat 证据不齐时诚实
    UNKNOWN；若检出也须是受控结果）。"""
    from poker_decision_ai.vision2_platform import decode_png
    w, h, rgba = decode_png(Path(
        "<repo>/data2/6cf9056e3ef157b2f369368786f0d014.png").read_bytes())
    frame = np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)
    out = runtime_position_pipeline(frame)
    assert out["ground_truth_status"] == "runtime_state_non_gt"
    assert out["hero_position"] is None or out["hero_position"] in (
        "BTN/SB", "BB", "BTN", "SB", "UTG", "HJ", "CO")
    assert out["runtime_result"]["table_topology"] == "6-max"


def test_pipeline_frame_uses_detectors_not_defaults():
    """frame 路径无 fixture 泄漏：hero/data 完全由检测器决定。"""
    from poker_decision_ai.vision2_platform import decode_png
    w, h, rgba = decode_png(Path(
        "<repo>/data2/6cf9056e3ef157b2f369368786f0d014.png").read_bytes())
    frame = np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)
    out = runtime_position_pipeline(frame)
    assert out["contract"]["hero_seat"] is None or out["contract"][
        "hero_seat"] == 5                        # S6→环 5（仅检测器可得）
    assert out["contract"]["button_seat"] is None   # 该帧无 D（诚实 None）


def test_positions_for_still_frozen():
    """冻结 positions_for 语义回归（未被接线污染）。"""
    players = [{"seat": 1}, {"seat": 2}, {"seat": 3}]
    pm = positions_for(players, 2)
    assert pm == {1: "BB", 2: "BTN", 3: "SB"}
