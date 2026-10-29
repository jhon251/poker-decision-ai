"""V3F-INTEGRATION-FIX-1 — occupancy 形状归一化修复测试。

覆盖：真实 frame path 从视觉 occupancy 得到正确 hero_position（3/4/5/6
人局）+ 空座 + UNKNOWN + dealer/hero 缺失 + 不在 occupied + 不完整 +
overlay UNKNOWN + 归一化函数幂等性 + positions_for 零修改。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from poker_decision_ai.vision2.pixel.realclient.seat_position import (  # noqa: E402
    _occupied_status_map, compute_positions, runtime_position_pipeline,
    to_assembly_contract)
from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402

EV = Path("data/vision2/coinpoker_evidence")


def _frame(fid):
    w, h, rgba = decode_png((EV / fid).read_bytes())
    return np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)


def _dictocc(statuses):
    return {s: {"status": st, "confidence": 0.9} for s, st in statuses.items()}


# ── 1. 归一化函数：dict-of-dicts → string map（幂等）─────────────────
def test_normalize_dict_of_dicts():
    assert _occupied_status_map(_dictocc({"S1": "OCCUPIED",
                                          "S2": "UNKNOWN",
                                          "S3": "EMPTY"})) == {
        "S1": "OCCUPIED", "S2": "UNKNOWN", "S3": "EMPTY"}


def test_normalize_idempotent_strings():
    inp = {"S1": "OCCUPIED", "S2": "UNKNOWN"}
    assert _occupied_status_map(inp) == inp


def test_normalize_empty_none():
    assert _occupied_status_map(None) == {}
    assert _occupied_status_map({}) == {}


# ── 2. 真实 frame path → 正确 hero_position（重点）──────────────────
def test_frame_path_6_handed():
    """f01 全 6 座占用 → RECOGNIZED + hero_position（非恒 UNKNOWN）。"""
    out = runtime_position_pipeline(frame=_frame(
        "BUTTON_CHANGED_167_1787992682749_full.png"))
    rr = out["runtime_result"]
    assert rr["uncertainty"] == "RECOGNIZED"
    assert out["hero_position"] is not None


def test_frame_path_hero_position_is_label():
    """hero_position 为合法位置标签。"""
    out = runtime_position_pipeline(frame=_frame(
        "BUTTON_CHANGED_167_1787992682749_full.png"))
    assert out["hero_position"] in (
        "BTN/SB", "BB", "BTN", "SB", "UTG", "HJ", "CO", "MP")


def test_frame_path_8bf8_recognized():
    out = runtime_position_pipeline(frame=_frame(
        "8bf82f134651db0add626be0304a2ff1.png"))
    assert out["runtime_result"]["uncertainty"] == "RECOGNIZED"
    assert out["hero_position"] is not None


# ── 3. compute_positions（string contract 原样）+ 多人数 ─────────────
def test_compute_3_handed():
    occ = {"S1": "OCCUPIED", "S2": "OCCUPIED", "S3": "OCCUPIED"}
    r = compute_positions(occ, "S2", "S3")
    assert r["uncertainty"] == "RECOGNIZED"


def test_compute_4_handed():
    occ = {"S1": "OCCUPIED", "S2": "OCCUPIED", "S3": "OCCUPIED",
           "S4": "OCCUPIED"}
    r = compute_positions(occ, "S4", "S1")
    assert r["uncertainty"] == "RECOGNIZED"


def test_compute_5_handed():
    occ = {"S1": "OCCUPIED", "S2": "OCCUPIED", "S3": "OCCUPIED",
           "S4": "OCCUPIED", "S5": "OCCUPIED"}
    r = compute_positions(occ, "S5", "S2")
    assert r["uncertainty"] == "RECOGNIZED"


def test_compute_6_handed():
    occ = {f"S{i}": "OCCUPIED" for i in range(1, 7)}
    r = compute_positions(occ, "S1", "S6")
    assert r["uncertainty"] == "RECOGNIZED"


# ── 4. 空座跳过 ────────────────────────────────────────────────────
def test_empty_seat_skipped():
    occ = {"S1": "OCCUPIED", "S2": "OCCUPIED", "S3": "OCCUPIED",
           "S4": "EMPTY", "S5": "OCCUPIED", "S6": "OCCUPIED"}
    r = compute_positions(occ, "S1", "S6")
    assert r["uncertainty"] == "RECOGNIZED"
    # 5 人（S4 空）→ 仍有 hero_position
    assert r["hero_position"] is not None


# ── 5. UNKNOWN 传播（不转 EMPTY/OCCUPIED）──────────────────────────
def test_unknown_propagates_not_empty():
    occ = {"S1": "OCCUPIED", "S2": "OCCUPIED", "S3": "UNKNOWN",
           "S4": "OCCUPIED", "S5": "OCCUPIED", "S6": "OCCUPIED"}
    r = compute_positions(occ, "S1", "S6")
    assert r["uncertainty"] == "UNKNOWN"
    assert r["reason"] == "occupied_incomplete_S3"


# ── 6. dealer 缺失 / hero 缺失 ─────────────────────────────────────
def test_dealer_missing():
    r = compute_positions({f"S{i}": "OCCUPIED" for i in range(1, 7)},
                          None, "S6")
    assert r["uncertainty"] == "UNKNOWN"
    assert r["reason"] == "dealer_or_hero_missing"


def test_hero_missing():
    r = compute_positions({f"S{i}": "OCCUPIED" for i in range(1, 7)},
                          "S1", None)
    assert r["uncertainty"] == "UNKNOWN"
    assert r["reason"] == "dealer_or_hero_missing"


# ── 7. dealer/hero 不在 occupied ───────────────────────────────────
def test_dealer_not_occupied():
    occ = {"S1": "OCCUPIED", "S2": "OCCUPIED", "S3": "OCCUPIED"}
    r = compute_positions(occ, "S4", "S1")
    assert r["uncertainty"] == "UNKNOWN"
    assert r["reason"] == "dealer_not_occupied"


def test_hero_not_occupied():
    occ = {"S1": "OCCUPIED", "S2": "OCCUPIED", "S3": "OCCUPIED"}
    r = compute_positions(occ, "S1", "S5")
    assert r["uncertainty"] == "UNKNOWN"
    assert r["reason"] == "hero_not_occupied"


# ── 8. 不完整 occupancy → UNKNOWN ──────────────────────────────────
def test_incomplete_occupancy_unknown():
    """某座 UNKNOWN（证据不足）混入 → 整体 UNKNOWN（不猜测）。"""
    r = compute_positions({"S1": "OCCUPIED", "S3": "OCCUPIED",
                           "S4": "OCCUPIED", "S5": "UNKNOWN"}, "S4", "S1")
    assert r["uncertainty"] == "UNKNOWN"
    assert r["reason"] == "occupied_incomplete_S5"


# ── 9. overlay 导致 UNKNOWN（真实帧 f03 S6 被亮牌遮）────────────────
def test_overlay_occupancy_unknown_frame():
    out = runtime_position_pipeline(frame=_frame(
        "POT_CHANGED_102_1787992505794_full.png"))
    rr = out["runtime_result"]
    assert rr["occupied_seats"]["S6"].get("status") == "UNKNOWN"
    assert rr["uncertainty"] == "UNKNOWN"          # 诚实传播（非形状 bug）
    assert out["hero_position"] is None


# ── 10. to_assembly_contract 归一化一致性 ──────────────────────────
def test_contract_from_dict_of_dicts_via_pipeline():
    """frame path 的 contract 也要有正确的 occupied_ringed（非空）。"""
    out = runtime_position_pipeline(frame=_frame(
        "BUTTON_CHANGED_167_1787992682749_full.png"))
    assert out["contract"]["occupied_ringed"]      # 修复后非空


# ── 11. positions_for 零修改 ───────────────────────────────────────
def test_positions_for_untouched():
    from poker_decision_ai.domain.poker.models import positions_for  # noqa: E401
    players = [{"seat": 1}, {"seat": 2}, {"seat": 3}]
    assert positions_for(players, 2) == {1: "BB", 2: "BTN", 3: "SB"}
