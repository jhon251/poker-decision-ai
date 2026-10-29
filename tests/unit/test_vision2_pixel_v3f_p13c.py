"""V3F-P13-C — Bet Runtime Wiring 单元测试。

覆盖：BetObserver 存在/接线（冻结 BetRecognizer 只读消费）、render-class
按几何选择（1280 recognized / 1419 evidence-limited）、seat ownership（几何键）、
no-bet/UNKNOWN-first、无 cache/previous、快照/DecisionInput 透传、Action 隔离、
回归语义（无 observer → 占位不变）。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from poker_decision_ai.vision2_platform.png_decode import decode_png
from poker_decision_ai.vision2.pixel.realclient.bet_observer import BetObserver
from poker_decision_ai.vision2.pixel.realclient.runtime_snapshot import (
    build_snapshot)
from poker_decision_ai.vision2.pixel.realclient.decision_input import (
    FieldStatus, from_snapshot)

EV = Path("data/vision2/coinpoker_evidence")


def load(name: str) -> np.ndarray:
    w, h, rgba = decode_png((EV / name).read_bytes())
    return np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)


def blank_with(region: tuple, src: np.ndarray) -> np.ndarray:
    x0, y0, x1, y1 = region
    out = np.zeros_like(src)
    out[y0:y1, x0:x1] = src[y0:y1, x0:x1]
    return out


# ------------------------------------------------------------------ wiring

def test_betobserver_consumes_frozen_recognizer_only():
    """BetObserver 只消费 BetRecognizer.recognize——不二次 OCR/NCC/识别。"""
    import inspect
    import poker_decision_ai.vision2.pixel.realclient.bet_observer as bo
    src = inspect.getsource(bo.BetObserver)
    # 只允许一次识别入口调用（recognize），无 re-recognition 调用
    assert ".recognize(frame)" in src
    assert "coinpoker_stack" not in src
    assert "StackRecognizer" not in src
    assert "PotRecognizer" not in src
    assert "_ncc_best" not in src            # 不二次 NCC


def test_betobserver_no_hardcode_no_frame_identity():
    """BetObserver 无 frame id / 金额值硬编码；render-class 仅由 (w,h) 几何。"""
    import inspect
    import poker_decision_ai.vision2.pixel.realclient.bet_observer as bo
    src = inspect.getsource(bo.BetObserver)
    for lit in ("193400", "199680", "102000", "128190", "frame_id",
                "expected_value"):
        assert lit not in src, lit
    assert "(1280, 779)" in src or "1280" in src   # 合法几何选择


# ------------------------------------------------------- render-class + value

def test_betobserver_1280_recognized_values():
    obs = BetObserver()
    for name, zone, value in [
        ("TABLE_CHANGED_100_1787928276694_full.png", "S_TL", 1000),
        ("TABLE_CHANGED_100_1787928276694_full.png", "S_TR", 2000),
        ("TABLE_CHANGED_117_1787928295861_full.png", "S_TL", 193400),
        ("TABLE_CHANGED_117_1787928295861_full.png", "S_TR", 199680),
        ("TABLE_CHANGED_413_1787928698111_full.png", "S_TL", 128190),
        ("TABLE_CHANGED_185_1787928391857_full.png", "S_TL", 102000),
    ]:
        arr = load(name)
        res = obs.observe(arr, arr.shape[1], arr.shape[0])
        seat = res["seats"][zone]
        assert seat["status"] == "RECOGNIZED", (name, zone, seat)
        assert seat["bet"] == value, (name, zone, seat)
        assert res["window"] == "1280"


def test_betobserver_1419_evidence_limited():
    obs = BetObserver()
    frame = np.zeros((1066, 1419, 4), dtype=np.uint8)
    res = obs.observe(frame, 1419, 1066)
    assert res["window"] == "1419"
    for zone, r in res["seats"].items():
        assert r["status"] == "UNKNOWN"
        assert r["bet"] is None
        assert r["reason"] == "no_bet_scale_glyphs"


def test_betobserver_no_bet_semantics():
    """no-bet 帧 → UNKNOWN no_bet_capsule（bet=None，≠0）。"""
    obs = BetObserver()
    arr = load("TABLE_CHANGED_39_1787928191464_full.png")
    res = obs.observe(arr, arr.shape[1], arr.shape[0])
    for zone in ("S_TL", "S_TR"):
        r = res["seats"][zone]
        assert r["status"] == "UNKNOWN"
        assert r["bet"] is None


def test_betobserver_no_previous_no_cache_deterministic():
    """帧间无污染：A→B→A，两次 A 结果一致（无 previous/cache 状态）。"""
    obs = BetObserver()
    a = load("TABLE_CHANGED_100_1787928276694_full.png")
    b = load("TABLE_CHANGED_39_1787928191464_full.png")
    r1 = obs.observe(a, a.shape[1], a.shape[0])
    obs.observe(b, b.shape[1], b.shape[0])          # no-bet 帧
    r3 = obs.observe(a, a.shape[1], a.shape[0])
    assert r1["seats"] == r3["seats"]
    assert r1["seats"]["S_TL"]["bet"] == 1000        # 未被 B 污染


# --------------------------------------------------------------- anti-confusion

def test_betobserver_bet_not_stack_pot_status():
    """只给 Stack/Pot/徽章真实像素 → BetObserver 全 UNKNOWN（无跨字段推导）。"""
    obs = BetObserver()
    frame = load("TABLE_CHANGED_100_1787928276694_full.png")
    for region in ((198, 287, 340, 320),    # stack 文本
                   (843, 353, 1030, 394),   # pot 胶囊
                   (219, 194, 287, 231)):   # 徽章
        only = blank_with(region, frame)
        res = obs.observe(only, only.shape[1], only.shape[0])
        for r in res["seats"].values():
            assert r["status"] == "UNKNOWN"


# ---------------------------------------------------------------- snapshot

def _obs_with_bet(frame: np.ndarray) -> BetObserver:
    obs = BetObserver()
    obs.observe(frame, frame.shape[1], frame.shape[0])
    return obs


def test_snapshot_bet_propagation_and_action_isolation():
    frame = load("TABLE_CHANGED_100_1787928276694_full.png")
    obs = _obs_with_bet(frame)
    snap = build_snapshot({"bet_runtime": obs})
    assert snap["bet"]["last_seats"]["S_TL"]["bet"] == 1000
    assert snap["bet"]["last_seats"]["S_TR"]["bet"] == 2000
    assert snap["bet"]["recognized"] == 1
    # Action 仍 NOT-READY（不因 Bet 成功推进）
    assert snap["action"]["status"] == "UNKNOWN"
    assert snap["action"]["reason"] == "not_ready"


def test_snapshot_bet_placeholder_preserved_without_observer():
    snap = build_snapshot({})
    assert snap["bet"]["status"] == "UNKNOWN"
    assert snap["bet"]["reason"] == "evidence_limited"


def test_decisioninput_bet_propagation():
    frame = load("TABLE_CHANGED_100_1787928276694_full.png")
    snap = build_snapshot({"bet_runtime": _obs_with_bet(frame)})
    di = from_snapshot(snap)
    assert di.evidence["bet"].status == FieldStatus.RECOGNIZED
    assert di.evidence["bet_S_TL"].value == 1000
    assert di.evidence["bet_S_TR"].value == 2000
    assert di.evidence["action"].status == FieldStatus.NOT_READY


def test_decisioninput_bet_evidence_limited_without_observer():
    di = from_snapshot(build_snapshot({}))
    assert di.evidence["bet"].status == FieldStatus.EVIDENCE_LIMITED
    assert di.evidence["bet"].value is None


def test_decisioninput_serializable_with_bet():
    frame = load("TABLE_CHANGED_100_1787928276694_full.png")
    snap = build_snapshot({"bet_runtime": _obs_with_bet(frame)})
    di = from_snapshot(snap)
    json.dumps(di.to_dict())          # 不抛异常 = 可序列化
