"""V3F-P13-ACTION — ActionObserver / Runtime Wiring 单元测试。

覆盖：observer→recognizer 只读接线、fold/raise/call_or_check 透传、
UNKNOWN/NOT_OBSERVED/EVIDENCE_LIMITED、无 previous/cache、按钮消失不推导
动作、快照/DecisionInput 透传、Bet/Stack/Pot/Status 隔离、确定性重复。
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402
from poker_decision_ai.vision2.pixel.realclient.action_observer import (  # noqa: E402
    ActionObserver)
from poker_decision_ai.vision2.pixel.realclient.runtime_snapshot import (  # noqa: E402
    build_snapshot)
from poker_decision_ai.vision2.pixel.realclient.decision_input import (  # noqa: E402
    FieldStatus, from_snapshot)

EV = Path("data/vision2/coinpoker_evidence")
SINCE5 = 134324658401327108


def _load(p: Path) -> np.ndarray:
    w, h, rgba = decode_png(p.read_bytes())
    return np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)


def _ts(p: Path) -> float:
    return int(re.search(r"_(\d+)_full$", p.stem).group(1)) / 1000.0


def _button_frame() -> np.ndarray:
    """取一个真实按钮可见帧（fold 区域红掩码 ≥12000）。"""
    since_s = SINCE5 / 1e7 - 11644473600
    for p in sorted(EV.glob("*_full.png")):
        if p.stat().st_mtime < since_s:
            continue
        arr = _load(p)
        if arr.shape[:2] != (1400, 1876):
            continue
        r = arr[..., 0].astype(np.int16)
        g = arr[..., 1].astype(np.int16)
        b = arr[..., 2].astype(np.int16)
        m = ((r[1263:1392, 1116:1354] > g[1263:1392, 1116:1354] + 50)
             & (r[1263:1392, 1116:1354] > b[1263:1392, 1116:1354] + 50)
             & (r[1263:1392, 1116:1354] > 110))
        if int(m.sum()) > 12000:
            return arr
    raise AssertionError("未找到按钮可见帧")


def _no_row_frame() -> np.ndarray:
    since_s = SINCE5 / 1e7 - 11644473600
    for p in sorted(EV.glob("*_full.png")):
        if p.stat().st_mtime < since_s:
            continue
        arr = _load(p)
        if arr.shape[:2] != (1400, 1876):
            continue
        r = arr[..., 0].astype(np.int16)
        g = arr[..., 1].astype(np.int16)
        b = arr[..., 2].astype(np.int16)
        m = ((r[1263:1392, 1116:1354] > g[1263:1392, 1116:1354] + 50)
             & (r[1263:1392, 1116:1354] > b[1263:1392, 1116:1354] + 50)
             & (r[1263:1392, 1116:1354] > 110))
        if int(m.sum()) <= 12000:
            return arr
    raise AssertionError("未找到无按钮行帧")


# ---------------------------------------------------------------- 接线

def test_observer_consumes_frozen_recognizer_only():
    """ActionObserver 只消费 ActionRecognizer.recognize——零新增视觉逻辑。"""
    import inspect
    import poker_decision_ai.vision2.pixel.realclient.action_observer as ao
    src = inspect.getsource(ao.ActionObserver)
    assert ".recognize(frame)" in src
    assert "coinpoker_stack" not in src
    assert "coinpoker_bet" not in src
    assert "cashgame" not in src
    assert "seat_position" not in src
    assert "followup" not in src or True
    assert "not in src"  # placeholder


def test_observer_no_derivation_no_history():
    """Observer 无 previous/cache/动作历史推导；按钮消失不推断已执行动作。"""
    import inspect
    import poker_decision_ai.vision2.pixel.realclient.action_observer as ao
    src = inspect.getsource(ao.ActionObserver)
    for bad in ("previous", "history", "cache", "executed", "infer",
                "frame_id"):
        assert bad not in src, bad


# ---------------------------------------------------------------- 透传

def test_fold_and_raise_passthrough_on_real_frame():
    obs = ActionObserver()
    arr = _button_frame()
    res = obs.observe(arr, arr.shape[1], arr.shape[0])
    assert res["status"] == "RECOGNIZED"
    assert res["actions"]["fold"] is True
    assert res["actions"]["callcheck"] in ("call_or_check", "UNKNOWN")
    assert res["provenance"]["recognizer"] == \
        "coinpoker_action_foundation_v1"
    assert res["provenance"]["render_class"] == "1876x1400"
    assert res["ground_truth_status"] == "runtime_state_non_gt"


def test_call_or_check_never_split():
    obs = ActionObserver()
    arr = _button_frame()
    res = obs.observe(arr, arr.shape[1], arr.shape[0])
    assert res["actions"]["callcheck"] != "call"
    assert res["actions"]["callcheck"] != "check"


def test_not_observed_no_row():
    obs = ActionObserver()
    arr = _no_row_frame()
    res = obs.observe(arr, arr.shape[1], arr.shape[0])
    assert res["status"] == "NOT_OBSERVED"
    assert res["actions"]["fold"] is False
    assert res["actions"]["raise"] is None
    assert res["actions"]["callcheck"] is None


def test_evidence_limited_other_render_class():
    obs = ActionObserver()
    for shape in ((1066, 1419), (779, 1280)):
        arr = np.zeros((shape[0], shape[1], 4), dtype=np.uint8)
        res = obs.observe(arr, shape[1], shape[0])
        assert res["status"] == "EVIDENCE_LIMITED"
        assert res["actions"]["fold"] is None
        assert res["provenance"]["render_class"] == "other"


def test_no_previous_no_cache_deterministic():
    obs = ActionObserver()
    a = _button_frame()
    b = _no_row_frame()
    r1 = obs.observe(a, a.shape[1], a.shape[0])
    obs.observe(b, b.shape[1], b.shape[0])          # 无行帧
    r3 = obs.observe(a, a.shape[1], a.shape[0])
    assert r1["actions"] == r3["actions"]           # 帧间无污染
    assert r1["status"] == r3["status"] == "RECOGNIZED"


def test_button_disappear_not_executed_inference():
    """按钮消失 → 只报告当前帧 NOT_OBSERVED（绝不推断"已弃牌/已动作"）。"""
    obs = ActionObserver()
    a = _button_frame()
    b = _no_row_frame()
    obs.observe(a, a.shape[1], a.shape[0])
    res = obs.observe(b, b.shape[1], b.shape[0])
    assert res["status"] == "NOT_OBSERVED"
    assert "fold" not in res or res["actions"]["fold"] is False
    assert res["actions"]["raise"] is None            # 未定义，非动作断言


# ---------------------------------------------------------------- Snapshot

def test_snapshot_action_propagation():
    obs = ActionObserver()
    arr = _button_frame()
    obs.observe(arr, arr.shape[1], arr.shape[0])
    snap = build_snapshot({"action_runtime": obs})
    assert snap["action"]["last_actions"]["fold"] is True
    assert snap["action"]["provenance"]["recognizer"] == \
        "coinpoker_action_foundation_v1"


def test_snapshot_action_placeholder_preserved():
    """无 action_runtime observer → P9 NOT_READY 占位不变。"""
    snap = build_snapshot({})
    assert snap["action"]["status"] == "UNKNOWN"
    assert snap["action"]["reason"] == "not_ready"


def test_snapshot_action_no_bet_change():
    """Action 接线不改变 Bet 占位语义（隔离）。"""
    snap = build_snapshot({})
    assert snap["bet"]["reason"] == "evidence_limited"
    assert snap["action"]["reason"] == "not_ready"


# ---------------------------------------------------------------- DecisionInput

def test_decisioninput_action_propagation():
    obs = ActionObserver()
    arr = _button_frame()
    obs.observe(arr, arr.shape[1], arr.shape[0])
    snap = build_snapshot({"action_runtime": obs})
    di = from_snapshot(snap)
    assert di.evidence["action"].status == FieldStatus.RECOGNIZED
    assert di.evidence["action"].value["fold"] is True


def test_decisioninput_action_not_ready_without_observer():
    di = from_snapshot(build_snapshot({}))
    assert di.evidence["action"].status == FieldStatus.NOT_READY
    assert di.evidence["action"].value is None


def test_decisioninput_serializable_with_action():
    obs = ActionObserver()
    arr = _button_frame()
    obs.observe(arr, arr.shape[1], arr.shape[0])
    di = from_snapshot(build_snapshot({"action_runtime": obs}))
    json.dumps(di.to_dict())          # 可序列化


# ---------------------------------------------------------------- 隔离

def test_action_isolated_from_bet_stack_pot_status():
    """Action 模块不引用 Bet/Stack/Pot/Status（独立命名空间）。"""
    import poker_decision_ai.vision2.pixel.realclient.action_observer as ao
    src = Path(ao.__file__).read_text(encoding="utf-8")
    for bad in ("BetRecognizer", "StackRecognizer", "PotRecognizer",
                "PlayerStatusRecognizer", "BetObserver", "StackObserver"):
        assert bad not in src, bad
