"""V3F-P14 — Complete State Closure 单元测试。

覆盖：完整字段矩阵（13 字段）、诚实状态（to_call/blind/legal_actions
NOT_READY 且 value=None 非 0）、UNCERTAIN 安全收敛 UNKNOWN、deterministic
repeat、可序列化 round-trip、跨字段不推导、render-class 语义保留。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from poker_decision_ai.vision2.pixel.realclient.complete_state import (  # noqa: E402
    CompleteState, build_complete_state)
from poker_decision_ai.vision2.pixel.realclient.decision_input import (  # noqa: E402
    FieldStatus)
from poker_decision_ai.vision2.pixel.realclient.runtime_snapshot import (  # noqa: E402
    build_snapshot)
from poker_decision_ai.vision2.pixel.realclient.action_observer import (  # noqa: E402
    ActionObserver)
from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402

EV = Path("data/vision2/coinpoker_evidence")
SINCE5 = 134324658401327108

REQUIRED_FIELDS = {
    "hand_context.street", "hand_context.hero_cards", "hand_context.board",
    "position", "pot", "stack", "status", "bet", "action",
    "blind", "to_call", "legal_actions",
}


def _button_frame() -> np.ndarray:
    since_s = SINCE5 / 1e7 - 11644473600
    for p in sorted(EV.glob("*_full.png")):
        if p.stat().st_mtime < since_s:
            continue
        w, h, rgba = decode_png(p.read_bytes())
        arr = np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)
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


def test_full_field_matrix():
    """13 字段全在场（Closure 字段矩阵完整性）。"""
    cs = build_complete_state(build_snapshot({}))
    assert set(cs.fields) == REQUIRED_FIELDS


def test_dependency_unclosed_fields_not_ready():
    """blind/to_call/legal_actions = NOT_READY，value=None（非 0、非猜测）。"""
    cs = build_complete_state(build_snapshot({}))
    for k in ("blind", "to_call", "legal_actions"):
        assert cs.fields[k].status == FieldStatus.NOT_READY
        assert cs.fields[k].value is None


def test_action_field_preserves_render_class_semantics():
    """action 字段透传 ActionObserver（1876 类 RECOGNIZED，其它 EVIDENCE_LIMITED）。"""
    arr = _button_frame()
    obs = ActionObserver()
    obs.observe(arr, arr.shape[1], arr.shape[0])
    snap = build_snapshot({"action_runtime": obs})
    cs = build_complete_state(snap)
    assert cs.fields["action"].status == FieldStatus.RECOGNIZED
    assert cs.fields["action"].value["fold"] is True
    # 无 observer → action NOT_READY
    cs2 = build_complete_state(build_snapshot({}))
    assert cs2.fields["action"].status == FieldStatus.NOT_READY


def test_no_cross_field_derivation():
    """pot 为 UNKNOWN 时，to_call/blind/legal_actions 仍 NOT_READY（不推 Bet）。"""
    cs = build_complete_state(build_snapshot({}))
    assert cs.fields["pot"].status != FieldStatus.RECOGNIZED
    assert cs.fields["to_call"].status == FieldStatus.NOT_READY


def test_uncertain_converges_to_unknown():
    """非枚举状态（如 UNCERTAIN）安全收敛为 UNKNOWN。"""
    snap = {"hero_cards": {"last_cards": None, "last_status": "UNCERTAIN"},
            "board": {"last_cards": None, "last_status": "UNCERTAIN"}}
    cs = build_complete_state(snap)
    assert cs.fields["hand_context.hero_cards"].status == \
        FieldStatus.UNKNOWN


def test_deterministic_repeat():
    a = build_complete_state(build_snapshot({}))
    b = build_complete_state(build_snapshot({}))
    assert json.dumps(a.to_dict(), sort_keys=True) == \
        json.dumps(b.to_dict(), sort_keys=True)


def test_serializable_round_trip():
    cs = build_complete_state(build_snapshot({}))
    d = cs.to_dict()
    json.dumps(d)                       # 可序列化
    cs2 = CompleteState.from_dict(d)
    assert set(cs2.fields) == REQUIRED_FIELDS
    assert cs2.fields["to_call"].status == FieldStatus.NOT_READY


def test_version_and_ground_truth():
    cs = build_complete_state(build_snapshot({}))
    assert cs.version == "1.0"
    assert cs.ground_truth_status == "runtime_state_non_gt"
