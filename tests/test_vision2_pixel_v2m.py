"""Vision V2M — Runtime Temporal Vision State Engine 测试。"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from poker_decision_ai.vision2.pixel.realclient.runtime import (  # noqa: E402
    EVENT_KINDS, FieldState, FrameStream, RealtimePokerVisionState,
    TemporalVisionState, VisualEventBus, board_transition, image_digest)


class _Img:
    def __init__(self, data=b"x" * 100):
        self.data = data


def test_frame_stream_dedupe_and_order():
    fs = FrameStream(fps=2.0)
    img = _Img()
    assert fs.accept(1, 0.0, img)["accepted"] is True
    assert fs.accept(2, 0.5, img)["reason"] == "duplicate_frame"
    assert fs.accept(1, 0.9, _Img(b"y" * 100))["reason"] == "out_of_order"


def test_frame_stream_stale_gap():
    fs = FrameStream(fps=2.0, stale_gap_s=5.0)
    fs.accept(1, 0.0, _Img(b"a"))
    r = fs.accept(2, 10.0, _Img(b"b"))
    assert r["accepted"] and r["stale"] is True


def test_image_digest_deterministic():
    assert image_digest(_Img(b"abc")) == image_digest(_Img(b"abc"))
    assert image_digest(_Img(b"abc")) != image_digest(_Img(b"abd"))


def test_temporal_observe_lifecycle():
    t = TemporalVisionState()
    r1 = t.observe("p3.stack", 100, 0.9, 1, "d1")
    assert r1["temporal_status"] == "APPEARED"
    r2 = t.observe("p3.stack", 100, 0.9, 2, "d1")
    assert r2["temporal_status"] == "STABLE"
    r3 = t.observe("p3.stack", 90, 0.9, 3, "d1")
    assert r3["temporal_status"] == "CHANGED"
    assert r3["evidence_frame_ids"] == [3]


def test_temporal_uncertain_does_not_reuse():
    t = TemporalVisionState()
    t.observe("p3.stack", 100, 0.9, 1, "d1")
    r = t.uncertain("p3.stack", 2)
    assert r["value"] is None
    assert r["temporal_status"] == "UNCERTAIN"


def test_temporal_conflict_blocks():
    t = TemporalVisionState()
    t.observe("p3.stack", 100, 0.9, 1, "d1")
    r = t.conflict("p3.stack", 2, {"reason": "x"})
    assert r["blocked"] is True
    assert t.fields["p3.stack"].value is None
    assert t.fields["p3.stack"].source == "conflict_blocked"


def test_temporal_reuse_rules():
    t = TemporalVisionState()
    t.observe("p3.stack", 100, 0.9, 1, "d1")
    assert t.reuse("p3.stack", 2, "d1")["reused"] is True        # 同 ROI
    assert t.reuse("p3.stack", 3, "d1")["reused"] is False       # 已复用不可再链
    t2 = TemporalVisionState()
    t2.observe("p3.stack", 100, 0.9, 1, "d1")
    assert t2.reuse("p3.stack", 2, "dX")["reason"] == "roi_changed"
    t3 = TemporalVisionState()
    assert t3.reuse("p3.stack", 2, "d1")["reason"] == "no_observed"


def test_board_transition_rules():
    assert board_transition(None, 3)["ok"] is True
    assert board_transition(0, 3)["ok"] is True
    assert board_transition(3, 4)["ok"] is True
    assert board_transition(4, 5)["ok"] is True
    assert board_transition(3, 3)["ok"] is True
    assert board_transition(3, 5)["conflict"] is True
    assert board_transition(0, 2)["conflict"] is True


def test_event_bus_whitelist_and_reject():
    bus = VisualEventBus()
    bus.emit("STACK_CHANGED", 1, {"seat": 3})
    assert bus.events[0].kind == "STACK_CHANGED"
    for bad in ("PLAYER_FOLDED", "VILLAIN_BET", "PLAYER_INTENDS"):
        with pytest.raises(ValueError):
            bus.emit(bad, 1)
    assert "PLAYER_FOLDED" not in EVENT_KINDS


def test_field_state_provenance():
    fs = FieldState(name="p3.stack", value=100, source="observed",
                    evidence_frame_ids=[7], roi_digest="d")
    d = fs.to_dict()
    assert d["evidence_frame_ids"] == [7]
    assert d["source"] == "observed"
    assert d["last_update_frame"] is None


def test_stale_check():
    t = TemporalVisionState(stale_gap_s=3)
    t.observe("p3.stack", 100, 0.9, 1, "d1")
    assert t.stale_check(2) == []
    assert t.stale_check(5) == ["p3.stack"]
    assert t.fields["p3.stack"].temporal_status == "STALE"


def test_snapshot_partial_valid():
    s = RealtimePokerVisionState(frame_id=1)
    assert s.vision_status == "PARTIAL_VALID_STATE"
    assert s.pot == {} and s.hero_cards == {}
    d = s.to_dict()
    assert d["frame_id"] == 1 and "players" in d


def test_runtime_module_redlines():
    for name in ("runtime.py", "phase_v2m_replay.py"):
        path = (Path(__file__).parent.parent / "poker_decision_ai"
                / "vision2" / "pixel" / "realclient" / name
                if name == "runtime.py" else
                Path(__file__).parent.parent / "scripts" / name)
        src = path.read_text(encoding="utf-8")
        banned = ("expected_value", "recommend", "exploit", "gto", "bluff",
                  "villain", "folded_player", "cv2", "torch")
        for bad in banned:
            assert bad not in src, f"{name} 含禁止 token {bad}"
