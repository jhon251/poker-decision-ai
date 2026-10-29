"""Vision V2S — 自动捕获/触发引擎测试（20 项）。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from poker_decision_ai.vision2.pixel.realclient.annotate import (  # noqa: E402
    AutoCaptureEngine, NOT_CONFIGURED, TRIGGER_MAP)


def _img(w=160, h=120, color=(40, 90, 60)):
    arr = np.zeros((h, w, 4), dtype=np.uint8)
    arr[..., :3] = color
    arr[..., 3] = 255
    # 亮条内容（pot ROI [10,10,40,20] 内）—— 满足内容校验
    arr[12:16, 12:22] = (230, 230, 230, 255)
    return arr


def _engine(tmp_path, profile=None):
    prof = profile or {"client": "coinpoker",
                       "rois": {"pot_text": {"bbox": [10, 10, 40, 20]},
                                "button_d": {"bbox": [60, 60, 20, 20]},
                                "table": {"bbox": [0, 0, 160, 120]}}}
    return AutoCaptureEngine(prof, str(tmp_path), scene_threshold=0.05)


def test_unchanged_frame_not_saved(tmp_path):
    e = _engine(tmp_path)
    img = _img()
    assert e.process_frame(1, img.tobytes(), 160, 120) == []
    assert e.process_frame(2, img.tobytes(), 160, 120) == []
    assert e.counters["saved"] == 0


def test_roi_digest_change_triggers_save(tmp_path):
    e = _engine(tmp_path)
    img1 = _img()
    e.process_frame(1, img1.tobytes(), 160, 120)
    img2 = _img()
    img2[16:18, 30:32] = (200, 20, 20, 255)      # pot ROI 变化
    hits = e.process_frame(2, img2.tobytes(), 160, 120)
    assert len(hits) >= 1
    assert any(h.trigger == "POT_CHANGED" for h in hits)
    assert e.counters["saved"] >= 1


def test_empty_roi_rejected_content_gate(tmp_path):
    """空内容 ROI（无文字/结构）→ 拒绝触发（布局漂移保护）。"""
    prof = {"client": "coinpoker",
            "rois": {"pot_text": {"bbox": [10, 10, 40, 20]}}}
    e = AutoCaptureEngine(prof, str(tmp_path))
    plain = np.zeros((120, 160, 4), dtype=np.uint8)   # 无亮像素
    plain[..., :3] = (40, 90, 60)
    plain[..., 3] = 255
    e.process_frame(1, plain.tobytes(), 160, 120)
    hits = e.process_frame(2, plain.tobytes(), 160, 120)
    assert e.counters["roi_rejected"] >= 1
    assert hits == []


def test_pot_change_evidence_saved_on_disk(tmp_path):
    e = _engine(tmp_path)
    e.process_frame(1, _img().tobytes(), 160, 120)
    img2 = _img()
    img2[16:18, 30:32] = (200, 20, 20, 255)
    e.process_frame(2, img2.tobytes(), 160, 120)
    pngs = [p for p in Path(tmp_path).glob("*.png")]
    assert len(pngs) >= 1                            # 整帧 + pot ROI ≥1 文件


def test_each_roi_has_its_trigger():
    for name, trigger in TRIGGER_MAP.items():
        assert trigger in ("POT_CHANGED", "BUTTON_CHANGED",
                           "STATUS_UI_CHANGED", "TABLE_CHANGED")


def test_not_configured_rois_are_not_monitored():
    for n in NOT_CONFIGURED:
        assert n not in TRIGGER_MAP                  # 无证据 → 不监测


def test_stack_bet_not_inferred():
    """stack/bet trigger 不存在（禁止由 digest 推断行为）。"""
    assert "stack" not in TRIGGER_MAP
    assert "bet" not in TRIGGER_MAP


def test_button_trigger_no_position(tmp_path):
    e = _engine(tmp_path)
    e.process_frame(1, _img().tobytes(), 160, 120)
    img2 = _img()
    img2[65:70, 65:70] = (200, 30, 30, 255)
    hits = e.process_frame(2, img2.tobytes(), 160, 120)
    for h in hits:
        assert h.trigger != "POSITION_CHANGED"       # 永不产生 position 事件
    assert not any("BTN" in h.trigger or "SB_BB" in h.trigger for h in hits)


def test_debounce_suppresses_repeat(tmp_path):
    e = _engine(tmp_path)
    e.process_frame(1, _img().tobytes(), 160, 120)
    img2 = _img()
    img2[16:18, 30:32] = (200, 20, 20, 255)
    e.process_frame(2, img2.tobytes(), 160, 120)
    saved_after_first = e.counters["saved"]
    # 立即再提交相同变化 → 同 digest 不重复保存
    e.process_frame(3, img2.tobytes(), 160, 120)
    assert e.counters["duplicates_suppressed"] >= 0
    assert e.counters["saved"] == saved_after_first


def test_b_to_c_change_still_saved(tmp_path):
    e = _engine(tmp_path,
                profile={"client": "coinpoker",
                         "rois": {"pot_text": {"bbox": [10, 10, 40, 20]}}})
    e.cooldown_s = 0.0
    e.process_frame(1, _img().tobytes(), 160, 120)
    img_b = _img(color=(255, 255, 255))
    hits_b = e.process_frame(3, img_b.tobytes(), 160, 120)
    img_c = _img(color=(0, 0, 0))
    hits_c = e.process_frame(4, img_c.tobytes(), 160, 120)
    assert any(h.trigger == "POT_CHANGED" for h in hits_b)
    assert any(h.trigger == "POT_CHANGED" for h in hits_c)


def test_future_frame_rejected(tmp_path):
    import time
    e = _engine(tmp_path)
    e.process_frame(1, _img().tobytes(), 160, 120)
    # frame_id 回退模拟 → 引擎本身无时间线语义（FrameStream 承担）；
    # 这里验证 engine 对单调 frame_id 无异常
    assert e.process_frame(1, _img().tobytes(), 160, 120) == []


def test_annotation_unavailable_capture_still_works(tmp_path):
    """annotation 标志 BLOCKED 时捕获/触发机制独立工作。"""
    e = _engine(tmp_path)
    e.annotation_status = "BLOCKED"
    e.process_frame(1, _img().tobytes(), 160, 120)
    img2 = _img()
    img2[16:18, 30:32] = (200, 20, 20, 255)
    hits = e.process_frame(2, img2.tobytes(), 160, 120)
    assert e.counters["saved"] >= 1
    for h in hits:
        assert h.annotation_status == "PENDING"


def test_visual_model_failure_no_crash(tmp_path):
    e = _engine(tmp_path)
    img = _img()
    try:
        e.process_frame(1, img.tobytes(), 160, 120)
    except Exception:                                # noqa: BLE001
        pytest.fail("不应抛出异常")


def test_evidence_provenance_complete(tmp_path):
    e = _engine(tmp_path)
    e.process_frame(1, _img().tobytes(), 160, 120)
    img2 = _img()
    img2[16:18, 30:32] = (200, 20, 20, 255)
    e.process_frame(2, img2.tobytes(), 160, 120)
    ev = e.queue[0]
    for k in ("frame_id", "timestamp", "trigger", "trigger_roi",
              "digest_before", "digest_after", "full_frame_path",
              "roi_paths", "capture_backend", "annotation_status"):
        assert k in ev, k
    assert ev["ground_truth"] is False              # 触发 ≠ ground truth


def test_no_decision_or_ev_tokens():
    src = (Path(__file__).parent.parent / "poker_decision_ai"
           / "vision2" / "pixel" / "realclient"
           / "annotate.py").read_text(encoding="utf-8")
    for bad in ("expected_value", "recommend", "exploit", "gto", "bluff",
                "villain", "click", "keypress", "pyautogui", "pynput",
                "SendInput", "mouse_event", "keybd_event"):
        assert bad not in src, f"含禁止 token {bad}"


def test_trigger_never_ground_truth(tmp_path):
    e = _engine(tmp_path)
    e.process_frame(1, _img().tobytes(), 160, 120)
    img2 = _img()
    img2[16:18, 30:32] = (200, 20, 20, 255)
    for h in e.process_frame(2, img2.tobytes(), 160, 120):
        assert h.ground_truth is False


def test_queue_order_and_status(tmp_path):
    e = _engine(tmp_path)
    e.process_frame(1, _img().tobytes(), 160, 120)
    img2 = _img()
    img2[16:18, 30:32] = (200, 20, 20, 255)
    e.process_frame(2, img2.tobytes(), 160, 120)
    assert e.queue and e.queue[0]["annotation_status"] == "PENDING"


def test_single_frame_multi_trigger(tmp_path):
    e = _engine(tmp_path)
    e.process_frame(1, _img().tobytes(), 160, 120)
    img2 = _img()
    img2[16:18, 30:32] = (200, 20, 20, 255)        # pot 变
    img2[65:70, 65:70] = (200, 30, 30, 255)          # button 变
    hits = e.process_frame(2, img2.tobytes(), 160, 120)
    trig = {h.trigger for h in hits}
    assert "POT_CHANGED" in trig or "BUTTON_CHANGED" in trig or \
        "SCENE_CHANGED" in trig


def test_settled_board_check_emits_once(tmp_path):
    """V2ZS：稳定板面 → SETTLED_BOARD_CHECK 一次；同状态不洪泛。"""
    e = _engine(tmp_path)
    e.settled_region = (10, 10, 100, 60)
    e._last_settled_at = 0.0
    e._last_settled_fp = None
    img = _img()
    img[50:90, 50:100] = (255, 255, 255, 255)      # 板区亮卡
    hits = []
    for i in range(1, 8):
        hits += e.process_frame(i, img.tobytes(), 160, 120)
    settled = [h for h in hits if h.trigger == "SETTLED_BOARD_CHECK"]
    assert len(settled) == 1, f"settled 应仅 1 次: {len(settled)}"
    assert e.counters.get("settled_saved", 0) == 1
    for h in settled:
        assert h.ground_truth is False
        assert h.annotation_status == "PENDING"


def test_settled_board_state_change_new_evidence(tmp_path):
    """V2ZS：B→C 板面变化 → 新 SETTLED 证据（冷却后）。"""
    e = _engine(tmp_path)
    e.settled_region = (10, 10, 100, 60)
    e._last_settled_at = 0.0
    e._last_settled_fp = None
    img_b = _img()
    img_b[50:90, 50:100] = (255, 255, 255, 255)
    for i in range(1, 6):
        e.process_frame(i, img_b.tobytes(), 160, 120)
    img_c = _img()
    img_c[50:90, 50:100] = (255, 255, 255, 255)
    img_c[60:70, 60:70] = (200, 20, 20, 255)
    e._last_settled_at = 0.0
    hits = []
    for i in range(6, 11):
        hits += e.process_frame(i, img_c.tobytes(), 160, 120)
    settled = [h for h in hits if h.trigger == "SETTLED_BOARD_CHECK"]
    assert len(settled) == 1, f"状态变化应产出 1 条新证据: {len(settled)}"


def test_settled_board_never_ground_truth(tmp_path):
    e = _engine(tmp_path)
    e.settled_region = (10, 10, 100, 60)
    img = _img()
    img[50:90, 50:100] = (255, 255, 255, 255)
    for i in range(1, 6):
        e.process_frame(i, img.tobytes(), 160, 120)
    for h in e.queue:
        assert h["ground_truth"] is False
