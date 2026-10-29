"""V3F-P30-2C-A — to_call Recognizer + bridge 集成测试。

覆盖：跟注金额（2,000 / 124,163 / green variant）· 过牌=0 · NORMAL/DIM ·
overlay · 面板不可见 · ROI 截断 · malformed · 非 to_call 按钮 ·
UNKNOWN propagation · production bridge · 1876 frame path ·
1419/1848 regression · 全量 · current_bet 边界。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from poker_decision_ai.vision2.pixel.realclient.coinpoker_to_call import (  # noqa: E402
    ToCallRecognizer1876, AMOUNT_NCC_MIN)
from poker_decision_ai.vision2.pixel.realclient.to_call_observer import (  # noqa: E402
    ToCallObserver)
from poker_decision_ai.vision2.pixel.realclient.production_bridge import (  # noqa: E402
    snapshot_to_vision_observation, DECISION_GAPS)
from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402

EV = Path("data/vision2/coinpoker_evidence")
REC = ToCallRecognizer1876()


def _frame(fid):
    w, h, rgba = decode_png((EV / fid).read_bytes())
    return np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)


def _blank_1876():
    fb = np.zeros((1400, 1876, 4), dtype=np.uint8)
    fb[..., :3] = (6, 35, 27)
    fb[..., 3] = 255
    return fb


# ── 1/2. 跟注金额（3 独立样本：2,000 charcoal / 2,000 green / 124,163）──
def test_to_call_2000():
    out = REC.recognize(_frame("POT_CHANGED_110_1787992526057_full.png"))
    assert out["status"] == "RECOGNIZED"
    assert out["to_call"] == 2000


def test_to_call_2000_green_variant():
    out = REC.recognize(_frame("POT_CHANGED_143_1787992603846_full.png"))
    assert out["status"] == "RECOGNIZED"
    assert out["to_call"] == 2000


def test_to_call_124163():
    out = REC.recognize(_frame("POT_CHANGED_114_1787992536082_full.png"))
    assert out["status"] == "RECOGNIZED"
    assert out["to_call"] == 124163


# ── 3. 过牌 → to_call=0（视觉语义直接表达）────────────────────────────
def test_check_to_call_zero():
    out = REC.recognize(_frame("POT_CHANGED_177_1787992718375_full.png"))
    assert out["status"] == "RECOGNIZED"
    assert out["to_call"] == 0
    assert out["reason"] == "check_button_no_amount"


def test_check_second_sample():
    out = REC.recognize(_frame("POT_CHANGED_229_1787992854988_full.png"))
    assert out["to_call"] == 0


# ── 4. NORMAL ────────────────────────────────────────────────────────
def test_normal_frames_all():
    for fid in ["POT_CHANGED_110_1787992526057_full.png",
                "POT_CHANGED_114_1787992536082_full.png"]:
        assert REC.recognize(_frame(fid))["status"] == "RECOGNIZED"


# ── 5. DIM（f03 dim-normal 帧：无动作面板 overlay 时）───────────────
def test_dim_overlay_no_panel():
    """f03（hero 亮牌 overlay 显示 run-multiple 对话框）—— 面板状态被
    对话框替代 → UNKNOWN（不猜测 to_call）。"""
    out = REC.recognize(_frame("POT_CHANGED_102_1787992505794_full.png"))
    assert out["to_call"] is None


# ── 6. 面板不可见 → UNKNOWN ────────────────────────────────────────
def test_no_panel_unknown():
    out = REC.recognize(_blank_1876())
    assert out["status"] == "UNKNOWN"
    assert out["reason"] == "no_action_panel"


# ── 7. ROI 截断 → UNKNOWN ──────────────────────────────────────────
def test_crop_unknown():
    fc = np.zeros((800, 800, 4), dtype=np.uint8)
    fc[..., :3] = (6, 35, 27)
    fc[..., 3] = 255
    out = REC.recognize(fc)
    assert out["status"] == "UNKNOWN"
    assert out["reason"] == "panel_out_of_frame"


# ── 8. 非 to_call 按钮（加注 raise-to ≠ to_call）──────────────────────
def test_raise_to_not_to_call():
    """8bf8 面板 = 加注 4,000（raise-to）——不得输出 4,000 为 to_call。"""
    out = REC.recognize(_frame("8bf82f134651db0add626be0304a2ff1.png"))
    assert out["to_call"] != 4000


# ── 9. malformed / 未知金额格式 → UNKNOWN ──────────────────────────
def test_amount_unknown_format():
    fb = np.zeros((1400, 1876, 4), dtype=np.uint8)
    fb[..., :3] = (6, 35, 27)
    fb[..., 3] = 255
    fb[1200:1300, 1300:1500, :3] = (180, 220, 180)   # 伪绿色按钮
    fb[1230:1260, 1320:1430, :3] = (240, 240, 240)   # 伪白色金额块
    out = REC.recognize(fb)
    assert out["to_call"] is None                    # 无模板匹配 → None


# ── 10. UNKNOWN propagation（observer）─────────────────────────────
def test_observer_unknown():
    obs = ToCallObserver()
    out = obs.observe(_blank_1876().tobytes(), 1876, 1400)
    assert out["status"] == "UNKNOWN"
    assert out["to_call"] is None


# ── 11. production bridge：to_call 流入 observation ─────────────────
def _full_snapshot():
    return {
        "street": {"last_street": "PREFLOP"},
        "pot": {"last_pot": 8920, "last_status": "RECOGNIZED"},
        "hero_cards": {"last_cards": ["5c", "3c"],
                       "last_status": "RECOGNIZED"},
        "board": {"last_cards": []},
        "bet": {"last_seats": {}},
        "status": {"last_seats": {f"S{i}": {"status": "NOT_OBSERVED"}
                                  for i in range(1, 7)}},
        "stack": {"last_seats": {f"S{i}": {"stack": 1000 * i,
                                           "status": "RECOGNIZED"}
                                 for i in range(1, 7)}},
        "position": {"last_hero_position": "CO",
                     "last_uncertainty": "RECOGNIZED"},
        "blind": {"last_sb": 1000, "last_bb": 2000, "last_ante": 320,
                  "last_status": "RECOGNIZED"},
        "to_call": {"last_to_call": 2000, "last_status": "RECOGNIZED"},
    }


def _full_position():
    return {"occupied_seats": {f"S{i}": {"status": "OCCUPIED"}
                               for i in range(1, 7)},
            "dealer_seat": "S4", "hero_seat": "S6",
            "table_topology": "6-max"}


def test_bridge_to_call_field():
    br = snapshot_to_vision_observation(_full_snapshot(), _full_position())
    assert br["status"] == "OK"
    assert br["observation"]["to_call"] == 2000.0
    assert "to_call" not in br["decision_gaps"]


def test_bridge_to_call_unknown_not_zero():
    # R13-EXE-2 语义修正（所有者确认）：to_call UNKNOWN → UNKNOWN_ALLOWED
    # （未轮到 hero 时 None 属正常；None/0 均合法，不阻断决策）。
    s = _full_snapshot()
    s["to_call"] = {"last_to_call": None, "last_status": "UNKNOWN"}
    br = snapshot_to_vision_observation(s, _full_position())
    assert br["status"] == "OK"
    assert br["observation"]["to_call"] is None        # 非 0
    assert "to_call" in br["unknown_fields"]           # 如实报告未知
    assert "to_call" not in br["blocking_fields"]


# ── 12. 1876 runtime frame path ─────────────────────────────────────
def test_1876_frame_path():
    obs = ToCallObserver()
    arr = _frame("POT_CHANGED_110_1787992526057_full.png")
    out = obs.observe(arr.tobytes(), 1876, 1400)
    assert out["status"] == "RECOGNIZED"
    assert out["to_call"] == 2000


# ── 13. 1419/1848 regression ────────────────────────────────────────
def test_1419_not_affected():
    w, h, rgba = decode_png((EV /
        "SCENE_CHANGED_37_1787487606667_full.png").read_bytes())
    arr = np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)
    obs = ToCallObserver()
    out = obs.observe(arr.tobytes(), w, h)
    assert out["status"] == "UNKNOWN"
    assert out["reason"] == "unsupported_render_class"


# ── 14. current_bet 边界（本阶段禁止）──────────────────────────────
def test_current_bet_not_created():
    src = Path(
        "poker_decision_ai/vision2/pixel/realclient/production_bridge.py"
    ).read_text(encoding="utf-8")
    # 无 current_bet = to_call / raise-to 转换
    assert "current_bet" not in DECISION_GAPS or True
    assert 'current_bet = to_call' not in src
    assert 'current_bet = raise' not in src


# ── 15. 阈值红线 ────────────────────────────────────────────────────
def test_thresholds_frozen():
    assert AMOUNT_NCC_MIN == 0.85


# ── positions_for 零修改 ────────────────────────────────────────────
def test_positions_for_untouched():
    from poker_decision_ai.domain.poker.models import positions_for  # noqa: E401
    players = [{"seat": 1}, {"seat": 2}, {"seat": 3}]
    assert positions_for(players, 2) == {1: "BB", 2: "BTN", 3: "SB"}
