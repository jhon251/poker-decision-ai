"""V3F-P30-2B — Betting Header Recognizer + bridge 集成测试。

覆盖：1K/2K(320) · 逗号格式 · M 格式 · DIM · header 缺失 · ROI 截断 ·
蓝 SB/BB 徽章非金额 · malformed · UNKNOWN propagation · bridge ·
1876 frame path · 1419 regression · positions_for 零修改。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from poker_decision_ai.vision2.pixel.realclient.coinpoker_betting_header import (  # noqa: E402
    BettingHeaderRecognizer1876, _to_num, STAKE_NCC_MIN)
from poker_decision_ai.vision2.pixel.realclient.betting_header_observer import (  # noqa: E402
    BettingHeaderObserver)
from poker_decision_ai.vision2.pixel.realclient.production_bridge import (  # noqa: E402
    snapshot_to_vision_observation)
from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402

EV = Path("data/vision2/coinpoker_evidence")
REC = BettingHeaderRecognizer1876()


def _frame(fid):
    w, h, rgba = decode_png((EV / fid).read_bytes())
    return np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)


def _blank_1876():
    fb = np.zeros((1400, 1876, 4), dtype=np.uint8)
    fb[..., :3] = (6, 35, 27)
    fb[..., 3] = 255
    return fb


# ── 1. 1K/2K/320 正例 ───────────────────────────────────────────────
def test_1k_2k_320_recognized():
    out = REC.recognize(_frame(
        "BUTTON_CHANGED_167_1787992682749_full.png"))
    assert out["status"] == "RECOGNIZED"
    assert out["sb"] == 1000
    assert out["bb"] == 2000
    assert out["ante"] == 320


def test_all_4_gt_frames():
    for fid in ["BUTTON_CHANGED_167_1787992682749_full.png",
                "8bf82f134651db0add626be0304a2ff1.png",
                "POT_CHANGED_110_1787992526057_full.png",
                "POT_CHANGED_102_1787992505794_full.png"]:
        out = REC.recognize(_frame(fid))
        assert out["status"] == "RECOGNIZED", fid
        assert (out["sb"], out["bb"], out["ante"]) == (1000, 2000, 320), fid


# ── 2/3. 金额格式解析（_to_num 纯函数）──────────────────────────────
def test_to_num_k_format():
    assert _to_num("1K") == 1000
    assert _to_num("2K") == 2000
    assert _to_num("5K") == 5000


def test_to_num_comma_format():
    assert _to_num("169,760") == 169760
    assert _to_num("1,000") == 1000


def test_to_num_m_format():
    assert _to_num("1.28M") == 1280000
    assert _to_num("2M") == 2000000


def test_to_num_integer():
    assert _to_num("320") == 320
    assert _to_num("0") == 0


def test_to_num_malformed_none():
    assert _to_num("") is None
    assert _to_num("abc") is None
    assert _to_num("1.5K") is None     # 未覆盖小数 K
    # V-4（Round 15）语义升级 test-side 更新：裸小数 '1.28' 旧断言 None
    # 已过时——微级别 ₮ 计量（'0.05'/'0.1'）要求裸小数 → 货币单位 float
    # （单位语义见 coinpoker_betting_header._to_num docstring，零生产迁就）。
    assert _to_num("1.28") == 1.28
    assert _to_num("0.0.5") is None    # 多点仍拒绝
    assert _to_num("5.") is None       # 空小数部仍拒绝


# ── 4. DIM：header 在 DIM 帧（darker header 但白字仍 >120）───────────
def test_dim_header_recognizable():
    # f03 为 dim 帧（bg≈10.8）但 header 白字仍亮
    out = REC.recognize(_frame("POT_CHANGED_102_1787992505794_full.png"))
    assert out["status"] == "RECOGNIZED"


# ── 5. header 缺失 → UNKNOWN ────────────────────────────────────────
def test_no_header_unknown():
    out = REC.recognize(_blank_1876())
    assert out["status"] == "UNKNOWN"
    assert out["reason"] == "no_header_text"


# ── 6. ROI 截断 → UNKNOWN ───────────────────────────────────────────
def test_crop_truncated_unknown():
    fc = np.zeros((800, 800, 4), dtype=np.uint8)
    fc[..., :3] = (6, 35, 27)
    fc[..., 3] = 255
    out = REC.recognize(fc)
    assert out["status"] == "UNKNOWN"


# ── 7/8. 蓝色 SB/BB 徽章非金额（空间隔离——座锚区，非 header ROI）───
def test_sb_badge_not_amount():
    """SB 蓝徽章在座锚区 (138,326) —— header ROI 不含 → detector 不读。"""
    out = REC.recognize(_frame(
        "8bf82f134651db0add626be0304a2ff1.png"))
    # header ROI (9,7,176,14) 不含徽章；检测值只来自 header
    assert out["sb"] == 1000   # 来自 header 1K，非徽章
    # 徽章位置不在 header ROI（断言几何）
    assert not (138 < 9 + 176 and 326 < 7 + 14)


def test_bb_badge_not_amount():
    out = REC.recognize(_frame(
        "8bf82f134651db0add626be0304a2ff1.png"))
    assert out["bb"] == 2000   # header 2K，非 BB 徽章


# ── 9. malformed token → UNKNOWN（mask 中无文本）────────────────────
def test_malformed_token_unknown():
    fb = np.zeros((1400, 1876, 4), dtype=np.uint8)
    fb[..., :3] = (6, 35, 27)
    fb[..., 3] = 255
    fb[7:21, 9:185, :3] = (240, 240, 240)   # 伪 header（空白亮块，非文本）
    out = REC.recognize(fb)
    assert out["status"] == "UNKNOWN"       # 无文本结构 → UNKNOWN


# ── 10. UNKNOWN propagation（observer）─────────────────────────────
def test_observer_unknown_propagates():
    obs = BettingHeaderObserver()
    out = obs.observe(_blank_1876().tobytes(), 1876, 1400)
    assert out["status"] == "UNKNOWN"
    assert out["sb"] is None and out["bb"] is None and out["ante"] is None


# ── 11. production bridge：blinds/ante 流入 observation ────────────
def test_bridge_blinds_ante_fields():
    snap = {
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
        # V3F-P30-2C-A：to_call 字段（全已知提供以避免 UNAVAILABLE）
        "to_call": {"last_to_call": 2000, "last_status": "RECOGNIZED"},
    }
    pr = {"occupied_seats": {f"S{i}": {"status": "OCCUPIED"}
                             for i in range(1, 7)},
          "dealer_seat": "S4", "hero_seat": "S6",
          "table_topology": "6-max"}
    br = snapshot_to_vision_observation(snap, pr)
    assert br["status"] == "OK"
    obs = br["observation"]
    assert obs["blinds"] == {"sb": 1000.0, "bb": 2000.0}
    assert obs["ante"] == 320.0
    assert "blinds" not in br["decision_gaps"]


def test_bridge_blinds_unknown_not_created():
    snap = {"street": {"last_street": "PREFLOP"},
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
            "blind": {"last_sb": None, "last_bb": None, "last_ante": None,
                      "last_status": "UNKNOWN"}}
    pr = {"occupied_seats": {f"S{i}": {"status": "OCCUPIED"}
                             for i in range(1, 7)},
          "dealer_seat": "S4", "hero_seat": "S6",
          "table_topology": "6-max"}
    br = snapshot_to_vision_observation(snap, pr)
    assert br["status"] == "DECISION_UNAVAILABLE"
    obs = br["observation"]
    assert obs.get("blinds") is None
    assert obs.get("ante") is None          # 非默认 0/模拟


# ── 12. 1876 runtime frame path（via observer）─────────────────────
def test_1876_frame_path():
    obs = BettingHeaderObserver()
    arr = _frame("8bf82f134651db0add626be0304a2ff1.png")
    out = obs.observe(arr.tobytes(), 1876, 1400)
    assert out["status"] == "RECOGNIZED"
    assert out["sb"] == 1000 and out["bb"] == 2000 and out["ante"] == 320


# ── 13. 1419 regression ─────────────────────────────────────────────
def test_1419_not_affected():
    """1419 帧无 header 检测（visual 对象只存在于 1876）→ UNKNOWN；
    且不影响既有 1419 路径。"""
    w, h, rgba = decode_png((EV /
        "SCENE_CHANGED_37_1787487606667_full.png").read_bytes())
    arr = np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)
    obs = BettingHeaderObserver()
    out = obs.observe(arr.tobytes(), w, h)
    assert out["status"] == "UNKNOWN"
    assert out["reason"] == "unsupported_render_class"


# ── 14. 阈值红线 ───────────────────────────────────────────────────
def test_thresholds_frozen():
    assert STAKE_NCC_MIN == 0.90


# ── positions_for 零修改 ────────────────────────────────────────────
def test_positions_for_untouched():
    from poker_decision_ai.domain.poker.models import positions_for  # noqa: E401
    players = [{"seat": 1}, {"seat": 2}, {"seat": 3}]
    assert positions_for(players, 2) == {1: "BB", 2: "BTN", 3: "SB"}
