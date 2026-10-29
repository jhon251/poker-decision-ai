"""Vision V2ZG — CoinPoker Board Runtime Integration 测试。"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from poker_decision_ai.vision2.pixel.realclient.coinpoker_board import (  # noqa: E402
    DEFAULT_ASSET_DIRS, CoinPokerBoardRecognizer)


def _rec():
    return CoinPokerBoardRecognizer(list(DEFAULT_ASSET_DIRS))


def test_template_load():
    r = _rec()
    assert r.available() is True
    cov = r.coverage()
    # v2zh 完整字形覆盖（v2x/v2y 碎片兜底）
    assert set(cov["ranks"]) >= {"2", "7", "8", "A", "K", "T"}
    assert set(cov["suits"]) >= {"c", "d", "h", "s"}


def test_slot_crop_oob_safe():
    import numpy as np
    r = _rec()
    frame = np.zeros((100, 100, 4), dtype=np.uint8)
    res = r.recognize_slot(frame, 4)          # 槽位越界 → UNCERTAIN 不崩溃
    assert res["status"] == "UNCERTAIN"


def test_color_separation():
    """颜色族机制（合成字形确定性验证）；碎片模板另记质量缺陷。"""
    import numpy as np
    from poker_decision_ai.vision2.realtime.image import ImageData
    r = _rec()
    red = np.zeros((20, 20, 4), dtype=np.uint8)
    red[..., :3] = (255, 255, 255)
    red[..., 3] = 255
    red[6:14, 6:14] = (215, 50, 60, 255)          # 红字形
    red_img = ImageData.from_rgba_bytes(20, 20, red.tobytes())
    black = red.copy()
    black[6:14, 6:14] = (20, 20, 20, 255)          # 黑字形
    black_img = ImageData.from_rgba_bytes(20, 20, black.tobytes())
    assert r._red_dominant(red_img)          # numpy bool 真值
    assert not r._red_dominant(black_img)


def test_uncertainty_path():
    import numpy as np
    blank = np.zeros((1066, 1419, 4), dtype=np.uint8)
    blank[..., :3] = (40, 90, 60)
    blank[..., 3] = 255
    res = _rec().recognize(blank.tobytes(), 1419, 1066, 1)
    assert res["status"] == "UNCERTAIN"
    for c in res["cards"]:
        assert c["rank"] is None and c["suit"] is None


def test_replay_556_187_rank_consistency():
    """回放一致性（rank 级）：识别出的卡 rank 必须 ∈ GT rank 集合。

    当前已知缺陷（如实）：suit 判别弱（2c→d、Ks→d 等）——模板质量
    与角标花色定位待修；rank 为强判据。
    """
    from poker_decision_ai.vision2_platform import decode_png
    import numpy as np
    r = _rec()
    cases = [("SCENE_CHANGED_188_1787489345211_full.png", "2c Ac 8d 9c Ks"),
             ("SCENE_CHANGED_556_1787489611440_full.png", "5h Td 7s")]
    total_ok = 0
    for fname, truth in cases:
        w, h, rgba = decode_png(Path(
            f"data/vision2/coinpoker_evidence/{fname}").read_bytes())
        truth_ranks = {t[0] for t in truth.split()}
        res = r.recognize(rgba, w, h, 1)
        for c in res["cards"]:
            if c["status"] == "OK":
                assert c["rank"] in truth_ranks, f"rank 误判: {c['raw_text']}"
                total_ok += 1
    assert total_ok >= 1, "期望至少 1 卡识别（rank 级）"


def test_output_fields():
    import numpy as np
    r = _rec()
    blank = np.zeros((1066, 1419, 4), dtype=np.uint8)
    blank[..., 3] = 255
    res = r.recognize(blank.tobytes(), 1419, 1066, 7)
    for c in res["cards"]:
        for k in ("rank", "suit", "confidence", "raw_text", "status",
                  "source", "evidence_frame_id", "slot_index"):
            assert k in c, k


def test_source_redlines():
    src = (Path(__file__).parent.parent / "poker_decision_ai" / "vision2"
           / "pixel" / "realclient"
           / "coinpoker_board.py").read_text(encoding="utf-8")
    for bad in ("expected_value", "recommend", "exploit", "gto", "bluff",
                "pyautogui", "pynput", "click", "keypress"):
        assert bad not in src, f"含禁止 token {bad}"


def test_missing_suit_path():
    """无 suit 模板（含中心） → null + UNCERTAIN（不猜）。"""
    import numpy as np
    r = _rec()
    r.suits = {}
    r.center_suits = {}
    blank = np.zeros((1066, 1419, 4), dtype=np.uint8)
    blank[..., :3] = (40, 90, 60)
    blank[..., 3] = 255
    res = r.recognize(blank.tobytes(), 1419, 1066, 1)
    assert res["status"] == "UNCERTAIN"
    for c in res["cards"]:
        assert c["suit"] is None


def test_556_5h_center_evidence_ok():
    """V2ZJ 中心大花色模板：SCENE_556 slot0 = 5h（首例完全正确板卡）。"""
    from poker_decision_ai.vision2_platform import decode_png
    w, h, rgba = decode_png(Path(
        "data/vision2/coinpoker_evidence/"
        "SCENE_CHANGED_556_1787489611440_full.png").read_bytes())
    res = _rec().recognize(rgba, w, h, 1)
    c0 = res["cards"][0]
    assert c0["status"] == "OK"
    assert c0["raw_text"] == "5h", c0
    assert c0["confidence"] >= 0.7
    ev = c0["evidence"]
    assert ev["center"]["suit"] == "h"


def test_ks_no_longer_kd():
    """Ks 此前误判 Kd —— V2ZJ 后冲突阻断 → null+UNCERTAIN。"""
    from poker_decision_ai.vision2_platform import decode_png
    w, h, rgba = decode_png(Path(
        "data/vision2/coinpoker_evidence/"
        "SCENE_CHANGED_188_1787489345211_full.png").read_bytes())
    res = _rec().recognize(rgba, w, h, 1)
    c4 = res["cards"][4]
    if c4["status"] == "OK":
        assert c4["raw_text"] == "Ks", f"Ks 误判: {c4['raw_text']}"
    else:
        assert c4.get("reason") in ("suit_conflict", "suit_gate",
                                    "rank_gate")


def test_2c_correctly_recognized():
    """V2ZV：2c 已从安全阻断恢复为**正确识别**（conf≥0.7，中心 Pad 修复）。"""
    from poker_decision_ai.vision2_platform import decode_png
    w, h, rgba = decode_png(Path(
        "data/vision2/coinpoker_evidence/"
        "SCENE_CHANGED_188_1787489345211_full.png").read_bytes())
    res = _rec().recognize(rgba, w, h, 1)
    c0 = res["cards"][0]
    assert c0["status"] == "OK"
    assert c0["raw_text"] == "2c", c0
    assert c0["confidence"] >= 0.7


def test_complete_T_template_loaded():
    """S5-1/2：完整 T 模板（39×46）覆盖碎片；碎片 T 不再作为 production。"""
    r = _rec()
    t = r.templates["T"]
    assert t.width >= 22 and t.height >= 40, f"T 仍为碎片: {t.width}x{t.height}"
    assert r._red_dominant(t)          # numpy bool 真值


def test_Th_cross_slot_eliminated():
    """S5-3/4 更新（V2ZV）：碎片时代 Th 吸引 conf≈0.74-0.78 → 现已消除；
    完整模板+中心 Pad 后的 Th 匹配 conf≥0.9（真实 10♥，非吸引）。"""
    import glob
    from poker_decision_ai.vision2_platform import decode_png
    r = _rec()
    low_conf_th = 0
    for p in glob.glob(
            "data/vision2/coinpoker_evidence/SCENE_CHANGED_*.png"):
        w, h, rgba = decode_png(Path(p).read_bytes())
        res = r.recognize(rgba, w, h, 1)
        low_conf_th += sum(1 for c in res["cards"]
                           if c["status"] == "OK" and c["raw_text"] == "Th"
                           and c["confidence"] < 0.9)
    assert low_conf_th == 0, f"Th 低置信吸引残留 {low_conf_th} 次"


def test_slot_spatial_isolation():
    import numpy as np
    from poker_decision_ai.vision2.pixel.realclient.coinpoker_board import (
        SLOT_W, SLOT_X0, SLOT_Y0)
    # 构造层面：相邻槽位搜索区不重叠
    for i in range(4):
        assert SLOT_X0 + (i + 1) * SLOT_W >= SLOT_X0 + i * SLOT_W + 140
    # 行为层面：slot1 的结果不受 slot0 内容影响
    r = _rec()
    frame = np.zeros((1066, 1419, 4), dtype=np.uint8)
    frame[..., :3] = (40, 90, 60)
    frame[..., 3] = 255
    res_before = r.recognize_slot(frame, 1)
    frame[SLOT_Y0:SLOT_Y0 + 100, SLOT_X0:SLOT_X0 + 100] = (255, 255, 255,
                                                          255)
    res_after = r.recognize_slot(frame, 1)
    assert (res_before["rank"], res_before["suit"]) == (
        res_after["rank"], res_after["suit"])


def test_corner_only_disabled_by_default():
    """corner-only 输出默认禁用（无验证正确先例）；配置开启为显式实验。"""
    import numpy as np
    from poker_decision_ai.vision2.pixel.realclient.coinpoker_board import (
        CORNER_ONLY_SUIT_ALLOWED)
    assert CORNER_ONLY_SUIT_ALLOWED is False
    # 配置开启时仅影响 corner-only 路径（不改阈值）——不做正确性承诺
    r = CoinPokerBoardRecognizer(list(DEFAULT_ASSET_DIRS),
                                 corner_only_allowed=True)
    blank = np.zeros((1066, 1419, 4), dtype=np.uint8)
    blank[..., 3] = 255
    res = r.recognize(blank.tobytes(), 1419, 1066, 1)
    assert res["status"] == "UNCERTAIN"          # 空帧仍 UNCERTAIN
