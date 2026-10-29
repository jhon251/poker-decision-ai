"""Vision V2ZE — CoinPoker Hero Runtime Integration 测试。"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from poker_decision_ai.vision2.pixel.realclient.coinpoker_hero import (  # noqa: E402
    CoinPokerHeroRecognizer)


def _recognizer():
    return CoinPokerHeroRecognizer("data/vision2/coinpoker_assets_v2zd_hero",
                                   "data/vision2/hero_card_layout_v1.json")


def test_template_load():
    r = _recognizer()
    assert r.available() is True
    assert set(r.ranks) == {"Q", "9"}
    assert set(r.suits) == {"s", "d"}


def test_bbox_crop_and_recognition_scene188():
    """SCENE_188 回放：hero_rank=[Q,9] hero_suit=[s,d]（一致性验证）。"""
    from poker_decision_ai.vision2_platform import decode_png
    import numpy as np
    w, h, rgba = decode_png(Path(
        "data/vision2/coinpoker_evidence/"
        "SCENE_CHANGED_188_1787489345211_full.png").read_bytes())
    r = _recognizer()
    res = r.recognize(rgba, w, h, 188)
    assert res["status"] == "OK"
    # 真值（V2V 视觉读取）：Q s / 9 d —— 一致性断言（非 accuracy）
    truth = [("Q", "s"), ("9", "d")]
    got = [(c["rank"], c["suit"]) for c in res["cards"]]
    matching = [(t, g) for t, g in zip(truth, got) if t == g]
    assert len(matching) >= 1, f"无任何卡匹配: {got}"
    print(f"SCENE_188 hero 一致性: {got} vs truth {truth} → "
          f"{len(matching)}/2 卡一致")


def test_recognition_output_fields():
    from poker_decision_ai.vision2_platform import decode_png
    w, h, rgba = decode_png(Path(
        "data/vision2/coinpoker_evidence/"
        "SCENE_CHANGED_188_1787489345211_full.png").read_bytes())
    res = _recognizer().recognize(rgba, w, h, 188)
    for c in res["cards"]:
        for k in ("rank", "suit", "confidence", "raw_text", "status",
                  "source", "evidence_frame_id", "card_bbox"):
            assert k in c, k
        if c["status"] == "OK":
            assert c["evidence_frame_id"] == 188
            assert c["source"].startswith("coinpoker_hero")
            assert 0.7 <= c["confidence"] <= 1.0


def test_uncertainty_path():
    """无字形帧 → null + UNCERTAIN（不猜测）。"""
    import numpy as np
    blank = np.zeros((1066, 1419, 4), dtype=np.uint8)
    blank[..., :3] = (40, 90, 60)
    blank[..., 3] = 255
    res = _recognizer().recognize(blank.tobytes(), 1419, 1066, 999)
    assert res["status"] == "UNCERTAIN"
    for c in res["cards"]:
        assert c["rank"] is None and c["suit"] is None
        assert c["status"] == "UNCERTAIN"


def test_unrecognized_card_null():
    from poker_decision_ai.vision2.pixel.realclient.coinpoker_hero import (
        CoinPokerHeroRecognizer)
    r = _recognizer()
    # layout 中 rank/suit bbox 为 null 的卡 → null（模拟未知）
    layout = dict(r.layout)
    layout["cards"] = [{"card_index": 9, "card_bbox": [0, 0, 1, 1],
                        "rank_bbox": None, "suit_bbox": None,
                        "label": None, "verified": "none"}]
    r.layout = layout
    import numpy as np
    blank = np.zeros((100, 100, 4), dtype=np.uint8)
    blank[..., 3] = 255
    res = r.recognize_card(blank, layout["cards"][0])
    assert res["rank"] is None and res["suit"] is None
    assert res["status"] == "UNCERTAIN"


def test_source_redlines():
    src = (Path(__file__).parent.parent / "poker_decision_ai" / "vision2"
           / "pixel" / "realclient"
           / "coinpoker_hero.py").read_text(encoding="utf-8")
    for bad in ("expected_value", "recommend", "exploit", "gto", "bluff",
                "pyautogui", "pynput", "click", "keypress"):
        assert bad not in src, f"含禁止 token {bad}"


def test_color_family_separation():
    """红/黑族分离：Q/9 模板分属 black/red。"""
    r = _recognizer()
    assert r._color_family(r.ranks["Q"]) == "black"
    assert r._color_family(r.ranks["9"]) == "red"
    assert r._color_family(r.suits["s"]) == "black"
    assert r._color_family(r.suits["d"]) == "red"


def test_same_rank_different_color():
    """同字形异色：红字形不得匹配黑模板（族过滤后不匹配）。"""
    r = _recognizer()
    fam_red = r._family_templates(r.ranks, "red")
    fam_black = r._family_templates(r.ranks, "black")
    assert set(fam_red) == {"9"}
    assert set(fam_black) == {"Q"}
    assert set(r._family_templates(r.suits, "red")) == {"d"}
    assert set(r._family_templates(r.suits, "black")) == {"s"}


def test_9d_replay_ok_with_color_gate():
    """SCENE_188 颜色门控回放：Qs 与 9d 均 OK（严格门槛不变）。"""
    from poker_decision_ai.vision2_platform import decode_png
    w, h, rgba = decode_png(Path(
        "data/vision2/coinpoker_evidence/"
        "SCENE_CHANGED_188_1787489345211_full.png").read_bytes())
    res = _recognizer().recognize(rgba, w, h, 188)
    assert res["status"] == "OK"
    got = [(c["rank"], c["suit"]) for c in res["cards"]]
    assert got == [("Q", "s"), ("9", "d")], got
    for c in res["cards"]:
        assert c["confidence"] >= 0.7
