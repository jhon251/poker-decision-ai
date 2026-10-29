"""P15 — 1876 模板匹配探针（只读）：用计划 1876 几何 + 既有模板跑真实帧，
判定 board/hero/pot 模板是否在 1876 尺度可复用。不修改任何识别器。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from poker_decision_ai.vision2.pixel.realclient.coinpoker_board import (  # noqa: E402
    CoinPokerBoardRecognizer)
from poker_decision_ai.vision2.pixel.realclient.coinpoker_hero import (  # noqa: E402
    CoinPokerHeroRecognizer)
from poker_decision_ai.vision2.pixel.realclient.cashgame import PotRecognizer  # noqa: E402
from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402


def main() -> int:
    # 直接构造识别器 + 临时打补丁几何（不改源码）
    board = CoinPokerBoardRecognizer()
    # 覆盖模块级槽位常量（探针内联 1876 槽位）
    import poker_decision_ai.vision2.pixel.realclient.coinpoker_board as bm
    bm.SLOT_X0, bm.SLOT_Y0, bm.SLOT_W, bm.SLOT_H = 590, 551, 140, 180

    hero = CoinPokerHeroRecognizer(
        "data/vision2/coinpoker_assets_v2zd_hero",
        "data/vision2/hero_card_layout_v1.json")
    # 覆盖 layout 为 1876 布局（临时）
    hero.layout = {"cards": [
        {"card_index": 0, "card_bbox": [810, 1005, 120, 160],
         "rank_bbox": [828, 1015, 44, 44], "suit_bbox": [825, 1062, 50, 40]},
        {"card_index": 1, "card_bbox": [930, 1005, 120, 160],
         "rank_bbox": [948, 1015, 44, 44], "suit_bbox": [945, 1062, 50, 40]},
    ]}

    pot = PotRecognizer()

    frames = [p for p in sorted(Path("data/vision2/coinpoker_evidence")
                                .glob("*_full.png"))]
    # 只取有 board 卡或 hero 卡的代表帧（快速探针）：步进采样
    cand = [p for p in frames]
    # 快速预筛：先看哪些是 1876 且 board 区有白卡
    n1876 = 0
    board_cand = []
    hero_cand = []
    pot_cand = []
    for p in cand:
        w, h, rgba = decode_png(p.read_bytes())
        if (w, h) != (1876, 1400):
            continue
        n1876 += 1
        arr = np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)
        # board 白卡检测（快）
        bcrop = arr[540:740, 520:1220]
        lum = (bcrop[..., 0].astype(np.int16) * 0.299
               + bcrop[..., 1].astype(np.int16) * 0.587
               + bcrop[..., 2].astype(np.int16) * 0.114)
        if (lum > 200).sum() > 8000:
            board_cand.append(p)
        # hero 白卡检测（快）
        hcrop = arr[990:1200, 700:1120]
        lum2 = (hcrop[..., 0].astype(np.int16) * 0.299
                + hcrop[..., 1].astype(np.int16) * 0.587
                + hcrop[..., 2].astype(np.int16) * 0.114)
        if (lum2 > 200).sum() > 4000:
            hero_cand.append(p)
        # pot 金文本（快）
        pcrop = arr[80:170, 760:1060]
        r = pcrop[..., 0].astype(np.int16)
        g = pcrop[..., 1].astype(np.int16)
        b = pcrop[..., 2].astype(np.int16)
        if ((r > 120) & (g > 85) & (r >= g)).sum() > 200:
            pot_cand.append(p)
    print(f"1876 帧总数: {n1876}")
    print(f"board 候选帧: {len(board_cand)}, hero 候选: {len(hero_cand)}, "
          f"pot 候选: {len(pot_cand)}")

    # board 模板匹配（抽样最多 12 帧）
    board_ok = 0
    board_samples = []
    for p in board_cand[:12]:
        w, h, rgba = decode_png(p.read_bytes())
        br = board.recognize(rgba, w, h, 0)
        bc = [c["raw_text"] for c in br["cards"] if c["status"] == "OK"]
        if bc:
            board_ok += 1
        board_samples.append((p.name, bc, [c.get("status") for c in
                                           br["cards"]]))
    print(f"board 识别成功帧: {board_ok}/{len(board_cand[:12])}")
    for s in board_samples:
        print("   board:", s)

    # hero 模板匹配（抽样最多 12 帧）
    hero_ok = 0
    hero_samples = []
    for p in hero_cand[:12]:
        w, h, rgba = decode_png(p.read_bytes())
        hr = hero.recognize(rgba, w, h, 0)
        hc = hr.get("value") or []
        if hc:
            hero_ok += 1
        hero_samples.append((p.name, hc, [c.get("status") for c in
                                          hr["cards"]]))
    print(f"hero 识别成功帧: {hero_ok}/{len(hero_cand[:12])}")
    for s in hero_samples:
        print("   hero:", s)

    # pot 分割+匹配（抽样最多 12 帧）
    pot_hits = 0
    pot_texts = []
    for p in pot_cand[:12]:
        w, h, rgba = decode_png(p.read_bytes())
        arr = np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)
        roi = arr[80:170, 760:1060]
        luma = (roi[..., 0].astype(np.int16) * 0.299
                + roi[..., 1].astype(np.int16) * 0.587
                + roi[..., 2].astype(np.int16) * 0.114)
        runs = pot._seg(roi)
        if not runs:
            continue
        chars = []
        ok = True
        for (s, e) in runs:
            crop = roi[:, s:e]
            ch, sc, margin = pot._match_char(crop)
            if ch is None or sc is None or sc < pot.min_score \
                    or margin < pot.min_margin:
                ok = False
                break
            chars.append(ch)
        text = "".join(chars) if ok else None
        if text and __import__("re").match(r"^\d{1,3}(,\d{3})*$", text):
            pot_hits += 1
            pot_texts.append(text)
    print(f"pot 组装成功: {pot_hits}/{len(pot_cand[:12])}")
    print("pot 文本样本:", pot_texts[:12])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
