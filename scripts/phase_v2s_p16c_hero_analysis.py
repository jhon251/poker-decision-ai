"""P16-C — HeroCards 1876 READ-ONLY PRE-AUDIT。

用 GT K♠9♥ @ (802-1043,1005-1180) 实测：hero 卡 ROI / rank/suit 几何 /
颜色 / 既有模板覆盖（v2zd = Q/s/9/d）。
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))
from poker_decision_ai.vision2.pixel.realclient.coinpoker_hero import (  # noqa: E402
    CoinPokerHeroRecognizer)
from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402

EV = Path("data/vision2/coinpoker_evidence")


def _tight_ink(roi, mode="dark"):
    """roi → tight 墨迹 bbox（mode=dark: luma<150 或红; mode=bright: luma>150）。"""
    r = roi[..., 0].astype(np.int16)
    g = roi[..., 1].astype(np.int16)
    b = roi[..., 2].astype(np.int16)
    luma = r * 0.299 + g * 0.587 + b * 0.114
    if mode == "dark":
        m = (luma < 150) & ((r < 240) | (g < 240))
    elif mode == "red":
        m = (r > 100) & (r > g + 40) & (r > b + 40)
    else:
        m = luma > 150
    ys, xs = np.where(m)
    if len(ys) < 8:
        return None
    return (xs.min(), ys.min(), xs.max() - xs.min() + 1, ys.max() - ys.min() + 1)


def main() -> int:
    hero = CoinPokerHeroRecognizer(
        "data/vision2/coinpoker_assets_v2zd_hero",
        "data/vision2/hero_card_layout_v1.json")
    # 现有模板覆盖
    print("现有模板 ranks:", sorted(hero.ranks.keys()))
    print("现有模板 suits:", sorted(hero.suits.keys()))
    print("现有 layout cards bbox:", [c["card_bbox"] for c in hero.layout["cards"]])

    names = [
        "BUTTON_CHANGED_167_1787992682749_full.png",
        "POT_CHANGED_102_1787992505794_full.png",
        "BUTTON_CHANGED_598_1787993753587_full.png",
    ]
    for name in names:
        p = EV / name
        w, h, rgba = decode_png(p.read_bytes())
        if (w, h) != (1876, 1400):
            continue
        arr = np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)
        print(f"\n===== {name} =====")
        # hero 卡区 (802-1043, 1005-1180)
        hc = arr[1005:1180, 802:1043]
        r = hc[..., 0].astype(np.int16)
        g = hc[..., 1].astype(np.int16)
        b = hc[..., 2].astype(np.int16)
        luma = r * 0.299 + g * 0.587 + b * 0.114
        white = (luma > 200).sum()
        red = ((r > 100) & (r > g + 40) & (r > b + 40)).sum()
        print(f"  hero 区白卡像素={int(white)} 红像素={int(red)}")
        # 两张卡分别分析（card0 x 802-922, card1 x 922-1043）
        for ci, (cx0, cx1) in enumerate([(802, 922), (922, 1043)]):
            card = arr[1005:1180, cx0:cx1]
            # rank 区（顶部左，卡内 y 0-60）
            rank = card[0:60, 0:60]
            tb = _tight_ink(rank, "dark")
            rb = _tight_ink(rank, "red")
            # suit 区（rank 下方，卡内 y 55-120）
            suit = card[55:120, 0:70]
            sb = _tight_ink(suit, "dark")
            srb = _tight_ink(suit, "red")
            print(f"  card{ci} (x{cx0}-{cx1}): rank_dark={tb} rank_red={rb} "
                  f"suit_dark={sb} suit_red={srb}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
