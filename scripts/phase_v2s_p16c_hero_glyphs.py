"""P16-C — HeroCards 1876 rank/suit 几何 + 模板覆盖实测（READ-ONLY）。

GT K♠9♥：card0=K♠(黑), card1=9♥(红)。精确测量 rank/suit 墨迹 bbox，
并测试既有模板（rank{Q,9}, suit{d,s}）对实际字形的匹配（scale 验证）。
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))
from poker_decision_ai.vision2.pixel.realclient.coinpoker_hero import (  # noqa: E402
    CoinPokerHeroRecognizer)
from poker_decision_ai.vision2.pixel.realclient.backends import _match_family  # noqa: E402
from poker_decision_ai.vision2.realtime.image import ImageData  # noqa: E402
from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402

EV = Path("data/vision2/coinpoker_evidence")


def _ink_bbox(roi, mode):
    r = roi[..., 0].astype(np.int16)
    g = roi[..., 1].astype(np.int16)
    b = roi[..., 2].astype(np.int16)
    luma = r * 0.299 + g * 0.587 + b * 0.114
    if mode == "black":
        m = luma < 100
    elif mode == "red":
        m = (r > 100) & (r > g + 40) & (r > b + 40)
    ys, xs = np.where(m)
    if len(ys) < 10:
        return None
    return (xs.min(), ys.min(), xs.max() - xs.min() + 1, ys.max() - ys.min() + 1)


def _crop_glyph(frame, x0, y0, w, h, mode):
    roi = frame[y0:y0 + h, x0:x0 + w]
    r = roi[..., 0].astype(np.int16)
    g = roi[..., 1].astype(np.int16)
    b = roi[..., 2].astype(np.int16)
    luma = r * 0.299 + g * 0.587 + b * 0.114
    if mode == "black":
        m = (luma < 100).astype(np.uint8)
    else:
        m = ((r > 100) & (r > g + 40) & (r > b + 40)).astype(np.uint8)
    rgba = np.stack([m * 255] * 4, axis=-1).astype(np.uint8)
    return ImageData.from_rgba_bytes(w, h, rgba.tobytes())


def main() -> int:
    hero = CoinPokerHeroRecognizer(
        "data/vision2/coinpoker_assets_v2zd_hero",
        "data/vision2/hero_card_layout_v1.json")
    # 颜色族模板
    ranks = hero.ranks
    suits = hero.suits
    print("rank 模板:", sorted(ranks), " suit 模板:", sorted(suits))

    name = "POT_CHANGED_102_1787992505794_full.png"
    w, h, rgba = decode_png((EV / name).read_bytes())
    arr = np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)
    print(f"\n===== {name} =====")
    # 每张卡：rank 区 (卡内 y 0-60, x 0-60)，suit 区 (卡内 y 50-120, x 0-70)
    for ci, (cx0, cx1) in enumerate([(802, 922), (922, 1043)]):
        card = arr[1005:1180, cx0:cx1]
        # rank：顶左
        rank_zone = card[0:65, 0:65]
        # suit：rank 下方
        suit_zone = card[50:125, 0:75]
        for zone_name, zone, mode in [("rank", rank_zone, "red"),
                                      ("rank", rank_zone, "black"),
                                      ("suit", suit_zone, "red"),
                                      ("suit", suit_zone, "black")]:
            bb = _ink_bbox(zone, mode)
            if bb is None:
                continue
            (ix, iy, iw, ih) = bb
            # 绝对坐标
            ax, ay = cx0 + ix, 1005 + (50 if "suit" in zone_name else 0) + iy
            glyph = _crop_glyph(arr, ax, ay, iw, ih, mode)
            # 测试匹配
            fam = ranks if zone_name == "rank" else suits
            best = _match_family(glyph, fam, binary=False)
            print(f"  card{ci} {zone_name}({mode}) bbox=({ax},{ay},{iw},{ih}) "
                  f"-> 模板匹配 {best[0]} score={best[1] if best[1] is not None else None}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
