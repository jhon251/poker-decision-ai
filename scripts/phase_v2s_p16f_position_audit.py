"""P16-F — Position 1876 READ-ONLY PRE-AUDIT。

A/C: 1848→1876 几何迁移（board/hero 卡偏移实测）；
B: 占座(金)/Dealer(红D)/empty 候选检测；
E: gate 路径审计（源码文本，不运行）。
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))
from poker_decision_ai.vision2.pixel.realclient.coinpoker_stack import (  # noqa: E402
    SEAT_ANCHORS_1848)
from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402

EV = Path("data/vision2/coinpoker_evidence")


def _board_x0(arr):
    """board 区 y540-740 白卡左缘（最小 x）。"""
    crop = arr[540:740, 400:1300]
    r = crop[..., 0].astype(np.int16)
    g = crop[..., 1].astype(np.int16)
    b = crop[..., 2].astype(np.int16)
    luma = r * 0.299 + g * 0.587 + b * 0.114
    white = luma > 200
    cols = white.sum(axis=0)
    active = np.where(cols > 40)[0]
    if len(active) == 0:
        return None
    # 找第一个簇
    runs = []
    run = None
    for i in active:
        if run is None or i == run[-1] + 1:
            run = (run[0] if run else i, i)
        else:
            runs.append(run)
            run = (i, i)
    if run:
        runs.append(run)
    return 400 + runs[0][0]


def main() -> int:
    # 1848/1846 截图 board x0
    for name in ["data2/004d23eb6e87077fbb4377d5fb63913f.png",
                 "data2/f4652edf931adbdb56579e4c9eb5e971.png"]:
        p = Path(name)
        w, h, rgba = decode_png(p.read_bytes())
        arr = np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)
        bx = _board_x0(arr)
        print(f"{name} ({w}x{h}) board_x0={bx}")
    # 1876 帧 board x0
    for name in ["BUTTON_CHANGED_167_1787992682749_full.png",
                 "POT_CHANGED_12_1787995290794_full.png"]:
        p = EV / name
        w, h, rgba = decode_png(p.read_bytes())
        arr = np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)
        bx = _board_x0(arr)
        print(f"{name} ({w}x{h}) board_x0={bx}")

    # B: 1876 帧 per-seat 候选（金=占座 / 红D=dealer）
    for name in ["BUTTON_CHANGED_167_1787992682749_full.png",
                 "POT_CHANGED_12_1787995290794_full.png"]:
        p = EV / name
        w, h, rgba = decode_png(p.read_bytes())
        arr = np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)
        r = arr[..., 0].astype(np.int16)
        g = arr[..., 1].astype(np.int16)
        b = arr[..., 2].astype(np.int16)
        luma = r * 0.299 + g * 0.587 + b * 0.114
        print(f"\n===== {name} per-seat 候选 =====")
        for seat, (ax, ay) in SEAT_ANCHORS_1848.items():
            # 锚点周围 ±120px 的 金(占座) / 红D / 白 信号
            x0, y0 = max(0, ax - 120), max(0, ay - 120)
            x1, y1 = min(w, ax + 120), min(h, ay + 120)
            rr = r[y0:y1, x0:x1]
            gg = g[y0:y1, x0:x1]
            bb = b[y0:y1, x0:x1]
            ll = luma[y0:y1, x0:x1]
            gold_broad = int(((rr > 120) & (gg > 85) & (rr >= gg)
                              & (gg >= bb - 30) & (ll > 90)).sum())
            gold_bright = int(((rr > 150) & (gg > 110) & (bb < 130)
                               & (rr >= gg)).sum())
            redD = int(((rr > 150) & (gg < 100) & (bb < 100)).sum())
            white = int((ll > 200).sum())
            print(f"  {seat}({ax},{ay}): 金broad={gold_broad} 金bright={gold_bright} "
                  f"红D={redD} 白={white}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
