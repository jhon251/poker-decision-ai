"""P25-B.3 — 精确测量「盖牌」tag：颜色 / 圆角矩 / 白字 glyph / 相对锚点偏移。"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))
from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402

EV = Path("data/vision2/coinpoker_evidence")


def load(frame):
    w, h, rgba = decode_png((EV / frame).read_bytes())
    return np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)


def probe(frame, x0, y0, x1, y1, tag="probe"):
    arr = load(frame)
    roi = arr[y0:y1, x0:x1]
    r = roi[..., 0].astype(np.int16)
    g = roi[..., 1].astype(np.int16)
    b = roi[..., 2].astype(np.int16)
    l = r * 0.299 + g * 0.587 + b * 0.114
    # 墨绿 teal：g > r 且 g > b？先打印直方图关键值
    teal = ((g > r) & (g >= b) & (l > 25) & (l < 110)).astype(np.uint8)
    ys, xs = np.where(teal > 0)
    if len(xs) == 0:
        print(f"[{tag}] no teal pixels in roi")
        return
    print(f"[{tag}] teal pixels={len(xs)} bbox=({x0+xs.min()},{y0+ys.min()},"
          f"{xs.max()-xs.min()+1},{ys.max()-ys.min()+1})")
    # 采样中心颜色
    cx = int(xs.mean()); cy = int(ys.mean())
    print(f"    center px=({x0+cx},{y0+cy}) rgb="
          f"{tuple(int(v) for v in arr[y0+cy, x0+cx][:3])}")
    # 白字：亮 luma > 140 且在 teal bbox 内
    tx0, ty0 = xs.min(), ys.min()
    tsub = roi[ty0:, tx0:]
    tl = tsub[..., 0] * 0.299 + tsub[..., 1] * 0.587 + tsub[..., 2] * 0.114
    white = (tl > 140)
    wys, wxs = np.where(white)
    if len(wxs) > 0:
        print(f"    white-glyph bbox rel=({tx0+wxs.min()},{ty0+wys.min()},"
              f"{wxs.max()-wxs.min()+1},{wys.max()-wys.min()+1}) "
              f"abs=({x0+tx0+wxs.min()},{y0+ty0+wys.min()})")


# f01 S5：从目视估算 tag 在 (1533,893)-(1603,928)，放大搜索
probe("BUTTON_CHANGED_167_1787992682749_full.png",
      1500, 870, 1700, 990, tag="f01_S5_tag")
# f01 S6（hero 折叠）
probe("BUTTON_CHANGED_167_1787992682749_full.png",
      820, 1100, 1060, 1220, tag="f01_S6_tag")
