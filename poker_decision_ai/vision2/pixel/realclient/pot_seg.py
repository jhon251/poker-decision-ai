"""V2E — Pot 金色数字串字符级分割（真实客户端域适配，仅 visual 域）。

实测（10 张真实截图只读分析）：
  - 客户端 pot 金额为金色字体（大约 r>170,g>120,b<110 且 r>=g），
    位于 pot ROI 内 x≈200-355、y≈65-112（字符高 ~39px，字宽 19-20px，
    间隙 4-7px，逗号 ~7px）；
  - 灰度阈值会混入卡面/装饰亮色 → 字符切分必须用金色掩码；
  - 掩码列投影的字符间隙（≥2 列空带）即可稳定切分，无需猜内容。

只做几何切分；字符语义由标注者（视觉确认）决定。
"""
from __future__ import annotations

import numpy as _np

from ...realtime.coords import Rect
from ...realtime.image import ImageData, to_gray


def gold_mask(img: ImageData) -> _np.ndarray:
    """金色掩码（确定性颜色规则）；无 numpy 时由调用方保证已安装。"""
    gray = _np.asarray(to_gray(img).data, dtype="uint8")
    arr = img.array if img.array is not None else gray
    if img.channels >= 3:
        a = _np.frombuffer(img.data or b"", dtype="uint8").reshape(
            img.height, img.width, img.channels)[..., :3].astype("int32")
    else:
        a = _np.stack([gray.astype("int32")] * 3, axis=-1)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    return (r > 170) & (g > 120) & (b < 110) & (r >= g)


def segments_gold_chars(img: ImageData, *, min_area: int = 20,
                        min_gap: int = 2) -> tuple:
    """金色字符分割 → (Rect 列表, 列投影串)。

    - 列投影：字符列（含金币前景）；间隙 ≥ min_gap 列为切分点；
    - 每段按掩码亮行取 y 范围；面积 < min_area 丢弃；
    - 返回按 x 排序的字符候选 Rect（ROI 内坐标）。
    """
    mask = gold_mask(img)
    h, w = mask.shape
    col_bright = mask.sum(axis=0)
    col_on = col_bright > 0
    runs = []
    x = 0
    while x < w:
        if not col_on[x]:
            x += 1
            continue
        x0 = x
        while x < w and col_on[x]:
            x += 1
        runs.append((x0, x))
    out = []
    for (x0, x1) in runs:
        if x1 - x0 < min_gap:
            continue
        ys, xs = _np.where(mask[:, x0:x1])
        if len(ys) == 0 or len(ys) < min_area:
            continue
        out.append(Rect(float(x0), float(ys.min()),
                        float(x1 - x0), float(ys.max() - ys.min() + 1)))
    proj = "".join("#" if col_on[i] else "." for i in range(w))
    return tuple(out), proj
