"""像素统计基元：变化掩码/比率/包围盒 + RGB/HSV 特征 + 主色。

- 纯 Python 为基线（项目无第三方图像依赖），numpy 可用时自动加速；
  两条路径语义一致（cross-path 测试固化）。
- 只回答"像素层面发生了什么"，不解释任何牌局语义。
- 几何不一致 → None（调用方记 uncertainty，不猜测）。
"""
from __future__ import annotations

from typing import Any

try:
    import numpy as _np
except Exception:  # pragma: no cover - 环境缺 numpy 时回退纯 Python
    _np = None

from ..realtime.coords import Rect
from ..realtime.image import ImageData


def _channels_array(img: ImageData) -> Any:
    """numpy 视图（h, w, c）；不可用 → None。"""
    if _np is None:
        return None
    if img.array is not None:
        return img.array
    data = img.data or b""
    return _np.frombuffer(data, dtype="uint8").reshape(
        img.height, img.width, img.channels)


def changed_pixel_mask(a: ImageData, b: ImageData,
                       threshold: int = 12) -> Any:
    """逐像素判定：max(|dr|,|dg|,|db|) > threshold → True。

    几何不一致 → None。返回 numpy bool 数组或 list[list[bool]]。
    """
    if not a.same_geometry(b):
        return None
    if _np is not None:
        aa = _channels_array(a)
        bb = _channels_array(b)
        if aa is None or bb is None:
            return None
        if a.channels >= 3:
            diff = _np.abs(aa[..., :3].astype("int16")
                           - bb[..., :3].astype("int16")).max(axis=2)
        else:
            diff = _np.abs(aa.astype("int16") - bb.astype("int16")).max(axis=2)
        return diff > threshold
    # 纯 Python
    da, db = a.data or b"", b.data or b""
    if not da or not db:
        return None
    c, w = a.channels, a.width
    rows = []
    for y in range(a.height):
        row = []
        base = y * w * c
        for x in range(w):
            i = base + x * c
            dmax = 0
            nch = min(c, 3)
            for k in range(nch):
                d = abs(da[i + k] - db[i + k])
                if d > dmax:
                    dmax = d
            row.append(dmax > threshold)
        rows.append(row)
    return rows


def _mask_size(mask: Any) -> int:
    if mask is None:
        return 0
    if _np is not None and not isinstance(mask, list):
        return int(mask.size)
    return len(mask) * len(mask[0]) if mask else 0


def _mask_true_count(mask: Any) -> int | None:
    if mask is None:
        return None
    if _np is not None and not isinstance(mask, list):
        return int(mask.sum())
    return sum(1 for row in mask for v in row if v)


def changed_pixel_ratio(a: ImageData, b: ImageData,
                        threshold: int = 12) -> float | None:
    """变化像素占比 [0,1]；几何不一致/无数据 → None。"""
    mask = changed_pixel_mask(a, b, threshold)
    if mask is None:
        return None
    total = _mask_size(mask)
    if total == 0:
        return None
    count = _mask_true_count(mask)
    return count / total


def changed_pixel_count(a: ImageData, b: ImageData,
                        threshold: int = 12) -> int | None:
    mask = changed_pixel_mask(a, b, threshold)
    return _mask_true_count(mask)


def changed_bbox(a: ImageData, b: ImageData,
                 threshold: int = 12) -> Rect | None:
    """变化像素的包围盒；无变化/几何不一致 → None。"""
    mask = changed_pixel_mask(a, b, threshold)
    if mask is None:
        return None
    if _np is not None and not isinstance(mask, list):
        idx = _np.argwhere(mask)
        if idx.size == 0:
            return None
        y0, x0 = int(idx[:, 0].min()), int(idx[:, 1].min())
        y1, x1 = int(idx[:, 0].max()), int(idx[:, 1].max())
        return Rect(float(x0), float(y0), float(x1 - x0 + 1),
                    float(y1 - y0 + 1))
    min_x = min_y = None
    max_x = max_y = -1
    for y, row in enumerate(mask):
        for x, v in enumerate(row):
            if v:
                if min_x is None:
                    min_x = max_x = x
                    min_y = max_y = y
                else:
                    min_x = min(min_x, x)
                    max_x = max(max_x, x)
                    min_y = min(min_y, y)
                    max_y = max(max_y, y)
    if min_x is None:
        return None
    return Rect(float(min_x), float(min_y), float(max_x - min_x + 1),
                float(max_y - min_y + 1))


# ------------------------------------------------------------------ 颜色
def hsv_from_rgb(r: int, g: int, b: int) -> tuple:
    """RGB(0-255) → HSV（h∈[0,360), s∈[0,1], v∈[0,1]），确定性公式。"""
    rn, gn, bn = r / 255.0, g / 255.0, b / 255.0
    maxc, minc = max(rn, gn, bn), min(rn, gn, bn)
    delta = maxc - minc
    v = maxc
    s = 0.0 if maxc == 0.0 else delta / maxc
    if delta == 0.0:
        h = 0.0
    elif maxc == rn:
        h = 60.0 * (((gn - bn) / delta) % 6.0)
    elif maxc == gn:
        h = 60.0 * ((bn - rn) / delta + 2.0)
    else:
        h = 60.0 * ((rn - gn) / delta + 4.0)
    return (float(h), float(s), float(v))


def rgb_stats(img: ImageData) -> dict | None:
    """RGB 均值/标准差；通道 <3 或无数据 → None。"""
    if img.channels < 3:
        return None
    if _np is not None:
        arr = _channels_array(img)
        if arr is None:
            return None
        rgb = arr[..., :3].astype("float64")
        mean = rgb.reshape(-1, 3).mean(axis=0)
        std = rgb.reshape(-1, 3).std(axis=0)
        return {"mean_rgb": [float(mean[0]), float(mean[1]), float(mean[2])],
                "std_rgb": [float(std[0]), float(std[1]), float(std[2])]}
    data = img.data or b""
    if not data:
        return None
    c, w, n = img.channels, img.width, 0
    sr = sg = sb = 0
    s2r = s2g = s2b = 0
    for y in range(img.height):
        base = y * w * c
        for x in range(w):
            i = base + x * c
            r, g, b = data[i], data[i + 1], data[i + 2]
            sr += r
            sg += g
            sb += b
            s2r += r * r
            s2g += g * g
            s2b += b * b
            n += 1
    if n == 0:
        return None
    mr, mg, mb = sr / n, sg / n, sb / n
    return {"mean_rgb": [mr, mg, mb],
            "std_rgb": [(s2r / n - mr * mr) ** 0.5,
                        (s2g / n - mg * mg) ** 0.5,
                        (s2b / n - mb * mb) ** 0.5]}


def hsv_stats(img: ImageData) -> dict | None:
    """逐像素 HSV 的算术均值（h 为算术均值，环形性质注明于文档）。"""
    if img.channels < 3:
        return None
    if _np is not None:
        arr = _channels_array(img)
        if arr is None:
            return None
        rgb = arr[..., :3].astype("float64") / 255.0
        maxc = rgb.max(axis=2)
        minc = rgb.min(axis=2)
        delta = maxc - minc
        v = maxc
        with _np.errstate(divide="ignore", invalid="ignore"):
            s = _np.where(maxc == 0.0, 0.0, delta / maxc)
            r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
            h = _np.where(
                delta == 0.0, 0.0,
                _np.where(maxc == r, 60.0 * ((g - b) / delta % 6.0),
                          _np.where(maxc == g,
                                    60.0 * ((b - r) / delta + 2.0),
                                    60.0 * ((r - g) / delta + 4.0))))
        return {"mean_hsv": [float(h.mean()), float(s.mean()),
                             float(v.mean())]}
    data = img.data or b""
    if not data:
        return None
    c, w, n = img.channels, img.width, 0
    sh = ss = sv = 0.0
    for y in range(img.height):
        base = y * w * c
        for x in range(w):
            i = base + x * c
            h, s, v = hsv_from_rgb(data[i], data[i + 1], data[i + 2])
            sh += h
            ss += s
            sv += v
            n += 1
    if n == 0:
        return None
    return {"mean_hsv": [sh / n, ss / n, sv / n]}


def dominant_color(img: ImageData, levels: int = 4) -> dict | None:
    """量化主色：每通道 levels 级（默认 4 → 64 桶），返回桶中心色与占比。

    几何/无数据 → None。deterministic。
    """
    if img.channels < 3:
        return None
    step = 256 // levels
    if _np is not None:
        arr = _channels_array(img)
        if arr is None:
            return None
        rgb = arr[..., :3].astype("int32")
        idx = ((rgb[..., 0] // step) * levels + rgb[..., 1] // step) * levels \
            + rgb[..., 2] // step
        flat = idx.reshape(-1)
        counts = _np.bincount(flat, minlength=levels ** 3)
        best = int(_np.argmax(counts))
        total = int(flat.size)
        if total == 0:
            return None
        rl = (best // (levels * levels)) * step
        gl = ((best // levels) % levels) * step
        bl = (best % levels) * step
        return {"dominant_rgb": [rl + step // 2, gl + step // 2,
                                 bl + step // 2],
                "dominant_ratio": float(counts[best] / total)}
    data = img.data or b""
    if not data:
        return None
    c, w = img.channels, img.width
    buckets: dict = {}
    total = 0
    for y in range(img.height):
        base = y * w * c
        for x in range(w):
            i = base + x * c
            key = ((data[i] // step) * levels + data[i + 1] // step) * levels \
                + data[i + 2] // step
            buckets[key] = buckets.get(key, 0) + 1
            total += 1
    if total == 0:
        return None
    best = max(buckets, key=buckets.get)
    rl = (best // (levels * levels)) * step
    gl = ((best // levels) % levels) * step
    bl = (best % levels) * step
    return {"dominant_rgb": [rl + step // 2, gl + step // 2, bl + step // 2],
            "dominant_ratio": buckets[best] / total}


def color_distance(a: dict, b: dict) -> float:
    """两个 dominant_color 结果的 RGB 欧氏距离（缺任一 → None 语义由调用方处理）。"""
    if not a or not b:
        return float("inf")
    ra, ga, ba = a["dominant_rgb"]
    rb, gb, bb = b["dominant_rgb"]
    return ((ra - rb) ** 2 + (ga - gb) ** 2 + (ba - bb) ** 2) ** 0.5
