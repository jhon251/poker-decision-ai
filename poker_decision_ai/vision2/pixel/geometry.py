"""几何特征（确定性；非 OpenCV 级检测器，文档如实标注）。

提供：大津阈值、连通域（两遍 BFS）、边缘密度、矩形度/圆似度/线似度。
所有输出是启发式几何统计，不携带任何语义识别。
numpy 用于阈值/边缘的向量加速；BFS 单实现（list[bool]），双路径结果一致。
"""
from __future__ import annotations

from dataclasses import dataclass

try:
    import numpy as _np
except Exception:  # pragma: no cover
    _np = None

from ..realtime.coords import Rect
from ..realtime.image import GrayImage


@dataclass
class Component:
    id: int
    area: int
    bbox: Rect
    centroid: tuple
    aspect_ratio: float
    fill_ratio: float
    rect_score: float
    circle_score: float
    line_score: float

    def to_dict(self) -> dict:
        return {
            "id": self.id, "area": self.area, "bbox": self.bbox.to_dict(),
            "centroid": list(self.centroid),
            "aspect_ratio": round(self.aspect_ratio, 4),
            "fill_ratio": round(self.fill_ratio, 4),
            "rect_score": round(self.rect_score, 4),
            "circle_score": round(self.circle_score, 4),
            "line_score": round(self.line_score, 4),
        }


def otsu_threshold(gray: GrayImage) -> int:
    """大津法阈值（确定性；单一实现，双路径同结果）。"""
    hist = [0] * 256
    for y in range(gray.height):
        row = gray.data[y]
        for x in range(gray.width):
            hist[int(row[x]) & 0xFF] += 1
    total = gray.width * gray.height
    if total == 0:
        return 128
    sum_all = sum(i * hist[i] for i in range(256))
    sum_b = 0
    w_b = 0
    best_t = 0
    best_var = -1.0
    for t in range(256):
        w_b += hist[t]
        if w_b == 0:
            continue
        w_f = total - w_b
        if w_f == 0:
            break
        sum_b += t * hist[t]
        m_b = sum_b / w_b
        m_f = (sum_all - sum_b) / w_f
        var = w_b * w_f * (m_b - m_f) ** 2
        if var > best_var:
            best_var = var
            best_t = t
    return best_t


def threshold_mask(gray: GrayImage, threshold: int | None = None,
                   *, invert: bool | None = None) -> list:
    """灰度二值化 → list[list[bool]]（numpy 加速，结果与纯路径一致）。

    invert=None（默认）：自动极性——区域边框像素视为背景，其均值相对
    大津阈值的极性决定前景（背景亮 → 前景=暗；背景暗 → 前景=亮）。
    该规则对"符号占区域过半"（如花色符号）也稳定。显式 True/False 覆盖。
    """
    t = threshold if threshold is not None else otsu_threshold(gray)
    if invert is None:
        h, w = gray.height, gray.width
        border = []
        if h > 0 and w > 0:
            for x in range(w):
                border.append(int(gray.data[0][x]))
                if h > 1:
                    border.append(int(gray.data[h - 1][x]))
            for y in range(1, h - 1):
                border.append(int(gray.data[y][0]))
                if w > 1:
                    border.append(int(gray.data[y][w - 1]))
        if border:
            s = sorted(border)
            border_ref = s[len(s) // 2]     # 中位数：对少数贴边字形稳健
        else:
            border_ref = 128.0
        invert = border_ref > t             # 背景亮 → 前景=暗
    if _np is not None and not isinstance(gray.data, list):
        arr = _np.asarray(gray.data, dtype="int32")
        m = arr > t if not invert else arr <= t
        return [[bool(v) for v in row] for row in m.tolist()]
    rows = []
    for y in range(gray.height):
        row = gray.data[y]
        rows.append([(int(row[x]) > t) != invert for x in range(gray.width)])
    return rows


def connected_components(mask: list, min_area: int = 4) -> list:
    """两遍 BFS 标号（8 连通，字形/区域候选标准做法）→ Component 列表。

    组件统计为启发式：rect_score=填充率；circle_score=填充率/(π/4) 截断；
    line_score=细长且高填充。
    """
    h = len(mask)
    w = len(mask[0]) if h else 0
    visited = [[False] * w for _ in range(h)]
    comps: list = []
    next_id = 0
    for y0 in range(h):
        for x0 in range(w):
            if not mask[y0][x0] or visited[y0][x0]:
                continue
            # BFS（8 连通）
            stack = [(y0, x0)]
            visited[y0][x0] = True
            cells = []
            min_x = max_x = x0
            min_y = max_y = y0
            while stack:
                y, x = stack.pop()
                cells.append((x, y))
                if x < min_x:
                    min_x = x
                if x > max_x:
                    max_x = x
                if y < min_y:
                    min_y = y
                if y > max_y:
                    max_y = y
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        if dy == 0 and dx == 0:
                            continue
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < h and 0 <= nx < w and mask[ny][nx] \
                                and not visited[ny][nx]:
                            visited[ny][nx] = True
                            stack.append((ny, nx))
            area = len(cells)
            if area < min_area:
                continue
            bw, bh = max_x - min_x + 1, max_y - min_y + 1
            fill = area / (bw * bh)
            aspect = bw / bh if bh else 0.0
            cx = sum(x for x, _ in cells) / area
            cy = sum(y for _, y in cells) / area
            circle = min(1.0, fill / 0.7853981633974483)  # π/4
            line = fill if (aspect >= 3.0 or 1.0 / max(aspect, 1e-9) >= 3.0) \
                else 0.0
            comps.append(Component(
                id=next_id, area=area,
                bbox=Rect(float(min_x), float(min_y), float(bw), float(bh)),
                centroid=(cx, cy), aspect_ratio=aspect, fill_ratio=fill,
                rect_score=fill, circle_score=circle, line_score=line))
            next_id += 1
    comps.sort(key=lambda c: (-c.area, c.id))
    return comps


def edge_density(gray: GrayImage, threshold: int = 40) -> float:
    """边缘密度：|gx|+|gy| > threshold 的像素占比（中心差分）。"""
    h, w = gray.height, gray.width
    if h < 3 or w < 3:
        return 0.0
    if _np is not None and not isinstance(gray.data, list):
        arr = _np.asarray(gray.data, dtype="int32")
        gx = _np.abs(arr[1:-1, 2:] - arr[1:-1, :-2])
        gy = _np.abs(arr[2:, 1:-1] - arr[:-2, 1:-1])
        mag = gx + gy
        return float((mag > threshold).mean())
    total, edges = 0, 0
    data = gray.data
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            gx = abs(int(data[y][x + 1]) - int(data[y][x - 1]))
            gy = abs(int(data[y + 1][x]) - int(data[y - 1][x]))
            if gx + gy > threshold:
                edges += 1
            total += 1
    return edges / total if total else 0.0
