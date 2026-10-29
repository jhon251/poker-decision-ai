"""纯 Python 图像基元 + 可选加速路径。

职责边界（硬约束）：
  - 本模块只做 cheap 检测：感知哈希 / 像素差 / 归一化互相关模板匹配；
  - 只能回答「画面/ROI 是否发生变化」「模板在哪里」；
  - 严禁任何把像素解释为牌面/数字语义的逻辑（那是识别层职责）。

实现策略：标准库纯 Python 为基线（项目无第三方图像依赖），numpy 若可用则
自动加速（语义与纯 Python 路径一致；cross-path 一致性由测试覆盖）。
大图（整帧）的逐像素路径在纯 Python 下较慢，基准脚本会实测并如实报告。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

try:  # 可选加速（非必需依赖）
    import numpy as _np
except Exception:  # pragma: no cover - 环境缺 numpy 时回退纯 Python
    _np = None

from .coords import Rect


def luma_rgba(r: int, g: int, b: int, _a: int = 255) -> int:
    """整数亮度（确定性，两路径共用，避免浮点差异）。"""
    return (77 * r + 150 * g + 29 * b) >> 8


@dataclass
class GrayImage:
    """灰度亮度矩阵。data 为 numpy 数组或 list[list[int]]（行主序）。"""
    width: int
    height: int
    data: Any

    def pixel(self, x: int, y: int) -> int:
        return int(self.data[y][x])


@dataclass
class ImageData:
    """RGBA/灰度像素载体（row-major）。data=raw bytes；array=可选 numpy 视图。"""
    width: int
    height: int
    channels: int
    data: bytes | None = None
    array: Any = None

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError(f"非法图像尺寸 {self.width}x{self.height}")
        if self.channels not in (1, 3, 4):
            raise ValueError(f"非法通道数 {self.channels}")
        expected = self.width * self.height * self.channels
        if self.data is None and self.array is None:
            raise ValueError("data 与 array 至少提供一个")
        if self.data is not None and len(self.data) != expected:
            raise ValueError(f"字节数 {len(self.data)} != 期望 {expected}")
        if self.array is not None and int(self.array.size) != expected:
            raise ValueError("numpy array 尺寸与 width/height/channels 不符")

    @classmethod
    def from_rgba_bytes(cls, width: int, height: int, data: bytes) -> "ImageData":
        return cls(width=width, height=height, channels=4, data=data)

    @classmethod
    def from_gray_bytes(cls, width: int, height: int, data: bytes) -> "ImageData":
        return cls(width=width, height=height, channels=1, data=data)

    @classmethod
    def from_array(cls, arr: Any) -> "ImageData":
        h, w = arr.shape[:2]
        c = 1 if arr.ndim == 2 else int(arr.shape[2])
        return cls(width=w, height=h, channels=c, array=arr)

    def rgba_bytes(self) -> bytes:
        if self.data is not None and self.channels == 4:
            return self.data
        if self.array is not None:
            if self.channels == 4:
                return self.array.tobytes()
            return self.array.tobytes()  # 1/3 通道原样返回（仅内部用途）
        if self.channels == 4:
            return self.data or b""
        raise ValueError("无法提供 RGBA bytes")

    def same_geometry(self, other: "ImageData") -> bool:
        return (self.width == other.width and self.height == other.height
                and self.channels == other.channels)

    def crop(self, rect: Rect) -> "ImageData | None":
        x0, y0 = int(round(rect.x)), int(round(rect.y))
        x1, y1 = int(round(rect.x + rect.width)), int(round(rect.y + rect.height))
        if x0 < 0 or y0 < 0 or x1 > self.width or y1 > self.height \
                or x1 <= x0 or y1 <= y0:
            return None
        w, h = x1 - x0, y1 - y0
        if self.array is not None:
            return ImageData(width=w, height=h, channels=self.channels,
                             array=self.array[y0:y1, x0:x1])
        c = self.channels
        rows = []
        stride = self.width * c
        src = self.data or b""
        for y in range(y0, y1):
            rows.append(src[y * stride + x0 * c: y * stride + x1 * c])
        return ImageData(width=w, height=h, channels=c, data=b"".join(rows))

    def to_gray(self) -> GrayImage:
        return to_gray(self)


def to_gray(img: ImageData) -> GrayImage:
    if img.array is not None:
        a = img.array.astype("int32")
        if img.channels >= 3:
            g = ((77 * a[..., 0] + 150 * a[..., 1] + 29 * a[..., 2]) >> 8)
        else:
            g = a
        return GrayImage(width=img.width, height=img.height, data=g)
    if _np is not None:
        c = img.channels
        a = _np.frombuffer(img.data or b"", dtype="uint8").reshape(
            img.height, img.width, c).astype("int32")
        if c >= 3:
            g = ((77 * a[..., 0] + 150 * a[..., 1] + 29 * a[..., 2]) >> 8)
        else:
            g = a[..., 0]
        return GrayImage(width=img.width, height=img.height, data=g)
    c, w = img.channels, img.width
    data = img.data or b""
    rows = []
    for y in range(img.height):
        row = []
        base = y * w * c
        if c == 4:
            for x in range(w):
                i = base + x * 4
                row.append(luma_rgba(data[i], data[i + 1], data[i + 2]))
        elif c == 3:
            for x in range(w):
                i = base + x * 3
                row.append(luma_rgba(data[i], data[i + 1], data[i + 2]))
        else:
            row = list(data[base: base + w])
        rows.append(row)
    return GrayImage(width=w, height=img.height, data=rows)


def block_average(gray: GrayImage, rows: int, cols: int) -> list:
    """分块均值降采样（rows x cols 网格，块边界取整，两路径一致）。"""
    h, w = gray.height, gray.width
    y_edges = [int(r * h / rows) for r in range(rows + 1)]
    x_edges = [int(c * w / cols) for c in range(cols + 1)]
    if _np is not None and not isinstance(gray.data, list):
        arr = _np.asarray(gray.data, dtype="float64")
        out = []
        for r in range(rows):
            row = []
            for c in range(cols):
                block = arr[y_edges[r]:y_edges[r + 1],
                            x_edges[c]:x_edges[c + 1]]
                row.append(float(block.mean()) if block.size else 0.0)
            out.append(row)
        return out
    out = []
    for r in range(rows):
        row = []
        for c in range(cols):
            s, n = 0, 0
            for y in range(y_edges[r], max(y_edges[r + 1], y_edges[r])):
                for x in range(x_edges[c], max(x_edges[c + 1], x_edges[c])):
                    s += gray.data[y][x]
                    n += 1
            row.append(s / n if n else 0.0)
        out.append(row)
    return out


def dhash(img: ImageData, hash_size: int = 16) -> int:
    """分块均值 + 行内相邻比较的感知哈希（只判"变化"，不判语义）。"""
    if hash_size < 2:
        raise ValueError("hash_size 必须 >= 2")
    grid = block_average(to_gray(img), hash_size, hash_size + 1)
    bits = 0
    for r in range(hash_size):
        for c in range(hash_size):
            if float(grid[r][c]) < float(grid[r][c + 1]):
                bits |= 1 << (r * hash_size + c)
    return bits


def dhash_hex(img: ImageData, hash_size: int = 16) -> str:
    n = hash_size * hash_size
    return format(dhash(img, hash_size), f"0{n // 4}x")


def hamming_distance(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def byte_equality_ratio(a: bytes, b: bytes) -> float:
    """字节级相等比例（整帧去重用；整数 XOR 实现，纯 Python 亦为 C 速度）。

    只表达「两帧字节层面有多少完全相同」，无语义。
    """
    if len(a) == 0 or len(b) == 0 or len(a) != len(b):
        return 0.0
    n = len(a)
    if n > 0 and _np is not None:
        xa = _np.frombuffer(a, dtype="uint8")
        xb = _np.frombuffer(b, dtype="uint8")
        return float((xa == xb).mean())
    xor = int.from_bytes(a, "little") ^ int.from_bytes(b, "little")
    # bit 级差异计数的上界即字节差异数（保守估计：变化字节不会多于变化位）
    diff_bytes = min(n, xor.bit_count())
    return (n - diff_bytes) / n


def pixel_diff_ratio(a: ImageData, b: ImageData) -> float:
    """平均通道差 / 255（RGB 逐通道；0=完全相同，1=完全不同）。

    尺寸不一致 → 1.0。用通道差而非亮度差：亮度近似但颜色不同的变化
    （cheap 层必须能发现）不会被漏掉。
    """
    if not a.same_geometry(b):
        return 1.0
    if a.channels >= 3:
        if _np is not None:
            aa = _np.frombuffer(a.data or b"", dtype="uint8")
            bb = _np.frombuffer(b.data or b"", dtype="uint8")
            aa = aa.reshape(a.height, a.width, a.channels)[..., :3].astype(
                "float64")
            bb = bb.reshape(b.height, b.width, b.channels)[..., :3].astype(
                "float64")
            return float(_np.abs(aa - bb).mean() / 255.0)
        total, n = 0, 0
        c, w = a.channels, a.width
        da, db = a.data or b"", b.data or b""
        for y in range(a.height):
            base = y * w * c
            for x in range(w):
                i = base + x * c
                total += abs(da[i] - db[i]) + abs(da[i + 1] - db[i + 1]) \
                    + abs(da[i + 2] - db[i + 2])
                n += 3
        return (total / n) / 255.0 if n else 1.0
    ga, gb = to_gray(a), to_gray(b)
    if _np is not None and not isinstance(ga.data, list) \
            and not isinstance(gb.data, list):
        da = _np.asarray(ga.data, dtype="float64")
        db = _np.asarray(gb.data, dtype="float64")
        return float((_np.abs(da - db).mean()) / 255.0)
    total, n = 0, 0
    for y in range(ga.height):
        ra, rb = ga.data[y], gb.data[y]
        for x in range(ga.width):
            total += abs(int(ra[x]) - int(rb[x]))
            n += 1
    return (total / n) / 255.0 if n else 1.0


def ncc_template_match(image: ImageData, template: ImageData,
                       search_rect: Rect | None = None,
                       score_threshold: float = 0.6) -> tuple | None:
    """归一化互相关模板匹配：返回 (x, y, score) 或 None（找不到）。

    用途仅限定位 UI 锚点（「在哪里看」）；绝不用于识别牌面。
    numpy 路径走 FFT 相关 + 积分图归一化（整帧亦快）；纯 Python 回退
    只适合小图（测试/裁剪）。
    """
    if template.width > image.width or template.height > image.height:
        return None
    if search_rect is None:
        search_rect = Rect(0, 0, image.width - template.width,
                           image.height - template.height)
    x0, y0 = int(round(search_rect.x)), int(round(search_rect.y))
    x1 = min(int(round(search_rect.x + search_rect.width)),
             image.width - template.width)
    y1 = min(int(round(search_rect.y + search_rect.height)),
             image.height - template.height)
    if x1 < x0 or y1 < y0:
        return None

    gi = to_gray(image)
    gt = to_gray(template)
    t_w, t_h = template.width, template.height

    if _np is not None and not isinstance(gi.data, list):
        return _ncc_fft(gi, gt, x0, y0, x1, y1, score_threshold)

    t_mean = sum(v for row in gt.data for v in row) / (t_w * t_h)
    t_std = (sum((v - t_mean) ** 2 for row in gt.data for v in row)
             / (t_w * t_h)) ** 0.5
    if t_std == 0.0:
        return None
    tn = [[(v - t_mean) / t_std for v in row] for row in gt.data]
    best = None
    for y in range(y0, y1 + 1):
        for x in range(x0, x1 + 1):
            win = [gi.data[yy][x:x + t_w] for yy in range(y, y + t_h)]
            w_mean = sum(sum(r) for r in win) / (t_w * t_h)
            w_std = (sum((v - w_mean) ** 2 for r in win for v in r)
                     / (t_w * t_h)) ** 0.5
            if w_std == 0.0:
                score = 0.0
            else:
                s = sum((v - w_mean) * tn[yy - y][xx]
                        for yy in range(y, y + t_h)
                        for xx, v in enumerate(win[yy - y]))
                score = s / (w_std * t_w * t_h)
            if best is None or score > best[2]:
                best = (x, y, score)
    return best if best and best[2] >= score_threshold else None


def _ncc_fft(gi: GrayImage, gt: GrayImage, x0: int, y0: int,
             x1: int, y1: int, score_threshold: float) -> tuple | None:
    """FFT 加速 NCC（与暴力版同公式：score = corr / sqrt(Σt·Σw)）。"""
    im = _np.asarray(gi.data, dtype="float64")
    t = _np.asarray(gt.data, dtype="float64")
    t_h, t_w = t.shape
    n = float(t_w * t_h)
    t_mean = float(t.mean())
    sum_t = float(((t - t_mean) ** 2).sum())
    if sum_t <= 0.0:
        return None
    sub = im[y0:y1 + t_h, x0:x1 + t_w]
    H, W = sub.shape
    T = _np.zeros_like(sub)
    T[:t_h, :t_w] = t - t_mean
    F = _np.fft.rfft2(sub, s=(H, W))
    G = _np.fft.rfft2(T, s=(H, W))
    corr = _np.fft.irfft2(F * _np.conj(G), s=(H, W))[:H - t_h + 1, :W - t_w + 1]
    # 积分图：窗口和 / 平方和（确定性与暴力版一致）
    integ = _np.zeros((H + 1, W + 1))
    _np.cumsum(_np.cumsum(sub, axis=0), axis=1, out=integ[1:, 1:])
    s1 = (integ[:H - t_h + 1, :W - t_w + 1] + integ[t_h:, t_w:]
          - integ[:H - t_h + 1, t_w:] - integ[t_h:, :W - t_w + 1])
    integ2 = _np.zeros((H + 1, W + 1))
    _np.cumsum(_np.cumsum(sub * sub, axis=0), axis=1, out=integ2[1:, 1:])
    s2 = (integ2[:H - t_h + 1, :W - t_w + 1] + integ2[t_h:, t_w:]
          - integ2[:H - t_h + 1, t_w:] - integ2[t_h:, :W - t_w + 1])
    var_w = _np.maximum(s2 - (s1 * s1) / n, 0.0)
    denom = _np.sqrt(var_w * sum_t)
    with _np.errstate(divide="ignore", invalid="ignore"):
        score = _np.where(denom > 0.0, corr / denom, 0.0)
    idx = int(_np.argmax(score))
    cy, cx = divmod(idx, score.shape[1])
    best = (x0 + int(cx), y0 + int(cy), float(score[cy, cx]))
    return best if best[2] >= score_threshold else None
