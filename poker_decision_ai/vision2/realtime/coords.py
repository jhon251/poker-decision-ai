"""坐标层：screen / window / client / normalized 四坐标系确定性转换。

尺度链（纯函数、确定性，无状态）：
    screen(物理px, 绝对) -> window(相对窗口左上, 物理px)
    window(物理px) -> client(相对客户区左上, 物理px)     [减去标题栏/边框偏移]
    client(物理px) <-> normalized(客户区相对 [0,1])

DPI：dpi_scale = 物理px / 逻辑px。normalized 坐标对 DPI 不变（分子分母同除），
物理坐标换算经 client_size_physical 体现；另提供 client->logical 换算供识别层
预处理使用。禁止把绝对屏幕像素坐标作为长期稳定坐标（固定坐标仅 bootstrap 种子）。
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Union


class CoordinateSystem(str, Enum):
    SCREEN = "screen"
    WINDOW = "window"
    CLIENT = "client"
    NORMALIZED = "normalized"


class CoordinateError(ValueError):
    """坐标/几何不合法。确定性失败，调用方不得猜测替代值。"""


@dataclass(frozen=True)
class Point:
    x: float
    y: float

    def __add__(self, other: "Point") -> "Point":
        return Point(self.x + other.x, self.y + other.y)

    def __sub__(self, other: "Point") -> "Point":
        return Point(self.x - other.x, self.y - other.y)

    def to_tuple(self) -> tuple:
        return (self.x, self.y)


@dataclass(frozen=True)
class Rect:
    """左上原点矩形（与屏幕/窗口像素坐标一致）。"""
    x: float
    y: float
    width: float
    height: float

    @property
    def topleft(self) -> Point:
        return Point(self.x, self.y)

    @property
    def bottomright(self) -> Point:
        return Point(self.x + self.width, self.y + self.height)

    @property
    def is_empty(self) -> bool:
        return self.width <= 0.0 or self.height <= 0.0

    @property
    def area(self) -> float:
        if self.is_empty:
            return 0.0
        return self.width * self.height

    def contains(self, p: Point, *, inclusive: bool = True) -> bool:
        if inclusive:
            return (self.x <= p.x <= self.x + self.width
                    and self.y <= p.y <= self.y + self.height)
        return (self.x < p.x < self.x + self.width
                and self.y < p.y < self.y + self.height)

    def intersects(self, other: "Rect") -> bool:
        return not (other.x >= self.x + self.width
                    or other.x + other.width <= self.x
                    or other.y >= self.y + self.height
                    or other.y + other.height <= self.y)

    def intersection(self, other: "Rect") -> "Rect":
        x0, y0 = max(self.x, other.x), max(self.y, other.y)
        x1, y1 = min(self.x + self.width, other.x + other.width), \
            min(self.y + self.height, other.y + other.height)
        if x1 <= x0 or y1 <= y0:
            return Rect(0.0, 0.0, 0.0, 0.0)
        return Rect(x0, y0, x1 - x0, y1 - y0)

    def shifted(self, dx: float, dy: float) -> "Rect":
        return Rect(self.x + dx, self.y + dy, self.width, self.height)

    def scaled(self, sx: float, sy: float) -> "Rect":
        return Rect(self.x * sx, self.y * sy, self.width * sx, self.height * sy)

    def to_dict(self) -> dict:
        return {"x": self.x, "y": self.y,
                "width": self.width, "height": self.height}


@dataclass(frozen=True)
class GeometryContext:
    """一帧的几何事实（只观察记录，绝不修改画面/窗口）。"""
    screen_size: tuple = (1920.0, 1080.0)      # 物理像素
    window_rect: Rect | None = None            # screen 坐标；None=全屏捕获
    client_offset: tuple = (0.0, 0.0)          # window→client 原点偏移（物理px）
    client_size_physical: tuple = (1920.0, 1080.0)
    dpi_scale: float = 1.0                     # 物理px / 逻辑px，必须 > 0

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if self.screen_size[0] <= 0 or self.screen_size[1] <= 0:
            raise CoordinateError(f"非法屏幕尺寸 {self.screen_size}")
        if self.client_size_physical[0] <= 0 or self.client_size_physical[1] <= 0:
            raise CoordinateError(f"非法客户区尺寸 {self.client_size_physical}")
        if self.dpi_scale <= 0:
            raise CoordinateError(f"dpi_scale 必须 > 0，实际 {self.dpi_scale}")
        if self.window_rect is not None:
            w = self.window_rect
            if w.width <= 0 or w.height <= 0:
                raise CoordinateError(f"非法窗口矩形 {w}")
            cr = self.client_rect_in_screen()
            screen = Rect(0.0, 0.0, self.screen_size[0], self.screen_size[1])
            if not screen.contains(cr.topleft) or not screen.contains(cr.bottomright):
                raise CoordinateError(f"客户区 {cr.to_dict()} 超出屏幕 {screen.to_dict()}")

    @property
    def client_size_logical(self) -> tuple:
        return (self.client_size_physical[0] / self.dpi_scale,
                self.client_size_physical[1] / self.dpi_scale)

    def client_rect_in_screen(self) -> Rect:
        """客户区在 screen 坐标中的位置（无窗口信息时=全屏）。"""
        if self.window_rect is None:
            return Rect(0.0, 0.0, self.client_size_physical[0],
                        self.client_size_physical[1])
        return Rect(self.window_rect.x + self.client_offset[0],
                    self.window_rect.y + self.client_offset[1],
                    self.client_size_physical[0], self.client_size_physical[1])

    def to_dict(self) -> dict:
        return {
            "screen_size": list(self.screen_size),
            "window_rect": None if self.window_rect is None
            else self.window_rect.to_dict(),
            "client_offset": list(self.client_offset),
            "client_size_physical": list(self.client_size_physical),
            "dpi_scale": self.dpi_scale,
        }


_CHAIN = (CoordinateSystem.SCREEN, CoordinateSystem.WINDOW,
          CoordinateSystem.CLIENT, CoordinateSystem.NORMALIZED)

GeomObj = Union[Point, Rect]


class CoordinateTransformer:
    """四坐标系之间的确定性转换。禁止任何隐式猜测/取整歧义。"""

    def __init__(self, ctx: GeometryContext):
        ctx.validate()
        self.ctx = ctx

    # ---------------------------------------------------------- 相邻坐标系步进
    def _point_step(self, p: Point, a: CoordinateSystem,
                    b: CoordinateSystem) -> Point:
        w = self.ctx.window_rect
        if (a, b) == (CoordinateSystem.SCREEN, CoordinateSystem.WINDOW):
            return p - (w.topleft if w is not None else Point(0.0, 0.0))
        if (a, b) == (CoordinateSystem.WINDOW, CoordinateSystem.SCREEN):
            return p + (w.topleft if w is not None else Point(0.0, 0.0))
        if (a, b) == (CoordinateSystem.WINDOW, CoordinateSystem.CLIENT):
            return p - Point(self.ctx.client_offset[0], self.ctx.client_offset[1])
        if (a, b) == (CoordinateSystem.CLIENT, CoordinateSystem.WINDOW):
            return p + Point(self.ctx.client_offset[0], self.ctx.client_offset[1])
        if (a, b) == (CoordinateSystem.CLIENT, CoordinateSystem.NORMALIZED):
            return Point(p.x / self.ctx.client_size_physical[0],
                         p.y / self.ctx.client_size_physical[1])
        if (a, b) == (CoordinateSystem.NORMALIZED, CoordinateSystem.CLIENT):
            return Point(p.x * self.ctx.client_size_physical[0],
                         p.y * self.ctx.client_size_physical[1])
        raise CoordinateError(f"无相邻坐标步进 {a.value} -> {b.value}")

    def _step(self, obj: GeomObj, a: CoordinateSystem,
              b: CoordinateSystem) -> GeomObj:
        if isinstance(obj, Point):
            return self._point_step(obj, a, b)
        p = self._point_step(Point(obj.x, obj.y), a, b)
        if {a, b} == {CoordinateSystem.CLIENT, CoordinateSystem.NORMALIZED}:
            # Rect 尺寸必须按轴缩放：x/width 用宽度因子，y/height 用高度因子
            if a is CoordinateSystem.NORMALIZED:
                sx, sy = (self.ctx.client_size_physical[0],
                          self.ctx.client_size_physical[1])
            else:
                sx, sy = (1.0 / self.ctx.client_size_physical[0],
                          1.0 / self.ctx.client_size_physical[1])
            return Rect(p.x, p.y, obj.width * sx, obj.height * sy)
        return Rect(p.x, p.y, obj.width, obj.height)

    # ---------------------------------------------------------- 通用转换
    def transform(self, obj: GeomObj, from_sys: CoordinateSystem,
                  to_sys: CoordinateSystem) -> GeomObj:
        """任意两坐标系间转换（沿固定链 SCREEN->WINDOW->CLIENT->NORMALIZED）。"""
        if from_sys == to_sys:
            return obj
        if from_sys not in _CHAIN or to_sys not in _CHAIN:
            raise CoordinateError(f"未知坐标系 {from_sys} / {to_sys}")
        i, j = _CHAIN.index(from_sys), _CHAIN.index(to_sys)
        step = 1 if j > i else -1
        cur = obj
        for k in range(i, j, step):
            cur = self._step(cur, _CHAIN[k], _CHAIN[k + step])
        return cur

    # ---------------------------------------------------------- 便捷方法
    def screen_to_window(self, p: Point) -> Point:
        return self._point_step(p, CoordinateSystem.SCREEN, CoordinateSystem.WINDOW)

    def window_to_screen(self, p: Point) -> Point:
        return self._point_step(p, CoordinateSystem.WINDOW, CoordinateSystem.SCREEN)

    def window_to_client(self, p: Point) -> Point:
        return self._point_step(p, CoordinateSystem.WINDOW, CoordinateSystem.CLIENT)

    def client_to_window(self, p: Point) -> Point:
        return self._point_step(p, CoordinateSystem.CLIENT, CoordinateSystem.WINDOW)

    def client_to_normalized(self, p: Point) -> Point:
        return self._point_step(p, CoordinateSystem.CLIENT, CoordinateSystem.NORMALIZED)

    def normalized_to_client(self, p: Point) -> Point:
        return self._point_step(p, CoordinateSystem.NORMALIZED, CoordinateSystem.CLIENT)

    def client_to_logical(self, p: Point) -> Point:
        """物理px -> 逻辑px（DPI 换算，供识别层预处理）。"""
        return Point(p.x / self.ctx.dpi_scale, p.y / self.ctx.dpi_scale)
