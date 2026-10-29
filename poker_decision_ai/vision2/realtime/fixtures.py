"""Synthetic 场景与帧工厂（synthetic_test_only=True；测试/演示专用）。

场景 = 几何 + anchor 地面真值 + ROI 内容地面真值；渲染纯确定性：
  - anchor 以确定性棋盘格图案绘制（NCC 模板匹配定位用）；
  - 各 ROI 以内容哈希着色 → 内容变化必然引起像素变化（cheap 检测可见）。
synthetic OCR/card/multimodal 后端从场景 ground truth 读值（绝不从像素猜）。
所有产出帧恒 synthetic_test_only=True，不代表真实视觉能力。
"""
from __future__ import annotations

import zlib
from dataclasses import dataclass, replace

from ..schemas import RawAction, RawSeat
from .capture import CapturedFrame
from .coords import GeometryContext, Rect
from .image import ImageData
from .roi import ROIName, ROISpec


@dataclass
class SeatTruth:
    seat: int
    occupied: bool = True
    stack: float | None = None
    current_bet: float | None = None
    folded: bool | None = None
    all_in: bool | None = None
    in_hand: bool | None = None
    visible_action: str | None = None


@dataclass
class ROITruth:
    number: float | None = None
    text: str | None = None
    actor_seat: int | None = None
    cards: tuple | None = None        # None=未知；元素可为 None（不确定）
    seat_stacks: dict | None = None   # seat -> stack
    seats: dict | None = None         # seat -> SeatTruth
    blinds: dict | None = None        # {sb, bb, ante}
    to_call: float | None = None      # hero 需跟注额（table_meta 显示）


DEFAULT_GEOMETRY = GeometryContext(
    screen_size=(1920.0, 1080.0),
    window_rect=Rect(100.0, 50.0, 1280.0, 800.0),
    client_offset=(0.0, 30.0),
    client_size_physical=(1280.0, 770.0),
    dpi_scale=1.0,
)

#: 校准后（ground truth）anchor：table 边界 + 各关键区锚点（client 物理px）
DEFAULT_ANCHORS_CLIENT = {
    "table": Rect(40.0, 40.0, 1200.0, 690.0),
    "board": Rect(448.0, 220.0, 384.0, 140.0),
    "pot": Rect(448.0, 120.0, 384.0, 60.0),
}

#: 校准后（ground truth）ROI spec：anchor-relative 归一化偏移（互不重叠）
DEFAULT_ROI_SPECS = {
    ROIName.HERO_CARDS: ROISpec(ROIName.HERO_CARDS, "table",
                                Rect(0.36, 0.62, 0.28, 0.14),
                                required=True, read_layer="cards"),
    ROIName.BOARD: ROISpec(ROIName.BOARD, "table",
                           Rect(0.28, 0.24, 0.44, 0.22),
                           required=True, read_layer="cards"),
    ROIName.POT: ROISpec(ROIName.POT, "table",
                         Rect(0.28, 0.10, 0.44, 0.10),
                         required=True, read_layer="number"),
    ROIName.ACTION_AREA: ROISpec(ROIName.ACTION_AREA, "table",
                                 Rect(0.60, 0.52, 0.30, 0.12),
                                 required=False, read_layer="text"),
    ROIName.BET_AREA: ROISpec(ROIName.BET_AREA, "table",
                              Rect(0.36, 0.78, 0.28, 0.10),
                              required=False, read_layer="number"),
    ROIName.STACKS: ROISpec(ROIName.STACKS, "table",
                            Rect(0.02, 0.40, 0.14, 0.30),
                            required=False, read_layer="seat_stacks"),
    ROIName.PLAYERS: ROISpec(ROIName.PLAYERS, "table",
                             Rect(0.02, 0.04, 0.20, 0.18),
                             required=False, read_layer="seats"),
    ROIName.TABLE_META: ROISpec(ROIName.TABLE_META, "table",
                                Rect(0.84, 0.02, 0.14, 0.10),
                                required=False, read_layer="table_meta"),
}


def _anchor_pattern(anchor_id: str, size: int = 10) -> bytes:
    """anchor 确定性棋盘格图案（RGBA）。"""
    seed = zlib.crc32(anchor_id.encode("utf-8"))
    out = bytearray()
    for y in range(size):
        for x in range(size):
            v = 255 if ((x + y + seed) % 2) == 0 else 0
            out += bytes((v, v, v, 255))
    return bytes(out)


def _content_color(truth: ROITruth) -> tuple:
    """ROI 内容 → 填充色（内容变化 → 像素变化，供 cheap 检测）。"""
    payload = repr((truth.number, truth.text, truth.actor_seat,
                    truth.cards, truth.seat_stacks,
                    tuple(sorted((truth.seats or {}).keys())),
                    truth.blinds)).encode("utf-8")
    h = zlib.crc32(payload)
    return (h & 0xFF, (h >> 8) & 0xFF, (h >> 16) & 0xFF, 255)


@dataclass
class SyntheticScene:
    scene_id: str
    geometry: GeometryContext = DEFAULT_GEOMETRY
    anchors_client: dict = None            # anchor_id -> Rect（client 物理px）
    roi_specs: dict = None                 # ROIName -> ROISpec
    truth: dict = None                     # ROIName -> ROITruth
    hero_seat: int | None = 3
    button_seat: int | None = 2
    table_size: int = 6
    poker_variant: str = "nlhe"
    game_type: str = "cash"

    def __post_init__(self) -> None:
        if self.anchors_client is None:
            self.anchors_client = dict(DEFAULT_ANCHORS_CLIENT)
        if self.roi_specs is None:
            self.roi_specs = dict(DEFAULT_ROI_SPECS)
        if self.truth is None:
            self.truth = {}

    def render(self, now: float = 0.0, source: str = "fixture") -> CapturedFrame:
        """渲染确定性 RGBA 画面（客户区尺寸）。"""
        w = int(self.geometry.client_size_physical[0])
        h = int(self.geometry.client_size_physical[1])
        buf = bytearray(bytes((24, 32, 48, 255)) * (w * h))
        # table anchor 区域底色
        table = self.anchors_client.get("table")
        if table is not None:
            self._fill(buf, w, table, (40, 96, 40, 255))
        # ROI 内容色
        for name, truth in self.truth.items():
            spec = self.roi_specs.get(name)
            if spec is None:
                continue
            rect = self._resolve(spec)
            self._fill(buf, w, rect, _content_color(truth))
        # anchor 图案（定位用）
        for anchor_id, rect in self.anchors_client.items():
            self._blit_pattern(buf, w, rect, _anchor_pattern(anchor_id))
        return CapturedFrame(
            image=ImageData.from_rgba_bytes(w, h, bytes(buf)),
            captured_at=now, geometry=self.geometry, source=source,
            synthetic_test_only=True)

    def _resolve(self, spec: ROISpec) -> Rect:
        anchor = self.anchors_client[spec.anchor]
        return Rect(anchor.x + spec.offset.x * anchor.width,
                    anchor.y + spec.offset.y * anchor.height,
                    spec.offset.width * anchor.width,
                    spec.offset.height * anchor.height)

    @staticmethod
    def _fill(buf, w, rect, color) -> None:
        x0, y0 = int(round(rect.x)), int(round(rect.y))
        x1, y1 = int(round(rect.x + rect.width)), int(round(rect.y + rect.height))
        for y in range(max(y0, 0), min(y1, len(buf) // (w * 4))):
            row = bytes(color) * (min(x1, w) - max(x0, 0))
            start = (y * w + max(x0, 0)) * 4
            buf[start: start + len(row)] = row

    @staticmethod
    def _blit_pattern(buf, w, rect, pattern_rgba: bytes) -> None:
        size = int((len(pattern_rgba) // 4) ** 0.5)
        x0, y0 = int(round(rect.x)), int(round(rect.y))
        for y in range(size):
            row = pattern_rgba[y * size * 4:(y + 1) * size * 4]
            start = ((y0 + y) * w + x0) * 4
            buf[start: start + len(row)] = row

    def anchor_template(self, anchor_id: str, size: int = 10) -> ImageData:
        return ImageData.from_rgba_bytes(size, size,
                                         _anchor_pattern(anchor_id, size))

    def with_truth(self, roi_name: ROIName, **kw) -> "SyntheticScene":
        truth = dict(self.truth)
        base = truth.get(roi_name, ROITruth())
        truth[roi_name] = replace(base, **kw)
        return replace(self, truth=truth)

    def with_geometry(self, **kw) -> "SyntheticScene":
        return replace(self, geometry=replace(self.geometry, **kw))

    def with_anchors(self, anchors_client: dict) -> "SyntheticScene":
        return replace(self, anchors_client=anchors_client)

    def raw_fields(self) -> dict:
        """场景 ground truth → RawVisionState 字段字典（synthetic 多模态用）。

        返回 dict 直接对应 RawVisionState 字段（seats 为 RawSeat 列表、
        action_history 为 RawAction 列表、board 为 V0 board dict）。
        """
        hero = self.truth.get(ROIName.HERO_CARDS)
        board = self.truth.get(ROIName.BOARD)
        pot = self.truth.get(ROIName.POT)
        meta = self.truth.get(ROIName.TABLE_META)
        players = self.truth.get(ROIName.PLAYERS)
        action = self.truth.get(ROIName.ACTION_AREA)
        bet = self.truth.get(ROIName.BET_AREA)

        fields = {
            "source": "vision2_realtime",
            "synthetic_test_only": True,
            "poker_variant": self.poker_variant,
            "game_type": self.game_type,
            "table_size": self.table_size,
            "hero_seat": self.hero_seat,
            "button_seat": self.button_seat,
            "hero_hole_cards": (list(hero.cards) if hero is not None
                                and hero.cards is not None else None),
            "pot": pot.number if pot is not None else None,
            "hero_current_bet": bet.number if bet is not None else None,
        }
        if meta is not None:
            fields["blinds"] = meta.blinds
        if board is not None:
            cards = board.cards
            if cards is None:
                fields["board"] = None
            else:
                n = len(cards)
                fields["board"] = {
                    "flop": list(cards[:3]) if n >= 3 else None,
                    "turn": list(cards[3:4]) if n >= 4 else None,
                    "river": list(cards[4:]) if n >= 5 else None,
                }
                fields["street"] = {0: "preflop", 3: "flop", 4: "turn",
                                    5: "river"}.get(n)
        if players is not None and players.seats:
            fields["seats"] = [
                RawSeat(seat=t.seat, occupied=t.occupied,
                        anonymous_player_id=f"player_{t.seat}",
                        stack=t.stack, current_bet=t.current_bet,
                        folded=t.folded, all_in=t.all_in, in_hand=t.in_hand,
                        visible_action=t.visible_action)
                for s, t in sorted(players.seats.items())
            ]
        if action is not None and action.text and action.actor_seat is not None:
            word, *rest = action.text.strip().split()
            if word.lower() in ("fold", "check", "call", "bet", "raise",
                                "allin", "all-in"):
                amount = float(rest[0]) if rest else None
                street = fields.get("street")
                fields["action_history"] = [RawAction(
                    player=f"player_{action.actor_seat}",
                    action=word.lower().replace("all-in", "allin"),
                    amount=amount, street=street, order=1, confidence=0.99)]
        return fields


# ------------------------------------------------------------------ 序列工厂
def six_seat_truths(hero_seat: int = 3, button_seat: int = 2) -> dict:
    seats = {}
    for n in range(1, 7):
        seats[n] = SeatTruth(seat=n, occupied=True, stack=100.0,
                             current_bet=0.0, folded=False, all_in=False,
                             in_hand=(n == hero_seat or n in (1, 2)),
                             visible_action=None)
    seats[1].stack, seats[1].current_bet = 99.5, 0.5
    seats[2].stack, seats[2].current_bet = 99.0, 1.0
    return seats


def preflop_scene(scene_id: str = "preflop", pot: float = 1.5) -> SyntheticScene:
    seats = six_seat_truths()
    return SyntheticScene(
        scene_id=scene_id,
        truth={
            ROIName.HERO_CARDS: ROITruth(cards=("As", "Kd")),
            ROIName.BOARD: ROITruth(cards=()),
            ROIName.POT: ROITruth(number=pot),
            ROIName.ACTION_AREA: ROITruth(text=None, actor_seat=None),
            ROIName.BET_AREA: ROITruth(number=0.0),
            ROIName.STACKS: ROITruth(
                seat_stacks={s: t.stack for s, t in seats.items()}),
            ROIName.PLAYERS: ROITruth(seats=seats),
            ROIName.TABLE_META: ROITruth(
                blinds={"sb": 0.5, "bb": 1.0, "ante": None}, to_call=1.0),
        },
        hero_seat=3, button_seat=2)


def flop_scene(scene_id: str = "flop", pot: float = 10.0) -> SyntheticScene:
    scene = preflop_scene(scene_id, pot)
    truth = dict(scene.truth)
    truth[ROIName.BOARD] = ROITruth(cards=("Qh", "7d", "2c"))
    truth[ROIName.POT] = ROITruth(number=pot)
    truth[ROIName.ACTION_AREA] = ROITruth(text="Bet 4.0", actor_seat=3)
    truth[ROIName.BET_AREA] = ROITruth(number=4.0)
    truth[ROIName.TABLE_META] = ROITruth(
        blinds={"sb": 0.5, "bb": 1.0, "ante": None}, to_call=0.0)
    return replace(scene, truth=truth)


def turn_scene(pot: float = 14.0) -> SyntheticScene:
    scene = flop_scene("turn", pot)
    truth = dict(scene.truth)
    truth[ROIName.BOARD] = ROITruth(cards=("Qh", "7d", "2c", "9s"))
    truth[ROIName.POT] = ROITruth(number=pot)
    return replace(scene, truth=truth)


def river_scene(pot: float = 18.0) -> SyntheticScene:
    scene = turn_scene(pot)
    truth = dict(scene.truth)
    truth[ROIName.BOARD] = ROITruth(cards=("Qh", "7d", "2c", "9s", "Ad"))
    truth[ROIName.POT] = ROITruth(number=pot)
    return replace(scene, scene_id="river", truth=truth)


def duplicate_card_flop_scene(pot: float = 10.0) -> SyntheticScene:
    """hero As Kd 与 board 含 As → validator duplicate_card → STATE_CONFLICT。"""
    scene = flop_scene("conflict", pot=pot)
    truth = dict(scene.truth)
    truth[ROIName.BOARD] = ROITruth(cards=("As", "7d", "2c"))
    return replace(scene, truth=truth)


def scenes_pot_increases() -> list:
    return [preflop_scene("p0", pot=1.5),
            preflop_scene("p1", pot=4.0),
            flop_scene("p2", pot=10.0),
            flop_scene("p3", pot=14.0),
            flop_scene("p4", pot=18.0)]


def scenes_street_progression() -> list:
    return [preflop_scene("s0"), flop_scene("s1", pot=10.0),
            turn_scene(pot=14.0), river_scene(pot=18.0)]


def scenes_new_hand() -> list:
    return [flop_scene("h0", pot=12.0),
            preflop_scene("h1", pot=1.5)]
