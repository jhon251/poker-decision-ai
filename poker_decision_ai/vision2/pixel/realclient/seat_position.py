"""V3D-P1 — Position Runtime Foundation（C4 CANONICAL_RING_MAP / C2 Dealer 映射 /
C1 座占用检测）。

- C4：视觉环 S1→S2→S3→S5→S6→S4→S1 → 环序号（桥接冻结 positions_for 的
  数字排序环；positions_for 零修改）；
- C2：DealerRecognizer 的红 D bbox → 最近 canonical 座锚点 → dealer_seat；
  无 D / 无法映射 → None（UNKNOWN）；
- C1：per-seat ROI（v3a 验证锚点，1419×1066 窗口）→ OCCUPIED/EMPTY/UNKNOWN；
  '空的' 标签 → EMPTY；座牌（名牌+金堆叠）存在 → OCCUPIED；无法可靠 → UNKNOWN；
  S2/S6 在 1419 窗口被裁剪 → 恒 UNKNOWN（不冒充 EMPTY）。
- 语义：UNKNOWN ≠ EMPTY ≠ OCCUPIED；无硬编码 Hero/Dealer/桌型结论。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ...realtime.image import ImageData, to_gray
from .render_class import EVIDENCE_LIMITED, window_for_frame


def _gate_window(w: int, h: int, supported: tuple) -> str | None:
    """P15 几何门：帧几何 → 规范 class，且必须在 supported 集合内；
    否则 → None（消费者返回 EVIDENCE_LIMITED，绝不 fallback 到 1419）。"""
    cls = window_for_frame(w, h)
    if cls is None:
        return None
    return cls if cls in supported else None

# ── C4：CANONICAL_RING_MAP ──────────────────────────────────────────────
CANONICAL_SEATS = ("S1", "S2", "S3", "S5", "S6", "S4")   # 视觉环序（V3C CONFIRMED）
RING_ORDER = {s: i + 1 for i, s in enumerate(CANONICAL_SEATS)}


def ring_number(seat: str | None) -> int | None:
    """视觉 seat → 环序号；非法/UNKNOWN → None；仅顺序桥接，不改 seat 编号。"""
    if seat is None or not isinstance(seat, str):
        return None
    return RING_ORDER.get(seat)


def ring_unmap(rn: int | None) -> str | None:
    if rn is None or not isinstance(rn, int) or not 1 <= rn <= 6:
        return None
    return CANONICAL_SEATS[rn - 1]


# ── C2：Dealer → seat 映射 ──────────────────────────────────────────────
#: canonical 座锚点（full_png 1419×1066 窗口；来源 V3A 标注 SETTLED_961 + V3B）
SEAT_ANCHORS_1419 = {
    "S1": (255, 320),    # top-left（v3a (240,190,175,275) 中心附近）
    "S3": (1000, 200),   # top-right（v3a (870,60,280,290)）
    "S4": (300, 880),    # bottom-left（v3a (190,755,210,270)）
    "S5": (1000, 1000),  # bottom-right（部分可见）
}
#: 完整窗口（1848×1388）canonical 锚点（来源 V3B 决定性截图测量）
SEAT_ANCHORS_1848 = {
    "S1": (255, 310),    # top-left
    "S2": (918, 210),    # top-center
    "S3": (1578, 320),   # top-right
    "S4": (218, 868),    # bottom-left
    "S5": (1618, 868),   # bottom-right
    "S6": (945, 1160),   # bottom-center（本地玩家约定座）
}
ANCHORS_BY_WINDOW = {"1419": SEAT_ANCHORS_1419, "1848": SEAT_ANCHORS_1848}
#: 各窗口不可见座（1419 裁剪 S2/S6；1848 全见）
CROPPED_BY_WINDOW = {"1419": ("S2", "S6"), "1848": ()}
#: 客户端本地玩家座约定（V3B 目视：own UI 于 bottom-center）——仅用作
#: hero 检测的候选座锚（证据门=own UI presence；本身非结论）
OWN_SEAT_CONVENTION = "S6"


def window_anchors(window: str) -> dict:
    return ANCHORS_BY_WINDOW.get(window, ANCHORS_BY_WINDOW["1419"])


def cropped_in(window: str) -> tuple:
    return CROPPED_BY_WINDOW.get(window, CROPPED_BY_WINDOW["1419"])
MAX_ASSOC_DIST = 300.0      # D 中心到锚点最大关联距离（px）


def dealer_seat_from_d(d_center: tuple | None,
                       window: str = "1419") -> str | None:
    """红 D 中心 → canonical 座；无 D/无窗口支持/超距 → None（UNKNOWN）。"""
    if d_center is None:
        return None
    anchors = SEAT_ANCHORS_1419
    best, best_d = None, None
    for seat, (ax, ay) in anchors.items():
        dist = float(((d_center[0] - ax) ** 2 + (d_center[1] - ay) ** 2) ** 0.5)
        if dist <= MAX_ASSOC_DIST and (best_d is None or dist < best_d):
            best, best_d = seat, dist
    return best


class DealerSeatMapper:
    """将 cashgame.DealerRecognizer 输出（button_detected/bbox）映射到 seat。

    输入：dealer_recognizer_result（dict，含 bbox/button_detected）
    输出：{"dealer_seat": str|None, "confidence": float, "evidence": dict}
    """

    name = "coinpoker_dealer_seat_mapping"

    def map(self, dealer_result: dict) -> dict:
        if not dealer_result.get("button_detected"):
            return {"dealer_seat": None, "confidence": 0.0,
                    "reason": "no_detected", "evidence": {}}
        bbox = dealer_result.get("bbox")
        if not bbox or len(bbox) < 2:
            return {"dealer_seat": None, "confidence": 0.0,
                    "reason": "no_bbox", "evidence": {}}
        cx = bbox[0] + (bbox[2] // 2 if len(bbox) > 2 else 0)
        cy = bbox[1] + (bbox[3] // 2 if len(bbox) > 3 else 0)
        seat = dealer_seat_from_d((cx, cy))
        if seat is None:
            return {"dealer_seat": None, "confidence": 0.0,
                    "reason": "unassociable", "evidence": {"d_center":
                                                           (cx, cy)}}
        return {"dealer_seat": seat, "confidence": 0.9,
                "reason": "d_nearest_seat", "evidence": {"d_center": (cx, cy)}}


# ── C1：座占用检测 ──────────────────────────────────────────────────────
SEAT_ROIS_1419 = {           # full_png（v3a 验证）
    "S1": (240, 250, 200, 240),
    "S3": (870, 150, 280, 230),
    "S4": (180, 800, 240, 240),
    "S5": (900, 970, 380, 96),
}
OCCUPIED, EMPTY, UNKNOWN = "OCCUPIED", "EMPTY", "UNKNOWN"


class SeatOccupancyDetector:
    """per-seat occupied/empty/unknown。

    模式（与 C2 同型，抗帧间坐标漂移）：
      1) 全帧金色堆叠 blob（座牌金文本）→ 中心关联最近 canonical 座锚点
         → OCCUPIED（距离置信）；
      2) 未关联座：座上部小亮文本（"空的"标签）→ EMPTY；
      3) 其余 → UNKNOWN；S2/S6（1419 窗口裁剪）恒 UNKNOWN。
    """

    name = "coinpoker_seat_occupancy"

    def __init__(self, anchors: dict | None = None,
                 max_assoc_dist: float = 320.0, window: str = "1419"):
        self.anchors = anchors or window_anchors(window)
        self.window = window
        self.max_assoc_dist = max_assoc_dist

    @staticmethod
    def _gold_blobs(frame: np.ndarray) -> list:
        from ..geometry import connected_components
        r = frame[..., 0].astype(np.int16)
        g = frame[..., 1].astype(np.int16)
        b = frame[..., 2].astype(np.int16)
        gold = ((r > 150) & (g > 110) & (b < 130) & (r >= g)).astype("uint8")
        return [(int(c.bbox.x + c.bbox.width / 2),
                 int(c.bbox.y + c.bbox.height / 2))
                for c in connected_components(gold, min_area=200)
                if c.bbox.height >= 15 and c.bbox.width >= 35]

    @staticmethod
    def _empty_tag_at(frame: np.ndarray, seat: str,
                      anchors: dict) -> bool:
        """座"空的"标签候选：座上部区域的亮文本小簇（无金堆叠）。"""
        x0, y0 = anchors[seat]
        probe = frame[max(0, y0 - 90):y0 + 60, max(0, x0 - 90):x0 + 90]
        if probe.size == 0:
            return False
        luma = (probe[..., 0].astype(np.int16) * 0.299
                + probe[..., 1].astype(np.int16) * 0.587
                + probe[..., 2].astype(np.int16) * 0.114)
        return float((luma > 200).mean()) >= 0.002

    def detect(self, frame: np.ndarray) -> dict:
        result = {}
        gold_centers = self._gold_blobs(frame)
        associated = {}
        for (cx, cy) in gold_centers:
            best, best_d = None, None
            for seat, (ax, ay) in self.anchors.items():
                dist = float(((cx - ax) ** 2 + (cy - ay) ** 2) ** 0.5)
                if dist <= self.max_assoc_dist and (best_d is None
                                                    or dist < best_d):
                    best, best_d = seat, dist
            if best is not None:
                associated.setdefault(best, []).append(best_d)
        for seat in CANONICAL_SEATS:
            if seat in cropped_in(self.window) or seat not in self.anchors:
                result[seat] = {"status": UNKNOWN, "confidence": 0.0,
                                "reason": "cropped_or_no_anchor"}
                continue
            if seat in associated:
                d = min(associated[seat])
                conf = round(max(0.6, 1.0 - d / (self.max_assoc_dist * 1.2)), 3)
                result[seat] = {"status": OCCUPIED, "confidence": conf,
                                "reason": "gold_stack_nearest_seat",
                                "distance": round(d, 1)}
                continue
            if self._empty_tag_at(frame, seat, self.anchors):
                result[seat] = {"status": EMPTY, "confidence": 0.6,
                                "reason": "empty_tag_like"}
                continue
            result[seat] = {"status": UNKNOWN, "confidence": 0.2,
                            "reason": "no_reliable_signal"}
        return result


# ── C3：Hero own-seat detection ─────────────────────────────────────────
class HeroSeatDetector:
    """Hero 身份 = 客户端本地玩家 UI 证据（own-seat presence）。

    - 候选座：客户端本地玩家约定座（OWN_SEAT_CONVENTION，V3B 目视事实），
      仅作为检测锚点（不是结果）；
    - 证据门：该座存在占座证据（金堆叠/名牌亮文本 presence）——
      hero_seat = 该座；无证据 → UNKNOWN；冲突（多座有 own 风格证据）→ UNKNOWN；
    - 禁用：dealer 反推 / position 反推 / 固定输出 S6。
    """

    name = "coinpoker_hero_own_seat"

    def __init__(self, window: str = "1848"):
        self.window = window
        self.anchors = window_anchors(window)

    def detect(self, frame: np.ndarray) -> dict:
        occ = SeatOccupancyDetector(anchors=self.anchors,
                                    window=self.window).detect(frame)
        cand = OWN_SEAT_CONVENTION
        if cand in cropped_in(self.window):
            return {"hero_seat": None, "confidence": 0.0,
                    "reason": "own_seat_cropped", "occupancy": occ}
        if occ.get(cand, {}).get("status") == OCCUPIED:
            return {"hero_seat": cand, "confidence": 0.9,
                    "reason": "own_seat_presence_evidence",
                    "occupancy": occ}
        return {"hero_seat": None, "confidence": 0.0,
                "reason": "no_own_seat_evidence", "occupancy": occ}


# ── C5：positions_for 接线（冻结零修改）──────────────────────────────────
def _occupied_status_map(occupied: dict) -> dict:
    """归一化 occupancy detector 输出 → {seat: status-string}（integration
    boundary）。

    - detector 可能返回 dict-of-dicts `{seat: {"status": "OCCUPIED", ...}}`
      （SeatOccupancyDetector / SeatOccupancyRecognizer1876），也可能返回
      既有 string map `{seat: "OCCUPIED"}`（测试/证据注入）；
    - 两种都归一化为 string map，仅取 status 字段；
    - 语义绝不改变：UNKNOWN 不转 EMPTY/OCCUPIED；缺失 status → 保留原值
      交给 compute_positions 按未知处理（UNKNOWN-first）。
    """
    out = {}
    for seat, st in (occupied or {}).items():
        if isinstance(st, dict):
            out[seat] = st.get("status")
        else:
            out[seat] = st
    return out


def compute_positions(occupied: dict, dealer_seat: str | None,
                      hero_seat: str | None) -> dict:
    """occupied: {seat: status}；dealer/hero: seat|None。

    - 任一关键证据缺失/冲突 → position 结果 UNKNOWN（null），不 fallback；
    - 视觉座 → 环序号（CANONICAL_RING_MAP）→ 冻结 positions_for。
    """
    from ....domain.poker.models import positions_for

    def unknown(reason):
        return {"position_map": None, "hero_position": None,
                "uncertainty": "UNKNOWN", "reason": reason}

    if dealer_seat is None or hero_seat is None:
        return unknown("dealer_or_hero_missing")
    ringed = []
    for seat, st in occupied.items():
        if st not in ("OCCUPIED",):
            if st == "UNKNOWN":
                return unknown(f"occupied_incomplete_{seat}")
            continue                      # EMPTY 跳过（空座）
        rn = ring_number(seat)
        if rn is None:
            return unknown(f"illegal_seat_{seat}")
        ringed.append({"seat": rn})
    if not ringed:
        return unknown("no_occupied_seats")
    d_rn = ring_number(dealer_seat)
    h_rn = ring_number(hero_seat)
    if d_rn is None or h_rn is None:
        return unknown("illegal_dealer_or_hero")
    if d_rn not in {p["seat"] for p in ringed}:
        return unknown("dealer_not_occupied")
    if h_rn not in {p["seat"] for p in ringed}:
        return unknown("hero_not_occupied")
    pos_map = positions_for(ringed, d_rn)
    hero_position = pos_map.get(h_rn)
    if hero_position is None:
        return unknown("hero_position_unresolved")
    # 环序号 → 视觉座
    visual_map = {ring_unmap(rn): label
                  for rn, label in pos_map.items()}
    return {"position_map": visual_map, "hero_position": hero_position,
            "uncertainty": "RECOGNIZED", "reason": "positions_for_derived"}


# ── C6：输出包装 ───────────────────────────────────────────────────────
class PositionRuntimeResult:

    def __init__(self, occupied_seats=None, dealer_seat=None,
                 hero_seat=None, position=None, topology="6-max",
                 provenance="runtime_state_non_gt"):
        self.table_topology = topology
        self.occupied_seats = occupied_seats or {}
        self.dealer_seat = dealer_seat
        self.hero_seat = hero_seat
        self.hero_position = (position or {}).get("hero_position")
        self.uncertainty = (position or {}).get("uncertainty", "UNKNOWN")
        self.provenance = provenance

    def to_dict(self) -> dict:
        return {"table_topology": self.table_topology,
                "occupied_seats": self.occupied_seats,
                "dealer_seat": self.dealer_seat,
                "hero_seat": self.hero_seat,
                "hero_position": self.hero_position,
                "uncertainty": self.uncertainty,
                "provenance": self.provenance,
                "ground_truth_status": "runtime_state_non_gt"}


# ── V3E-P1：最小 contract adapter（V3D 视觉域 → 冻结 assembly int 域）──
def to_assembly_contract(occupied: dict, dealer_seat: str | None,
                         hero_seat: str | None) -> dict:
    """PositionRuntimeResult → 冻结 assembly 输入域。

    - 视觉座 → C4 环序号（冻结 positions_for 的数字环即视觉环序）；
    - 任何 UNKNOWN/missing → 对应字段 None（如实传播）；
    - 返回 {button_seat, hero_seat, occupied_ringed}（int 域）。
    """
    ringed = {}
    for seat, st in occupied.items():
        if st not in ("OCCUPIED",):
            continue                       # EMPTY 跳过；UNKNOWN 不计入
        rn = ring_number(seat)
        if rn is not None:
            ringed[rn] = True
    return {"button_seat": ring_number(dealer_seat),
            "hero_seat": ring_number(hero_seat),
            "occupied_ringed": ringed}


# ── V3E-P2：Runtime 接线（frame → V3D → result → adapter → assembly）────
def runtime_position_pipeline(frame: np.ndarray | None = None, *,
                              occupied: dict | None = None,
                              dealer_seat: str | None = None,
                              hero_seat: str | None = None,
                              window: str = "1848") -> dict:
    """当前帧 → V3D Position Runtime → PositionRuntimeResult → adapter →
    冻结 assembly 位置步骤（vision2.assembly._positions_for）→ hero.position。

    - frame 提供时：C1 占用检测 + C2 dealer 映射 + C3 hero 检测（全部由帧证据）；
      evidence 参数可显式覆盖（测试/证据注入——不得用 fixture 冒充 runtime，
      E2E 测试区分）；
    - UNKNOWN 全程贯穿：接口层 None → 冻结 _positions_for 的既有 None 语义。
    """
    from .cashgame import DealerRecognizer

    if frame is not None:
        # V3F-P27-C（最小 adapter）：1876 帧 → 三态占用检测器
        # （OCCUPIED/EMPTY/UNKNOWN，P27-A/B.0 冻结证据）；其余窗口保持
        # 既有 SeatOccupancyDetector（1419/1848 regression 不变）。
        if frame.shape[1] == 1876 and frame.shape[0] == 1400:
            from .seat_occupancy import SeatOccupancyRecognizer1876
            occupied = SeatOccupancyRecognizer1876().recognize(frame)
        else:
            occupied = SeatOccupancyDetector(window=window).detect(frame)
        # V3F-P22-C（最小 adapter）：1876 render class（帧几何 1876×1400）→
        # 双D模板 dealer button 检测（白D/暗D，D 字母为主导）；其余窗口保持
        # 既有 DealerRecognizer（1419 regression 不变）。
        if frame.shape[1] == 1876 and frame.shape[0] == 1400:
            from .dealer_button import detect_dealer_button_1876
            dealer_seat = detect_dealer_button_1876(frame)["dealer_seat"]
        else:
            dres = DealerRecognizer().recognize(frame)
            dealer_seat = DealerSeatMapper().map(dres)["dealer_seat"]
        # V3F-P28-B（最小 adapter）：1876 帧 → own-seat pill hero 检测
        # （P28-A 冻结证据；UNKNOWN-first）；其余窗口保持既有
        # HeroSeatDetector（1419/1848 regression 不变）。
        if frame.shape[1] == 1876 and frame.shape[0] == 1400:
            from .hero_seat import HeroSeatDetector1876
            hero_seat = HeroSeatDetector1876().detect(frame)["hero_seat"]
        else:
            hero_seat = HeroSeatDetector(window=window).detect(frame)[
                "hero_seat"]
    # V3F-INTEGRATION-FIX-1：归一化 occupancy 形状（dict-of-dicts → string
    # map）后再消费——修复 FINAL AUDIT §8 的 no_occupied_seats 恒 UNKNOWN。
    occ_map = _occupied_status_map(occupied)
    pos = compute_positions(occ_map, dealer_seat, hero_seat)
    result = PositionRuntimeResult(occupied_seats=occupied or {},
                                   dealer_seat=dealer_seat,
                                   hero_seat=hero_seat, position=pos)
    contract = to_assembly_contract(occ_map, dealer_seat, hero_seat)
    # UNKNOWN 门控：证据不完整/冲突 → 不进入冻结消费步（防止部分环伪结果）
    if pos.get("uncertainty") != "RECOGNIZED":
        return {"runtime_result": result.to_dict(),
                "contract": contract,
                "positions_ring": None,
                "hero_position": None,
                "ground_truth_status": "runtime_state_non_gt"}
    # 冻结 assembly 位置步骤（zero-modification 消费）
    from ....vision2.assembly import _positions_for
    players_view = [{"seat": rn} for rn in sorted(contract["occupied_ringed"])]
    pos_map = _positions_for(players_view, contract["button_seat"])
    hero_ring = contract["hero_seat"]
    hero_position = pos_map.get(hero_ring) if hero_ring is not None else None
    return {"runtime_result": result.to_dict(),
            "contract": contract,
            "positions_ring": pos_map,
            "hero_position": hero_position,
            "ground_truth_status": "runtime_state_non_gt"}


# ── V3E-P3：Capture Pipeline 消费者 hook ───────────────────────────────
class PositionObserver:
    """采集管线逐帧消费者（新增，不改 capture 行为）。

    - observe(frame) 仅消费当前帧 → Position Runtime → 记录；
    - 无缓存/历史回退（每帧独立，Case F 天然安全）；
    - last/summary 供报告输出。
    """

    name = "coinpoker_position_runtime_observer"

    def __init__(self, window: str = "1419"):
        self.window = window
        self.last = None
        self.count = 0
        self.recognized = 0
        self.unknown = 0
        self.evidence_limited = 0

    @staticmethod
    def _window_for(w: int, h: int) -> str | None:
        """1876 → 1848 锚点（S6 底中验证）；1419/1848 → 原样；其它 → None。"""
        cls = window_for_frame(w, h)
        if cls == "1876":
            return "1848"
        if cls in ("1419", "1848"):
            return cls
        return None

    def observe(self, frame_rgba: bytes | np.ndarray, w: int, h: int) -> dict:
        win = self._window_for(w, h)
        if win is None:
            out = {"runtime_result": {"uncertainty": "UNKNOWN"},
                   "hero_position": None, "positions_ring": None,
                   "contract": {}, "status": EVIDENCE_LIMITED,
                   "reason": "unsupported_render_class",
                   "window": f"{int(w)}x{int(h)}",
                   "ground_truth_status": "runtime_state_non_gt"}
            self.last = out
            self.count += 1
            self.evidence_limited += 1
            return out
        frame = (np.frombuffer(frame_rgba, dtype=np.uint8).reshape(h, w, 4)
                 if isinstance(frame_rgba, bytes) else frame_rgba)
        out = runtime_position_pipeline(frame=frame, window=win)
        self.last = out
        self.count += 1
        if out["runtime_result"]["uncertainty"] == "RECOGNIZED":
            self.recognized += 1
        else:
            self.unknown += 1
        return out

    def summary(self) -> dict:
        return {"frames": self.count, "recognized": self.recognized,
                "unknown_frames": self.unknown,
                "evidence_limited": self.evidence_limited,
                "last_hero_position": (self.last or {}).get("hero_position"),
                "last_uncertainty": (self.last or {}).get(
                    "runtime_result", {}).get("uncertainty"),
                "ground_truth_status": "runtime_state_non_gt"}


# ── V3F-P1：Pot 运行时消费者 ──────────────────────────────────────────
class PotObserver:
    """采集管线逐帧消费者（与 PositionObserver 同型，纯消费者）。

    - observe(frame) 仅消费当前帧 → 冻结 PotRecognizer → 记录；
    - 无缓存/历史回退；UNKNOWN 原样保留（**不转 0/前值/默认**）；
    - 结果不影响 capture（不回馈 trigger/ROI/保存）。
    """

    name = "coinpoker_pot_runtime_observer"

    def __init__(self):
        from .cashgame import PotRecognizer
        self._recognizers = {}
        self.last = None
        self.count = 0
        self.recognized = 0
        self.unknown = 0
        self.evidence_limited = 0

    def _recognizer_for(self, w: int, h: int):
        cls = _gate_window(w, h, ("1419", "1876"))
        if cls is None:
            return None, None
        if cls not in self._recognizers:
            from .cashgame import PotRecognizer
            self._recognizers[cls] = PotRecognizer(window=cls)
        return cls, self._recognizers[cls]

    def observe(self, frame_rgba: bytes | np.ndarray, w: int, h: int) -> dict:
        cls, rec = self._recognizer_for(w, h)
        if rec is None:
            reason = ("unsupported_render_class"
                      if window_for_frame(w, h) is not None
                      else "unknown_render_class")
            self.last = {"pot": None, "status": EVIDENCE_LIMITED,
                         "reason": reason,
                         "window": f"{int(w)}x{int(h)}",
                         "ground_truth_status": "runtime_state_non_gt",
                         "raw_text": None}
            self.count += 1
            self.evidence_limited += 1
            return self.last
        frame = (np.frombuffer(frame_rgba, dtype=np.uint8).reshape(h, w, 4)
                 if isinstance(frame_rgba, bytes) else frame_rgba)
        res = rec.recognize(frame)
        status = res.get("status")
        self.last = {
            "pot": res.get("value"),
            "status": status,
            "reason": res.get("reason"),
            "ground_truth_status": "runtime_state_non_gt",
            "raw_text": res.get("raw_text"),
            "window": cls,
        }
        self.count += 1
        if status == "RECOGNIZED":
            self.recognized += 1
        else:
            self.unknown += 1
        return self.last

    def summary(self) -> dict:
        return {"frames": self.count, "recognized": self.recognized,
                "unknown_frames": self.unknown,
                "evidence_limited": self.evidence_limited,
                "last_pot": (self.last or {}).get("pot"),
                "last_status": (self.last or {}).get("status"),
                "last_window": (self.last or {}).get("window"),
                "ground_truth_status": "runtime_state_non_gt"}


# ── V3F-P2：Board 运行时消费者 ────────────────────────────────────────
class BoardObserver:
    """采集管线逐帧消费者（与 Position/PotObserver 同型，纯消费者）。

    - observe(bytes, w, h) 仅消费当前帧 → 冻结 CoinPokerBoardRecognizer；
    - 输出板卡 rank/suit + count + status；**street 显式 None**——
      V3F-P2 不推导 Street（Board→Street 属后续阶段，本阶段红线隔离）；
    - 无缓存/历史回退；UNKNOWN 原样（不转空板/前板/默认板）。
    """

    name = "coinpoker_board_runtime_observer"

    def __init__(self):
        from .coinpoker_board import CoinPokerBoardRecognizer
        self._recognizers = {}
        self.last = None
        self.count = 0
        self.recognized = 0
        self.unknown = 0
        self.evidence_limited = 0
        self._fid = 0

    def _recognizer_for(self, w: int, h: int):
        cls = _gate_window(w, h, ("1419", "1876"))
        if cls is None:
            return None, None
        if cls not in self._recognizers:
            from .coinpoker_board import CoinPokerBoardRecognizer
            self._recognizers[cls] = CoinPokerBoardRecognizer(window=cls)
        return cls, self._recognizers[cls]

    def observe(self, frame_rgba: bytes, w: int, h: int) -> dict:
        self._fid = getattr(self, "_fid", 0) + 1
        cls, rec = self._recognizer_for(w, h)
        if rec is None:
            self.last = {
                "cards": [], "count": 0, "card_count": 0,
                "status": EVIDENCE_LIMITED, "slot_states": [],
                "street": None, "window": f"{int(w)}x{int(h)}",
                "reason": "unsupported_render_class",
                "ground_truth_status": "runtime_state_non_gt",
            }
            self.count += 1
            self.evidence_limited += 1
            return self.last
        res = rec.recognize(frame_rgba, w, h, self._fid)
        cards = res.get("cards") or []
        ok = [c for c in cards if c.get("status") == "OK"
              and c.get("rank") and c.get("suit")]
        rank_only = [c for c in cards if c.get("status") == "RANK_ONLY"
                     and c.get("rank")]
        status = res.get("status")
        self.last = {
            "cards": [f"{c['rank']}{c['suit']}" for c in ok],
            "count": len(ok),
            "card_count": len(ok),
            # V-2 半置信分层（additive）：suit 未定的 rank 卡——仅计数，
            # 不进最终牌面（牌面产品只消费 FULL）
            "rank_only_cards": [c["rank"] for c in rank_only],
            "countable_count": len(ok) + len(rank_only),
            "status": "RECOGNIZED" if (status == "OK" and ok)
            else (status or "UNKNOWN"),
            "slot_states": [c.get("status") for c in cards],
            "street": None,                      # 红线：本阶段不推导 Street
            "window": cls,
            "ground_truth_status": "runtime_state_non_gt",
        }
        self.count += 1
        if self.last["status"] == "RECOGNIZED":
            self.recognized += 1
        else:
            self.unknown += 1
        return self.last

    def summary(self) -> dict:
        return {"frames": self.count, "recognized": self.recognized,
                "unknown_frames": self.unknown,
                "evidence_limited": self.evidence_limited,
                "last_cards": (self.last or {}).get("cards"),
                "last_count": (self.last or {}).get("count"),
                # V-2 additive：可计数控（FULL+RANK_ONLY）与 rank-only 明细
                "last_countable_count": (self.last or {}).get(
                    "countable_count"),
                "last_rank_only_cards": (self.last or {}).get(
                    "rank_only_cards"),
                "last_street": (self.last or {}).get("street"),
                "last_window": (self.last or {}).get("window"),
                "ground_truth_status": "runtime_state_non_gt"}


# ── V3F-P3：Street 运行时消费者 ───────────────────────────────────────
class StreetObserver:
    """街线运行时消费者。

    - 输入：当前帧 →（复用冻结 CoinPokerBoardRecognizer 的 slot 证据）→
      冻结 BoardStateMachine（合法状态机：0→3→4→5 严格转换、确认/泄漏门）
    - 输出：street + state + confidence + reason；UNKNOWN/leak 语义来自
      状态机本身（无 previous-value recognition fallback——仅合法 state
      continuity）；
    - 不复制"count→street"映射（唯一映射在冻结机器内）。
    """

    name = "coinpoker_street_runtime_observer"

    def __init__(self, session_id: str = "capture_runtime",
                 settled: bool = False):
        from .board_state import BoardStateMachine
        self._board_recognizers = {}
        self.machine = BoardStateMachine(session_id=session_id)
        self.settled = settled
        self.last = None
        self.count = 0
        self.evidence_limited = 0
        self._fid = 0

    def _board_for(self, w: int, h: int):
        cls = _gate_window(w, h, ("1419", "1876"))
        if cls is None:
            return None, None
        if cls not in self._board_recognizers:
            from .coinpoker_board import CoinPokerBoardRecognizer
            self._board_recognizers[cls] = CoinPokerBoardRecognizer(window=cls)
        return cls, self._board_recognizers[cls]

    def observe(self, frame_rgba: bytes, w: int, h: int) -> dict:
        self._fid += 1
        cls, rec = self._board_for(w, h)
        if rec is None:
            self.last = {"street": None, "state": None, "cardinality": None,
                         "observed_cardinality": None,
                         "confidence": None, "reason": None,
                         "status": EVIDENCE_LIMITED,
                         "window": f"{int(w)}x{int(h)}",
                         "ground_truth_status": "runtime_state_non_gt"}
            self.count += 1
            self.evidence_limited += 1
            return self.last
        res = rec.recognize(frame_rgba, w, h, self._fid)
        # V-2 半置信分层：street 只需要"几张"——张数消费 FULL+RANK_ONLY；
        # 机器 token = rank 字符（花色不进状态机：泄漏门按 rank 稳定性
        # 比较——RANK_ONLY→FULL 的 suit 精化不构成卡变化，rank 变化才构成，
        # BoardStateMachine 冻结语义零改动，仅输入表示分层）。
        cards = [(int(c.get("slot_index", i)), str(c.get("rank")))
                 for i, c in enumerate(res.get("cards") or [])
                 if c.get("status") in ("OK", "RANK_ONLY") and c.get("rank")]
        tiers = [c.get("tier") for c in (res.get("cards") or [])
                 if c.get("status") in ("OK", "RANK_ONLY")]
        rec_res = self.machine.observe(self._fid, cards,
                                       settled=self.settled)
        # 无记录（确认中）→ 输出机器当前状态（pending_confirm，非回退）
        state = rec_res.get("state") or self.machine.state
        self.last = {"street": state,
                     "state": state,
                     "cardinality": rec_res.get("cardinality")
                     if rec_res.get("cardinality") is not None
                     else self.machine.cardinality,
                     # V-1 FM-1：本帧识别卡数（与 machine.cardinality 严格
                     # 区分——pending_confirm 期间机器 cardinality 停留旧值/0，
                     # 单帧豁免决议需要本帧真实识别证据）
                     # V-2：口径 = FULL+RANK_ONLY 可计数控
                     "observed_cardinality": len(cards),
                     "observed_tiers": tiers,
                     "full_cards": res.get("value"),
                     "confidence": rec_res.get("confidence"),
                     "reason": rec_res.get("transition_reason")
                     or ("pending_confirm" if not rec_res else None),
                     "window": cls,
                     "ground_truth_status": "runtime_state_non_gt"}
        self.count += 1
        return self.last

    def summary(self) -> dict:
        return {"frames": self.count,
                "evidence_limited": self.evidence_limited,
                "last_street": (self.last or {}).get("street"),
                "last_cardinality": (self.last or {}).get("cardinality"),
                "last_observed_cardinality": (self.last or {}).get(
                    "observed_cardinality"),
                "last_window": (self.last or {}).get("window"),
                "ground_truth_status": "runtime_state_non_gt"}


# ── V3F-P4：Hero Cards 运行时消费者 ────────────────────────────────────
class HeroCardsObserver:
    """采集管线逐帧消费者（与其它 observer 同型，纯消费者）。

    - observe(bytes, w, h) 仅消费当前帧 → 冻结 CoinPokerHeroRecognizer；
    - UNKNOWN/UNCERTAIN 原样（value=None——"未识别"≠"无手牌"，不转
      []/默认/前值/推测；无 cache/previous-frame fallback）；
    - 构造注入冻结配置（assets + hero_card_layout_v1 —— 客户端 own-card
      区域布局标注，非 fixture 冒充 runtime）。
    """

    name = "coinpoker_hero_cards_runtime_observer"

    def __init__(self, assets_dir: str = "data/vision2/coinpoker_assets_v2zd_hero",
                 layout_path: str = "data/vision2/hero_card_layout_v1.json"):
        from .coinpoker_hero import CoinPokerHeroRecognizer
        self.recognizer = CoinPokerHeroRecognizer(assets_dir, layout_path)
        self.last = None
        self.count = 0
        self.recognized = 0
        self.unknown = 0
        self.evidence_limited = 0
        self._fid = 0

    def observe(self, frame_rgba: bytes, w: int, h: int) -> dict:
        self._fid += 1
        cls = window_for_frame(w, h)
        if cls == "1876":
            # V3F-P29-B（最小 adapter）：1876 帧 → GT-provenance 模板
            # HeroCardsRecognizer1876（P29-A 证据：双卡角标模板，UNKNOWN-first）；
            # 1419 路径保持既有 recognizer（零改动）。
            from .coinpoker_herocards1876 import HeroCardsRecognizer1876
            frame = np.frombuffer(frame_rgba, dtype=np.uint8).reshape(h, w, 4)
            res = HeroCardsRecognizer1876().recognize(frame)
            ok = res.get("status") == "OK" and res.get("value")
            self.last = {
                "cards": res.get("value") if ok else None,
                "value": res.get("value") if ok else None,
                "card_states": [c.get("status") for c in
                                (res.get("cards") or [])],
                "status": "RECOGNIZED" if ok else "UNCERTAIN",
                "reason": res.get("reason"),
                "window": f"{int(w)}x{int(h)}",
                "ground_truth_status": "runtime_state_non_gt"}
            self.count += 1
            if ok:
                self.recognized += 1
            else:
                self.unknown += 1
            return self.last
        if cls != "1419":
            # V3F-P15：1419 之外的渲染类未提取 → EVIDENCE_LIMITED。
            reason = ("hero_cards_1876_layout_not_extracted"
                      if cls is None else "unsupported_render_class")
            self.last = {"cards": None, "value": None, "card_states": [],
                         "status": EVIDENCE_LIMITED, "reason": reason,
                         "window": f"{int(w)}x{int(h)}",
                         "ground_truth_status": "runtime_state_non_gt"}
            self.count += 1
            self.evidence_limited += 1
            return self.last
        res = self.recognizer.recognize(frame_rgba, w, h, self._fid)
        ok = (res.get("status") == "OK" and res.get("value"))
        self.last = {"cards": res.get("value") if ok else None,
                     "value": res.get("value") if ok else None,
                     "card_states": [c.get("status") for c in
                                     (res.get("cards") or [])],
                     "status": "RECOGNIZED" if ok else "UNCERTAIN",
                     "window": "1419",
                     "ground_truth_status": "runtime_state_non_gt"}
        self.count += 1
        if ok:
            self.recognized += 1
        else:
            self.unknown += 1
        return self.last

    def summary(self) -> dict:
        return {"frames": self.count, "recognized": self.recognized,
                "unknown_frames": self.unknown,
                "evidence_limited": self.evidence_limited,
                "last_cards": (self.last or {}).get("cards"),
                "last_status": (self.last or {}).get("status"),
                "last_window": (self.last or {}).get("window"),
                "ground_truth_status": "runtime_state_non_gt"}


# ── V3F-P7：Player Status 运行时消费者 ────────────────────────────────
class StatusObserver:
    """采集管线逐帧消费者（与其它 observer 同型，纯消费者）。

    - observe(bytes, w, h) → 冻结 PlayerStatusRecognizer（全帧徽章搜索）→
      徽章 bbox → 座锚 ownership（复用 C1 关联模式）；
    - 语义：RECOGNIZED（徽章）· NOT_OBSERVED（无徽章匹配）· UNKNOWN
      （无模板）——分离不混淆；无缓存/previous/fixture/默认状态。
    """

    name = "coinpoker_status_runtime_observer"

    def __init__(self, window: str = "1419"):
        from .cashgame import PlayerStatusRecognizer
        self.recognizer = PlayerStatusRecognizer()
        self.anchors = SEAT_ANCHORS_1419
        self.window = window
        self.last = None
        self.count = 0
        self.recognized = 0
        self.not_observed = 0
        self.unknown = 0
        self.evidence_limited = 0

    def _associate(self, bbox) -> str | None:
        cx = bbox[0] + bbox[2] // 2
        cy = bbox[1] + bbox[3] // 2
        best, best_d = None, None
        for seat, (ax, ay) in self.anchors.items():
            d = float(((cx - ax) ** 2 + (cy - ay) ** 2) ** 0.5)
            if d <= 320.0 and (best_d is None or d < best_d):
                best, best_d = seat, d
        return best

    def observe(self, frame_rgba: bytes, w: int, h: int) -> dict:
        cls = window_for_frame(w, h)
        if cls == "1876":
            # V3F-P25-C：1876 Status 生产路径（P25-B.3 冻结判据 + P25-C
            # disc/away/allin）——独立识别器；1419 路径不变。
            from .coinpoker_status import StatusRecognizer1876
            rec = getattr(self, "_rec1876", None)
            if rec is None:
                rec = StatusRecognizer1876()
                self._rec1876 = rec
            frame = np.frombuffer(frame_rgba, dtype=np.uint8).reshape(h, w, 4)
            res = rec.recognize(frame)
            seats = {seat: {"status": v["status"], "reason": v["reason"],
                            "evidence": v["evidence"]}
                     for seat, v in res["seats"].items()}
            self.last = {"status": res["status"], "seats": seats,
                         "reason": res["reason"], "window": "1876",
                         "ground_truth_status": "runtime_state_non_gt"}
            self.count += 1
            if res["status"] == "RECOGNIZED":
                self.recognized += 1
            else:
                self.not_observed += 1
            return self.last
        if cls != "1419":
            # V3F-P15：未校准 render class → 诚实 EVIDENCE_LIMITED（绝不
            # 用 1419 锚点/模板误关联）。
            self.last = {"status": EVIDENCE_LIMITED, "seats": {},
                         "reason": "unsupported_render_class",
                         "window": f"{int(w)}x{int(h)}",
                         "ground_truth_status": "runtime_state_non_gt"}
            self.count += 1
            self.evidence_limited += 1
            return self.last
        frame = np.frombuffer(frame_rgba, dtype=np.uint8).reshape(h, w, 4)
        res = self.recognizer.recognize(frame)
        seats = {}
        for label, info in (res.get("badges") or {}).items():
            seat = self._associate(info["bbox"])
            seats[seat or "unseated"] = {"status": label,
                                         "score": info["score"],
                                         "bbox": info["bbox"]}
        status = res.get("status")
        self.last = {"status": status, "seats": seats,
                     "reason": res.get("reason"),
                     "window": "1419",
                     "ground_truth_status": "runtime_state_non_gt"}
        self.count += 1
        if status == "RECOGNIZED":
            self.recognized += 1
        elif status == "NOT_OBSERVED":
            self.not_observed += 1
        else:
            self.unknown += 1
        return self.last

    def summary(self) -> dict:
        return {"frames": self.count, "recognized": self.recognized,
                "not_observed": self.not_observed, "unknown": self.unknown,
                "evidence_limited": self.evidence_limited,
                "last_status": (self.last or {}).get("status"),
                "last_seats": (self.last or {}).get("seats"),
                "last_window": (self.last or {}).get("window"),
                "ground_truth_status": "runtime_state_non_gt"}


# ── V3F-P8：Stack 运行时消费者 ────────────────────────────────────────
#: 座级 stack 文本 region（table-stable 几何——P5 稳定源座区；1419 窗口）
SEAT_STACK_REGIONS_1419 = {
    "S1": (200, 360, 470, 500),      # top-left stack line（953,909/388,080）
    "S3": (900, 80, 1180, 330),      # top-right stack line
    "S4": (220, 950, 420, 1060),     # bottom-left stack line（318,871/434,504）
    "S5": (880, 950, 1240, 1060),    # bottom-right stack line
}


class StackObserver:
    """采集管线逐帧消费者（与其它 observer 同型，纯消费者）。

    - observe(bytes, w, h) → 冻结 StackRecognizer.recognize_in_region
      （raw 行带工作流）per-seat → {seat:{stack,raw_text,status}}；
    - UNKNOWN-first（未识别≠0/前值/默认）；无缓存/previous/fixture；
    - S2/S6（1419 裁剪）恒 UNKNOWN；不引用 Pot/Bet。
    """

    name = "coinpoker_stack_runtime_observer"

    def __init__(self, window: str = "1419",
                 regions: dict | None = None):
        from .coinpoker_stack import StackRecognizer
        self._recognizers = {}
        self.regions = regions or SEAT_STACK_REGIONS_1419
        self.window = window
        self.last = None
        self.count = 0
        self.recognized = 0
        self.unknown = 0
        self.evidence_limited = 0

    def _recognizer_for(self, w: int, h: int):
        cls = window_for_frame(w, h)
        if cls == "1876":
            # V3F-P26-B：1876 Stack 专用路径（P26-A 深分割修正 ROI +
            # 背景相对 gold 判据）；不用 1848 blob 关联旧路径。
            key = "1876"
        elif cls == "1419":
            key = "1419"
        else:
            return None, None
        if key not in self._recognizers:
            from .coinpoker_stack import StackRecognizer
            if key == "1876":
                from .coinpoker_stack import StackRecognizer1876
                self._recognizers[key] = StackRecognizer1876()
            else:
                self._recognizers[key] = StackRecognizer(window=key)
        return cls, key, self._recognizers[key]

    def observe(self, frame_rgba: bytes, w: int, h: int) -> dict:
        got = self._recognizer_for(w, h)
        if got[1] is None:
            reason = ("stack_1876_anchor_unverified"
                      if window_for_frame(w, h) == "1876"
                      else "unsupported_render_class")
            self.last = {"seats": {}, "status": EVIDENCE_LIMITED,
                         "reason": reason, "window": f"{int(w)}x{int(h)}",
                         "ground_truth_status": "runtime_state_non_gt"}
            self.count += 1
            self.evidence_limited += 1
            return self.last
        cls, key, rec = got
        frame = np.frombuffer(frame_rgba, dtype=np.uint8).reshape(h, w, 4)
        if key == "1876":
            # V3F-P26-B：1876 Stack 专用路径（per-seat ROI → 背景相对 gold
            # → 8-conn → NCC；UNKNOWN-first）
            raw = rec.recognize(frame)
            seats = {seat: {"stack": r.get("stack"), "raw_text":
                            r.get("raw_text"), "status": r.get("status"),
                            "reason": r.get("reason")}
                     for seat, r in raw.items()}
        elif key == "1848":
            # 1876 全窗：1848 锚点 blob 关联（S1-S6 全见，无裁剪座）
            raw = rec.recognize(frame)
            seats = {seat: {"stack": r.get("value"), "raw_text":
                            r.get("raw_text"), "status": r.get("status"),
                            "reason": r.get("reason")}
                     for seat, r in raw.items()}
        else:
            seats = {}
            for seat, region in self.regions.items():
                res = rec.recognize_in_region(frame, region)
                seats[seat] = {"stack": res.get("value"), "raw_text":
                               res.get("raw_text"), "status": res.get("status"),
                               "reason": res.get("reason")}
            # 裁剪座（S2/S6）→ UNKNOWN（诚实——非 EMPTY/0）
            for seat in ("S2", "S6"):
                if seat not in seats:
                    seats[seat] = {"stack": None, "raw_text": None,
                                   "status": "UNKNOWN", "reason": "cropped"}
        self.last = {"seats": seats, "window": cls,
                     "ground_truth_status": "runtime_state_non_gt"}
        self.count += 1
        if any(v["status"] == "RECOGNIZED" for v in seats.values()):
            self.recognized += 1
        else:
            self.unknown += 1
        return self.last

    def summary(self) -> dict:
        return {"frames": self.count, "recognized": self.recognized,
                "unknown_frames": self.unknown,
                "evidence_limited": self.evidence_limited,
                "last_seats": (self.last or {}).get("seats"),
                "last_window": (self.last or {}).get("window"),
                "ground_truth_status": "runtime_state_non_gt"}


def seat_state_from_frame(rgba_frame: bytes, w: int, h: int) -> dict:
    """便捷入口：帧 → 占用状态表（1419 窗口）。"""
    frame = np.frombuffer(rgba_frame, dtype=np.uint8).reshape(h, w, 4)
    return SeatOccupancyDetector().detect(frame)
