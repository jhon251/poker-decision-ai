"""V2I — RealClientLayoutProfile（真实客户端布局固化，纯几何事实）。

原则（本阶段）：
  - ROI 是几何事实；绝不由识别结果反推 ROI；
  - 每个 ROI 保存 provenance：semantic_name / coordinate_system /
    bbox(client_px + normalized) / source / calibration_version /
    confidence / evidence_frame_id；
  - screen → window → client → normalized 逐级，禁止只存绝对像素；
  - 未测量的子 ROI = null + UNCERTAIN（不猜测）；
  - 碰撞检查：任何 ROI 不得与另一语义区域重叠（S4）。

测量来源（10 张真实截图只读分析，V2F/V2H 已核验）：
  - pot 文本:    gold bbox (852,319,144,39) → 容器 (846,314,156,49)
  - hero_cards:  白双卡 (805,1010,245,178)（帧1-9 几何一致）
  - hero stack:  金色堆叠 (850,1242,146,90)（=bet_area 内容）
  - board:       卡面字形 y 385-500 → 容器 (620,350,620,200)
  - button(D):   红六边形 (750,970,60,60)（帧6 真值）
  - action_area: call(800,1140,120,65) + fold(1110,1295,230,65) 带
  - players:     6 座沿桌缘扇区（hero=座0 底中；其余 5 座确定性扇区；
                 各座 stack/bet/status 子 ROI 未测量 → null+UNCERTAIN）
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ...realtime.coords import Rect
from ...realtime.roi import ROIName, ROISpec, default_roi_specs

CLIENT_SIZE = (1848.0, 1388.0)

#: 实测 hero 卡区（向后兼容：V2E/V2F 使用）
HERO_CARDS_BBOX = Rect(805.0, 1010.0, 245.0, 178.0)


def real_layout_specs() -> dict:
    """默认布局 + hero_cards 实测修正（V2E/V2F 提取脚本沿用）。"""
    specs = dict(default_roi_specs())
    norm = Rect(HERO_CARDS_BBOX.x / CLIENT_SIZE[0],
                HERO_CARDS_BBOX.y / CLIENT_SIZE[1],
                HERO_CARDS_BBOX.width / CLIENT_SIZE[0],
                HERO_CARDS_BBOX.height / CLIENT_SIZE[1])
    specs[ROIName.HERO_CARDS] = ROISpec(
        ROIName.HERO_CARDS, "table",
        Rect(norm.x, norm.y, norm.width, norm.height),
        required=True, read_layer="cards")
    return specs


def _norm(rect: Rect) -> dict:
    return {"x": rect.x / CLIENT_SIZE[0], "y": rect.y / CLIENT_SIZE[1],
            "width": rect.width / CLIENT_SIZE[0],
            "height": rect.height / CLIENT_SIZE[1]}


@dataclass
class ROIEntry:
    semantic_name: str
    coordinate_system: str                      # "client"
    bbox: Rect | None                           # client 物理 px；null=未测量
    source: str                                 # bootstrap|measured|derived
    calibration_version: int
    confidence: float | None                    # null=未校准
    evidence_frame_id: list = field(default_factory=list)

    def to_dict(self) -> dict:
        bbox = None
        if self.bbox is not None:
            bbox = {"x": self.bbox.x, "y": self.bbox.y,
                    "width": self.bbox.width, "height": self.bbox.height,
                    "normalized": _norm(self.bbox)}
        return {"semantic_name": self.semantic_name,
                "coordinate_system": self.coordinate_system,
                "bbox": bbox, "source": self.source,
                "calibration_version": self.calibration_version,
                "confidence": self.confidence,
                "evidence_frame_id": self.evidence_frame_id}


class RealClientLayoutProfile:
    """真实客户端布局档案（v1；几何事实，非识别结果）。"""

    def __init__(self, client_size=CLIENT_SIZE, calibration_version=1):
        self.client_size = client_size
        self.calibration_version = calibration_version
        self.entries: dict = {}
        self._seats = 6

    def add(self, name: str, bbox, source, confidence,
            evidence_frame_id=None) -> None:
        self.entries[name] = ROIEntry(
            name, "client", bbox, source, self.calibration_version,
            confidence, list(evidence_frame_id or []))

    # ------------------------------------------------------------------
    @classmethod
    def measured(cls) -> "RealClientLayoutProfile":
        """基于 10 张真实截图实测的布局（只读测量的固化结果）。"""
        p = cls()
        p.add("table", Rect(0.0, 0.0, 1848.0, 1388.0), "measured", 1.0,
              evidence_frame_id=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        p.add("hero_cards", Rect(805.0, 1010.0, 245.0, 178.0), "measured",
              0.95, evidence_frame_id=[1, 2, 3, 5, 7, 9])
        p.add("board", Rect(620.0, 380.0, 620.0, 180.0), "measured", 0.8,
              evidence_frame_id=[1, 2, 3, 7])
        p.add("pot", Rect(846.0, 314.0, 156.0, 49.0), "measured", 0.9,
              evidence_frame_id=list(range(1, 11)))
        p.add("button", Rect(748.0, 968.0, 55.0, 55.0), "measured", 0.85,
              evidence_frame_id=[6])
        p.add("bet_area", Rect(850.0, 1242.0, 146.0, 90.0), "measured", 0.85,
              evidence_frame_id=[5, 6])
        p.add("action_call", Rect(700.0, 1120.0, 240.0, 120.0), "measured",
              0.8, evidence_frame_id=[1, 6])
        p.add("action_fold", Rect(1000.0, 1240.0, 400.0, 140.0), "measured",
              0.8, evidence_frame_id=[1, 6])
        # 逐座扇区（hero=座0 底中；其余确定性几何扇区，堆叠/下注/状态子区未测量）
        p._build_seat_sectors()
        return p

    # ------------------------------------------------------------------
    @classmethod
    def measured_v2(cls) -> "RealClientLayoutProfile":
        """V2J：逐座实测（6 座全部可见并测量；pot 修正为客户端不显示）。

        实测事实（10 帧帧1 个例 + 10 帧存在性统计）：
          - 桌心 10 帧零金文本 → 客户端不显示 pot 金额（pot=null，视觉事实）；
          - V2F 的"pot"文本实为上座(stack)堆叠文本——语义已修正；
          - 6 座 stack 金文本 bbox 全部实测（存在帧数各异：座满才显示 stack）。
        """
        p = cls()
        p.add("table", Rect(0.0, 0.0, 1848.0, 1388.0), "measured", 1.0,
              evidence_frame_id=list(range(1, 11)))
        # pot：客户端无显示（桌心 10 帧零金文本）→ null + UNCERTAIN
        p.entries["pot"] = ROIEntry("pot", "client", None,
                                    "measured_absent", 1, None,
                                    evidence_frame_id=list(range(1, 11)))
        p.add("hero_cards", Rect(805.0, 1010.0, 245.0, 178.0), "measured",
              0.95, evidence_frame_id=[1, 2, 3, 5, 7, 9])
        p.add("board", Rect(620.0, 380.0, 620.0, 180.0), "measured", 0.8,
              evidence_frame_id=[1, 2, 3, 7])
        p.add("button", Rect(748.0, 968.0, 55.0, 55.0), "measured", 0.85,
              evidence_frame_id=[6])
        p.add("bet_area", Rect(850.0, 1242.0, 146.0, 90.0), "measured", 0.85,
              evidence_frame_id=[5, 6])
        p.add("action_call", Rect(700.0, 1120.0, 240.0, 120.0), "measured",
              0.8, evidence_frame_id=[1, 6])
        p.add("action_fold", Rect(1000.0, 1240.0, 400.0, 140.0), "measured",
              0.8, evidence_frame_id=[1, 6])
        # 逐座（V2J 实测；帧1 个例 bbox + 10 帧存在性；未观察子 ROI=null）
        SEATS = {
            # player_region, stack_roi, bet_roi, status_roi, identity_roi
            0: (Rect(700.0, 950.0, 400.0, 450.0),
                Rect(850.0, 1242.0, 147.0, 95.0),
                None, None, None),
            1: (Rect(90.0, 740.0, 240.0, 285.0),
                Rect(148.0, 988.0, 143.0, 39.0),
                None, None, Rect(95.0, 930.0, 225.0, 40.0)),
            2: (Rect(1510.0, 745.0, 330.0, 280.0),
                Rect(1576.0, 988.0, 104.0, 33.0),
                None, Rect(1525.0, 885.0, 250.0, 35.0),
                Rect(1528.0, 925.0, 300.0, 50.0)),
            3: (Rect(795.0, 80.0, 255.0, 270.0),
                Rect(851.0, 318.0, 145.0, 39.0),
                None, None, Rect(795.0, 265.0, 250.0, 40.0)),
            4: (Rect(120.0, 200.0, 210.0, 275.0),
                Rect(203.0, 431.0, 105.0, 33.0),
                None, None, Rect(125.0, 385.0, 200.0, 40.0)),
            5: (Rect(1460.0, 200.0, 240.0, 280.0),
                Rect(1520.0, 431.0, 143.0, 40.0),
                Rect(1475.0, 330.0, 90.0, 35.0),
                None, Rect(1540.0, 375.0, 135.0, 40.0)),
        }
        for i, (region, stack, bet, status, ident) in SEATS.items():
            p.add(f"player_{i}_region", region, "measured", 0.85,
                  evidence_frame_id=[1])
            p.add(f"player_{i}_stack_roi", stack, "measured", 0.85,
                  evidence_frame_id=[1])
            p.add(f"player_{i}_bet_roi", bet, "observed", 0.8,
                  evidence_frame_id=[1] if bet else [])
            p.add(f"player_{i}_status_roi", status, "observed", 0.8,
                  evidence_frame_id=[1] if status else [])
            p.add(f"player_{i}_identity_roi", ident, "observed", 0.8,
                  evidence_frame_id=[1] if ident else [])
        return p

    def _build_seat_sectors(self) -> None:
        """逐座几何：座0=hero（实测）；座1-5 未测量 → null+UNCERTAIN。

        不伪造座位几何；每个 player_i 有独立条目，但未测量者 bbox=null
        （诚实：几何事实缺失时 null + UNCERTAIN，禁止反推）。
        """
        # 座0 = hero：卡片+堆叠+标记的联合可视区（实测）
        self.add("player_0", Rect(700.0, 950.0, 400.0, 450.0), "measured",
                 0.9, evidence_frame_id=[1, 2, 3, 5, 7, 9])
        for i in range(1, self._seats):
            self.add(f"player_{i}", None, "unmeasured", None,
                     evidence_frame_id=[])

    # ------------------------------------------------------------------
    def collision_report(self) -> dict:
        """ROI 碰撞矩阵（S4）。

        - table / player_*_region 为容器（层级包含非碰撞）→ 排除；
        - hero_cards↔action_call 为真实 Z 序覆盖（UI 事实）；
        - pot（null）不参与；
        - 其余对等对象重叠 → collision（含各座 stack/bet/status/identity 子 ROI 间）。
        """
        objects = [n for n in self.entries
                   if not n.startswith("player_") and n != "table"
                   and not n.endswith("_region")]
        collisions = []
        overlays = []
        for i in range(len(objects)):
            for j in range(i + 1, len(objects)):
                a = self.entries[objects[i]].bbox
                b = self.entries[objects[j]].bbox
                if a is None or b is None:
                    continue
                if _rects_overlap(a, b):
                    pair = (objects[i], objects[j])
                    if set(pair) == {"hero_cards", "action_call"}:
                        overlays.append((objects[i], objects[j],
                                         _overlap_area(a, b)))
                    else:
                        collisions.append((objects[i], objects[j],
                                           _overlap_area(a, b)))
        return {"collision_count": len(collisions), "collisions": collisions,
                "z_order_overlays": overlays}

    def seat_coverage(self) -> dict:
        """逐座几何覆盖：有 bbox 的座位数 / 总数（未测量座位 null）。"""
        seats = sorted(n for n in self.entries if n.startswith("player_"))
        measured = [n for n in seats if self.entries[n].bbox is not None]
        return {"total_seats": len(seats),
                "measured_seats": len(measured),
                "measured": measured,
                "unmeasured": [n for n in seats if n not in measured]}

    def to_dict(self) -> dict:
        return {"client_size": list(self.client_size),
                "calibration_version": self.calibration_version,
                "rois": {n: e.to_dict() for n, e in
                         sorted(self.entries.items())}}


def _rects_overlap(a: Rect, b: Rect) -> bool:
    return not (a.x + a.width <= b.x or b.x + b.width <= a.x or
                a.y + a.height <= b.y or b.y + b.height <= a.y)


def _overlap_area(a: Rect, b: Rect) -> float:
    w = min(a.x + a.width, b.x + b.width) - max(a.x, b.x)
    h = min(a.y + a.height, b.y + b.height) - max(a.y, b.y)
    return round(max(0.0, w) * max(0.0, h), 1)
