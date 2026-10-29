"""像素时序状态机：appearance / disappearance / stable / transient / moved。

- appeared/disappeared 事件需连续 confirm_frames 帧确认（防闪烁）；
- transient：短暂消失又出现（消失帧数 ≤ transient_window）→ 只记状态，
  不产生 ROI_DISAPPEARED/ROI_APPEARED 事件（闪烁不是语义事实）；
- stable：连续 stable_frames 帧未变化 → state="stable"；
- moved：changed_bbox 质心位移 ≥ moved_centroid_px → state="moved"；
- 只描述 ROI 级事实，禁止任何行动者/行动语义推断。
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..realtime.coords import Rect
from .events import PixelEvent, PixelEventKind


@dataclass
class PixelTemporalConfig:
    confirm_frames: int = 2
    stable_frames: int = 3
    transient_window_frames: int = 3
    moved_centroid_px: float = 4.0


@dataclass
class _ROIState:
    presence: bool | None = None
    presence_run: int = 0
    absent_run: int = 0
    reappeared_at: int | None = None
    disappeared_at: int | None = None
    stable_run: int = 0
    ever_present: bool = False
    last_absent_gap: int | None = None
    last_bbox: Rect | None = None


def _centroid(rect: Rect) -> tuple:
    return (rect.x + rect.width / 2.0, rect.y + rect.height / 2.0)


class PixelTemporalTracker:
    """逐 ROI 的像素时序状态机（确定性；无 future information）。"""

    def __init__(self, config: PixelTemporalConfig | None = None):
        self.config = config or PixelTemporalConfig()
        self.states: dict = {}

    def reset(self) -> None:
        self.states.clear()

    def update(self, frame_id: int, presence: dict, changed: dict,
               changed_bboxes: dict) -> tuple:
        """返回 (temporal_features: dict[str, dict], events: list[PixelEvent])。

        presence/changed/changed_bboxes 均以 roi_name 为键；changed 为 bool，
        changed_bboxes 值为 Rect 或 None。
        """
        features: dict = {}
        events: list = []
        for roi_name in sorted(presence):
            present = bool(presence[roi_name])
            bbox = changed_bboxes.get(roi_name)
            feats, evs = self._update_roi(frame_id, roi_name, present,
                                          bool(changed.get(roi_name, False)),
                                          bbox)
            features[roi_name] = feats
            events.extend(evs)
        return features, events

    def _update_roi(self, frame_id: int, roi_name: str, present: bool,
                    changed: bool, bbox: Rect | None) -> tuple:
        st = self.states.setdefault(roi_name, _ROIState())
        cfg = self.config
        feats = {"state": None, "stable_frames": 0, "presence_run": 0,
                 "absent_run": 0, "last_absent_gap": None, "moved": None}
        events: list = []

        if st.presence is None:
            st.presence = present
            st.presence_run = 1
            feats["presence_run"] = 1
            feats["state"] = "present" if present else "absent"
            if present:
                st.ever_present = True
                st.stable_run = 1
                feats["stable_frames"] = 1
            st.last_bbox = bbox
            return feats, events

        if present:
            if st.presence is False:
                gap = st.absent_run
                st.last_absent_gap = gap
                st.absent_run = 0
                st.reappeared_at = frame_id
                feats["last_absent_gap"] = gap
                feats["state"] = ("transient"
                                  if gap <= cfg.transient_window_frames
                                  and st.ever_present else "appeared")
            st.presence = True
            st.presence_run += 1
            st.ever_present = True
            feats["presence_run"] = st.presence_run
            if st.reappeared_at is not None \
                    and frame_id - st.reappeared_at + 1 >= cfg.confirm_frames:
                events.append(PixelEvent(
                    kind=PixelEventKind.ROI_APPEARED, roi=roi_name,
                    frame_id=frame_id,
                    evidence_frame_ids=(st.reappeared_at, frame_id),
                    description=f"{roi_name} ROI appeared (confirmed "
                                f"{cfg.confirm_frames} frames)"))
                st.reappeared_at = None
            if not changed:
                st.stable_run += 1
                feats["stable_frames"] = st.stable_run
                if st.stable_run >= cfg.stable_frames \
                        and feats["state"] not in ("transient", "appeared"):
                    feats["state"] = "stable"
            else:
                st.stable_run = 0
                if feats["state"] not in ("transient", "appeared"):
                    feats["state"] = "changed"
                if st.last_bbox is not None and bbox is not None:
                    c0, c1 = _centroid(st.last_bbox), _centroid(bbox)
                    dist = ((c0[0] - c1[0]) ** 2 + (c0[1] - c1[1]) ** 2) ** 0.5
                    feats["moved"] = round(dist, 3)
                    if dist >= cfg.moved_centroid_px:
                        feats["state"] = "moved"
            st.last_bbox = bbox
            return feats, events

        if st.presence is True:
            st.disappeared_at = frame_id
        st.presence = False
        st.presence_run = 0
        st.absent_run += 1
        st.stable_run = 0
        feats["absent_run"] = st.absent_run
        feats["state"] = "absent"
        st.last_bbox = None
        if st.disappeared_at is not None \
                and frame_id - st.disappeared_at + 1 >= cfg.confirm_frames:
            events.append(PixelEvent(
                kind=PixelEventKind.ROI_DISAPPEARED, roi=roi_name,
                frame_id=frame_id,
                evidence_frame_ids=(st.disappeared_at, frame_id),
                description=f"{roi_name} ROI disappeared (confirmed "
                            f"{cfg.confirm_frames} frames)"))
            st.disappeared_at = None
        return feats, events
