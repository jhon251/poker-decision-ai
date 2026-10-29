"""V2S — 自动视觉捕获与状态触发证据采集引擎（只读，Observe→Capture→Annotate）。

- ROI digest 触发（已测量 ROI：pot_text/button_d/table/sb_badge）：
  变化 → 保存整帧+ROI 裁剪+trigger 元数据 → annotation queue（PENDING）；
- 未测量 ROI（hero/board/action/stack/bet/status）→ trigger NOT_CONFIGURED
  （无真实 bbox 不监测——禁止无证据监测/推断）；
- 补充 SCENE_CHANGED 兜底（整窗像素变化 ≥ 阈值）——只作"值得采集的证据"
  触发，绝不当作 ground truth/语义事件；
- cooldown + 重复抑制（同 digest 不重复保存；B→C 变化仍保存）；
- provenance 完整（frame_id/timestamp/trigger/digest_before/after/路径/...）；
- 零输入注入/窗口控制/决策——只读屏幕。
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .runtime import image_digest


def _render_class_for(w: int, h: int) -> str | None:
    """V3F-P17-A：帧几何 (w,h) → 规范 render class（只读几何键控）。
    未知几何 → None（不猜测）。"""
    from .render_class import window_for_frame
    return window_for_frame(w, h)

#: ROI 语义名 → 触发器事件（仅视觉变化事实，非行为语义）
TRIGGER_MAP = {
    "pot_text": "POT_CHANGED",
    "button_d": "BUTTON_CHANGED",
    "sb_badge_poder": "STATUS_UI_CHANGED",
    "table": "TABLE_CHANGED",
}
#: 队列优先级（SCENE=高；POT=中；BUTTON/STATUS=低；TABLE=最低——噪声降权）
TRIGGER_PRIORITY = {"SCENE_CHANGED": 0, "POT_CHANGED": 1,
                    "SETTLED_BOARD_CHECK": 1,
                    "BUTTON_CHANGED": 2, "STATUS_UI_CHANGED": 2,
                    "TABLE_CHANGED": 3}
#: 采样粒度（结构量化 digest）：小 ROI 细粒度（保真 2px 变化），table 粗粒度
TRIGGER_SCALE = {"table": 4, "pot_text": 2, "button_d": 2,
                 "sb_badge_poder": 2}
#: V2ZS — settled board 证据区（V2ZH 实测：5 卡槽 (664,551,140,177)×5 的并集
#: 区域，full_png 1419×1066 窗口坐标）；board fingerprint 仅用于
#: 稳定状态检测，≠ ground truth。
BOARD_REGION = (664, 551, 700, 177)
SETTLED_MIN_STABLE_FRAMES = 3      # 连续稳定帧数（@2fps ≈ 1.5s）
SETTLED_COOLDOWN_S = 10.0          # 同状态最大采样频率 1/10s
#: 未测量 ROI（无真实 bbox → 不监测）
NOT_CONFIGURED = ("board", "hero_cards", "action_panels", "folded_marker",
                  "allin_marker", "stack", "bet")


@dataclass
class TriggeredEvidence:
    frame_id: int
    timestamp: float
    trigger: str
    trigger_roi: str
    digest_before: str | None
    digest_after: str
    full_frame_path: str
    roi_paths: list = field(default_factory=list)
    capture_backend: str = "full_screen_bitblt+crop"
    annotation_status: str = "PENDING"
    ground_truth: bool = False        # trigger 永不为 ground truth
    priority: int = 3
    layout_version: str | None = None
    session_id: str | None = None
    screenshot_ref: str | None = None
    #: V3F-P17-A — 数据记录层：每证据帧显式记录几何 + render class
    #: （render_class 由 (w,h) 派生，非 frame-id/scene/值——只读几何键控）
    frame_geometry: list | None = None
    render_class: str | None = None

    def to_dict(self) -> dict:
        return {"frame_id": self.frame_id, "timestamp": self.timestamp,
                "trigger": self.trigger, "trigger_roi": self.trigger_roi,
                "digest_before": self.digest_before,
                "digest_after": self.digest_after,
                "full_frame_path": self.full_frame_path,
                "roi_paths": list(self.roi_paths),
                "capture_backend": self.capture_backend,
                "annotation_status": self.annotation_status,
                "ground_truth": self.ground_truth,
                "priority": self.priority,
                "layout_version": self.layout_version,
                "session_id": self.session_id,
                "screenshot_ref": self.screenshot_ref,
                "frame_geometry": self.frame_geometry,
                "render_class": self.render_class}


class AutoCaptureEngine:
    """自动捕获引擎（ROI digest 触发 + 场景兜底 + 去重/cooldown）。"""

    def __init__(self, profile: dict, evidence_dir: str,
                 scene_threshold: float = 0.05,
                 cooldown_s: float = 0.5, fps: float = 2.0,
                 session_id: str | None = None,
                 settled_region: tuple | None = None):
        self.profile = profile
        self.evidence_dir = Path(evidence_dir)
        self.evidence_dir.mkdir(parents=True, exist_ok=True)
        self.session_id = session_id or f"session_{int(time.time())}"
        self.layout_version = str(profile.get("calibration_version", "v1"))
        self.scene_threshold = scene_threshold
        self.cooldown_s = cooldown_s
        self.fps = fps
        self.trigger_rois = {
            name: data["bbox"] for name, data in profile.get("rois",
                                                             {}).items()
            if data.get("bbox") and name in TRIGGER_MAP}
        self.queue: list = []
        self.counters = {"triggers": 0, "duplicates_suppressed": 0,
                         "debounced": 0, "saved": 0, "roi_rejected": 0}
        self._last_digests: dict = {}
        self._last_emit_at: dict = {}
        self._last_scene_digest: str | None = None
        self._last_scene_arr = None
        self._last_scene_at = 0.0
        self.raw_digest = False
        self.settled_region = tuple(settled_region) if settled_region \
            else BOARD_REGION
        #: V2ZS settled board 状态
        self._board_fp: str | None = None
        self._board_stable = 0
        self._last_settled_at = 0.0
        self._last_settled_fp: str | None = None
        self.counters.setdefault("settled_saved", 0)
        self.counters.setdefault("settled_skipped", 0)
        #: ROI 内容校验：亮度 >150 的亮像素下限（空毡面/无文字区域不得触发）
        self.roi_min_content = 8

    def _roi_has_content(self, sub: np.ndarray) -> bool:
        luma = (sub[..., 0].astype(np.int16) * 0.299
                + sub[..., 1].astype(np.int16) * 0.587
                + sub[..., 2].astype(np.int16) * 0.114)
        return int((luma > 150).sum()) >= self.roi_min_content

    def process_frame(self, frame_id: int, image_bytes: bytes, w: int, h: int,
                      *, save: bool = True) -> list:
        """处理一帧 → 触发器命中列表（保存 + 入队）。"""
        arr = np.frombuffer(image_bytes, dtype=np.uint8).reshape(h, w, 4)
        now = time.time()
        hits = []
        # ROI 触发（已测量 ROI；结构化量化 digest —— 抑制动画噪声；
        # 内容校验：空 ROI（无文字/结构）→ 拒绝触发（布局漂移保护））
        for name, bbox in self.trigger_rois.items():
            x0, y0, cw, chh = [int(v) for v in bbox]
            if x0 + cw > w or y0 + chh > h:
                continue
            sub = arr[y0:y0 + chh, x0:x0 + cw]
            if not self._roi_has_content(sub):
                self.counters["roi_rejected"] += 1
                continue
            digest = _structural_digest(sub, scale=TRIGGER_SCALE.get(name, 2)) \
                if not self.raw_digest else image_digest(
                    ImageDataShim(sub.tobytes()))
            prev = self._last_digests.get(name)
            self._last_digests[name] = digest
            if prev is not None and prev != digest:
                if now - self._last_emit_at.get(name, 0.0) < self.cooldown_s:
                    self.counters["debounced"] += 1
                    continue
                self._last_emit_at[name] = now
                hits.append(self._emit(
                    frame_id, now, TRIGGER_MAP.get(name, "ROI_CHANGED"),
                    name, prev, digest, image_bytes, w, h, arr, bbox, save))
        # 场景兜底（整窗 ≥ threshold 像素变化——仅证据，非语义）
        whole = image_digest(ImageDataShim(image_bytes))
        scene_changed = (self._last_scene_digest is not None
                         and whole != self._last_scene_digest
                         and _pixel_change_ratio(arr, self._last_scene_arr,
                                                 self.scene_threshold)
                         if self._last_scene_arr is not None else False)
        if scene_changed and now - self._last_scene_at >= self.cooldown_s:
            self._last_scene_at = now
            hits.append(self._emit(frame_id, now, "SCENE_CHANGED", "whole",
                                   self._last_scene_digest, whole,
                                   image_bytes, w, h, arr, None, save))
        self._last_scene_digest = whole
        self._last_scene_arr = arr
        # V2ZS — settled board 采样（保守：稳定 ≥3 帧 + 同状态 10s 冷却 +
        # 新状态才保存；fingerprint ≠ GT）
        bx0, by0, bw, bh = self.settled_region
        if bx0 + bw <= w and by0 + bh <= h:
            fp = _structural_digest(arr[by0:by0 + bh, bx0:bx0 + bw],
                                    scale=2)
            if fp == self._board_fp:
                self._board_stable += 1
            else:
                self._board_fp = fp
                self._board_stable = 0
            if (self._board_stable >= SETTLED_MIN_STABLE_FRAMES
                    and fp != self._last_settled_fp
                    and now - self._last_settled_at >= SETTLED_COOLDOWN_S):
                self._last_settled_fp = fp
                self._last_settled_at = now
                self.counters["settled_saved"] += 1
                hits.append(self._emit(frame_id, now, "SETTLED_BOARD_CHECK",
                                       "board_region", None, fp,
                                       image_bytes, w, h, arr,
                                       (bx0, by0, bw, bh), save))
            elif (self._board_stable >= SETTLED_MIN_STABLE_FRAMES
                  and (fp == self._last_settled_fp
                       or now - self._last_settled_at
                       < SETTLED_COOLDOWN_S)):
                self.counters["settled_skipped"] += 1
        return hits

    def _emit(self, frame_id, now, trigger, roi, prev, cur, image_bytes,
              w, h, arr, bbox, save) -> TriggeredEvidence:
        tag = f"{trigger}_{frame_id}_{int(now * 1000)}"
        full_path = self.evidence_dir / f"{tag}_full.png"
        roi_paths = []
        if save:
            self._save_png(full_path, image_bytes, w, h)
            if bbox is not None:
                x0, y0, cw, chh = [int(v) for v in bbox]
                roi_paths.append(str(self.evidence_dir /
                                     f"{tag}_roi_{roi}.png"))
                self._save_png(Path(roi_paths[-1]),
                               arr[y0:y0 + chh, x0:x0 + cw].tobytes(),
                               cw, chh)
        ev = TriggeredEvidence(frame_id=frame_id, timestamp=now,
                               trigger=trigger, trigger_roi=roi,
                               digest_before=prev, digest_after=cur,
                               full_frame_path=str(full_path),
                               roi_paths=roi_paths,
                               priority=TRIGGER_PRIORITY.get(trigger, 3),
                               layout_version=self.layout_version,
                               session_id=self.session_id,
                               screenshot_ref=str(full_path),
                               frame_geometry=[int(w), int(h)],
                               render_class=_render_class_for(w, h))
        self.queue.append(ev.to_dict())
        self.counters["triggers"] += 1
        if save:
            self.counters["saved"] += 1
        return ev

    @staticmethod
    def _save_png(path: Path, rgba: bytes, w: int, h: int) -> None:
        from ....vision2_platform import encode_rgba_png
        path.write_bytes(encode_rgba_png(w, h, rgba))

    def to_dict(self) -> dict:
        return {"counters": dict(self.counters),
                "queue": sorted(self.queue, key=lambda q: q["priority"]),
                "configured_triggers": list(self.trigger_rois),
                "not_configured": list(NOT_CONFIGURED),
                "session_id": self.session_id,
                "layout_version": self.layout_version}


class ImageDataShim:
    def __init__(self, data: bytes):
        self.data = data


def _structural_digest(arr: np.ndarray, scale: int = 4,
                       levels: int = 8) -> str:
    """结构化量化摘要：降采样 + 8 级量化。

    用途：ROI 触发检测——抑制动画/光效的亚感知像素噪声（如桌面光晕、
    徽章发光），同时仍能捕获卡牌出现、pot/stack 数值变化等结构变化。
    这是对"真实变化"的保真（量化步长固定，非以阈值掩盖变化）。
    """
    import hashlib
    small = arr[::scale, ::scale, :3].astype(np.int16)
    q = (small * levels // 256)
    return hashlib.sha1(q.tobytes()).hexdigest()


def _pixel_change_ratio(arr, prev_arr, threshold: float) -> bool:
    if prev_arr is None or arr.shape != prev_arr.shape:
        return False
    a = arr[::8, ::8].astype(np.int16).mean(axis=-1)
    b = prev_arr[::8, ::8].astype(np.int16).mean(axis=-1)
    changed = (np.abs(a - b) > 24.0).mean()
    return changed >= threshold
