"""Realtime 视觉流水线：Frame → 布局/ROI → 分层识别 → V0 装配校验 →
Temporal → RealtimeVisionObservation。

红线：
  - 纯感知层：不输出任何决策/期望收益/策略字段；
  - STATE_CONFLICT 与关键 uncertainty 硬阻断下游（blocked=True）；
  - 多模态兜底仅事件触发且受 cooldown/预算约束（绝不每帧调用）；
  - 所有模型/后端输出必须经过 V0 assemble/validator，不得绕过；
  - position 只由冻结 positions_for 在 V0 assembly 内确定性推导。
"""
from __future__ import annotations

import time as _time
from dataclasses import dataclass, field, replace
from datetime import datetime

from ..assembly import assemble
from ..schemas import (ACTIONS, NormalizedVisionState, RawAction, RawSeat,
                       RawVisionState, UncertaintyEntry)
from .anchors import bootstrap_profile, calibrate, detect_anchor_drift
from .capture import FixtureCaptureProvider, ScreenCaptureProvider
from .coords import CoordinateError, CoordinateTransformer, Rect
from .fallback import (FallbackCall, FallbackGate, FallbackPolicyConfig,
                       FallbackTrigger, MultimodalBackend, MultimodalResult,
                       MultimodalStatus, UnavailableMultimodalBackend)
from .frame_manager import (ALLOWED_FPS, DEFAULT_FPS, Frame, FrameManager,
                            FrameStatus)
from .image import ImageData, ncc_template_match, pixel_diff_ratio
from .recognition import (CardRecognitionStatus, OCRBackend, OCRStatus,
                          UnavailableCardBackend, UnavailableOCRBackend)
from .roi import ROIManager, ROIName, default_roi_specs
from .status import RecognitionStatus
from .temporal import ObservationSnapshot, StateTransition, TemporalTracker


# ------------------------------------------------------------------ 配置
@dataclass
class RealtimeVisionConfig:
    fps: float = DEFAULT_FPS
    stale_after_s: float = 3.0
    dedup_threshold: float = 0.0      # 帧级去重=字节完全相同（小变化交给 ROI 层）
    roi_change_threshold: float = 0.02
    min_field_confidence: float = 0.5
    ocr_failure_streak_trigger: int = 3
    anchor_match_threshold: float = 0.75
    anchor_search_margin: float = 0.3
    critical_fields: tuple = (
        "hero.hole_cards", "hero.seat", "button_seat", "pot",
        "board.flop", "board.turn", "board.river",
        "game.street", "game.blinds",
    )
    fallback: FallbackPolicyConfig = field(default_factory=FallbackPolicyConfig)

    def validate(self) -> None:
        if self.fps not in ALLOWED_FPS:
            raise ValueError(f"fps 必须属于 {ALLOWED_FPS}，实际 {self.fps}")
        if not 0.0 <= self.min_field_confidence <= 1.0:
            raise ValueError("min_field_confidence 必须在 [0,1]")


# ------------------------------------------------------------------ 观测输出
@dataclass(frozen=True)
class RealtimeVisionObservation:
    status: RecognitionStatus
    blocked: bool
    failure_reason: str | None
    unchanged: bool
    waiting: bool
    timestamp: float | None
    frame_id: int | None
    synthetic_test_only: bool
    normalized: NormalizedVisionState | None = None
    provenance: dict = field(default_factory=dict)
    transition: dict | None = None
    events: tuple = ()

    def consumable(self) -> bool:
        return (not self.blocked and not self.waiting
                and self.normalized is not None)

    def to_dict(self) -> dict:
        return {
            "status": self.status.value,
            "blocked": self.blocked,
            "failure_reason": self.failure_reason,
            "unchanged": self.unchanged,
            "waiting": self.waiting,
            "timestamp": self.timestamp,
            "frame_id": self.frame_id,
            "synthetic_test_only": self.synthetic_test_only,
            "normalized": None if self.normalized is None
            else self.normalized.to_dict(),
            "provenance": self.provenance,
            "transition": self.transition,
            "events": [e.to_dict() for e in self.events],
        }


class StageTimer:
    """基准用分阶段计时器（写入 dict[str, list[float]]）。"""

    def __init__(self, collector: dict, stage: str,
                 clock=_time.perf_counter):
        self.collector = collector
        self.stage = stage
        self.clock = clock

    def __enter__(self):
        self._t0 = self.clock()
        return self

    def __exit__(self, *exc):
        self.collector.setdefault(self.stage, []).append(
            self.clock() - self._t0)
        return False


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts).isoformat(timespec="milliseconds")


def _street_from_board_len(n: int) -> str | None:
    return {0: "preflop", 3: "flop", 4: "turn", 5: "river"}.get(n)


def _cards_to_board(cards) -> dict | None:
    """识别到的牌序列 → V0 board dict（确定性；长度非法 → None）。"""
    if cards is None:
        return None
    n = len(cards)
    if n == 0:
        return {"flop": None, "turn": None, "river": None}
    if n == 3:
        return {"flop": list(cards), "turn": None, "river": None}
    if n == 4:
        return {"flop": list(cards[:3]), "turn": list(cards[3:]), "river": None}
    if n == 5:
        return {"flop": list(cards[:3]), "turn": list(cards[3:4]),
                "river": list(cards[4:])}
    return None


_ACTION_WORDS = {"fold": "fold", "check": "check", "call": "call",
                 "bet": "bet", "raise": "raise", "allin": "allin",
                 "all-in": "allin"}


def parse_action_text(text) -> tuple:
    """行动文字 → (action, amount)；解析失败 → (None, None)（不猜测）。"""
    if not text or not isinstance(text, str):
        return None, None
    parts = text.strip().split()
    if not parts:
        return None, None
    action = _ACTION_WORDS.get(parts[0].lower())
    if action is None or action not in ACTIONS:
        return None, None
    amount = None
    if len(parts) >= 2:
        try:
            amount = float(parts[1])
        except ValueError:
            amount = None
    return action, amount


def _good_number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and v == v


#: ROI 处理顺序：先牌面/元数据，后数字/座位，行动区最后（需要 street 与 seats）
_ROI_PRIORITY = {
    ROIName.BOARD: 0, ROIName.HERO_CARDS: 1, ROIName.TABLE_META: 2,
    ROIName.POT: 3, ROIName.BET_AREA: 4, ROIName.PLAYERS: 5,
    ROIName.STACKS: 6, ROIName.ACTION_AREA: 7,
}


# ------------------------------------------------------------------ 流水线
class RealtimeVisionPipeline:
    def __init__(self, *, config: RealtimeVisionConfig | None = None,
                 capture_provider: ScreenCaptureProvider | None = None,
                 ocr: OCRBackend | None = None,
                 card=None, multimodal: MultimodalBackend | None = None,
                 roi_specs: dict | None = None,
                 bootstrap_anchors: dict | None = None,
                 anchor_templates: dict | None = None,
                 anchor_sizes: dict | None = None,
                 profile=None, clock=None):
        self.config = config or RealtimeVisionConfig()
        self.config.validate()
        self.clock = clock or _time.monotonic
        provider = capture_provider or FixtureCaptureProvider([])
        self.frame_manager = FrameManager(
            provider, fps=self.config.fps,
            dedup_threshold=self.config.dedup_threshold,
            stale_after_s=self.config.stale_after_s, clock=self.clock)
        self.ocr = ocr or UnavailableOCRBackend()
        self.card = card or UnavailableCardBackend()
        self.multimodal = multimodal or UnavailableMultimodalBackend()
        self.roi_specs = dict(roi_specs or default_roi_specs())
        self.bootstrap_anchors = bootstrap_anchors or {
            "table": (0.0, 0.0, 1.0, 1.0)}
        self.anchor_templates = anchor_templates or {}
        self.anchor_sizes = anchor_sizes or {}
        self.profile = profile
        self.gate = FallbackGate(self.config.fallback, clock=self.clock)
        self.tracker = TemporalTracker()
        self.ocr_fail_streak = 0
        self.last_obs: RealtimeVisionObservation | None = None
        self.last_prov: dict = {}
        self.last_roi_fields: dict = {}
        self.last_raw: RawVisionState | None = None
        self.last_fresh_image: ImageData | None = None
        self.last_crops: dict = {}
        self.timers: dict = {}
        self.stats = {"frames_processed": 0, "fallback_calls": 0,
                      "fallback_denied": 0, "layout_unknown": 0,
                      "state_conflicts": 0}

    # ---------------------------------------------------------- 对外接口
    def step(self, now: float | None = None
             ) -> RealtimeVisionObservation | None:
        now = self.clock() if now is None else now
        with StageTimer(self.timers, "capture"):
            res = self.frame_manager.poll(now)
        if res.status is FrameStatus.WAIT_RATE_LIMIT:
            if self.last_obs is None:
                return None
            return replace(self.last_obs, waiting=True, unchanged=True)
        if res.status is FrameStatus.CAPTURE_FAILED:
            return self._blocked_obs(RecognitionStatus.CAPTURE_FAILED,
                                     res.error or "capture failed", now)
        if res.status is FrameStatus.FRAME_STALE:
            return self._blocked_obs(RecognitionStatus.FRAME_STALE,
                                     res.error or "frame stale", now)
        if res.status is FrameStatus.DUPLICATE:
            return self._duplicate_obs(res.frame, now)
        return self.process(res.frame)

    def process(self, frame: Frame) -> RealtimeVisionObservation:
        self.stats["frames_processed"] += 1
        with StageTimer(self.timers, "total_fast_path"):
            return self._process(frame)

    def _process(self, frame: Frame) -> RealtimeVisionObservation:
        # 1) 几何合法性
        try:
            transformer = CoordinateTransformer(frame.geometry)
        except CoordinateError as e:
            self.stats["layout_unknown"] += 1
            return self._blocked_obs(
                RecognitionStatus.LAYOUT_UNKNOWN, f"geometry invalid: {e}",
                frame.captured_at, fallback_frame=frame,
                fallback_trigger=FallbackTrigger.LAYOUT_UNKNOWN)

        # 2) 布局 / 校准 / ROI
        with StageTimer(self.timers, "layout"):
            ok, rois, missing, reason = self._ensure_layout(frame, transformer)
        if not ok:
            self.stats["layout_unknown"] += 1
            return self._blocked_obs(
                RecognitionStatus.LAYOUT_UNKNOWN, reason or "layout unknown",
                frame.captured_at, fallback_frame=frame,
                fallback_trigger=FallbackTrigger.LAYOUT_UNKNOWN)

        roiman = ROIManager(self.profile, self.roi_specs, transformer)
        with StageTimer(self.timers, "roi_extraction"):
            crops, crop_missing = roiman.extract_all(frame.image, rois)
        for name, why in crop_missing.items():
            missing[name] = why

        # 3) cheap 变化检测（只判"是否变化"，绝不判牌）
        with StageTimer(self.timers, "cheap_detection"):
            changed = self._changed_rois(frame, crops)

        # 4) 分层识别 → RawVisionState + provenance
        raw, prov = self._recognize(frame, rois, crops, changed, missing)
        with StageTimer(self.timers, "validation"):
            normalized = assemble(raw)

        # 5) 时序预计算（用于兜底触发判断）
        snap = ObservationSnapshot(frame.frame_id, frame.captured_at, raw,
                                   normalized, "processing", False)
        prelim: StateTransition | None = None
        if self.tracker.current is not None:
            prelim = self.tracker.compute_prelim(self.tracker.current, snap)
        if prelim is not None and any(
                e.kind.value == "NEW_HAND" for e in prelim.events):
            self.gate.on_new_hand()

        # 6) 兜底触发判断 + 调用（预算/冷却受控，绝不每帧调用）
        triggers = self._collect_triggers(frame, normalized, prelim, missing)
        if triggers:
            if self.gate.request(triggers[0], frame_id=frame.frame_id,
                                 now=frame.captured_at).allowed:
                with StageTimer(self.timers, "multimodal_fallback"):
                    raw, normalized = self._apply_fallback(frame, crops, raw,
                                                           triggers)
            else:
                self.stats["fallback_denied"] += 1

        # 7) 最终时序追踪
        with StageTimer(self.timers, "temporal"):
            transition = self._finalize_temporal(frame, raw, normalized)

        # 8) 状态/阻断判定
        status, blocked, reason = self._compute_status(normalized, missing,
                                                       transition)

        obs = RealtimeVisionObservation(
            status=status, blocked=blocked, failure_reason=reason,
            unchanged=False, waiting=False, timestamp=frame.captured_at,
            frame_id=frame.frame_id,
            synthetic_test_only=frame.synthetic_test_only,
            normalized=normalized, provenance=prov.to_dict(),
            transition=None if transition is None else transition.to_dict(),
            events=tuple(transition.events) if transition else ())
        self.last_obs = obs
        self.last_prov = prov.to_dict()
        self.last_raw = raw
        self.last_fresh_image = frame.image
        self.last_crops = crops
        return obs

    # ---------------------------------------------------------- 布局
    def _ensure_layout(self, frame: Frame, transformer):
        needs_calibration = False
        fresh_bootstrap = False
        if self.profile is None:
            # 无校准 profile：先用固定坐标 bootstrap 种子（低置信初始值）
            self.profile = bootstrap_profile(
                "vision2_rt", frame.geometry, self.bootstrap_anchors,
                calibrated_at=_iso(frame.captured_at))
            fresh_bootstrap = True
            needs_calibration = True
        elif detect_anchor_drift(self.profile, frame.geometry):
            needs_calibration = True

        if needs_calibration and self.anchor_templates:
            detections = self._locate_anchors(frame.image)
            if detections is None:
                return False, {}, {}, "anchor detection failed (no reliable layout)"
            self.profile = calibrate(
                "vision2_rt", frame.geometry, detections,
                calibrated_at=_iso(frame.captured_at),
                base_version=self.profile.calibration_version + 1)
        elif needs_calibration and not fresh_bootstrap:
            # 几何已漂移但无模板可重定位 → 不猜测
            return False, {}, {}, ("geometry drift without anchor templates "
                                   "(recalibration impossible)")

        roiman = ROIManager(self.profile, self.roi_specs, transformer)
        rois, missing = roiman.resolve()
        return True, rois, missing, None

    def _locate_anchors(self, image: ImageData) -> dict | None:
        """模板匹配定位 anchor（cheap 层；只定位 UI 元素，不识别牌）。

        返回 {anchor_id: (语义矩形 Rect, score)}；模板只标记 anchor 一角，
        语义矩形尺寸来自 anchor_sizes（缺省=模板尺寸）。
        """
        detections = {}
        for anchor_id, template in self.anchor_templates.items():
            sr = Rect(0, 0, image.width - template.width,
                      image.height - template.height)
            found = ncc_template_match(
                image, template, search_rect=sr,
                score_threshold=self.config.anchor_match_threshold)
            if found is None:
                return None
            x, y, score = found
            aw, ah = self.anchor_sizes.get(
                anchor_id, (float(template.width), float(template.height)))
            detections[anchor_id] = (Rect(float(x), float(y),
                                          float(aw), float(ah)), score)
        return detections

    # ---------------------------------------------------------- cheap 检测
    def _changed_rois(self, frame: Frame, crops: dict) -> dict:
        changed = {}
        if self.last_fresh_image is None:
            for name in crops:
                changed[name] = True
            return changed
        for name, crop in crops.items():
            prev = self.last_crops.get(name)
            if prev is None:
                changed[name] = True
            else:
                ratio = pixel_diff_ratio(prev, crop)
                changed[name] = ratio > self.config.roi_change_threshold
        return changed

    # ---------------------------------------------------------- 识别
    def _recognize(self, frame, rois, crops, changed, missing):
        raw = RawVisionState(source="vision2_realtime",
                             synthetic_test_only=frame.synthetic_test_only,
                             captured_at=_iso(frame.captured_at),
                             uncertainty=[])
        prov: dict = {}
        roi_fields: dict = {}
        self._ocr_failed_now = 0
        self._card_uncertain = False

        def add_prov(field_name, value, layer, roi_name, bbox, conf, status):
            from .provenance import FieldProvenance
            prov[field_name] = FieldProvenance(
                field=field_name, value=value, source_layer=layer,
                roi=roi_name, bbox=bbox, frame_id=frame.frame_id,
                confidence=conf, status=status).to_dict()
            roi_fields.setdefault(roi_name, []).append(field_name)

        def uncertain(field_name, status="uncertain", conf=None):
            raw.uncertainty.append(UncertaintyEntry(field=field_name,
                                                    status=status,
                                                    confidence=conf))

        rois_by_name = {name.value: (name, roi) for name, roi in rois.items()}
        ordered = sorted(rois.keys(),
                         key=lambda n: _ROI_PRIORITY.get(n, 99))
        for name in ordered:
            spec = self.roi_specs.get(name)
            if spec is None or spec.read_layer == "none":
                continue
            crop = crops.get(name)
            if crop is None:
                add_prov(f"{name.value}", None, spec.read_layer, name.value,
                         None, None, "missing")
                if name is ROIName.HERO_CARDS:
                    uncertain("hero.hole_cards", "missing")
                elif name is ROIName.BOARD:
                    uncertain("board", "missing")
                else:
                    uncertain(f"{name.value}", "missing")
                continue
            if not changed.get(name, True):
                for fname in self.last_roi_fields.get(name.value, []):
                    prev = self.last_prov.get(fname)
                    if prev:
                        p = dict(prev)
                        p["status"] = "reused"
                        p["frame_id"] = frame.frame_id
                        prov[fname] = p
                        roi_fields.setdefault(name.value, []).append(fname)
                    self._restore_field(raw, fname)
                continue

            layer = spec.read_layer
            if layer == "cards":
                with StageTimer(self.timers, "card_recognition"):
                    self._recognize_cards(frame, name, crop, raw, prov,
                                          add_prov, uncertain, rois_by_name)
            elif layer in ("number", "text", "seats", "seat_stacks",
                           "table_meta"):
                with StageTimer(self.timers, "ocr"):
                    self._recognize_ocr(frame, name, crop, layer, raw, prov,
                                        add_prov, uncertain, rois_by_name)

        # street 无法确定性推导 → 明确 uncertainty（视觉层不猜 street）
        if raw.street is None:
            uncertain("game.street", "missing")
            add_prov("game.street", None, "derived", "board", None, None,
                     "missing")
        self.last_roi_fields = roi_fields
        return raw, _ProvWrap(prov)

    _ATTR_OF_FIELD = {
        "pot": "pot", "board": "board", "hero.hole_cards": "hero_hole_cards",
        "game.street": "street", "hero.current_bet": "hero_current_bet",
        "game.blinds": "blinds", "hero.seat": "hero_seat",
        "button_seat": "button_seat", "action_history": "action_history",
        "hero.amount_to_call": "hero_amount_to_call",
    }

    def _restore_field(self, raw: RawVisionState, fname: str) -> None:
        """unchanged ROI → 从上一帧恢复对应字段值（与 provenance reused 同步）。

        只恢复上一帧已确认的值；上一帧为 None 的字段保持 None（诚实）。
        """
        import copy
        if self.last_raw is None:
            return
        if fname in ("players", "players.stack") and raw.seats is None:
            raw.seats = copy.deepcopy(self.last_raw.seats)
            self._sync_hero_from_seats(raw)
            return
        attr = self._ATTR_OF_FIELD.get(fname)
        if attr is not None and getattr(raw, attr) is None:
            setattr(raw, attr, copy.deepcopy(getattr(self.last_raw, attr)))

    def _sync_hero_from_seats(self, raw: RawVisionState) -> None:
        """hero 自身数值：从座位事实同步（只填当前为 None 的字段，不覆盖）。"""
        if raw.hero_seat is None or raw.seats is None:
            return
        hero_row = next((s for s in raw.seats if s.seat == raw.hero_seat), None)
        if hero_row is None:
            return
        if raw.hero_stack is None and hero_row.stack is not None:
            raw.hero_stack = hero_row.stack
        if raw.hero_all_in is None and hero_row.all_in is not None:
            raw.hero_all_in = hero_row.all_in

    def _recognize_ocr(self, frame, name, crop, kind, raw, prov, add_prov,
                       uncertain, rois_by_name):
        res = self.ocr.recognize(crop, name.value, kind)
        if res.status is OCRStatus.FAILED:
            self._ocr_failed_now += 1
        if res.status in (OCRStatus.FAILED, OCRStatus.UNAVAILABLE):
            add_prov(f"{name.value}", None, "ocr", name.value, None, None,
                     res.status.value.lower())
            uncertain(f"{name.value}", "missing")
            return
        conf = res.confidence
        if conf is not None and conf < self.config.min_field_confidence:
            res.status = OCRStatus.UNCERTAIN
        if res.status is OCRStatus.UNCERTAIN:
            add_prov(f"{name.value}", None, "ocr", name.value, None, conf,
                     "uncertain")
            uncertain(f"{name.value}", "uncertain", conf)
            return
        bbox = rois_by_name[name.value][1].rect()
        if kind == "number":
            self._apply_number(name, res, raw, prov, add_prov, uncertain, bbox)
        elif kind == "text":
            self._apply_text(name, res, raw, prov, add_prov, uncertain, bbox)
        elif kind == "seats":
            self._apply_seats(name, res, raw, prov, add_prov, uncertain, bbox)
        elif kind == "seat_stacks":
            self._apply_seat_stacks(name, res, raw, prov, add_prov, uncertain,
                                    bbox)
        elif kind == "table_meta":
            self._apply_table_meta(name, res, raw, prov, add_prov, uncertain,
                                   bbox)

    def _apply_number(self, name, res, raw, prov, add_prov, uncertain, bbox):
        v = res.value
        if not _good_number(v):
            add_prov(f"{name.value}", None, "ocr", name.value, bbox,
                     res.confidence, "uncertain")
            uncertain(f"{name.value}", "uncertain", res.confidence)
            return
        if name is ROIName.POT:
            raw.pot = float(v)
            add_prov("pot", float(v), "ocr", name.value, bbox, res.confidence,
                     "recognized")
        elif name is ROIName.BET_AREA:
            raw.hero_current_bet = float(v)
            add_prov("hero.current_bet", float(v), "ocr", name.value, bbox,
                     res.confidence, "recognized")
        else:
            add_prov(f"{name.value}", float(v), "ocr", name.value, bbox,
                     res.confidence, "recognized")

    def _apply_text(self, name, res, raw, prov, add_prov, uncertain, bbox):
        text = res.value if isinstance(res.value, str) else None
        action, amount = parse_action_text(text)
        actor = getattr(res, "extra", None) or {}
        actor_seat = actor.get("actor_seat") if isinstance(actor, dict) else None
        add_prov("action_history", None if text is None else text, "ocr",
                 name.value, bbox, res.confidence,
                 "recognized" if action else "uncertain")
        if action is None:
            if text is not None:
                uncertain("action_history", "uncertain", res.confidence)
            return
        street = raw.street
        raw.action_history = list(raw.action_history or [])
        raw.action_history.append(RawAction(
            player=(f"player_{actor_seat}" if actor_seat is not None else None),
            action=action, amount=amount, street=street,
            order=len(raw.action_history) + 1, confidence=res.confidence))
        if actor_seat is not None:
            for s in (raw.seats or []):
                if s.seat == actor_seat and s.visible_action is None:
                    s.visible_action = action

    def _apply_seats(self, name, res, raw, prov, add_prov, uncertain, bbox):
        v = res.value
        if not isinstance(v, dict) or not v:
            add_prov("players", None, "ocr", name.value, bbox, res.confidence,
                     "uncertain")
            uncertain("players.seats", "uncertain", res.confidence)
            return
        seats = []
        for seat in sorted(v):
            if not _good_number(seat):
                continue
            d = v[seat] or {}
            seats.append(RawSeat(
                seat=int(seat),
                occupied=(d.get("occupied") if isinstance(d.get("occupied"),
                                                          bool) else None),
                anonymous_player_id=f"player_{int(seat)}",
                stack=(float(d["stack"]) if _good_number(d.get("stack"))
                       else None),
                current_bet=(float(d["current_bet"])
                             if _good_number(d.get("current_bet")) else None),
                folded=(d.get("folded") if isinstance(d.get("folded"), bool)
                        else None),
                all_in=(d.get("all_in") if isinstance(d.get("all_in"), bool)
                        else None),
                in_hand=(d.get("in_hand") if isinstance(d.get("in_hand"), bool)
                         else None),
                visible_action=(d.get("visible_action")
                                if d.get("visible_action") in ACTIONS
                                else None),
            ))
        raw.seats = seats
        add_prov("players", [s.seat for s in seats], "ocr", name.value, bbox,
                 res.confidence, "recognized")
        # hero 自身数值：从座位事实同步（只填当前为 None 的字段，不覆盖）
        self._sync_hero_from_seats(raw)

    def _apply_seat_stacks(self, name, res, raw, prov, add_prov, uncertain,
                           bbox):
        v = res.value
        if not isinstance(v, dict):
            add_prov("stacks", None, "ocr", name.value, bbox, res.confidence,
                     "uncertain")
            uncertain("players.stacks", "uncertain", res.confidence)
            return
        raw.seats = list(raw.seats or [])
        applied = []
        for s in raw.seats:
            if s.seat in v and _good_number(v[s.seat]):
                s.stack = float(v[s.seat])
                applied.append(s.seat)
        add_prov("players.stack", applied, "ocr", name.value, bbox,
                 res.confidence, "recognized")

    def _apply_table_meta(self, name, res, raw, prov, add_prov, uncertain,
                          bbox):
        v = res.value if isinstance(res.value, dict) else {}
        blinds = v.get("blinds")
        if isinstance(blinds, dict):
            raw.blinds = {
                "sb": blinds.get("sb") if _good_number(blinds.get("sb"))
                else None,
                "bb": blinds.get("bb") if _good_number(blinds.get("bb"))
                else None,
                "ante": blinds.get("ante") if _good_number(blinds.get("ante"))
                else None,
            }
            add_prov("game.blinds", raw.blinds, "ocr", name.value, bbox,
                     res.confidence, "recognized")
        else:
            uncertain("game.blinds", "uncertain", res.confidence)
        hs, bs = v.get("hero_seat"), v.get("button_seat")
        if _good_number(hs):
            raw.hero_seat = int(hs)
            add_prov("hero.seat", int(hs), "ocr", name.value, bbox,
                     res.confidence, "recognized")
        if _good_number(bs):
            raw.button_seat = int(bs)
            add_prov("button_seat", int(bs), "ocr", name.value, bbox,
                     res.confidence, "recognized")
        if isinstance(v.get("table_size"), (int, float)) and not isinstance(
                v.get("table_size"), bool):
            raw.table_size = int(v["table_size"])
        for k, attr in (("poker_variant", "poker_variant"),
                        ("game_type", "game_type")):
            if isinstance(v.get(k), str):
                setattr(raw, attr, v[k])
        if _good_number(v.get("to_call")):
            raw.hero_amount_to_call = float(v["to_call"])
            add_prov("hero.amount_to_call", float(v["to_call"]), "ocr",
                     name.value, bbox, res.confidence, "recognized")

    def _recognize_cards(self, frame, name, crop, raw, prov, add_prov,
                         uncertain, rois_by_name):
        res = self.card.recognize(crop, name.value)
        bbox = rois_by_name[name.value][1].rect()
        hero = name is ROIName.HERO_CARDS
        if res.status is CardRecognitionStatus.UNAVAILABLE:
            add_prov(f"{name.value}", None, "card_recognition", name.value,
                     bbox, None, "unavailable")
            uncertain("hero.hole_cards" if hero else "board", "missing")
            return
        if res.status is CardRecognitionStatus.FAILED:
            self._card_uncertain = True
            add_prov(f"{name.value}", None, "card_recognition", name.value,
                     bbox, None, "failed")
            uncertain("hero.hole_cards" if hero else "board", "uncertain")
            return
        cards = res.cards
        if cards is None or res.status is CardRecognitionStatus.UNCERTAIN:
            self._card_uncertain = True
            add_prov(f"{name.value}", None, "card_recognition", name.value,
                     bbox, None, "uncertain")
            uncertain("hero.hole_cards" if hero else "board", "uncertain")
            return
        symbols = []
        for c in cards:
            if c.rank and c.suit and len(str(c.rank)) == 1 \
                    and str(c.suit).lower() in "cdhs":
                symbols.append(str(c.rank).upper() + str(c.suit).lower())
            else:
                symbols.append(None)
        if hero:
            if len(symbols) != 2:
                uncertain("hero.hole_cards", "uncertain")
                add_prov("hero.hole_cards", None, "card_recognition",
                         name.value, bbox, None, "uncertain")
                return
            raw.hero_hole_cards = symbols
            add_prov("hero.hole_cards", symbols, "card_recognition",
                     name.value, bbox,
                     min((c.confidence for c in cards
                          if c.confidence is not None), default=None),
                     "recognized")
        else:
            bd = _cards_to_board(symbols)
            if bd is None:
                uncertain("board", "uncertain")
                add_prov("board", None, "card_recognition", name.value, bbox,
                         None, "uncertain")
                return
            raw.board = bd
            add_prov("board", bd, "card_recognition", name.value, bbox,
                     min((c.confidence for c in cards
                          if c.confidence is not None), default=None),
                     "recognized")
        # street 由 board 结构确定性推导（视觉层不猜 street）
        bd = raw.board
        if bd is not None:
            n = len(list(bd.get("flop") or []) + list(bd.get("turn") or [])
                     + list(bd.get("river") or []))
            raw.street = _street_from_board_len(n)
            add_prov("game.street", raw.street, "derived", name.value, bbox,
                     None, "recognized" if raw.street else "uncertain")

    # ---------------------------------------------------------- 兜底
    def _collect_triggers(self, frame, normalized, prelim, missing) -> list:
        triggers = []
        if not normalized.validation.valid:
            triggers.append(FallbackTrigger.STATE_CONFLICT)
            crit_errors = [e for e in normalized.validation.errors
                           if self._is_critical_field(e.get("field", ""))]
            if crit_errors:
                triggers.append(FallbackTrigger.KEY_FIELD_CONFLICT)
        if prelim is not None:
            if any(e.kind.value == "NEW_HAND" for e in prelim.events):
                triggers.append(FallbackTrigger.NEW_HAND)
            if any(e.kind.value == "ACTION_AREA_CHANGED"
                   and e.description.startswith("unexplained:")
                   for e in prelim.events):
                triggers.append(FallbackTrigger.TEMPORAL_UNEXPLAINED)
        if self._ocr_failed_now > 0:
            self.ocr_fail_streak += 1
        else:
            self.ocr_fail_streak = 0
        if self.ocr_fail_streak >= self.config.ocr_failure_streak_trigger:
            triggers.append(FallbackTrigger.OCR_FAILURE_STREAK)
        if self._card_uncertain:
            triggers.append(FallbackTrigger.CARD_UNCERTAIN)
        # 关键字段 uncertain（非冲突）也允许兜底复核
        crit_unc = [u for u in normalized.uncertainty
                    if isinstance(u, dict)
                    and self._is_critical_field(u.get("field", ""))]
        if crit_unc:
            triggers.append(FallbackTrigger.KEY_FIELD_CONFLICT)
        return triggers

    def _is_critical_field(self, field: str) -> bool:
        return any(field == c or field.startswith(c + ".") or
                   field.startswith(c + "[") for c in self.config.critical_fields)

    def _apply_fallback(self, frame, crops, raw, triggers):
        # 任何兜底尝试（含后端不可用）都必须计入 gate 调用与 stats，
        # 否则 cooldown/预算门控失效、stats 失真
        self.stats["fallback_calls"] += 1
        if not self.multimodal.available():
            self.gate.record(FallbackCall(
                trigger=triggers[0], at=frame.captured_at,
                frame_id=frame.frame_id, outcome="UNAVAILABLE", latency_s=None))
            raw.uncertainty = list(raw.uncertainty or [])
            raw.uncertainty.append(UncertaintyEntry(
                field="vision.multimodal", status="missing"))
            return raw, assemble(raw)
        result: MultimodalResult = self.multimodal.observe(
            frame.image, crops,
            context={"triggers": [t.value for t in triggers],
                     "frame_id": frame.frame_id})
        self.gate.record(FallbackCall(
            trigger=triggers[0], at=frame.captured_at,
            frame_id=frame.frame_id, outcome=result.status.value,
            latency_s=None))
        if result.status is not MultimodalStatus.OK or not result.raw_fields:
            raw.uncertainty = list(raw.uncertainty or [])
            raw.uncertainty.append(UncertaintyEntry(
                field="vision.multimodal",
                status=("uncertain" if result.status
                        is MultimodalStatus.UNCERTAIN else "missing")))
            return raw, assemble(raw)
        raw = _merge_raw_fields(raw, result.raw_fields)
        return raw, assemble(raw)

    # ---------------------------------------------------------- 时序
    def _finalize_temporal(self, frame, raw, normalized):
        snap = ObservationSnapshot(frame.frame_id, frame.captured_at, raw,
                                   normalized, "processing", False)
        if self.tracker.current is None:
            self.tracker.update(snap)
            return None
        return self.tracker.update(snap)

    # ---------------------------------------------------------- 状态
    def _compute_status(self, normalized, missing, transition):
        errors = normalized.validation.errors
        if errors:
            self.stats["state_conflicts"] += 1
            codes = ",".join(sorted({e["code"] for e in errors}))
            return RecognitionStatus.STATE_CONFLICT, True, \
                f"validation errors: {codes}"
        crit_missing = [f"{n.value}:{w}" for n, w in missing.items()
                        if self.roi_specs.get(n) is not None
                        and self.roi_specs[n].required]
        if crit_missing:
            return RecognitionStatus.ROI_MISSING, True, \
                f"critical ROI missing: {';'.join(crit_missing)}"
        crit_unc = [u for u in normalized.uncertainty
                    if isinstance(u, dict) and self._is_critical_field(
                        u.get("field", ""))]
        if crit_unc:
            fields = ",".join(sorted({u["field"] for u in crit_unc}))
            return RecognitionStatus.RECOGNITION_UNCERTAIN, True, \
                f"critical uncertainty: {fields}"
        any_unc = bool(normalized.uncertainty) or bool(missing)
        if any_unc:
            return RecognitionStatus.RECOGNITION_UNCERTAIN, False, None
        return RecognitionStatus.RECOGNITION_OK, False, None

    # ---------------------------------------------------------- 便捷构造
    def _blocked_obs(self, status, reason, now, *, fallback_frame=None,
                     fallback_trigger=None):
        if fallback_frame is not None and fallback_trigger is not None:
            decision = self.gate.request(fallback_trigger,
                                         frame_id=fallback_frame.frame_id,
                                         now=now)
            if decision.allowed:
                result = self.multimodal.observe(
                    fallback_frame.image, {},
                    context={"triggers": [fallback_trigger.value],
                             "frame_id": fallback_frame.frame_id})
                self.gate.record(FallbackCall(
                    trigger=fallback_trigger, at=now,
                    frame_id=fallback_frame.frame_id,
                    outcome=result.status.value, latency_s=None))
                self.stats["fallback_calls"] += 1
            else:
                self.stats["fallback_denied"] += 1
        return RealtimeVisionObservation(
            status=status, blocked=True, failure_reason=reason,
            unchanged=False, waiting=False, timestamp=now, frame_id=None,
            synthetic_test_only=True, normalized=None, provenance={},
            transition=None, events=())

    def _duplicate_obs(self, frame, now):
        if self.last_obs is None:
            return self._blocked_obs(RecognitionStatus.RECOGNITION_UNCERTAIN,
                                     "duplicate without prior observation", now)
        return replace(self.last_obs, unchanged=True, waiting=False,
                       timestamp=now, frame_id=frame.frame_id)

    def run(self, *, stop_event=None, max_frames: int | None = None,
            on_observation=None, sleep_s: float | None = None
            ) -> list:
        """持续轮询循环（生产用；测试请直接调用 step() 保持确定性）。"""
        import time as _sleep_time
        out = []
        delay = self.frame_manager.interval if sleep_s is None else sleep_s
        while stop_event is None or not stop_event.is_set():
            if max_frames is not None and len(out) >= max_frames:
                break
            obs = self.step()
            if obs is not None:
                out.append(obs)
                if on_observation is not None:
                    on_observation(obs)
            _sleep_time.sleep(delay)
        return out


class _ProvWrap:
    def __init__(self, entries: dict):
        self._entries = entries

    def to_dict(self) -> dict:
        return dict(self._entries)


def _merge_raw_fields(raw: RawVisionState, fields: dict) -> RawVisionState:
    """兜底结果合并：只填 None 字段（不覆盖已识别值），绝不静默修正。"""
    for key, value in (fields or {}).items():
        if not hasattr(raw, key):
            continue
        if getattr(raw, key) is None:
            setattr(raw, key, value)
    return raw
