"""Cash-Game AI — 一键分析链 Runner（UI 后端）。

一次点击 = 完整合法链：
  Capture（BitBlt 屏幕捕获 → 1876 帧）
    → V3F observers（视觉入口 —— 唯一视觉管线）
    → snapshot → production_bridge（readiness gate）
    → RealtimeRunner.decide()（StateAgent→Equity→Solver→Judge）
    → DecisionSafetyResult + Agent 输出快照
    → UI（结论优先）

红线：Vision 是唯一视觉入口（结构化输出喂后续 Agent——无第二套
pipeline）；输出-only（无 click/下注）；UNKNOWN-first；不修改核心。
"""
from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from typing import Any


def board_blocking(obs: dict) -> str | None:
    """P30-9-GA：公共牌张数不足 ⇒ 返回阻断串（None=可继续）。**不伪造缺牌**。

    起因（真实会话取证）：19/25 个真实翻后局面的 state 里 `board=['Ac']` 之类，
    引擎仍在牌面不全的情况下产出建议 ⇒ 牌面相关计算不可信（已实测确认）。
    """
    need = {"preflop": 0, "flop": 3, "turn": 4, "river": 5}
    street = str((obs or {}).get("street") or "preflop")
    want = need.get(street)
    if want is None:
        return None
    got = len([c for c in ((obs or {}).get("board") or []) if c])
    if got < want:
        return f"board_incomplete:{got}/{want}@{street}"
    return None


@dataclass
class AgentOutput:
    agent: str
    status: str                       # OK | UNKNOWN | UNAVAILABLE | SKIPPED
    output: dict = field(default_factory=dict)
    provenance: str = ""

    def to_dict(self) -> dict:
        return {"agent": self.agent, "status": self.status,
                "output": dict(self.output), "provenance": self.provenance}


@dataclass
class AnalysisResult:
    status: str                        # OK | UNAVAILABLE | DECISION_UNAVAILABLE | ERROR | ADVICE_REFERENCE
    decision_action: str | None = None
    recommended_size: float | str | None = None
    confidence: float | None = None
    solver_mode: str = "UNKNOWN"
    strategy_mode: str = "UNKNOWN"
    reasoning: str | None = None
    reason: str | None = None
    blocking_fields: list = field(default_factory=list)
    advice_json: str | None = None
    rule_size: dict | None = None
    unknown_fields: list = field(default_factory=list)
    # R14-EXE-3：参考建议模式契约（仅 status=ADVICE_REFERENCE 非空）
    advice_type: str | None = None
    legal_actions_status: str | None = None
    agents: list = field(default_factory=list)      # AgentOutput[]
    vision: dict = field(default_factory=dict)      # 结构化视觉输出
    frame_provenance: dict = field(default_factory=dict)
    raw: dict = field(default_factory=dict)          # 展开详情（可折叠）

    def to_dict(self) -> dict:
        return {"status": self.status,
                "decision_action": self.decision_action,
                "recommended_size": self.recommended_size,
                "confidence": self.confidence,
                "solver_mode": self.solver_mode,
                "strategy_mode": self.strategy_mode,
                "reasoning": self.reasoning,
                "reason": self.reason,
                "blocking_fields": list(self.blocking_fields),
                "advice_json": self.advice_json,
                "rule_size": self.rule_size,
                "unknown_fields": list(self.unknown_fields),
                "advice_type": self.advice_type,
                "legal_actions_status": self.legal_actions_status,
                "agents": [a.to_dict() for a in self.agents],
                "vision": dict(self.vision),
                "frame_provenance": dict(self.frame_provenance),
                "raw": self.raw}


# ── P30-9-IC：**窗口几何归一化** ─────────────────────────────────────────
#   实测（用户报告）：牌桌窗口变成 **1866x1400** ✗，而全部识别器都按
#   **1876x1400** 标定（`action_observer`/`betting_header_observer`/`bet_observer`
#   一律 `!= (1876,1400) ⇒ UNKNOWN` ✓）⇒ 整条视觉链诚实拒绝 ⇒ UI 显示"信息不足" ✗。
#   ⇒ 在**唯一入口**做归一化：小幅偏差（≤8%）**缩放到参考几何** ✓ 并把缩放记入结果 ✓；
#     偏差过大 ⇒ **不猜** ✗，保持原样并标注 `geometry_out_of_tolerance` ✓（由上层如实报错 ✓）。
# P30-9-IM：**观测缓存**（按帧哈希 ✓）—— 开关与容量（可回退 ✓）
_VISION_CACHE_TTL_S = 120.0
_VISION_CACHE_MAX = 6


def _vision_cache_enabled() -> bool:
    return str(__import__("os").environ.get("VISION_CACHE", "1")).lower() \
        not in ("0", "off", "false", "no")


def _vision_cache_enabled_ok() -> bool:      # 兼容别名（无实际用途）
    return _vision_cache_enabled()


REF_W, REF_H = 1876, 1400
GEOM_TOL = 0.08


def normalize_geometry(frame_rgba: bytes, w: int, h: int) -> tuple:
    """把捕获帧归一到参考几何 `1876x1400` ✓（返回 `(rgba, w, h, info)`）。

    * 已是参考尺寸 ⇒ 原样返回 ✓（幂等 ✓）；
    * 偏差 ≤8% ⇒ 双线性缩放 ✓，`info["scale"]` 记录缩放比 ✓；
    * 偏差 >8% ⇒ **不归一** ✗（`info["reason"]` 标注 ✓）⇒ 上层照常诚实 UNKNOWN ✓。
    """
    info = {"reference": (REF_W, REF_H), "from": (int(w), int(h)),
             "normalized": False, "scale": None, "reason": None}
    try:
        if (int(w), int(h)) == (REF_W, REF_H):
            return frame_rgba, REF_W, REF_H, info
        dw = abs(int(w) - REF_W) / REF_W
        dh = abs(int(h) - REF_H) / REF_H
        if dw > GEOM_TOL or dh > GEOM_TOL:
            info["reason"] = f"geometry_out_of_tolerance:{int(w)}x{int(h)}"
            return frame_rgba, int(w), int(h), info
        from PIL import Image
        im = Image.frombytes("RGBA", (int(w), int(h)), bytes(frame_rgba))
        im = im.resize((REF_W, REF_H), Image.BILINEAR)
        info["normalized"] = True
        info["scale"] = (round(REF_W / int(w), 5), round(REF_H / int(h), 5))
        return im.tobytes(), REF_W, REF_H, info
    except Exception as e:                                      # noqa: BLE001
        info["reason"] = f"normalize_failed:{type(e).__name__}"
        return frame_rgba, int(w), int(h), info


class CashGameRunner:
    """UI 后端：一键分析（复用既有核心——零核心修改）。"""

    def __init__(self):
        self._runtime = None
        # P30-9-IM：观测缓存（帧哈希 → (时间, 观测) ✓）；只缓存**视觉观测** ✓
        self._vision_cache: dict = {}
        self._vision_cache_hit = False

    @staticmethod
    def blocking_reason(blocking: list) -> str:
        """MANDATORY 缺失字段 → 人话原因（不编造，逐字段映射；
        R13-EXE-2：两个 GUI（app/preview_gui）共用）。"""
        mapping = {
            "hero_cards": "缺手牌（未识别到 hero 底牌）",
            "blinds": "盲注未识别（header 金额不可读；可用下方兜底输入或等自动预填）",
        "pot": "底池未识别（数字区不可读——避免发牌动画帧、确保数字区可见）",
        "stack": "筹码量未识别（座位数字被遮挡/动画中）",
        "hero": "未定位到你的座位（庄家按钮/座位几何未识别）",
        "hero_position": "位置未识别（庄家按钮不可读）",
        "hero_cards": "未识别到你的底牌（牌面动画/遮挡；已自动重读 2 次）",
        "street": "街道不可判定（公共牌/占用状态未识别）",
            "hero": "未定位到 hero 座位",
            "dealer": "未定位到按钮位",
            "pot": "底池未识别",
            "stack": "筹码量未识别",
            "status": "座位状态未识别",
            "hero_position": "位置未识别",
            "board": "公共牌未识别",
            "street": "街未确认",
        }
        return "；".join(mapping.get(b, b) for b in blocking)

    @staticmethod
    def _fallback_board_read(frame_rgba, w: int, h: int) -> list[str] | None:
        """P30-9-GT：本地识别器**只读**补读公共牌（V3F 快照残缺时的回退）。

        * 只在**已知 render 几何**下尝试（未知几何 ⇒ 返回 None，绝不猜测 ✓）；
        * 只读同一帧，不点击、不注入、不改变任何游戏状态 ✓；
        * 任何失败都返回 None（调用方按原样诚实阻断 ✓）。
        """
        if frame_rgba is None or not w or not h:
            return None
        try:
            from poker_decision_ai.vision2.pixel.realclient.coinpoker_board import (
                CoinPokerBoardRecognizer)
            from poker_decision_ai.vision2.pixel.realclient.render_class import (
                window_for_frame)
            win = window_for_frame(int(w), int(h))
            if not win:
                return None                       # 未知几何 ⇒ 不猜测 ✓
            rec = CoinPokerBoardRecognizer(window=win)
            if not rec.available():
                return None
            out = rec.recognize(bytes(frame_rgba), int(w), int(h), 0)
            cards = [str(c) for c in (out.get("value") or []) if c]
            return cards or None
        except Exception:                          # noqa: BLE001
            return None

    @staticmethod
    def record_board_evidence(kind: str, v3f_board: list, local_board: list | None,
                               street: str, detail: str, w: int | None = None,
                               h: int | None = None,
                               frame_rgba=None) -> str | None:
        """P30-9-GU：**补读成败双向留痕**（JSONL 一行一事件）⇒ 可算恢复率 ✓。

        * `kind`：`fallback_ok`（本地补读成功 ⇒ 采用）或 `fallback_failed`（仍残缺 ⇒ 阻断）；
        * 记录 V3F 牌面、本地牌面、街、几何、时间 ⇒ 事后可算
          **恢复率 = fallback_ok / (fallback_ok + fallback_failed)** ✓；
        * 只在**证据上限内**附带保存该帧 PNG（磁盘可控 ✓）；任何失败静默 ✓。
        """
        try:
            import json as _json
            import pathlib as _pl
            import time as _t
            d = _pl.Path("logs")
            d.mkdir(parents=True, exist_ok=True)
            rec = {"ts": _t.strftime("%Y-%m-%dT%H:%M:%S"), "kind": kind,
                    # P30-9-HW：**来源标记**（三级判定 ✓）——让"恢复率"只统计**真实**事件 ✓
                    #   ① 显式 `DSH_EVIDENCE_SOURCE`（脚本/门禁 ✓）
                    #   ② 在 pytest 下运行 ⇒ `test` ✓（合成帧/夹具 ≠ 真实对局 ✗，
                    #      此前这些事件被误记成 runtime ✗ ⇒ 分母被污染 ✗）
                    #   ③ 否则 `runtime`（真实对局 ✓）
                    "source": (__import__("os").environ.get("DSH_EVIDENCE_SOURCE")
                                or ("test" if __import__("os").environ.get(
                                    "PYTEST_CURRENT_TEST") else "runtime")),
                    "street": street, "v3f_board": list(v3f_board or []),
                    "local_board": (list(local_board) if local_board else None),
                    "detail": detail, "w": w, "h": h}
            if frame_rgba is not None and w and h:
                try:
                    import hashlib
                    h16 = hashlib.sha256(bytes(frame_rgba)).hexdigest()[:16]
                    rec["sha16"] = h16
                    frames = d / "frames"
                    frames.mkdir(parents=True, exist_ok=True)
                    png = frames / f"{_t.strftime('%Y%m%d-%H%M%S')}-{kind}-{h16}.png"
                    if len(list(frames.glob("*.png"))) < int(
                            __import__("os").environ.get("MAX_BOARD_EVIDENCE", "20")):
                        from PIL import Image
                        Image.frombytes("RGBA", (int(w), int(h)),
                                         bytes(frame_rgba)).save(str(png))
                        rec["png"] = png.name
                except Exception:                      # noqa: BLE001
                    pass
            with (d / "board_fallback.jsonl").open("a", encoding="utf-8") as fh:
                fh.write(_json.dumps(rec, ensure_ascii=False) + "\n")
            return rec["ts"]
        except Exception:                          # noqa: BLE001
            return None

    @staticmethod
    def _save_board_incomplete_frame(frame_rgba, w: int, h: int, obs: dict,
                                      detail: str) -> str | None:
        """P30-9-GM：牌面残缺时保存**证据帧**（便于定位"只识别到 1–2 张公共牌"的根因）。

        真实日志里那些残缺帧**没有被保存** ⇒ 根因无法定位 ✗。此处按内容哈希去重、
        最多保留 `MAX_BOARD_EVIDENCE`（默认 20）份；保存失败一律静默（不影响决策 ✓）。
        """
        if frame_rgba is None or not w or not h:
            return None
        try:
            import hashlib
            import json as _json
            import pathlib as _pl
            import time as _t
            from PIL import Image
            raw = bytes(frame_rgba)
            h16 = hashlib.sha256(raw).hexdigest()[:16]
            d = _pl.Path("logs") / "frames"
            d.mkdir(parents=True, exist_ok=True)
            if list(d.glob(f"*BOARD_INCOMPLETE*{h16}*")):
                return None
            limit = int(__import__("os").environ.get("MAX_BOARD_EVIDENCE", "20"))
            if len(list(d.glob("*BOARD_INCOMPLETE*.png"))) >= limit:
                return None
            ts = _t.strftime("%Y%m%d-%H%M%S")
            base = d / f"{ts}-BOARD_INCOMPLETE-{h16}"
            Image.frombytes("RGBA", (w, h), raw).save(str(base) + ".png")
            (d / f"{base.name}.json").write_text(_json.dumps(
                {"detail": detail, "street": (obs or {}).get("street"),
                  "board": (obs or {}).get("board"), "sha16": h16, "w": w, "h": h,
                  "warnings": [(x or {}).get("code") for x in ((obs or {}).get("warnings") or [])]},
                ensure_ascii=False, indent=1), encoding="utf-8")
            return str(base)
        except Exception:                          # noqa: BLE001
            return None

    def _get_runtime(self):
        if self._runtime is None:
            from poker_decision_ai.core.orchestration.runtime import (
                build_runtime)
            self._runtime = build_runtime()
        return self._runtime

    def capture_frame(self) -> tuple:
        """BitBlt 捕获 CoinPoker 窗口 → (frame_rgba, w, h, bbox)。

        失败 → 返回 (None, None, None, None)（UI 显示 CAPTURE 错误）。
        仅复用既有 capture 逻辑（scripts/phase_v2s_auto_capture 同源）。
        """
        import numpy as np
        from poker_decision_ai.vision2.pixel.realclient.capture import (
            CAPTURE_OK, CtypesBitBltBackend)
        try:
            from scripts.phase_v2s_auto_capture import locate_coinpoker
            win = locate_coinpoker()
        except Exception:
            win = None
        if win is None:
            return None, None, None, None
        x0, y0, cw, chh = win["bbox"]
        backend = CtypesBitBltBackend()
        r = backend.capture()
        if r["status"] != CAPTURE_OK:
            return None, None, None, None
        arr = np.frombuffer(r["image_bytes"], dtype=np.uint8).reshape(
            r["height"], r["width"], 4)
        if x0 + cw > r["width"] or y0 + chh > r["height"]:
            cw = min(cw, r["width"] - x0)
            chh = min(chh, r["height"] - y0)
        sub = arr[y0:y0 + chh, x0:x0 + cw]
        # P30-9-IC：**几何归一化** ✓（窗口尺寸/DPI 变化时，全部识别器都按 1876x1400 标定 ✓）
        _rgba, _w, _h, _geo = normalize_geometry(sub.tobytes(), cw, chh)
        self._last_geometry = _geo
        return _rgba, _w, _h, win["bbox"]

    def analyze(self, frame_rgba=None, w=None, h=None, *,
                allow_recapture: bool | None = None,
                blinds_override: dict | None = None,
                multiway_conservative: bool | None = None) -> AnalysisResult:
        """一键分析：优先用给定帧；否则实时 Capture。

        `allow_recapture`：是否允许 hero 牌缺失时的只读重捕获。
        缺省 = 自行捕获时允许；**宿主（GUI）自行截图后传入帧时须显式传 True**，
        测试/fixture 轨道传 False（避免对固定帧重复读屏）。
        """
        from poker_decision_ai.vision2.pixel.realclient.production_bridge import (
            snapshot_to_vision_observation, frame_to_decision_request,
            vision_cache_new_epoch)
        runtime = self._get_runtime()
        # P30-9-IC：**宿主传入的帧也要归一化** ✓（GUI 自行截图时会传 1866x1400 之类 ✗；
        #   全部识别器按 1876x1400 标定 ⇒ 不归一就会"信息不足" ✗）。幂等 ✓。
        self._last_geometry = None
        if frame_rgba is not None and w and h:
            try:
                frame_rgba, w, h, _g = normalize_geometry(bytes(frame_rgba), int(w), int(h))
                self._last_geometry = _g
            except Exception:                      # noqa: BLE001
                pass
        # P30-9-HP：**请求边界** —— 本次 analyze 内缓存的新写入不可见 ✓（同输入同输出 ✓）
        try:
            from poker_decision_ai.core.providers.spot_table_6max import begin_request
            begin_request()
        except Exception:                          # noqa: BLE001
            pass
        # 同一次 analyze 内 bridge 去重作用域（跨点击不复用——见
        # production_bridge.vision_cache_new_epoch 注释）
        vision_cache_new_epoch()

        live = frame_rgba is None or w is None or h is None
        bbox = None
        if live:
            frame_rgba, w, h, bbox = self.capture_frame()
            if frame_rgba is None:
                return AnalysisResult(
                    status="UNAVAILABLE",
                    reason="capture_failed:coinpoker_window_not_found",
                    agents=[AgentOutput("Capture", "UNAVAILABLE",
                                        provenance="bitblt")])
        # 视觉入口（唯一 pipeline）→ bridge → decide
        # P30-9-IM：**观测按帧哈希缓存** ✓ —— 视觉观测**只由像素决定** ✓
        #   （同一份 RGBA ⇒ 同一观测 ✓），因此对"同一帧再次点击"复用是**正确**的 ✓；
        #   **决策永远重算** ✓（决策还依赖求解器缓存/时点 ✓，缓存它就会改变行为 ✗）。
        #   这正是"连点两次第二次秒回"的来源 ✓；`VISION_CACHE=0` 可关闭 ✓（可回退 ✓）。
        got = None
        _vkey = None
        if _vision_cache_enabled() and frame_rgba is not None:
            try:
                import hashlib as _hl
                _vkey = f"{w}x{h}:" + _hl.sha256(bytes(frame_rgba)).hexdigest()[:24]
                _hit = self._vision_cache.get(_vkey)
                if _hit is not None and (time.time() - _hit[0]) <= _VISION_CACHE_TTL_S:
                    got = copy.deepcopy(_hit[1])            # 深拷贝：下游会改写 got ✓
                    self._vision_cache_hit = True
            except Exception:                              # noqa: BLE001
                got = None
        if got is None:
            self._vision_cache_hit = False
            got = frame_to_decision_request(frame_rgba, w, h)
            if _vkey is not None:
                try:
                    self._vision_cache[_vkey] = (time.time(), copy.deepcopy(got))
                    while len(self._vision_cache) > _VISION_CACHE_MAX:
                        _old = min(self._vision_cache, key=lambda k: self._vision_cache[k][0])
                        self._vision_cache.pop(_old, None)
                except Exception:                          # noqa: BLE001
                    pass
        # P30-9-G：hero 底牌缺失多为**发牌/翻牌动画或瞬时遮挡**（同一张桌子时好时坏）。
        # 仅对**实时捕获**做有限次只读重捕获（重读同一屏，不注入任何内容、不点击）；
        # 重捕获后仍缺 ⇒ 按原样阻断（诚实）。
        recapture = 0
        can_recapture = live if allow_recapture is None else bool(allow_recapture)
        # P30-9-N：**瞬时类**才重试（动画/遮挡/帧态过渡），且不重试"纯 blinds"
        # （盲注不随帧变化，已有证据链/兜底路径）。预算 2 次 ≈ 每轮 1.2s。
        _TRANSIENT = ("hero", "hero_cards", "hero_position", "pot", "stack")
        # P30-9-II：**重捕上限 2 → 1** ✓ —— 实测每次重捕都会**重跑整条视觉链** ✗
        #   （`frame_to_decision_request` ≈3.4 秒 ✗）⇒ 真实点击出现 `recapture=2` 时
        #   要多付 **~7 秒** ✗✗（EXE 实跑 16.2 秒里很大一块 ✓，也是用户"还是很慢"的主因之一 ✓）。
        #   保留 1 次重试（应对动画/遮挡的**瞬时**缺失 ✓）；第 2 次收益低而代价高 ✗。
        #   可用 `RECAPTURE_MAX=2` 恢复旧行为 ✓（可回退 ✓，不改判定口径 ✓）。
        _MAX_RECAPTURE = max(0, int(__import__("os").environ.get("RECAPTURE_MAX", "1")))
        recapture_classes: list = []
        if can_recapture:
            import time as _time

            def _transient(g: dict) -> str:
                br = g.get("bridge") or {}
                blocking = list(br.get("blocking_fields") or [])
                reason = str(g.get("reason") or "")
                if "street_unrepresentable" in reason:
                    return "street_unrepresentable"
                # P30-9-GA：公共牌不足多为**翻牌/转牌动画或遮挡**（同 hero 牌类瞬时）
                if board_blocking(br.get("observation") or {}):
                    return "board"
                hits = [k for k in _TRANSIENT if k in blocking]
                if hits:
                    return ",".join(hits)
                return ""

            cls = _transient(got)
            while recapture < _MAX_RECAPTURE and cls:
                recapture_classes.append(cls)
                _time.sleep(0.25)
                f2, w2, h2, _b2 = self.capture_frame()
                if f2 is None:
                    break
                got = frame_to_decision_request(f2, w2, h2)
                frame_rgba, w, h = f2, w2, h2
                recapture += 1
                cls = _transient(got)
        self._last_recapture = recapture
        self._last_recapture_classes = recapture_classes
        # P30-9-GA：重拍后牌面仍不足 ⇒ **诚实阻断**（不产出牌面相关建议）
        _obs_now = (got.get("bridge") or {}).get("observation") or {}
        _board_blk = board_blocking(_obs_now)
        if _board_blk:
            # P30-9-GT：**V3F 快照牌面残缺 ⇒ 用本地识别器只读补读同一帧**。
            #   取证：全部 19 例残缺牌面都来自 `v3f_snapshot`（该来源 86% 残缺 ✗），
            #   另一来源 19/19 完整 ✓ ⇒ 缺陷在 V3F 生产端；本地识别器可作可靠回退 ✓。
            #   几何未知则跳过（render_class 契约：未知几何 ⇒ 不猜测 ✓）。
            _fb_board = self._fallback_board_read(frame_rgba, w, h)
            _need_n = {"flop": 3, "turn": 4, "river": 5}.get(
                str(_obs_now.get("street") or ""), 0)
            if _fb_board and len(_fb_board) == _need_n:
                _obs_new = dict(_obs_now)
                _obs_new["board"] = list(_fb_board)
                _obs_new["board_source"] = "py_reader_fallback"
                _obs_new["warnings"] = list(_obs_now.get("warnings") or []) + [{
                    "code": "BOARD_FALLBACK_READ",
                    "message": (f"V3F 快照牌面残缺（{_board_blk}）⇒ 本地识别器补读成功 "
                                 f"{''.join(_fb_board)}")}]
                _br2 = dict(got.get("bridge") or {})
                _br2["observation"] = _obs_new
                _br2["board_fallback"] = "py_reader"
                got = {**got, "bridge": _br2}
                _board_blk = board_blocking(_obs_new)
                # P30-9-GU：**成功留痕**（供算恢复率 ✓）
                try:
                    self.record_board_evidence(
                        "fallback_ok", _obs_now.get("board"), _fb_board,
                        str(_obs_now.get("street") or ""), str(_board_blk or "resolved"),
                        w, h, frame_rgba)
                except Exception:                      # noqa: BLE001
                    pass
            elif _fb_board is not None or True:
                # P30-9-GU：**失败留痕**（本地也没读全 ⇒ 仍阻断 ✓）
                try:
                    self.record_board_evidence(
                        "fallback_failed", _obs_now.get("board"), _fb_board,
                        str(_obs_now.get("street") or ""), str(_board_blk),
                        w, h, frame_rgba)
                except Exception:                      # noqa: BLE001
                    pass
        if _board_blk:
            # P30-9-GM：**保存证据帧**（真实日志里那些残缺帧没被保存 ⇒ 根因无法定位 ✗）
            try:
                self._save_board_incomplete_frame(frame_rgba, w, h, _obs_now, _board_blk)
            except Exception:                      # noqa: BLE001
                pass
            _br = dict(got.get("bridge") or {})
            _br["blocking_fields"] = sorted(set(
                list(_br.get("blocking_fields") or []) + ["board"]))
            _br["board_blocking"] = _board_blk
            got = {**got, "status": "DECISION_UNAVAILABLE", "bridge": _br,
                    "reason": (str(got.get("reason") or "") + f"; {_board_blk}").strip("; ")}
        # P30-9-HY：**对手档案库** —— 每次点击识别即按对手 ID 记录逐座位证据 ✓
        #   设计红线：只记**真实可见**字段（seats 的 seat/name/stack/in_hand ✓ 与
        #   `extra.v3f_bets` 的逐座位前置下注额 ✓）；实测 `v3f_action_history` 恒 None ✗、
        #   `v3f_status` 值恒 UNKNOWN ✗、`seats[].status` 不存在 ✗ ⇒ **单帧拿不到
        #   fold/check/raise/call 标签** ⇒ 绝不虚构动作序列 ✗；动作线索只在**同手牌连续遭遇**
        #   上做**可证据化**推断（筹码减少=投入 / 前置下注增加 / in_hand→消失=弃牌证据 ✓）。
        #   严格旁路：失败静默、不阻塞点击 ✓、可 `OPPONENT_DB=0` 关闭 ✓。
        self._last_opponents = None
        try:
            from poker_decision_ai.core import opponents as _opp
            if _opp.enabled() and isinstance(_obs_now, dict) and _obs_now.get("seats"):
                self._last_opponents = _opp.record_encounter(
                    _obs_now, frame_rgba=(bytes(frame_rgba) if frame_rgba else None))
        except Exception:                          # noqa: BLE001
            self._last_opponents = None
        # P30-9-HI：**翻后 to_call UNKNOWN ⇒ 证据采集**（定位"动作面板未识别"根因 ✓）
        #   取证：41 个真实翻后请求里 25 个 to_call 为 UNKNOWN ✗（完整牌面 19/22 ✗）
        #   一次性旁路：只读留痕 + 存帧（上限内 ✓），**不改变任何决策** ✓。
        try:
            _st_hi = str(_obs_now.get("street") or "")
            _tc_hi = _obs_now.get("to_call")
            if _st_hi in ("flop", "turn", "river") and _tc_hi is None:
                self.record_board_evidence(
                    "to_call_unknown", _obs_now.get("board"), None, _st_hi,
                    "to_call=UNKNOWN（动作面板/投注额未识别）", w, h, frame_rgba)
        except Exception:                          # noqa: BLE001
            pass
        self._click_seq = int(getattr(self, "_click_seq", 0)) + 1
        try:                                       # 杜绝跨调用残留（默认关=无影响）
            from poker_decision_ai.multiway_equity import clear_conservative
            clear_conservative()
        except Exception:                          # noqa: BLE001
            pass
        try:
            # P30-9-BE：安装"范围化联合胜率"挂钩（只计算与记录；是否**采用**由开关决定）
            from poker_decision_ai.joint_equity import (
                LAST_JOINT, USE_JOINT_AS_HERO_EQUITY, install_joint_hook)
            install_joint_hook()
            # None ⇒ 采用模块默认（P30-9-BF：默认开启范围化联合胜率）
            if multiway_conservative is not None:
                USE_JOINT_AS_HERO_EQUITY["on"] = bool(multiway_conservative)
            LAST_JOINT.clear()
        except Exception:                          # noqa: BLE001
            pass
        # P30-9-AK：多人保守模式（默认关）——开启时用**多人联合胜率**替换引擎的
        # 逐对手平均；未开启时挂钩为零影响（原样返回）。
        self._multiway_info = None
        if multiway_conservative:
            try:
                from poker_decision_ai.multiway_equity import (
                    clear_conservative, conservative_for, install_equity_hook,
                    set_conservative)
                _obs0 = (got.get("bridge") or {}).get("observation") or {}
                _est = conservative_for(_obs0)
                if _est.get("ok"):
                    install_equity_hook()
                    set_conservative(_est)
                    self._multiway_info = _est
            except Exception:                      # noqa: BLE001
                self._multiway_info = None
        # P30-9-AH：挂上本帧（供**失败自动存帧**）
        try:
            self._last_frame = (bytes(frame_rgba), int(w), int(h))
            self._last_frame_path = None
        except Exception:                          # noqa: BLE001
            self._last_frame = None
        # P30-9-AZ：跨帧线路追踪（从投入增量推导 action_history/current_player）
        try:
            from poker_decision_ai.line_tracker import (
                LineTracker, attach_to_observation)
            if getattr(self, "_line_tracker", None) is None:
                self._line_tracker = LineTracker()
            _obs_after = (got.get("bridge") or {}).get("observation") or {}
            _derived = self._line_tracker.observe(
                _obs_after, frame_id=str(getattr(self, "_click_seq", 0)))
            self._last_line = _derived
            got = {**got,
                   "bridge": {**(got.get("bridge") or {}),
                               "observation": attach_to_observation(
                                   _obs_after, _derived)}}
        except Exception:                          # noqa: BLE001
            self._last_line = None
        # P30-9-AD：用自校准发现器补齐观察里漏读的投注位（只补不改）
        try:
            got = {**got,
                   "bridge": {**(got.get("bridge") or {}),
                               "observation": self._fill_bets_from_finder(
                                   frame_rgba, w, h,
                                   (got.get("bridge") or {}).get("observation") or {})}}
        except Exception:                          # noqa: BLE001
            pass
        if got["status"] != "OK":
            br = got["bridge"]
            obs = br.get("observation") or {}
            blocking = list(br.get("blocking_fields") or [])
            # P30-9-F：header 金额不可读 ⇒ 唯一阻断项是 blinds。若**已识别证据**
            # （preflop 盲注座投入 + 庄家座）足以推导盲注 → 用证据补上并重跑决策；
            # 全程标注 blinds_source（证据优先；不改冻结层；无证据仍阻断）。
            # P30-9-O：header 盲注若与其他已识别证据矛盾（如 pot < 盲注、桌上投入
            # 远小于 BB）⇒ **不信任**，改走证据链/兜底（绝不静默缩放）
            _distrust = {}
            try:
                from poker_decision_ai.facing_evidence import blinds_consistent
                _c = blinds_consistent(obs)
                if not _c.get("ok"):
                    _distrust = _c
                    obs = {**obs, "blinds": None}
            except Exception:                          # noqa: BLE001
                _distrust = {}
            if _distrust:
                br = {**br, "blocking_fields": sorted(set(
                    list(br.get("blocking_fields") or []) + ["blinds"]))}
                blocking = list(br.get("blocking_fields") or [])
            # P30-9-S：只要有 blinds 阻断就解析（**不要求它是唯一阻断项**）——
            # 否则 hero 类同时缺失时，标题/兜底里的盲注永远用不上，用户看到的
            # 阻断清单也会一直挂着 blinds（实测：标题已写明 ¥0.05/¥0.1）。
            if "blinds" in blocking:
                # P30-9-K：证据优先级
                #   ① 桌上盲注座投入（preflop 直接证据）
                #   ② GUI 手动输入（user_setting——用户声明的级别，最可靠的兜底）
                #   ③ 同桌同局会话记忆（session_memo，TTL 900s）
                #   ④ 窗口标题（window_title：OS 元数据里常带 "0.02/0.05"）
                # P30-9-O：**每个来源都要过一致性检查**（防 header 误读 100× 被
                # 回写进兜底框后又被当真值使用）。
                rejected: list = []

                def _accept(cand: dict) -> dict:
                    if not (cand and cand.get("sb") and cand.get("bb")):
                        return {}
                    try:
                        from poker_decision_ai.facing_evidence import (
                            blinds_consistent)
                        chk = blinds_consistent(
                            {**obs, "blinds": {"sb": cand["sb"], "bb": cand["bb"]}})
                    except Exception:                  # noqa: BLE001
                        return cand
                    if chk.get("ok"):
                        return cand
                    rejected.append({"source": cand.get("source"),
                                     "reason": chk.get("reason"),
                                     "sb": cand.get("sb"), "bb": cand.get("bb")})
                    return {}

                ev = _accept(self._derive_blinds(obs))
                if not ev:
                    ev = _accept(self._blinds_override_ev(blinds_override) or {})
                if not ev:
                    ev = _accept(self._blinds_memo_lookup(obs, bbox))
                if not ev:
                    ev = _accept(self._blinds_from_window_title())
                self._last_blinds_rejected = rejected
                if ev and ev.get("sb") and ev.get("bb"):
                    # 把解析出的盲注写进观察（供 vision 摘要/GUI 显示与回写），
                    # 并从阻断清单中移除 blinds（它已被证据解决）
                    obs = {**obs, "blinds": {"sb": ev.get("sb"),
                                              "bb": ev.get("bb")}}
                    blocking = [b for b in blocking if b != "blinds"]
                    self._blinds_resolved = {
                        "source": ev.get("source"), "sb": ev.get("sb"),
                        "bb": ev.get("bb")}
                    self._blinds_memo_store(ev, obs, bbox)
                    if not blocking:
                        res = self._analyze_with_derived_blinds(
                            runtime, got, ev, w, h)
                        if res is not None:
                            return res
            self._log_attempt(got["status"], blocking,
                              getattr(self, "_last_recapture", 0), None, None)
            return AnalysisResult(
                status=got["status"],
                reason=got["reason"],
                blocking_fields=blocking,
                rule_size=None,
                unknown_fields=br.get("unknown_fields", []),
                vision=self._vision_summary(obs),
                agents=[self._agent("Vision", obs)],
                frame_provenance={"geometry": f"{w}x{h}",
                                  "status": got["status"],
                                  "recapture_attempts": getattr(
                                      self, "_last_recapture", 0),
                                  "recapture_classes": getattr(
                                      self, "_last_recapture_classes", []),
                                  "blinds_distrusted": (
                                      _distrust.get("reason") if _distrust else None),
                                  "blinds_sources_rejected": getattr(
                                      self, "_last_blinds_rejected", []),
                                  "blinds_resolved": getattr(
                                      self, "_blinds_resolved", None)})

        # P30-9-H：把本次**已识别/已推导**的盲注记入会话记忆（同桌同局复用）
        try:
            _ev = self._derive_blinds((got.get("bridge") or {}).get(
                "observation") or {})
            self._blinds_memo_store(_ev, (got.get("bridge") or {}).get(
                "observation") or {}, bbox)
        except Exception:                          # noqa: BLE001
            pass

        # OK → decide（RealtimeRunner——State/Equity/Solver/Judge 全链）。
        # R13-EXE-2：直接走 frame_to_decision（一次 decide），从 report
        # 提取 final_decision（动作+尺寸+置信+provenance）——原 tick 路径
        # 的 DecisionSafetyResult 不含 report 级字段（置信恒 n/a）。
        # R14-EXE-3（方案 Y）：ADVICE_REFERENCE = legal_actions UNKNOWN 时
        # 的参考建议（非决策）——action 只读 advice_reference 通道，带
        # 诚实标注（绝不显示为"针对当前下注的决策"）。
        from poker_decision_ai.frame_decision import frame_to_decision
        from poker_decision_ai.advice_reference import (
            ADVICE_TYPE_REFERENCE, ADVICE_WARNING_ZH,
            LEGAL_ACTIONS_STATUS_UNKNOWN)
        obs = (got["bridge"].get("observation") or {})
        # P30-9-AA：桥 status=OK 但 **to_call 未知**且看得到对手投入时，冻结入口会走
        # "求解器锚点"路径（weights={solver_anchor:1.0}）——它**绕过底池赔率**，实测会把
        # "Q-high 面对 75% 底池下注"输出成 check/raise（而其自身理由写 fold 更优）。
        # 处置：用"对手投入下界"补齐 to_call 后走同一决策核心（与盲注补齐路径一致）。
        _needs_evidence = False
        try:
            if obs.get("to_call") is None:
                from poker_decision_ai.facing_evidence import to_call_lower_bound
                _lb = to_call_lower_bound(obs)
                _needs_evidence = bool(_lb.get("to_call"))
        except Exception:                          # noqa: BLE001
            _needs_evidence = False
        if _needs_evidence:
            _bl = obs.get("blinds") or {}
            _ev = {"sb": _bl.get("sb"), "bb": _bl.get("bb"),
                   "source": "observed", "detail": {"from": "bridge"}}
            if _ev["sb"] and _ev["bb"]:
                _res = self._analyze_with_derived_blinds(runtime, got, _ev, w, h)
                if _res is not None:
                    return _res
        res = frame_to_decision(runtime, frame_rgba, w, h)
        prov = res.provenance or {}
        if res.status == "ADVICE_REFERENCE":
            adv = dict(prov.get("advice_reference") or {})
            return AnalysisResult(
                status="ADVICE_REFERENCE",
                decision_action=adv.get("action"),
                recommended_size=adv.get("size_bb"),
                confidence=adv.get("confidence"),
                solver_mode="BASELINE",
                strategy_mode="BASELINE",
                reasoning=ADVICE_WARNING_ZH,
                reason=res.reason,
                unknown_fields=list((got["bridge"].get("unknown_fields")
                                     or [])),
                advice_type=ADVICE_TYPE_REFERENCE,
                legal_actions_status=LEGAL_ACTIONS_STATUS_UNKNOWN,
                agents=self._advice_agents(adv, obs),
                vision=self._vision_summary(obs),
                frame_provenance={
                    **{k: prov.get(k) for k in (
                        "unknown_allowed_fields", "unknown_allowed_rationale",
                        "action_history") if k in prov},
                    "advice_type": ADVICE_TYPE_REFERENCE,
                    "legal_actions_status": LEGAL_ACTIONS_STATUS_UNKNOWN,
                    "status": res.status},
                # 决不让 judge final_decision 混入参考建议的 raw（红线：
                # 下游不得把 judge 的 to_call=0 假设动作当成决策）
                raw={"advice_reference": adv,
                     "judge_action_not_consumed": prov.get(
                         "judge_action_not_consumed")})
        rep = res.report or {}
        fd = (rep.get("final_decision") or {})
        try:
            prov = dict(prov or {})
            prov["recapture_attempts"] = getattr(self, "_last_recapture", 0)
        except Exception:                          # noqa: BLE001
            pass
        # P30-9-D：GUI 输出边界动作语义对齐（证据优先；冻结层/策略零改动）
        try:
            from poker_decision_ai.action_semantics import align_final_decision
            fd = align_final_decision(final_decision=fd, obs=obs or {})
        except Exception:                     # noqa: BLE001
            pass
        loop_safe = {
            "decision_status": "READY" if res.status == "OK" else "UNAVAILABLE",
            "action": fd.get("action"),
            "action_validated": bool(prov.get("action_validated", False)),
            "solver_mode": (fd.get("final_inputs_summary") or {}).get(
                "mode", "UNKNOWN"),
            "strategy_mode": self._strategy_mode(rep),
            "solver_provenance": (rep.get("provider_status") or {}).get(
                "solver") or {},
            "provenance": {"unknown_allowed_fields": prov.get(
                "unknown_allowed_fields", []),
                "unknown_allowed_rationale": prov.get(
                    "unknown_allowed_rationale", ""),
                "decision_provenance": fd.get("provenance") or {}},
            "decision_id": (rep.get("request_id") or ""),
        }
        advice_path = self._write_advice_json(
            fd, obs, prov if isinstance(prov, dict) else {}, "OK")
        self._log_attempt("OK", [], getattr(self, "_last_recapture", 0),
                          fd.get("action"), advice_path)
        return AnalysisResult(
            status="OK",
            advice_json=advice_path,
            rule_size=fd.get("rule_size_suggestion") or None,
            reason=((fd.get("state_incomplete") or {}).get("reason")
                    if fd.get("action") is None else None),
            decision_action=fd.get("action"),
            recommended_size=fd.get("recommended_size"),
            confidence=fd.get("confidence"),
            solver_mode=loop_safe["solver_mode"],
            strategy_mode=loop_safe["strategy_mode"],
            reasoning=fd.get("reasoning"),
            agents=self._agents_from(loop_safe, loop_safe, obs),
            vision=self._vision_summary(obs),
            frame_provenance=prov,
            raw={"final_decision": fd, "report_status": rep.get("status"),
                 "degraded": rep.get("degraded"),
                 "latency_ms": rep.get("latency_ms")})

    @staticmethod
    def _blinds_override_ev(override) -> dict:
        """GUI 手动输入的盲注（用户声明级别；source=user_setting）。"""
        if not isinstance(override, dict):
            return {}
        from poker_decision_ai.facing_evidence import parse_decimal_text
        bb = parse_decimal_text(override.get("bb"))
        if bb is None or not (bb > 0):
            return {}
        raw_sb = override.get("sb")
        if raw_sb is None or str(raw_sb).strip() == "":
            sb = None                      # SB 缺省 = 未知（下游按 0.0 处理）
        else:
            sb = parse_decimal_text(raw_sb)
            if sb is None or not (0 < sb < bb):
                return {}                  # 显式非法 SB ⇒ 整条拒绝（不静默丢弃）
        return {"sb": (sb if sb and sb > 0 else None), "bb": bb,
                "source": "user_setting",
                "detail": {"entered_by": "gui",
                           "sb_recognized": bool(sb and sb > 0)}}

    def _blinds_from_window_title(self) -> dict:
        """从牌桌窗口标题（OS 元数据）解析盲注；失败 ⇒ {}（不猜）。"""
        try:
            import ctypes
            from scripts.phase_v2s_auto_capture import locate_coinpoker
            from poker_decision_ai.facing_evidence import (
                parse_stakes_from_title)
            win = locate_coinpoker()
            if not win:
                return {}
            buf = ctypes.create_unicode_buffer(512)
            ctypes.windll.user32.GetWindowTextW(int(win["hwnd"]), buf, 512)
            title = buf.value or ""
            ev = parse_stakes_from_title(title)
            return ev if (ev.get("sb") and ev.get("bb")) else {}
        except Exception:                          # noqa: BLE001
            return {}

    @staticmethod
    def _blinds_memo_key(obs: dict, bbox) -> str:
        """会话记忆键：窗口几何 + 座位数（同桌同局）。"""
        seats = obs.get("seats") if isinstance(obs.get("seats"), list) else []
        return f"{bbox}|{len(seats)}"

    def _blinds_memo_store(self, ev: dict, obs: dict, bbox) -> None:
        try:
            import time as _t
            from poker_decision_ai.facing_evidence import memo_store
            if not hasattr(self, "_blinds_memo"):
                self._blinds_memo = {}
            memo_store(self._blinds_memo, self._blinds_memo_key(obs, bbox),
                       ev or {}, now=_t.time())
        except Exception:                          # noqa: BLE001
            pass

    def _blinds_memo_lookup(self, obs: dict, bbox) -> dict:
        try:
            import time as _t
            from poker_decision_ai.facing_evidence import memo_lookup
            return memo_lookup(getattr(self, "_blinds_memo", {}) or {},
                               self._blinds_memo_key(obs, bbox),
                               now=_t.time()) or {}
        except Exception:                          # noqa: BLE001
            return {}

    @staticmethod
    def _seat_bets_text(obs: dict) -> str:
        # P30-9-AG：1876 用**几何角色**标注（与 to_call 归属同源），避免bet环编号
        # 与 hero_seat 编号不同造成误读（用户截图曾显示"S4(SB)/S5(你 BB)"的错位标注）。
        try:
            from poker_decision_ai.facing_evidence import (
                BET_ZONE_ROLE_1876, implausible_preflop_posts, hero_opp_bets,
                looks_like_1876_bet_ring)
            _bets = (obs.get("extra") or {}).get("v3f_bets") or {}
            if _bets and looks_like_1876_bet_ring(_bets):
                _zh = {"top_left": "左上", "top_center": "上中", "top_right": "右上",
                        "bottom_left": "左下", "bottom_center_hero": "底中(你)",
                        "bottom_right": "右下"}
                _bad = set(implausible_preflop_posts(obs).get("seats") or [])
                hb = hero_opp_bets(obs)
                _parts = []
                for _k, _v in _bets.items():
                    _role = BET_ZONE_ROLE_1876.get(str(_k))
                    if _role is None:
                        continue
                    _name = _zh.get(_role, _role)
                    if _v is None:
                        _txt = "未识别/无投入"
                    elif str(_k) in _bad:
                        _txt = f"{_v}(⚠误读:翻前<小盲,已排除)"
                    else:
                        _txt = str(_v)
                    _parts.append(f"{_name}: {_txt}")
                _extra_txt = ""
                if hb["detail"].get("hero_post_floor_applied"):
                    _a = hb["detail"]["hero_post_floor_applied"]
                    _extra_txt = (f"；你的投入按盲注计({_a['read']}→{_a['floor']}，"
                                   f"位置 {_a['position']})")
                if _parts:
                    return ("座位下注(按桌面位置)： " + " · ".join(_parts)
                            + _extra_txt)
        except Exception:                          # noqa: BLE001
            pass

        """座位下注证据的可读注记（含"你/SB/BB/BTN"；失败 ⇒ 空串）。"""
        try:
            from poker_decision_ai.facing_evidence import annotate_seat_bets
            return str(annotate_seat_bets(obs).get("text") or "")
        except Exception:                          # noqa: BLE001
            return ""

    @staticmethod
    def _annotate_decision_context(fd: dict, rep: dict | None,
                                   obs: dict | None = None) -> dict:
        """把降级/偏离**显式**写进建议（只加说明与字段，不改动作）。"""
        if not isinstance(fd, dict):
            return fd
        r = rep if isinstance(rep, dict) else {}
        degraded = [str(x) for x in (r.get("degraded") or [])]
        skipped = [str(x) for x in (r.get("skipped") or [])]
        dev = bool(r.get("deviation_applied"))
        ctx = {"degraded": degraded, "skipped": skipped,
               "deviation_applied": dev}
        notes = []
        # 关键组件降级 ⇒ 结论基于回退路径（诚实标注，不隐藏）
        key = [x for x in degraded if x in ("solver", "equity", "population",
                                             "frequency_prior", "opponent",
                                             "fold_equity")]
        if key:
            notes.append("⚠ 本次以下组件未生效（降级/超时，结论基于回退路径）："
                          + "、".join(key)
                          + "。若机器繁忙或求解器超时，同一局面可能得到不同结论。")
        if dev:
            base = (r.get("base_decision") or {}).get("action")
            ctx["base_action"] = base
            notes.append("ℹ 裁决层随机化/偏离：EV 层建议 "
                          + str(base).upper()
                          + "，最终 " + str(fd.get("action")).upper()
                          + "（配置内固定种子，故同局面可复现）。")
        # P30-9-CI：公共牌不完整 ⇒ 明确说明（不静默给建议）
        try:
            _st = str((obs or {}).get("street") or "")
            _need = {"flop": 3, "turn": 4, "river": 5}.get(_st)
            _nb = len((obs or {}).get("board") or [])
            if _need is not None and _nb < _need:
                notes.append(f"⚠ 公共牌只识别到 {_nb}/{_need} 张（{_st}）"
                              "⇒ 牌面/权益相关结论不可信，建议重拍后再看。")
        except Exception:                          # noqa: BLE001
            pass
        # P30-9-BH：写明**实际使用**的胜率（范围化联合）——避免与裁决层证据串不一致
        try:
            from poker_decision_ai.joint_equity import (
                LAST_JOINT, USE_JOINT_AS_HERO_EQUITY)
            if LAST_JOINT.get("ok") and USE_JOINT_AS_HERO_EQUITY.get("on"):
                ctx["equity_used"] = {
                    "value": LAST_JOINT.get("equity"),
                    "basis": LAST_JOINT.get("basis"),
                    "n_villains": LAST_JOINT.get("n_villains"),
                    "source": "range_aware_joint"}
                notes.append(
                    "ℹ 本次采用**范围化联合胜率** "
                    f"{LAST_JOINT.get('equity')}（同时击败所有对手；"
                    f"依据 {LAST_JOINT.get('basis')}，{LAST_JOINT.get('n_villains')} 个对手）。"
                    "裁决层证据串里的 hero_equity 是引擎的逐对手平均值，两者不同。")
        except Exception:                          # noqa: BLE001
            pass
        if notes:
            fd["reasoning"] = (str(fd.get("reasoning") or "")
                                + "（" + " ".join(notes) + "）")
            fd["decision_context"] = ctx
        return fd

    def _fill_bets_from_finder(self, frame_rgba, w, h, obs: dict) -> dict:
        """用**自校准发现器**补齐 `extra.v3f_bets` 中未识别到的座位（只补不改）。

        * 只填空缺（None）位置的键；已有可读值**一律不覆盖**；
        * 值来源标注 `bets_filled_by_finder`，冲突记入 `bets_conflicts`；
        * 失败/无帧 ⇒ 原样返回（绝不发明值、绝不影响主流程）。
        """
        try:
            import numpy as np
            from poker_decision_ai.bet_capsule_finder import (
                find_bet_capsules, to_bet_ring_labels)
            from poker_decision_ai.vision2.pixel.realclient.hero_seat import (
                SEAT_ANCHORS_1876)
            if frame_rgba is None or w is None or h is None:
                return obs
            frame = np.frombuffer(frame_rgba, dtype=np.uint8).reshape(h, w, 4)
            stacks = {f"S{i}": None for i in range(1, 7)}
            for s in (obs.get("seats") or []):
                if isinstance(s, dict) and s.get("seat") is not None:
                    stacks[f"S{s['seat']}"] = s.get("stack")
            cc = {"pot": obs.get("pot"), "stacks": stacks}
            found = find_bet_capsules(frame, SEAT_ANCHORS_1876,
                                       cross_check=cc)
            seats_ring = to_bet_ring_labels(found.get("seats") or {})
            if not seats_ring:
                return obs
            extra = dict(obs.get("extra") or {})
            bets = dict(extra.get("v3f_bets") or {})
            filled, conflicts = {}, {}
            for k, v in seats_ring.items():
                cur = bets.get(k)
                if cur is None:
                    bets[k] = v
                    filled[k] = v
                elif v is not None and abs(float(cur) - float(v)) > 1e-9:
                    conflicts[k] = {"observer": cur, "finder": v}
            if filled or conflicts:
                extra["v3f_bets"] = bets
                if filled:
                    extra["bets_filled_by_finder"] = filled
                if conflicts:
                    extra["bets_conflicts"] = conflicts
            return {**obs, "extra": extra}
        except Exception:                          # noqa: BLE001
            return obs

    @staticmethod
    def _blinds_consistent(obs: dict) -> bool:
        """观察里的 blinds 是否与其他证据自洽（P30-9-O）。"""
        try:
            from poker_decision_ai.facing_evidence import blinds_consistent
            return bool(blinds_consistent(obs).get("ok"))
        except Exception:                          # noqa: BLE001
            return True

    @staticmethod
    def _derive_blinds(obs: dict) -> dict:
        """盲注证据推导（P30-9-F；无证据 ⇒ None ⇒ 调用方维持阻断）。"""
        try:
            from poker_decision_ai.facing_evidence import derive_blinds
            return derive_blinds(obs)
        except Exception:                          # noqa: BLE001
            return {}

    # ── P30-9-N：JSON 建议输出 + 尝试统计 ─────────────────────────────
    @staticmethod
    def _advice_dir() -> str:
        import os
        d = os.path.join(os.getcwd(), "advice")
        try:
            os.makedirs(d, exist_ok=True)
        except OSError:
            return os.getcwd()
        return d

    def _write_advice_json(self, fd: dict, obs: dict, prov: dict,
                           status: str) -> str | None:
        """把建议写成 **JSON 文件**（EXE 旁 advice/advice-<ts>.json）。"""
        try:
            import datetime
            import json
            import os
            ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            bb_chips = None
            b = obs.get("blinds") if isinstance(obs, dict) else None
            if isinstance(b, dict):
                try:
                    bb_chips = float(b.get("bb")) or None
                except (TypeError, ValueError):
                    bb_chips = None
            size_bb = fd.get("recommended_size")
            size_chips = (round(float(size_bb) * bb_chips, 4)
                          if (size_bb is not None and bb_chips) else None)
            payload = {
                "schema": "cash_game_advice/v1",
                "ts": datetime.datetime.now().isoformat(timespec="seconds"),
                "status": status,
                "action": fd.get("action"),
                "alternative_action": fd.get("alternative_action"),
                "size_bb": size_bb,
                "size_chips": size_chips,
                "size_rule": fd.get("rule_size_suggestion") or None,
                "advice_risk": fd.get("advice_risk") or None,
                "decision_context": fd.get("decision_context") or None,
                "hand_class": (fd.get("action_semantics") or {}).get("hand_class"),
                "state_incomplete": fd.get("state_incomplete") or None,
                "confidence": fd.get("confidence"),
                "strategy_mode": fd.get("strategy_mode"),
                "reasoning": fd.get("reasoning"),
                "vision": {
                    "hero_seat": obs.get("hero_seat"),
                    "hero_cards": obs.get("hole_cards"),
                    "board": obs.get("board"),
                    "street": obs.get("street"),
                    "pot": obs.get("pot"),
                    "to_call": obs.get("to_call"),
                    "blinds": obs.get("blinds"),
                    "position": (obs.get("extra") or {}).get("v3f_hero_position"),
                    "seat_bets": (obs.get("extra") or {}).get("v3f_bets"),
                },
                "provenance": {
                    "blinds_source": prov.get("blinds_source"),
                    "recapture_attempts": prov.get("recapture_attempts"),
                    "recapture_classes": prov.get("recapture_classes"),
                    "action_semantics": fd.get("action_semantics"),
                },
                "honesty": {
                    "strategy_mode_note": "BASELINE ≠ REAL/GTO；来源见 provenance",
                    "size_note": ("size_bb 仅在可证明合法时给出；否则用 size_rule"
                                   "（规则下限，非策略尺寸）"),
                },
            }
            path = os.path.join(self._advice_dir(),
                                f"advice-{ts}-{os.getpid()}.json")
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, ensure_ascii=False, indent=1)
            return path
        except Exception:                          # noqa: BLE001
            return None

    def _dump_failure_frame(self, status: str, blocking) -> str | None:
        """**失败时保存当帧截图**（logs/frames/<ts>-<status>.png）+ 诊断 sidecar。

        目的：失败不再依赖用户手动截图——下次出现即可离线复现定位。
        """
        try:
            import datetime
            import json
            import os
            if not getattr(self, "_last_frame", None):
                return None
            rgba, w, h = self._last_frame
            d = os.path.join(getattr(self, "_log_dir", None) or os.getcwd(),
                             "logs", "frames")
            os.makedirs(d, exist_ok=True)
            ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            base = f"{ts}-{status}"
            png = os.path.join(d, base + ".png")
            import numpy as np
            from PIL import Image
            arr = np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)
            Image.fromarray(arr, mode="RGBA").save(png)
            side = {"ts": ts, "status": status, "blocking": list(blocking or []),
                    "window": f"{int(w)}x{int(h)}",
                    "note": "失败帧存档（GUI 自动保存，供离线复现）"}
            with open(os.path.join(d, base + ".json"), "w",
                      encoding="utf-8") as fh:
                json.dump(side, fh, ensure_ascii=False, indent=1)
            self._last_frame_path = png
            return png
        except Exception:                          # noqa: BLE001
            return None

    def _log_attempt(self, status: str, blocking, recapture: int, action,
                     advice_path: str | None) -> None:
        """每次尝试追加一行 JSONL（成功率可量化）。"""
        try:
            import datetime
            import json
            import os
            d = os.path.join(getattr(self, "_log_dir", None) or os.getcwd(),
                             "logs")
            os.makedirs(d, exist_ok=True)
            frame_path = None
            if status != "OK":
                frame_path = self._dump_failure_frame(status, blocking)
            rec = {"ts": datetime.datetime.now().isoformat(timespec="seconds"),
                   "status": status, "blocking": list(blocking or []),
                   "recapture": recapture, "action": action,
                   "advice_json": advice_path, "frame": frame_path}
            with open(os.path.join(d, "attempts.jsonl"), "a",
                      encoding="utf-8") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except Exception:                          # noqa: BLE001
            pass

    def _analyze_with_derived_blinds(self, runtime, got: dict, ev: dict,
                                     w, h):
        """用证据补盲注 → RealtimeRunner.decide（同一决策核心）→ 结果。

        诚实标注：frame_provenance/vision 里写明 blinds 来源与证据明细；
        任何异常 ⇒ None（上层回退为原阻断行为，绝不静默）。
        """
        import asyncio
        import copy
        br = got.get("bridge") or {}
        obs_bridge = br.get("observation") or {}
        try:
            from poker_decision_ai.action_semantics import align_final_decision
            from poker_decision_ai.realtime import RealtimeRunner
            # P30-9-U（关键）：桥被阻断时 got["request"] 为 None。**必须**用桥的
            # 规范化观察（含 hero_cards/pot/street/seats/extra.v3f_bets）构造请求，
            # 否则决策在空观察上跑 ⇒ 恒 RAISE、置信恒定。
            req_in = got.get("request") or {}
            base_obs = dict(obs_bridge)
            if not base_obs:                      # 退化情形：桥没有观察 ⇒ 不猜
                base_obs = dict((req_in.get("vision_observation") or {}))
            req = {"kind": req_in.get("kind") or "cash_hand",
                   "hero_id": req_in.get("hero_id") or "Hero",
                   "vision_observation": base_obs}
            vo = req["vision_observation"]
            vo["blinds"] = {"sb": float(ev["sb"]), "bb": float(ev["bb"])}
            ex = vo.setdefault("extra", {})
            if not isinstance(ex, dict):
                ex = {}
                vo["extra"] = ex
            ex["v3f_blinds_source"] = ev.get("source")
            ex["v3f_blinds_evidence"] = ev.get("detail")
            # P30-9-T：to_call 未知 ⇒ 用"对手投入下界"替代（否则 pot odds 按 0 算，
            # 所需胜率=0 ⇒ 任何牌都显得值得跟/加 ⇒ 决策必然偏激进）
            tc_note = None
            if vo.get("to_call") is None:
                try:
                    from poker_decision_ai.facing_evidence import (
                        to_call_lower_bound)
                    lb = to_call_lower_bound(base_obs)
                    if lb.get("to_call"):
                        vo["to_call"] = float(lb["to_call"])
                        ex["v3f_to_call_source"] = lb.get("source")
                        ex["v3f_to_call_is_lower_bound"] = True
                        ex["v3f_to_call_detail"] = lb.get("detail")
                        tc_note = ("跟注额为**下界估计**（由对手投入推得，"
                                    f"非动作面板识别；{lb.get('detail', {}).get('max_opponent_bet')}"
                                    f" − {lb.get('detail', {}).get('hero_bet')}）")
                except Exception:                      # noqa: BLE001
                    tc_note = None
            rr = RealtimeRunner(runtime)
            report = asyncio.run(rr.decide(req))
        except Exception:                          # noqa: BLE001
            return None
        rep = report.to_dict() if hasattr(report, "to_dict") else dict(report)
        fd = dict(rep.get("final_decision") or {})
        if not fd:
            return None
        if tc_note and isinstance(fd.get("reasoning"), str):
            fd["reasoning"] = f"{fd['reasoning']}（{tc_note}）"
        fd = align_final_decision(final_decision=fd,
                                  obs={**obs_bridge, **vo})
        fd = self._annotate_decision_context(fd, rep, obs={**obs_bridge, **vo})
        obs = {**obs_bridge, "blinds": vo["blinds"]}
        # P30-9-Z：派生（盲注补齐）路径同样写建议 JSON（此前只写普通路径 ⇒ 目录为空）
        advice_path = self._write_advice_json(
            fd, obs,
            {"blinds_source": ev.get("source"),
             "to_call_source": (vo.get("extra") or {}).get("v3f_to_call_source"),
             "to_call_is_lower_bound": bool(
                 (vo.get("extra") or {}).get("v3f_to_call_is_lower_bound"))},
            "OK")
        self._log_attempt("OK", [], getattr(self, "_last_recapture", 0),
                          fd.get("action"), advice_path)
        return AnalysisResult(
            status="OK",
            advice_json=advice_path,
            decision_action=fd.get("action"),
            recommended_size=fd.get("recommended_size"),
            confidence=fd.get("confidence"),
            solver_mode=((fd.get("final_inputs_summary") or {}).get(
                "mode", "UNKNOWN")),
            strategy_mode=self._strategy_mode(rep),
            reasoning=fd.get("reasoning"),
            vision=self._vision_summary(obs),
            frame_provenance={
                "geometry": f"{w}x{h}",
                "status": "OK",
                "blinds_source": ev.get("source"),
                "multiway_mode": bool(getattr(self, "_multiway_info", None)),
                "joint_equity_range_aware": (lambda: (
                    (lambda j: (j.get("equity") if j else None)))(
                        __import__("poker_decision_ai.joint_equity",
                                   fromlist=["LAST_JOINT"]).LAST_JOINT))(),
                "joint_equity_basis": (lambda: (
                    (lambda j: (j.get("basis") if j else None)))(
                        __import__("poker_decision_ai.joint_equity",
                                   fromlist=["LAST_JOINT"]).LAST_JOINT))(),
                "line_events": ((getattr(self, "_last_line", None) or {})
                                 .get("action_history")),
                "line_current_player": ((getattr(self, "_last_line", None) or {})
                                         .get("current_player_role")),
                "line_to_call_derived": ((getattr(self, "_last_line", None) or {})
                                          .get("to_call_derived")),
                "multiway_estimate": (getattr(self, "_multiway_info", None) or {}
                                       ).get("equity"),
                "blinds_evidence": ev.get("detail"),
                "blinds_derived_value": {"sb": vo["blinds"]["sb"],
                                          "bb": vo["blinds"]["bb"]},
                "note": "header 金额不可读 → 用盲注座投入证据补齐（非 header 识别）",
                "to_call_source": (vo.get("extra") or {}).get("v3f_to_call_source"),
                "to_call_is_lower_bound": bool(
                    (vo.get("extra") or {}).get("v3f_to_call_is_lower_bound")),
                "to_call_note": tc_note,
            },
            raw={"final_decision": fd, "report_status": rep.get("status"),
                 "blinds_source": ev.get("source")})

    @staticmethod
    def _strategy_mode(rep: dict) -> str:
        """provider_status 诚实标签（REAL=外部可验证 solver / RULES=
        realengine / 其余 BASELINE——与 P38-F2 同判据，UI 侧只读）。"""
        s = ((rep.get("final_decision") or {}).get(
            "final_inputs_summary") or {}).get("solver")
        src = ((rep.get("provider_status") or {}).get("solver") or {}).get(
            "provider")
        if s == "ok":
            if src == "postflop":
                return "REAL"
            if src == "realengine":
                return "RULES"
            return "BASELINE"
        return "BASELINE"

    # ── helpers ──────────────────────────────────────────────────────
    @staticmethod
    def _advice_agents(adv: dict, obs: dict) -> list:
        """R14-EXE-3：ADVICE_REFERENCE 模式的 Agent 输出（诚实标注——
        Judge=参考通道（非合法性判定），Decision=参考打法（非决策））。"""
        return [
            AgentOutput(
                "Vision", "OK" if (obs or {}).get("hero_seat") else "UNKNOWN",
                output={"hero_seat": (obs or {}).get("hero_seat"),
                        "hero_cards": (obs or {}).get("hole_cards"),
                        "pot": (obs or {}).get("pot"),
                        "to_call": (obs or {}).get("to_call"),
                        "street": (obs or {}).get("street")},
                provenance="visual_observer"),
            AgentOutput(
                "Advice", "REFERENCE",
                output={"action": adv.get("action"),
                        "size_bb": adv.get("size_bb"),
                        "source": adv.get("source"),
                        "scenario": adv.get("scenario"),
                        "hand_class": adv.get("hand_class"),
                        "legal_actions_status":
                            adv.get("legal_actions_status")},
                provenance=str(adv.get("version") or "advice_reference")),
        ]

    def _vision_summary(self, obs: dict) -> dict:
        extra = obs.get("extra") or {}
        # P30-9-E：hero 牌失败原因（用户可见——"有时识别不到 hero 的牌"）
        try:
            from poker_decision_ai.facing_evidence import hero_cards_diagnosis
            hero_diag = hero_cards_diagnosis(obs)
        except Exception:                          # noqa: BLE001
            hero_diag = {}
        return {
            "hero_seat": obs.get("hero_seat"),
            "hero_cards": obs.get("hole_cards"),
            "dealer_seat": obs.get("button_seat"),
            "pot": obs.get("pot"),
            "to_call": obs.get("to_call"),
            "board": obs.get("board"),
            "street": obs.get("street"),
            "blinds": obs.get("blinds"),
            "ante": obs.get("ante"),
            "position": extra.get("v3f_hero_position"),
            "blinds_consistent": self._blinds_consistent(obs),
            "blinds_resolved_source": (
                (getattr(self, "_blinds_resolved", None) or {}).get("source")),
            "hero_cards_ok": hero_diag.get("ok"),
            "hero_cards_reason": hero_diag.get("reason"),
            "hero_cards_observer_reason": hero_diag.get("observer_reason"),
            "facing_from_seat_bets": extra.get("v3f_bets"),
            "seat_bets_text": self._seat_bets_text(obs),
            "status_unknown_fields": [k for k, c in
                                      (obs.get("field_confidences") or {}).items()
                                      if c < 1.0],
        }

    def _vision_ok(self, loop, safe: dict) -> dict:
        obs = (safe.get("input_provenance") or {})
        return {
            "hero_cards": None, "pot": None, "to_call": None,
            "solver_provenance": safe.get("solver_provenance"),
            "input_provenance": obs,
        }

    def _agent(self, name: str, obs: dict) -> AgentOutput:
        return AgentOutput(name, "UNKNOWN" if not obs.get("pot") else "OK",
                           provenance="visual_observer")

    def _agents_from(self, loop, safe: dict, obs: dict | None = None) -> list:
        """Agent 输出（结构化——status/output/provenance）。

        Vision/Position/State/Solver/Judge 均由同一 DecisionInput 链产生
        （无第二套视觉 pipeline——Vision 输出喂后续 Agent）。
        """
        ag = []
        # Vision（视觉入口）
        ag.append(AgentOutput(
            "Vision", "OK" if (obs or {}).get("hero_seat") else "UNKNOWN",
            output={
                "hero_seat": (obs or {}).get("hero_seat"),
                "hero_cards": (obs or {}).get("hole_cards"),
                "pot": (obs or {}).get("pot"),
                "to_call": (obs or {}).get("to_call"),
                "dealer_seat": (obs or {}).get("button_seat"),
                "street": (obs or {}).get("street"),
            },
            provenance="visual_observer"))
        # Position（由 vision 座位识别 + positions_for——无重复截图）
        pos = (obs or {}).get("extra", {}).get("v3f_hero_position")
        ag.append(AgentOutput(
            "Position", "OK" if pos else "UNKNOWN",
            output={"hero_position": pos,
                    "hero_seat": (obs or {}).get("hero_seat")},
            provenance="positions_for"))
        # State（bridge → state）
        ag.append(AgentOutput(
            "State", "READY" if safe.get("decision_status") == "READY"
            else "UNAVAILABLE",
            output={"blocking": safe.get("provenance", {}).get(
                "unknown_allowed_fields", [])},
            provenance="production_bridge"))
        # Solver / Strategy
        ag.append(AgentOutput(
            "Solver", safe.get("solver_mode", "UNKNOWN"),
            output={"solver_mode": safe.get("solver_mode"),
                    "solver_status": (safe.get("solver_provenance") or {}).get(
                        "solver_status")},
            provenance="solver_contract"))
        ag.append(AgentOutput(
            "Strategy", safe.get("strategy_mode", "UNKNOWN"),
            output={"strategy_mode": safe.get("strategy_mode"),
                    "approach": ("REAL external solver (postflop; verifiable)"
                                 if safe.get("strategy_mode") == "REAL"
                                 else ("RULES rule-engine (deterministic; "
                                       "not GTO)"
                                       if safe.get("strategy_mode") == "RULES"
                                       else "BASELINE (not GTO)"))},
            provenance="strategy_provider"))
        # Judge / Decision
        ag.append(AgentOutput(
            "Judge", "LEGAL" if safe.get("action_validated") else
            "REJECTED",
            output={"action": safe.get("action"),
                    "action_validated": safe.get("action_validated")},
            provenance="judge_safety_gate"))
        ag.append(AgentOutput(
            "Decision", "OK" if safe.get("action") else "UNAVAILABLE",
            output={"action": safe.get("action"),
                    "decision_id": safe.get("decision_id")},
            provenance="production_decision_loop"))
        return ag
