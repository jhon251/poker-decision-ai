"""V3F-P36 — 连续生产闭环组装层（只组装冻结组件，不重写）。

数据流（全部复用既有冻结组件）：
```
ScreenCaptureBackend（只读截图）/ replay 图源
  → FrameAcquirer（FrameStream：FPS 节流 + 精确 digest 去重 + 顺序/stale）
  → VisualEventBus → EventStreamConsumer（仅视觉事实——绝不归因行为）
  → ProductionDecisionLoop.tick（readiness gate → RealtimeRunner.decide → Judge gate）
  → DecisionSafetyResult（decision_id 幂等 + provenance）
  → sink 消费端（回调 / JSONL——供 person/服务消费）
```

安全边界（P36 授权）：
- 零输入注入：不点击/不键盘/不窗口控制（backend 仅只读截图 API）；
- UNKNOWN-first / readiness gate / Judge safety gate 全部由冻结组件保证，
  本层不加宽、不绕过、不为测试放宽；
- REAL solver 仅经 PostflopSolverCLIProvider（solver.path / POSTFLOP_SOLVER_PATH）；
  未配置 → 诚实降级 BASELINE（P34 语义），绝不伪造 REAL；
- action_history / 对家建模保持 BASELINE（不借 P36 偷渡建模）；
- 无自动下注：DecisionSafetyResult 只写入 sink，绝不执行任何牌桌动作；
- 安全停止：threading.Event，任何循环迭代边界检查；同帧幂等去重不变。
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from typing import Any, Callable, Iterable

import numpy as np

from .event_stream_consumer import EventStreamConsumer
from .production_decision_loop import ProductionDecisionLoop
from .vision2.pixel.realclient.acquisition import FrameAcquirer
from .vision2.pixel.realclient.capture import (CAPTURE_BACKEND_UNAVAILABLE,
                                               CAPTURE_OK, detect_backend)


class ContinuousProductionRunner:
    """P36 常驻运行单元：acquisition → event consumer → decision loop → sink。

    单帧入口（process_image / process_capture）与常驻循环（run_replay /
    run_capture）共用同一 acquirer + loop + sink——语义完全一致。
    """

    def __init__(self, config: Any = None, *, fps: float = 2.0,
                 backend: Any = None, throttle_s: float = 1.0,
                 sink: Callable[[dict], None] | None = None):
        from .core.orchestration.runtime import build_runtime
        self.runtime = build_runtime(config)
        self.acquirer = FrameAcquirer(fps=fps, backend=backend)
        self.event_consumer = EventStreamConsumer()
        self.loop = ProductionDecisionLoop(
            self.runtime, throttle_s=throttle_s,
            event_consumer=self.event_consumer)
        self.sink = sink
        self._stop_event = threading.Event()
        self._bus_drain_idx = 0
        self.counters = {"frames_in": 0, "frames_processed": 0,
                         "frames_skipped": 0, "capture_failures": 0,
                         "decisions": 0}

    # ── 安全停止 ────────────────────────────────────────────────────
    def stop(self) -> None:
        """安全停止：置位后任何循环在下一迭代边界退出（不中断半帧）。"""
        self._stop_event.set()

    @property
    def stopped(self) -> bool:
        return self._stop_event.is_set()

    # ── 视觉事实总线 → EventStreamConsumer（仅白名单事实） ───────────
    def _drain_bus(self) -> None:
        for ev in self.acquirer.bus.events[self._bus_drain_idx:]:
            self.event_consumer.ingest_visual_fact(
                ev.kind, ev.frame_id, dict(ev.detail or {}))
        self._bus_drain_idx = len(self.acquirer.bus.events)

    # ── 单帧处理（replay 语义 = 真实路径共用 FrameStream） ───────────
    def process_image(self, image, *, w: int, h: int,
                      captured_at: float | None = None) -> dict:
        """一帧 rgba → acquirer 去重 → loop.tick → DecisionSafetyResult。

        image 接受 HxWx4 ndarray 或等长 rgba bytes（decode_png 输出）；
        bytes → ndarray 归一化（冻结 image_digest 只采样 buffer 对象，
        裸 bytes 会得到空摘要——归一化是组装层职责，不改冻结组件）。
        """
        if self.stopped:
            return {"status": "STOPPED", "reason": "safe_stop"}
        if isinstance(image, (bytes, bytearray, memoryview)):
            image = np.frombuffer(bytes(image), dtype=np.uint8).reshape(
                h, w, 4)
        self.counters["frames_in"] += 1
        self._drain_bus()
        acc = self.acquirer.acquire_replay(
            image, w=w, h=h, captured_at=captured_at)
        if acc["status"] != CAPTURE_OK:
            self.counters["frames_skipped"] += 1
            return {"status": acc["status"], "reason": acc.get("reason")}
        f = acc["frame"]
        res = self.loop.tick(f["frame_id"], image, f["width"], f["height"])
        self.counters["frames_processed"] += 1
        self.counters["decisions"] += 1
        if self.sink is not None:
            self.sink(res)
        return res

    # ── 真实截图路径（只读） ────────────────────────────────────────
    def process_capture(self) -> dict:
        """backend.capture()（只读）→ rgba → process_image；失败=明确状态。"""
        if self.stopped:
            return {"status": "STOPPED", "reason": "safe_stop"}
        backend = self.acquirer.backend
        if backend is None or not backend.available():
            self.counters["capture_failures"] += 1
            return {"status": CAPTURE_BACKEND_UNAVAILABLE,
                    "reason": (backend.unavailable_reason()
                               if backend is not None
                               else "no backend configured")}
        res = backend.capture()
        if res["status"] != CAPTURE_OK:
            self.counters["capture_failures"] += 1
            return {"status": res["status"], "reason": res.get("reason")}
        rgba = np.frombuffer(res["image_bytes"], dtype=np.uint8).reshape(
            res["height"], res["width"], 4)
        return self.process_image(rgba, w=res["width"], h=res["height"],
                                  captured_at=res.get("captured_at"))

    # ── 常驻循环 ────────────────────────────────────────────────────
    def run_replay(self, images: Iterable[Any], *, w: int, h: int,
                   captured_at: float | None = None) -> dict:
        """回放图源常驻循环（测试/审计/复现——与真实路径同语义）。"""
        for img in images:
            if self.stopped:
                break
            self.process_image(img, w=w, h=h, captured_at=captured_at)
        return self.summary()

    def run_capture(self, *, max_frames: int | None = None,
                    poll_s: float | None = None) -> dict:
        """真实截图常驻循环（默认 fps 节流；stop()/Ctrl+C → 安全退出）。"""
        poll = poll_s if poll_s is not None else max(
            0.0, self.acquirer.stream.interval)
        n = 0
        while not self.stopped:
            if max_frames is not None and n >= max_frames:
                break
            self.process_capture()
            n += 1
            if poll:
                # 分片睡眠——stop() 能在 ≤0.05s 内生效
                waited = 0.0
                while waited < poll and not self.stopped:
                    time.sleep(min(0.05, poll - waited))
                    waited += 0.05
        return self.summary()

    # ── 汇总（全部来自真实运行计数） ────────────────────────────────
    def summary(self) -> dict:
        return {"counters": dict(self.counters),
                "acquisition": self.acquirer.metrics.to_dict(),
                "events": self.event_consumer.summary()["event_count"],
                "source": self.event_consumer.current_source().to_dict(),
                "decisions": len(self.loop.decisions),
                "backend": (self.acquirer.backend.name
                            if self.acquirer.backend is not None else None)}


# ── CLI 入口（python -m poker_decision_ai.continuous_runner） ────────
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="P36 连续生产闭环（只读视觉 → 决策建议输出；无自动下注）")
    ap.add_argument("--source", choices=("replay", "screen"),
                    required=True)
    ap.add_argument("--frames", default="",
                    help="replay 源：目录或 glob（PNG evidence）")
    ap.add_argument("--fps", type=float, default=2.0)
    ap.add_argument("--throttle", type=float, default=1.0,
                    help="同街重复帧去重窗口（s）")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--out", default="",
                    help="DecisionSafetyResult JSONL 输出路径（可选）")
    args = ap.parse_args(argv)

    from .core.config import load_config
    from .vision2_platform.png_decode import decode_png

    fh = open(args.out, "a", encoding="utf-8") if args.out else None

    def sink(res: dict) -> None:
        if fh is not None:
            fh.write(json.dumps(res.to_dict()
                                if hasattr(res, "to_dict") else res,
                                ensure_ascii=False) + "\n")
            fh.flush()

    runner = ContinuousProductionRunner(load_config(), fps=args.fps,
                                        throttle_s=args.throttle, sink=sink)
    try:
        if args.source == "replay":
            if not args.frames:
                ap.error("--source replay 需要 --frames")
            import glob as _glob
            paths = sorted(_glob.glob(args.frames)) if any(
                c in args.frames for c in "*?[") else sorted(
                p.name for p in __import__("pathlib").Path(
                    args.frames).glob("*.png"))
            for p in paths:
                if runner.stopped:
                    break
                w, h, rgba = decode_png(open(p, "rb").read())
                runner.process_image(rgba, w=w, h=h)
        else:
            st = detect_backend()
            print(json.dumps({"capture_backend": st}, ensure_ascii=False))
            runner.run_capture(max_frames=args.max_frames)
    except KeyboardInterrupt:
        runner.stop()                                # Ctrl+C → 安全停止
    finally:
        if fh is not None:
            fh.close()
    print(json.dumps(runner.summary(), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
