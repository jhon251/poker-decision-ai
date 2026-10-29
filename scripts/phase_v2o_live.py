"""Phase V2O — Live Vision Runtime（真实屏幕捕获 → 实时视觉状态）。

- 捕获：ctypes BitBlt 只读后端（无任何输入注入/窗口控制）；
- 布局：当前屏幕 1966×1106 ≠ profile 客户端 1848×1388 → LAYOUT_UNKNOWN
  → recalibration（不得使用旧坐标假装正常）；
- 运行：2 FPS 有界运行（--seconds 默认 60；5 分钟用 --seconds 300）；
- 指标：capture/ROI/digit/card/temporal/total 分阶段 P50/P95/P99 +
  RuntimeMetrics + 内存（psapi 只读进程内存）；
- 状态完整性：V2M 时序引擎（unchanged ROI 复用、board 单调、冲突阻断）。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import tracemalloc
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from poker_decision_ai.vision2.pixel.realclient import (  # noqa: E402
    RealClientCardRecognitionBackend, RealClientDigitOCRBackend,
    RealClientTemplateLibrary)
from poker_decision_ai.vision2.pixel.realclient.capture import (  # noqa: E402
    CAPTURE_OK, CtypesBitBltBackend, LAYOUT_UNKNOWN)
from poker_decision_ai.vision2.pixel.realclient.layout import (  # noqa: E402
    CLIENT_SIZE, RealClientLayoutProfile)
from poker_decision_ai.vision2.pixel.realclient.runtime import (  # noqa: E402
    FrameStream, TemporalVisionState, VisualEventBus, roi_digest)
from poker_decision_ai.vision2.realtime.coords import Rect  # noqa: E402
from poker_decision_ai.vision2.realtime.image import ImageData  # noqa: E402


def process_memory_mb() -> float:
    import ctypes
    from ctypes import wintypes

    class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t)]
    try:
        k32 = ctypes.windll.kernel32
        h = k32.GetCurrentProcess()
        pmc = PROCESS_MEMORY_COUNTERS()
        pmc.cb = ctypes.sizeof(pmc)
        k32.GetProcessMemoryInfo(h, ctypes.byref(pmc),
                                 ctypes.sizeof(pmc))
        return round(pmc.WorkingSetSize / (1024 * 1024), 1)
    except Exception:  # noqa: BLE001
        return -1.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=60.0)
    args = ap.parse_args()
    backend = CtypesBitBltBackend()
    manifest = Path("data/vision2/realclient_assets_v6/manifest.json")
    mdata = json.loads(manifest.read_text(encoding="utf-8"))
    refs = {a["template_id"]: a.get("png_ref") for a in mdata["assets"]}

    def loader(tid):
        r = refs.get(tid)
        if not r or not Path(r).exists():
            return None
        import poker_decision_ai.vision2_platform.png_decode as pd
        w_, h_, rgba_ = pd.decode_png(Path(r).read_bytes())
        return ImageData.from_rgba_bytes(w_, h_, rgba_)

    lib = RealClientTemplateLibrary.from_json(str(manifest), loader)
    digit = RealClientDigitOCRBackend(lib)
    card = RealClientCardRecognitionBackend(lib)
    prof = RealClientLayoutProfile.measured_v2()
    stack_rois = [prof.entries[f"player_{i}_stack_roi"].bbox
                  for i in range(6)]
    stream = FrameStream(fps=2.0)
    tvs = TemporalVisionState()
    bus = VisualEventBus()
    stage = {k: [] for k in
             ("capture", "roi", "digit", "card", "temporal", "total")}
    counters = {"frames_captured": 0, "frames_processed": 0,
                "frames_dropped": 0, "frames_deduplicated": 0,
                "capture_failures": 0, "layout_recalibrations": 0,
                "roi_changed": 0, "recognition": 0, "conflicts": 0,
                "stale": 0}
    tracemalloc.start()
    mem_start = process_memory_mb()
    layout_status = None
    t_end = time.time() + args.seconds
    frame_id = 0
    prev_board = None
    last_digests = {}
    while time.time() < t_end:
        t0 = time.perf_counter()
        res = backend.capture()
        counters["frames_captured"] += 1
        stage["capture"].append((time.perf_counter() - t0) * 1000)
        if res["status"] != CAPTURE_OK:
            counters["capture_failures"] += 1
            bus.emit("CAPTURE_FAILURE", frame_id + 1, res)
            continue
        w, h = res["width"], res["height"]
        if (w, h) != CLIENT_SIZE:
            if layout_status is None:
                layout_status = LAYOUT_UNKNOWN
                counters["layout_recalibrations"] += 1
                bus.emit("ROI_CHANGED", frame_id + 1,
                         {"reason": "RESOLUTION_CHANGED",
                          "expected": CLIENT_SIZE, "actual": (w, h)})
        t1 = time.perf_counter()
        img = ImageData.from_rgba_bytes(w, h, res["image_bytes"])
        acc = stream.accept(frame_id + 1, res["captured_at"], img)
        stage["roi"].append((time.perf_counter() - t1) * 1000)
        if not acc["accepted"]:
            if acc["reason"] == "duplicate_frame":
                counters["frames_deduplicated"] += 1
            else:
                counters["frames_dropped"] += 1
            continue
        frame_id += 1
        counters["frames_processed"] += 1
        t2 = time.perf_counter()
        # 仅布局未漂移时按 profile 识别（漂移 → LAYOUT_UNKNOWN，不动旧坐标）
        if layout_status is None:
            for i, rect in enumerate(stack_rois):
                if rect is None:
                    continue
                px, py = rect.width * 0.3, rect.height * 0.3
                exp = Rect(max(0, rect.x - px), max(0, rect.y - py),
                           min(w, rect.x + rect.width + px) -
                           max(0, rect.x - px),
                           min(h, rect.y + rect.height + py) -
                           max(0, rect.y - py))
                if exp.x + exp.width > w or exp.y + exp.height > h:
                    continue
                sub = img.crop(exp)
                d = roi_digest(sub)
                if d == last_digests.get(i):
                    continue
                last_digests[i] = d
                counters["roi_changed"] += 1
                counters["recognition"] += 1
                resr = digit.recognize(sub, "stacks", "number")
                if resr.status.value == "OK":
                    tvs.observe(f"player_{i}.stack", resr.value,
                                resr.confidence, frame_id, d)
                else:
                    tvs.uncertain(f"player_{i}.stack", frame_id)
        stage["digit"].append((time.perf_counter() - t2) * 1000)
        tvs.stale_check(frame_id)
        stage["temporal"].append((time.perf_counter() - t2) * 1000)
        stage["total"].append((time.perf_counter() - t0) * 1000)
        time.sleep(max(0.0, 0.5 - (time.perf_counter() - t0)))

    mem_end = process_memory_mb()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    def pct(xs):
        xs = sorted(xs)
        n = len(xs)
        if not xs:
            return None
        return {"p50": round(xs[n // 2], 1),
                "p95": round(xs[int(n * 0.95)] if n > 1 else xs[-1], 1),
                "p99": round(xs[int(n * 0.99)] if n > 1 else xs[-1], 1)}

    report = {"runtime_s": round(args.seconds, 1), "backend": backend.name,
              "layout_status": layout_status,
              "counters": counters, "bench_ms": {k: pct(v) for k, v in
                                                 stage.items()},
              "memory_mb": {"start": mem_start, "end": mem_end,
                            "delta": round(mem_end - mem_start, 1),
                            "peak_traced_mb": round(peak / 1048576, 1)},
              "events": [e.to_dict() for e in bus.events[-20:]],
              "fields": tvs.to_dict()}
    out = Path("data/vision2/live_runtime_report.json")
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False),
                   encoding="utf-8")
    print(json.dumps(report, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
