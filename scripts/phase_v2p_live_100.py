"""Phase V2P — CoinPoker 100+ 帧 Live Runtime（只读捕获 + 确定性证据）。

- 捕获：全屏 BitBlt（GetWindowDC 对 GPU/DWM 窗口失效 → 全屏+裁剪为正确路径）；
- 证据：毡面掩码边界（table ROI 的确定性几何证据）；
- 布局：历史 profile 尺寸/宽高比不匹配 → 除 table（实测）外全部 ROI=null
  + UNCERTAIN（无视觉 → 不测量座位几何，禁止猜测）；
- 状态：FrameStream 去重/顺序/未来拒绝 + 指标；识别链在布局未知时不启动。
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from poker_decision_ai.vision2.pixel.realclient.capture import (  # noqa: E402
    CAPTURE_OK, CtypesBitBltBackend)
from poker_decision_ai.vision2.pixel.realclient.runtime import (  # noqa: E402
    FrameStream, VisualEventBus)

COIN_RECT = (343, 144, 343 + 1280, 144 + 779)


def felt_bounds(arr) -> dict:
    """毡面（深绿）掩码边界 — 确定性几何证据。"""
    r = arr[..., 0].astype(int)
    g = arr[..., 1].astype(int)
    b = arr[..., 2].astype(int)
    felt = (g > 40) & (g < 130) & (b > 20) & (b < 110) & (g >= r + 25)
    ys, xs = np.where(felt)
    if len(xs) < 500:
        return {"visible": False}
    return {"visible": True,
            "bbox": {"x": int(xs.min()), "y": int(ys.min()),
                     "width": int(xs.max() - xs.min() + 1),
                     "height": int(ys.max() - ys.min() + 1)},
            "felt_px": int(felt.sum())}


def main() -> int:
    sys.argv = sys.argv[:1]
    backend = CtypesBitBltBackend()
    stream = FrameStream(fps=2.0)
    bus = VisualEventBus()
    stage = {k: [] for k in ("capture", "crop", "measure", "total")}
    counters = {"frames_captured": 0, "frames_processed": 0,
                "frames_dropped": 0, "frames_deduplicated": 0,
                "capture_failures": 0}
    table_samples = []
    t_end = time.time() + 65.0          # ≈120 帧 @2FPS
    frame_id = 0
    while time.time() < t_end:
        t0 = time.perf_counter()
        res = backend.capture()
        counters["frames_captured"] += 1
        stage["capture"].append((time.perf_counter() - t0) * 1000)
        if res["status"] != CAPTURE_OK:
            counters["capture_failures"] += 1
            bus.emit("CAPTURE_FAILURE", frame_id + 1, res)
            continue
        t1 = time.perf_counter()
        w, h = res["width"], res["height"]
        x0, y0, x1, y1 = COIN_RECT
        if x1 > w or y1 > h:
            x1, y1 = w, h
        import poker_decision_ai.vision2_platform.png_decode as _  # noqa: F401
        from poker_decision_ai.vision2.realtime.image import ImageData
        arr = np.frombuffer(res["image_bytes"], dtype=np.uint8).reshape(h, w, 4)
        sub = arr[y0:y1, x0:x1]
        img = ImageData.from_rgba_bytes(x1 - x0, y1 - y0, sub.tobytes())
        stage["crop"].append((time.perf_counter() - t1) * 1000)
        acc = stream.accept(frame_id + 1, res["captured_at"], img)
        if not acc["accepted"]:
            if acc["reason"] == "duplicate_frame":
                counters["frames_deduplicated"] += 1
            else:
                counters["frames_dropped"] += 1
            stage["total"].append((time.perf_counter() - t0) * 1000)
            continue
        frame_id += 1
        counters["frames_processed"] += 1
        t2 = time.perf_counter()
        tb = felt_bounds(sub)
        stage["measure"].append((time.perf_counter() - t2) * 1000)
        stage["total"].append((time.perf_counter() - t0) * 1000)
        if tb["visible"]:
            table_samples.append({"frame_id": frame_id, **tb["bbox"]})
        time.sleep(max(0.0, 0.5 - (time.perf_counter() - t0)))

    def pct(xs):
        xs = sorted(xs)
        n = len(xs)
        if not xs:
            return None
        return {"p50": round(xs[n // 2], 1),
                "p95": round(xs[int(n * 0.95)] if n > 1 else xs[-1], 1),
                "p99": round(xs[int(n * 0.99)] if n > 1 else xs[-1], 1)}

    import statistics
    if table_samples:
        tx = [s["x"] for s in table_samples]
        ty = [s["y"] for s in table_samples]
        tw = [s["width"] for s in table_samples]
        th = [s["height"] for s in table_samples]
        table = {"x": int(statistics.median(tx)),
                 "y": int(statistics.median(ty)),
                 "width": int(statistics.median(tw)),
                 "height": int(statistics.median(th)),
                 "frames": len(table_samples)}
    else:
        table = None
    report = {"capture": {"backend": "full_screen_bitblt+crop",
                          "coin_rect": COIN_RECT},
              "counters": counters, "bench_ms": {k: pct(v) for k, v in
                                                 stage.items()},
              "table_felt_measured": table,
              "layout_status": "CALIBRATION_REQUIRED",
              "note": ("历史 profile 尺寸/宽高比不匹配（1280×779 vs "
                       "1848×1388）；无视觉 → 除 table(毡面实测) 外全部 "
                       "ROI=null+UNCERTAIN；识别链未启动（诚实）")}
    Path("data/vision2/coinpoker_live_100.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
