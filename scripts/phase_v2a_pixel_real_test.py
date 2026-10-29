"""Phase V2A Pixel Core — 真实截图离线分析（10 张，只读，不修改截图）。

- 输入: <repo>\picture data 下真实 Poker 截图（PNG，纯标准库解码）；
- 每帧: 解码 → Frame（synthetic_test_only=False）→ bootstrap 布局 →
  8 ROI 裁剪 → PixelVisionEngine → RawPixelObservation + PixelEvent；
- 逐 ROI 输出: pixel_change_ratio / changed_bbox / dominant color /
  temporal state / events；证据帧全程可追溯；
- 输出 JSON（data/vision2/pixel_v2a_real_test.json）+ 控制台摘要。
"""
from __future__ import annotations

import argparse
import json
import sys
import time as _time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from poker_decision_ai.vision2.pixel import PixelVisionEngine  # noqa: E402
from poker_decision_ai.vision2.realtime.anchors import bootstrap_profile  # noqa: E402
from poker_decision_ai.vision2.realtime.capture import CapturedFrame  # noqa: E402
from poker_decision_ai.vision2.realtime.coords import GeometryContext  # noqa: E402
from poker_decision_ai.vision2.realtime.frame_manager import Frame  # noqa: E402
from poker_decision_ai.vision2.realtime.image import ImageData  # noqa: E402
from poker_decision_ai.vision2.realtime.roi import (  # noqa: E402
    ROIManager, default_roi_specs)
from poker_decision_ai.vision2_platform import decode_png  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="V2A pixel real screenshot test")
    ap.add_argument("--dir", type=str, default="picture data")
    ap.add_argument("--out", type=str,
                    default="data/vision2/pixel_v2a_real_test.json")
    args = ap.parse_args()
    png_dir = Path(args.dir).resolve()
    files = sorted(png_dir.glob("*.png"))
    if not files:
        print(f"未找到 PNG: {png_dir}")
        return 2

    engine = PixelVisionEngine()
    results = []
    print(f"=== V2A Pixel Core 真实截图离线分析（{len(files)} 张，只读） ===")
    for i, p in enumerate(files):
        t0 = _time.perf_counter()
        w, h, rgba = decode_png(p.read_bytes())
        decode_ms = (_time.perf_counter() - t0) * 1000.0
        geo = GeometryContext(screen_size=(float(w), float(h)),
                              window_rect=None, client_offset=(0.0, 0.0),
                              client_size_physical=(float(w), float(h)),
                              dpi_scale=1.0)
        cap = CapturedFrame(image=ImageData.from_rgba_bytes(w, h, rgba),
                            captured_at=float(i), geometry=geo,
                            source=f"real_screenshot:{p.name}",
                            synthetic_test_only=False)
        frame = Frame.from_capture(cap, i + 1)
        prof = bootstrap_profile("real", geo, {"table": (0.0, 0.0, 1.0, 1.0)})
        mgr = ROIManager(prof, default_roi_specs())
        rois, missing = mgr.resolve()
        crops, cm = mgr.extract_all(frame.image, rois)
        missing.update(cm)
        r = engine.process(frame, rois, crops, missing)
        entry = {
            "file": p.name, "frame_id": frame.frame_id,
            "size": [w, h], "decode_ms": round(decode_ms, 1),
            "result": r.to_dict(),
        }
        results.append(entry)
        events = [e.kind.value for e in r.events]
        pot = r.observations["pot"]
        print(f"  [{p.name[:10]}..] frame={frame.frame_id} "
              f"events={len(events)} pot_ratio={pot.pixel_change_ratio} "
              f"pot_state={pot.temporal_features['state']}")
        if events:
            print(f"      {events[:6]}")

    report = {"screenshots": results,
              "capability_notes": [
                  "Pixel Core 只能输出像素事实（变化/颜色/时序状态/事件）；",
                  "不识别牌面、不识别数字、不推断行动者或意图；",
                  "所有语义字段（pot 数值/牌/座位）仍属 OCR/卡牌/模板层。"]}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False),
                   encoding="utf-8")
    print(f"\n结果已写入 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
