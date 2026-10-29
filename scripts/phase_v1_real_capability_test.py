"""Real Screenshot Capability Test —— 真实 Poker 客户端截图的端到端链路测试。

原则（硬约束）：
  - 只用真实截图文件；解码用平台层纯标准库 PNG 解码器（Pillow 缺失时的
    自包含实现）；任何 synthetic fixture 不参与本测试；
  - 后端可用性如实报告：OCR/卡牌/多模态在本机均不可用 → 字段 null +
    uncertainty（绝不伪造识别）；
  - 逐帧输出：status / blocked / normalized / provenance（value、
    confidence、source、ROI、evidence frame、是否经 OCR、是否经兜底）；
  - 不实现任何自动操作/决策输出。

用法：
    python scripts/phase_v1_real_capability_test.py [--dir "picture data"]
                                                    [--out data/vision2/real_capability_test.json]
"""
from __future__ import annotations

import argparse
import json
import sys
import time as _time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from poker_decision_ai.vision2.realtime import (  # noqa: E402
    GeometryContext, RealtimeVisionConfig, RealtimeVisionPipeline,
    RecognitionStatus, dhash, hamming_distance, pixel_diff_ratio)
from poker_decision_ai.vision2.realtime.capture import (  # noqa: E402
    CapturedFrame, FixtureCaptureProvider)
from poker_decision_ai.vision2.realtime.image import ImageData  # noqa: E402
from poker_decision_ai.vision2.realtime.roi import (  # noqa: E402
    ROIManager, default_roi_specs)
from poker_decision_ai.vision2.realtime.backends import platform_status  # noqa: E402
from poker_decision_ai.vision2_platform import decode_png  # noqa: E402

ALLOWED_EXTS = (".png",)


def load_screenshots(directory: Path) -> list:
    """真实截图文件 → CapturedFrame（synthetic_test_only=False）。"""
    frames = []
    for p in sorted(directory.iterdir()):
        if p.suffix.lower() not in ALLOWED_EXTS:
            print(f"[skip] 非 PNG（本机无 JPEG 解码器）: {p.name}")
            continue
        data = p.read_bytes()
        t0 = _time.perf_counter()
        w, h, rgba = decode_png(data)
        dt = (_time.perf_counter() - t0) * 1000.0
        geo = GeometryContext(screen_size=(float(w), float(h)),
                              window_rect=None, client_offset=(0.0, 0.0),
                              client_size_physical=(float(w), float(h)),
                              dpi_scale=1.0)
        frames.append({
            "file": p.name, "path": str(p), "size_bytes": len(data),
            "decode_ms": round(dt, 1), "width": w, "height": h,
            "frame": CapturedFrame(
                image=ImageData.from_rgba_bytes(w, h, rgba),
                captured_at=0.0, geometry=geo,
                source=f"real_screenshot:{p.name}", synthetic_test_only=False),
        })
    return frames


def cheap_detection(frames: list) -> dict:
    """真实像素上的 cheap 检测（只判"是否变化"，不识别语义）。"""
    out = []
    for i in range(1, len(frames)):
        a, b = frames[i - 1]["frame"].image, frames[i]["frame"].image
        ratio = pixel_diff_ratio(a, b)
        ham = hamming_distance(dhash(a), dhash(b))
        out.append({"pair": f"{frames[i-1]['file'][:8]}..->"
                            f"{frames[i]['file'][:8]}..",
                    "pixel_diff_ratio": round(ratio, 6),
                    "dhash_hamming": ham,
                    "changed": ratio > 0.0})
    return out


def roi_crop_stats(frame_entry: dict) -> dict:
    """bootstrap 布局下 ROI 裁剪的机械性验证（不判语义）。"""
    geo = frame_entry["frame"].geometry
    profile = None
    from poker_decision_ai.vision2.realtime.anchors import bootstrap_profile
    profile = bootstrap_profile("real", geo, {"table": (0.0, 0.0, 1.0, 1.0)})
    mgr = ROIManager(profile, default_roi_specs())
    rois, missing = mgr.resolve()
    crops, crop_missing = mgr.extract_all(frame_entry["frame"].image, rois)
    return {
        "resolved": sorted(r.value for r in rois),
        "missing": {k.value: v for k, v in missing.items()},
        "crop_missing": {k.value: v for k, v in crop_missing.items()},
        "crops": {k.value: {"w": v.width, "h": v.height}
                  for k, v in crops.items()},
    }


def run_pipeline(frames: list) -> list:
    """真实帧 → 完整 V1 链路（真实后端可用性门控，全部诚实降级）。"""
    provider = FixtureCaptureProvider(
        [f["frame"] for f in frames], cycle=False)
    pipe = RealtimeVisionPipeline(
        config=RealtimeVisionConfig(fps=10), capture_provider=provider)
    obs_list = []
    for i, f in enumerate(frames):
        obs = pipe.step(now=float(i))
        obs_list.append({
            "file": f["file"],
            "frame_id": obs.frame_id,
            "status": obs.status.value,
            "blocked": obs.blocked,
            "failure_reason": obs.failure_reason,
            "consumable": obs.consumable(),
            "synthetic_test_only": obs.synthetic_test_only,
            "normalized": None if obs.normalized is None
            else obs.normalized.to_dict(),
            "provenance": obs.provenance,
            "transition": obs.transition,
            "events": [e.to_dict() for e in obs.events],
        })
    return obs_list


def main() -> int:
    ap = argparse.ArgumentParser(description="Vision V1 real screenshot test")
    ap.add_argument("--dir", type=str, default="picture data")
    ap.add_argument("--out", type=str,
                    default="data/vision2/real_capability_test.json")
    args = ap.parse_args()
    directory = Path(args.dir).resolve()

    print("=== 环境审计 ===")
    status = platform_status()
    for k, v in status.items():
        print(f"  {k:<10} available={v['available']}  {v['reason']}")

    print(f"\n=== 加载真实截图: {directory} ===")
    frames = load_screenshots(directory)
    print(f"  解码成功 {len(frames)} 张")
    for f in frames:
        print(f"  {f['file']:<40} {f['width']}x{f['height']} "
              f"{f['size_bytes']}B decode={f['decode_ms']}ms")

    print("\n=== cheap 变化检测（真实像素） ===")
    diffs = cheap_detection(frames)
    for d in diffs:
        print(f"  {d['pair']:<24} diff={d['pixel_diff_ratio']:.6f} "
              f"dhash_ham={d['dhash_hamming']} changed={d['changed']}")

    print("\n=== ROI 裁剪（bootstrap 布局，机械验证） ===")
    roi_info = roi_crop_stats(frames[0])
    print(f"  resolved={roi_info['resolved']}")
    print(f"  missing={roi_info['missing']} crop_missing={roi_info['crop_missing']}")
    print(f"  crops={json.dumps(roi_info['crops'], ensure_ascii=False)}")

    print("\n=== 全链路（真实帧 → NormalizedVisionState） ===")
    results = run_pipeline(frames)
    for r in results:
        n = r["normalized"]
        print(f"  [{r['file'][:12]}..] frame={r['frame_id']} "
              f"status={r['status']} blocked={r['blocked']} "
              f"reason={r['failure_reason']} synthetic={r['synthetic_test_only']}")
        if n:
            print(f"      pot={n['pot']} street={n['game']['street']} "
                  f"hero={n['hero']}")
            print(f"      uncertainty={[u['field'] for u in n['uncertainty']]}")

    report = {
        "environment": status,
        "screenshots": [{k: v for k, v in f.items() if k != "frame"}
                        for f in frames],
        "cheap_detection": diffs,
        "roi": roi_info,
        "observations": results,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False),
                   encoding="utf-8")
    print(f"\n结果已写入 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
