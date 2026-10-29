"""Phase V2N — Real Capture Runtime 集成回测（当前环境：capture 后端不可用）。"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from poker_decision_ai.vision2.pixel.realclient import (  # noqa: E402
    RealClientCardRecognitionBackend, RealClientDigitOCRBackend,
    RealClientTemplateLibrary)
from poker_decision_ai.vision2.pixel.realclient.acquisition import (  # noqa: E402
    FrameAcquirer, detected_backend_status)
from poker_decision_ai.vision2.pixel.realclient.layout import (  # noqa: E402
    RealClientLayoutProfile)
from poker_decision_ai.vision2.pixel.realclient.runtime import (  # noqa: E402
    roi_digest)
from poker_decision_ai.vision2.realtime.coords import Rect  # noqa: E402
from poker_decision_ai.vision2.realtime.image import ImageData  # noqa: E402
from poker_decision_ai.vision2_platform import decode_png  # noqa: E402


def main() -> int:
    sys.argv = sys.argv[:1]
    # 1) 后端可用性（诚实）
    det = detected_backend_status()
    print("capture backend:", det)
    # 2) 回放驱动运行（backend 不可用 → 采集路径如实 CAPTURE_BACKEND_UNAVAILABLE）
    acquirer = FrameAcquirer(fps=2.0, backend=None)
    r = acquirer.acquire_from_capture()
    assert r["status"] == "CAPTURE_BACKEND_UNAVAILABLE"

    manifest = Path("data/vision2/realclient_assets_v6/manifest.json")
    mdata = json.loads(manifest.read_text(encoding="utf-8"))
    refs = {a["template_id"]: a.get("png_ref") for a in mdata["assets"]}

    def loader(tid):
        rr = refs.get(tid)
        if not rr or not Path(rr).exists():
            return None
        w_, h_, rgba_ = decode_png(Path(rr).read_bytes())
        return ImageData.from_rgba_bytes(w_, h_, rgba_)

    lib = RealClientTemplateLibrary.from_json(str(manifest), loader)
    digit = RealClientDigitOCRBackend(lib)
    prof = RealClientLayoutProfile.measured_v2()
    stack_rois = [prof.entries[f"player_{i}_stack_roi"].bbox
                  for i in range(6)]

    files = sorted(Path("picture data").glob("*.png"))
    metrics = acquirer.metrics
    t_all = []
    last_digests = {}
    for fi, p in enumerate(files):
        t0 = time.perf_counter()
        w, h, rgba = decode_png(p.read_bytes())
        img = ImageData.from_rgba_bytes(w, h, rgba)
        acc = acquirer.acquire_replay(img, w=w, h=h, captured_at=float(fi))
        if acc["frame"] is None:
            continue
        metrics.frames_processed += 1
        # 增量 ROI：unchanged → 复用；changed → 重识别
        for i, rect in enumerate(stack_rois):
            if rect is None:
                continue
            px, py = rect.width * 0.3, rect.height * 0.3
            exp = Rect(max(0, rect.x - px), max(0, rect.y - py),
                       min(w, rect.x + rect.width + px) - max(0, rect.x - px),
                       min(h, rect.y + rect.height + py) - max(0, rect.y - py))
            sub = img.crop(exp)
            d = roi_digest(sub)
            if d == last_digests.get(i):
                continue                          # unchanged：复用上一状态
            last_digests[i] = d
            metrics.roi_changed_count += 1
            metrics.recognition_count += 1
            digit.recognize(sub, "stacks", "number")
        metrics.record_latency((time.perf_counter() - t0) * 1000)
        t_all.append((time.perf_counter() - t0) * 1000)

    out = Path("data/vision2/pixel_v2n_runtime.json")
    out.write_text(json.dumps({
        "backend": det, "metrics": metrics.to_dict(),
        "replay_latency_ms_percentiles": metrics.percentiles()},
        indent=2, ensure_ascii=False), encoding="utf-8")
    print("metrics:", metrics.to_dict())
    print("out:", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
