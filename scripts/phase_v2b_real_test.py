"""Phase V2B — 真实截图离线验证（10 张，只读，不修改截图）。

每张截图：解码 → 引擎（几何/分割/模板候选）→ 数字字形识别（pot/bet/
stacks 区）→ 卡牌模板识别（hero_cards/board 区）→ 逐 ROI 事实报告。

诚实边界：builtin_synthetic 字体/卡牌模板**不是任何真实客户端资产**，
真实截图上的命中只代表"与内置模板相似"，未标注验证不宣称识别成功；
未命中一律 null + UNCERTAIN。
"""
from __future__ import annotations

import argparse
import json
import sys
import time as _time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from poker_decision_ai.vision2.pixel import (  # noqa: E402
    CardTemplateRecognitionBackend, DigitTemplateOCRBackend,
    PixelEngineConfig, PixelVisionEngine, builtin_card_library,
    builtin_digit_templates, recognize_digits)
from poker_decision_ai.vision2.realtime.anchors import bootstrap_profile  # noqa: E402
from poker_decision_ai.vision2.realtime.capture import CapturedFrame  # noqa: E402
from poker_decision_ai.vision2.realtime.coords import GeometryContext  # noqa: E402
from poker_decision_ai.vision2.realtime.frame_manager import Frame  # noqa: E402
from poker_decision_ai.vision2.realtime.image import ImageData  # noqa: E402
from poker_decision_ai.vision2.realtime.roi import (  # noqa: E402
    ROIManager, default_roi_specs)
from poker_decision_ai.vision2_platform import decode_png  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="V2B real screenshot test")
    ap.add_argument("--dir", type=str, default="picture data")
    ap.add_argument("--out", type=str,
                    default="data/vision2/pixel_v2b_real_test.json")
    args = ap.parse_args()
    png_dir = Path(args.dir).resolve()
    files = sorted(png_dir.glob("*.png"))
    if not files:
        print(f"未找到 PNG: {png_dir}")
        return 2

    engine = PixelVisionEngine(PixelEngineConfig())
    digit_tpls = builtin_digit_templates()
    card_lib = builtin_card_library()
    card_backend = CardTemplateRecognitionBackend(card_lib)
    results = []
    print(f"=== V2B 真实截图离线验证（{len(files)} 张，只读） ===")
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

        digit_results = {}
        for name in ("pot", "bet_area", "stacks"):
            crop = crops.get(name)
            if crop is None:
                digit_results[name] = {"status": "ROI_MISSING", "text": None,
                                       "value": None, "confidence": None}
            else:
                rec = recognize_digits(crop, digit_tpls)
                digit_results[name] = rec.to_dict()
        card_results = {}
        for name in ("hero_cards", "board"):
            crop = crops.get(name)
            if crop is None:
                card_results[name] = {"status": "ROI_MISSING", "cards": None,
                                      "error": None}
            else:
                res = card_backend.recognize(crop, name)
                card_results[name] = {"status": res.status.value,
                                      "cards": None if res.cards is None
                                      else [c.to_dict() for c in res.cards],
                                      "error": res.error}

        entry = {
            "file": p.name, "frame_id": frame.frame_id,
            "size": [w, h], "decode_ms": round(decode_ms, 1),
            "events": [e.to_dict() for e in r.events],
            "digit": digit_results,
            "card": card_results,
            "note": ("digit/card 识别使用 builtin_synthetic 模板库，"
                     "未标注验证不宣称识别成功"),
        }
        results.append(entry)
        pot_rec = digit_results["pot"]
        print(f"  [{p.name[:10]}..] frame={frame.frame_id} "
              f"events={len(r.events)} "
              f"pot_digit={pot_rec['status']}('{pot_rec['text']}') "
              f"hero_cards={card_results['hero_cards']['status']}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"screenshots": results}, indent=2,
                              ensure_ascii=False), encoding="utf-8")
    print(f"\n结果已写入 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
