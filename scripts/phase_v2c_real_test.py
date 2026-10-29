"""Phase V2C — 真实截图识别后端离线测试（10 张，只读）。

- 加载资产库（当前全 unlabeled_candidate）→ 识别后端严格门控：
  无已标注真实资产 → UNAVAILABLE → 全部字段 null + uncertainty；
- 输出诚实能力边界 + 标注启用指引；
- 若用户未来完成标注（manifest 中 semantic_label 填写），同一脚本
  会自动切换到真实识别路径（后端复用 V2B 识别机制）。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from poker_decision_ai.vision2.pixel.realclient import (  # noqa: E402
    RealClientCardRecognitionBackend, RealClientDigitOCRBackend,
    RealClientTemplateLibrary)
from poker_decision_ai.vision2.realtime.anchors import bootstrap_profile  # noqa: E402
from poker_decision_ai.vision2.realtime.capture import CapturedFrame  # noqa: E402
from poker_decision_ai.vision2.realtime.coords import GeometryContext  # noqa: E402
from poker_decision_ai.vision2.realtime.frame_manager import Frame  # noqa: E402
from poker_decision_ai.vision2.realtime.image import ImageData  # noqa: E402
from poker_decision_ai.vision2.realtime.roi import (  # noqa: E402
    ROIManager, default_roi_specs)
from poker_decision_ai.vision2_platform import decode_png  # noqa: E402


def _load_images(assets_dir: Path):
    def loader(template_id: str):
        p = assets_dir / f"{template_id}.png"
        if not p.exists():
            return None
        w, h, rgba = decode_png(p.read_bytes())
        return ImageData.from_rgba_bytes(w, h, rgba)
    return loader


def main() -> int:
    ap = argparse.ArgumentParser(description="V2C real-client backend test")
    ap.add_argument("--dir", type=str, default="picture data")
    ap.add_argument("--assets", type=str,
                    default="data/vision2/realclient_assets_v1")
    ap.add_argument("--out", type=str,
                    default="data/vision2/pixel_v2c_real_test.json")
    args = ap.parse_args()
    assets_dir = Path(args.assets).resolve()
    manifest = assets_dir / "manifest.json"
    png_dir = Path(args.dir).resolve()
    files = sorted(png_dir.glob("*.png"))
    if not manifest.exists():
        print(f"未找到资产 manifest: {manifest}（先运行 phase_v2c_extract_assets.py）")
        return 2
    library = RealClientTemplateLibrary.from_json(str(manifest),
                                                  _load_images(assets_dir))
    labeled = library.labeled_assets()
    print(f"=== V2C 识别后端真实截图测试 ===")
    print(f"  资产: 总数 {len(library.assets)}，已标注 {len(labeled)}，"
          f"未标注 {len(library.unlabeled_assets())}")

    digit_backend = RealClientDigitOCRBackend(library)
    card_backend = RealClientCardRecognitionBackend(library)
    print(f"  digit backend available: {digit_backend.available()} "
          f"({digit_backend.unavailable_reason()})")
    print(f"  card backend available: {card_backend.available()} "
          f"({card_backend.unavailable_reason()})")

    results = []
    for i, p in enumerate(files):
        w, h, rgba = decode_png(p.read_bytes())
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
        entry = {"file": p.name, "frame_id": frame.frame_id}
        pot = crops.get("pot")
        entry["pot"] = (digit_backend.recognize(pot, "pot", "number").to_dict()
                        if pot is not None else {"status": "ROI_MISSING"})
        hero = crops.get("hero_cards")
        entry["hero_cards"] = (card_backend.recognize(hero, "hero_cards")
                               .to_dict() if hero is not None
                               else {"status": "ROI_MISSING"})
        board = crops.get("board")
        entry["board"] = (card_backend.recognize(board, "board").to_dict()
                          if board is not None
                          else {"status": "ROI_MISSING"})
        results.append(entry)
        print(f"  [{p.name[:10]}..] frame={i+1} "
              f"pot={entry['pot']['status']} "
              f"hero={entry['hero_cards']['status']} "
              f"board={entry['board']['status']}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"backend_available": {
        "digit": digit_backend.available(),
        "card": card_backend.available()},
        "results": results}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n结果已写入 {out}")
    print("启用真实识别路径: 对 manifest 中 cluster_id 批量填写 semantic_label"
          "（rank/suit/数字），并更新 ground_truth_status 为 visually_labeled"
          " 或 verified；此后本脚本自动进入识别模式。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
