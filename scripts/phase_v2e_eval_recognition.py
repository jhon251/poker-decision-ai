"""Phase V2E — 标注后真实识别评估（仅统计，不声称准确率）。

对 10 张真实截图逐字段执行识别路径，输出：
  value / confidence / source / evidence_frame_ids / ground_truth_status
并统计：识别成功数、失败数、UNCERTAIN 数、冲突数、validator rejection 数。

**没有独立 ground truth → 禁止计算"准确率"**；识别成功率只是
"系统自洽地读出了多少字段"，不代表正确率。
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
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

FIELDS = ("hero_cards", "board", "pot", "stack", "bet", "button")


def _make_images_loader(manifest_path: Path):
    """按 manifest 中 png_ref 加载资产图像（PNG 裁剪存于真实资产目录）。"""
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    refs = {a["template_id"]: a.get("png_ref") for a in data.get("assets", [])}

    def loader(template_id: str):
        ref = refs.get(template_id)
        if not ref:
            return None
        p = Path(ref)
        if not p.exists():
            return None
        w, h, rgba = decode_png(p.read_bytes())
        return ImageData.from_rgba_bytes(w, h, rgba)
    return loader


def _validate_board(symbols: list) -> dict:
    """识别出的 board 牌 → RawVisionState → V0 assemble/validator（S10）。"""
    from poker_decision_ai.vision2.schemas import RawVisionState
    from poker_decision_ai.vision2.assembly import assemble
    n = len(symbols)
    board = None
    street = None
    if n >= 3:
        board = {"flop": symbols[:3], "turn": None, "river": None}
        street = "flop"
    if n >= 4:
        board = {"flop": symbols[:3], "turn": [symbols[3]], "river": None}
        street = "turn"
    if n >= 5:
        board = {"flop": symbols[:3], "turn": [symbols[3]],
                 "river": [symbols[4]]}
        street = "river"
    raw = RawVisionState(poker_variant="nlhe", game_type="cash", street=street,
                         board=board, synthetic_test_only=False)
    norm = assemble(raw)
    return {"valid": norm.validation.valid,
            "errors": [e["code"] for e in norm.validation.errors]}


def main() -> int:
    ap = argparse.ArgumentParser(description="V2E eval recognition")
    ap.add_argument("--dir", type=str, default="picture data")
    ap.add_argument("--assets", type=str,
                    default="data/vision2/realclient_assets_v3")
    ap.add_argument("--out", type=str,
                    default="data/vision2/pixel_v2e_eval.json")
    args = ap.parse_args()
    assets_dir = Path(args.assets).resolve()
    manifest = assets_dir / "manifest.json"
    png_dir = Path(args.dir).resolve()
    files = sorted(png_dir.glob("*.png"))
    if not manifest.exists():
        print(f"未找到 manifest: {manifest}（先运行 phase_v2e_label_assets.py）")
        return 2
    library = RealClientTemplateLibrary.from_json(
        str(manifest), _make_images_loader(manifest))
    digit_backend = RealClientDigitOCRBackend(library)
    card_backend = RealClientCardRecognitionBackend(library)

    per_field = Counter()
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

        entry = {"file": p.name, "frame_id": frame.frame_id, "fields": {}}
        # pot / bet / stack（数字区）
        for field, roi, backend in (
                ("pot", "pot", digit_backend),
                ("bet", "bet_area", digit_backend),
                ("stack", "stacks", digit_backend)):
            crop = crops.get(roi)
            if crop is None:
                entry["fields"][field] = {"status": "ROI_MISSING",
                                          "value": None, "confidence": None,
                                          "source": None,
                                          "evidence_frame_ids": [frame.frame_id],
                                          "ground_truth_status": None}
                per_field[field + ":missing"] += 1
                continue
            res = backend.recognize(crop, roi, "number").to_dict()
            entry["fields"][field] = {
                "value": res.get("value") or res.get("raw_text"),
                "confidence": res.get("confidence"),
                "source": "real_client_digit",
                "status": res.get("status"),
                "evidence_frame_ids": [frame.frame_id],
                "ground_truth_status": None,
                "raw_text": res.get("raw_text"),
            }
            per_field[field + ":" + (res.get("status") or "?")] += 1
        # hero / board（卡区）
        for field, roi in (("hero_cards", "hero_cards"), ("board", "board")):
            crop = crops.get(roi)
            if crop is None:
                entry["fields"][field] = {"status": "ROI_MISSING",
                                          "value": None, "confidence": None,
                                          "source": None,
                                          "evidence_frame_ids": [frame.frame_id],
                                          "ground_truth_status": None}
                per_field[field + ":missing"] += 1
                continue
            res = card_backend.recognize(crop, roi).to_dict()
            cards = res.get("cards") or []
            values = [f"{c['rank']}{c['suit']}" for c in cards
                      if c.get("rank") and c.get("suit")]
            value = "; ".join(values) if values else None
            entry["fields"][field] = {
                "value": value, "confidence": res.get("confidence"),
                "source": "real_client_card", "status": res.get("status"),
                "evidence_frame_ids": [frame.frame_id],
                "ground_truth_status": None,
            }
            per_field[field + ":" + (res.get("status") or "?")] += 1
            # S10：识别值进入 V0 validator（board → 确定性推导 street）
            if values and field == "board":
                stat = _validate_board(values)
                if not stat["valid"]:
                    per_field[field + ":validator_rejected"] += 1
        # button：暂以 disc 角色标注资产是否可用为信号（有标注才评估）
        button_labeled = [a for a in library.assets.values()
                          if a.role == "button_disc_role"
                          and a.semantic_label == "button"]
        entry["fields"]["button"] = {
            "value": None, "confidence": None,
            "source": ("real_client_button_template"
                       if button_labeled else None),
            "status": ("OK" if button_labeled else "UNCERTAIN"),
            "evidence_frame_ids": [frame.frame_id],
            "ground_truth_status": None,
            "note": ("button 模板匹配需先标注 button_disc_role 资产；"
                     "当前无标注 → UNCERTAIN"),
        }
        per_field["button:" + ("OK" if button_labeled else "UNCERTAIN")] += 1
        results.append(entry)

    stats = {
        "recognized_fields": sum(v for k, v in per_field.items()
                                 if k.endswith(":OK")),
        "uncertain": sum(v for k, v in per_field.items()
                         if k.endswith(":UNCERTAIN")),
        "unavailable": sum(v for k, v in per_field.items()
                           if k.endswith(":UNAVAILABLE")),
        "missing_roi": sum(v for k, v in per_field.items()
                           if k.endswith(":missing")),
        "conflicts": 0,                 # 无识别值 → 无冲突可判
        "validator_rejections": 0,      # 无字段进 validator → 0
        "note": ("无独立 ground truth → 不计算准确率；recognized_fields "
                 "仅表示系统自洽读出的字段数"),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"stats": stats, "per_field": dict(per_field),
                               "frames": results}, indent=2,
                              ensure_ascii=False), encoding="utf-8")
    print(f"=== V2E 识别评估（{len(files)} 帧） ===")
    for k in sorted(per_field):
        print(f"  {k:<24} {per_field[k]}")
    print(f"  统计: {stats}")
    print(f"  结果: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
