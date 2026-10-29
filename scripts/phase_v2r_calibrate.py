"""Phase V2R — CoinPoker 一次性视觉标注与校准（annotation evidence）。

标注来源：真实截帧（frame 1/2）视觉核验（窗口坐标 1419×1066）：
  - pot:      "底池 3,960"  白字深底 (905-1245, 480-535)
  - poder2030 stack "201,000" (285-460, 425-480)
  - miyagi112  stack "1.3M"   (925-1110, 300-345)  ← M 单位实据
  - homefoodgood stack "197,360" (185-410, 940-990)
  - bet 1,000 (515-625, 415-480)；bet 960 (700-1035, 745-810)
  - dealer D  红六边形 (385-458, 722-798)
  - SB 徽章（Poder2030）——客户端 UI 徽章（视觉事实，非视觉推导 position）
  - hero_cards / board / action panels / folded-allin：本帧帧不可见 → null

产物：
  data/vision2/coinpoker/  assets(白字字符模板, coinpoker namespace)
  data/vision2/reallayout_coinpoker_v1.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from poker_decision_ai.vision2.pixel.realclient.capture import (  # noqa: E402
    CtypesBitBltBackend)
from poker_decision_ai.vision2_platform import (  # noqa: E402
    decode_png, encode_rgba_png)

WINDOW = (274, 2, 1419, 1066)
FRAME_EVIDENCE = "coinpoker_frame_1.png"

#: 标注区域（window 坐标，视觉核验）→ 读出字符串（标注证据）
ANNOTATIONS = {
    "pot_text": {"bbox": (905, 500, 340, 45), "text": "3,960"},
    "stack_poder": {"bbox": (285, 425, 180, 55), "text": "201,000"},
    "stack_miyagi": {"bbox": (925, 300, 190, 45), "text": "1.3M"},
    "stack_home": {"bbox": (185, 940, 230, 50), "text": "197,360"},
    "bet_1000": {"bbox": (515, 445, 115, 40), "text": "1,000"},
    "bet_960": {"bbox": (700, 770, 110, 40), "text": "960"},
}


def bright_text_chars(sub: np.ndarray) -> list:
    """白字深底 → 亮度掩码 → 列投影字符切分（确定性）。"""
    r = sub[..., 0].astype(int)
    g = sub[..., 1].astype(int)
    b = sub[..., 2].astype(int)
    luma = r * 0.299 + g * 0.587 + b * 0.114
    mask = luma > 150
    col_on = mask.sum(axis=0) > 2
    runs = []
    x = 0
    while x < len(col_on):
        if not col_on[x]:
            x += 1
            continue
        x0 = x
        while x < len(col_on) and col_on[x]:
            x += 1
        runs.append((x0, x))
    out = []
    for (x0, x1) in runs:
        ys = np.where(mask[:, x0:x1].any(axis=1))[0]
        if len(ys) < 6:
            continue
        out.append((int(x0), int(ys.min()), int(x1 - x0),
                    int(ys.max() - ys.min() + 1)))
    return out


def main() -> int:
    sys.argv = sys.argv[:1]
    out_dir = Path("data/vision2/coinpoker")
    out_dir.mkdir(parents=True, exist_ok=True)
    evidence = (Path("data/vision2/coinpoker_calibration")
                / FRAME_EVIDENCE)
    w, h, rgba = decode_png(evidence.read_bytes())
    full = np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)

    assets = []
    prof = {"client": "coinpoker", "resolution": [1419, 1066],
            "window_bbox": list(WINDOW), "calibration_version": 1,
            "source": "visual_annotation", "evidence_frame_id": [1, 2],
            "rois": {}, "note": ""}
    prof["rois"]["table"] = {"bbox": [215, 255, 1000, 500],
                             "source": "felt_mask_measured",
                             "confidence": 0.85, "measured": True}
    # 模板资产逐字符切分（按标注串位置标注——视觉读取证据）
    char_assets = {}
    for name, ann in ANNOTATIONS.items():
        x0, y0, cw, chh = ann["bbox"]
        sub = full[y0:y0 + chh, x0:x0 + cw]
        chars = bright_text_chars(sub)
        text = ann["text"]
        if len(chars) != len(text):
            prof["note"] += (f"{name}: 切分 {len(chars)} 字符 vs 标注 {len(text)}"
                             f" 字符不一致（{chars}）; ")
            continue
        prof["rois"][name] = {"bbox": list(ann["bbox"]),
                              "source": "visual_annotation",
                              "confidence": 0.9, "measured": True}
        for j, (cx, cy, cwid, chh2) in enumerate(chars):
            crop = sub[cy:cy + chh2, cx:cx + cwid]
            tid = f"coinpoker_{name}_{j}"
            p = out_dir / f"{tid}.png"
            p.write_bytes(encode_rgba_png(crop.shape[1], crop.shape[0],
                                          crop.tobytes()))
            char_assets[tid] = {
                "template_id": tid, "label": text[j],
                "source_frame": FRAME_EVIDENCE,
                "source_roi": name,
                "client": "coinpoker",
                "kind": "digit_glyph",
                "ground_truth_status": "visually_labeled",
                "label_source": "vision_annotation",
                "crop_bbox": [cx + x0, cy + y0, cwid, chh2],
                "png_ref": str(p)}
    # 按钮/徽章/牌背（区域资产，非字符模板）
    prof["rois"]["button_d"] = {"bbox": [385, 722, 73, 76],
                                "source": "visual_annotation",
                                "confidence": 0.9, "measured": True}
    prof["rois"]["sb_badge_poder"] = {"bbox": [410, 420, 50, 40],
                                      "source": "visual_annotation",
                                      "confidence": 0.9, "measured": True,
                                      "note": "UI 徽章（视觉事实）"}
    for nm in ("hero_cards", "board", "action_panels",
               "folded_marker", "allin_marker"):
        prof["rois"][nm] = {"bbox": None, "confidence": None,
                            "measured": False,
                            "note": "当前帧不可见 → null+UNCERTAIN"}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "coinpoker_assets.json").write_text(
        json.dumps({"client": "coinpoker", "assets": char_assets},
                   indent=1, ensure_ascii=False), encoding="utf-8")
    Path("data/vision2/reallayout_coinpoker_v1.json").write_text(
        json.dumps(prof, indent=1, ensure_ascii=False), encoding="utf-8")
    print("assets:", len(char_assets))
    print("profile rois:", {k: v["bbox"] for k, v in prof["rois"].items()})
    print("note:", prof["note"] or "(none)")
    print("saved: data/vision2/coinpoker/coinpoker_assets.json + "
          "data/vision2/reallayout_coinpoker_v1.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
