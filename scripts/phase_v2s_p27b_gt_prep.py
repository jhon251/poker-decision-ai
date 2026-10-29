"""P27-B.0 — 用户空座帧 GT 准备：
1) 复制帧进入 coinpoker_evidence（provenance 可追溯，不改原文件）
2) 生成 coinpoker_1876_gt_empty.json（独立空座 GT 文件——不触碰 22-GT）
3) 用冻结 validate_gt 校验（应 ERRORS: NONE）
"""
from __future__ import annotations
import json
import shutil
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

SRC = Path("data2/42b553c9468d1d1e55b9801bebad7b4f.png")
EVID = Path("data/vision2/coinpoker_evidence")
DST = EVID / SRC.name
if not DST.exists():
    shutil.copy2(SRC, DST)
    print(f"copied -> {DST} ({DST.stat().st_size} bytes)")
else:
    print(f"already in evidence: {DST}")

GT_PATH = Path("data/vision2/coinpoker_1876_gt_empty.json")
gt = {
    "schema_version": "v2",
    "render_class": "1876",
    "annotation_source": "user_provided_frame + read-only visual verification",
    "ground_truth_status": "user_annotated",
    "note": ("P27-B.0: 用户提供空座帧（5 人局，S4 空座）。S4=EMPTY 视觉结构: "
             "「空的」灰字标签(≈201,880,43,21) + ghost 幽灵圈(luma 25-48) + "
             "无座牌盒(panel/name/gold 均为暗区无文字) + 无头像(maroon=0)。"
             "其余 5 座=OCCUPIED（座牌盒+name+gold 三件套；S3/S5 显示暂离"
             "离座图标）。EMPTY 证据系用户帧目视+像素测量，无 detector 反推。"),
    "annotations": [
        {
            "frame_id": SRC.name,
            "geometry": [1876, 1400],
            "fields": {
                "stack": None,
                "status": None,
                "pot": None,
                "bet": None,
                "hero_cards": None,
                "position": {
                    "dealer": {"seat": "S2",
                               "button_bbox": [1065, 309, 12, 12],
                               "button_desc": "红圆白D钮（冻结 P22-C 视觉检测 "
                                              "score=0.886；仅作字段记录，"
                                              "不用于 S4 EMPTY 判定）"},
                    "hero": "S6",
                    "occupied": {
                        "S1": "occupied",
                        "S2": "occupied",
                        "S3": "occupied",
                        "S4": "empty",
                        "S5": "occupied",
                        "S6": "occupied",
                    },
                },
            },
            "visual": {
                "empty_label_bbox": [201, 880, 43, 21],
                "empty_label_luma_band": "dim gray 90-190 (no luma>170 white)",
                "empty_ghost_ring": {"center": [218, 868],
                                     "radius_px": 115,
                                     "luma_band": "25-48"},
                "empty_no_avatar": "maroon ring pixel count = 0",
                "occupied_seat_note": ("S1=competent7724(140,270) "
                                       "S2=ktoo(185,360) S3=amir7777(32,020) "
                                       "S5=traumatic777(75,360) "
                                       "S6=dualang(200,000)"),
                "away_vs_empty_discrimination": ("away badges (S3/S5 暂离) "
                    "are PURE WHITE luma 253-255 on avatar circle (maroon "
                    "present); EMPTY label is DIM gray 90-190, maroon=0. "
                    "Old _empty_tag_at bright-text heuristic wrongly fires "
                    "on away (S3/S5) — recorded as ATTESTED CONFUSION for "
                    "detector design; EMPTY must require seatbox+avatar "
                    "ABSENCE (name=0, gold=0, maroon=0) plus dim label."),
                "table_state": "5-player, waiting 6th (等待盲注/开始下一局 UI)",
            },
        }
    ],
}
GT_PATH.write_text(json.dumps(gt, indent=2, ensure_ascii=False),
                   encoding="utf-8")
print(f"GT written -> {GT_PATH}")

# 冻结校验
from poker_decision_ai.vision2.pixel.realclient.gt_loader import (  # noqa: E402
    validate_gt, gt_summary)
errors = validate_gt(gt, EVID)
print(f"validate_gt errors: {errors}")
print(f"gt_summary: {gt_summary(gt)}")
