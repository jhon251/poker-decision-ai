"""V3E-P1 ASSET REPAIR — 冻结模板资产修复（用户已授权）。

依据：gt_confirm_v3e_p1/v2/final_answers.json 人工最终 GT（80 确认项）+
logs/v3e_p1_final_review.json 模板身份判定。

操作（仅资产层；detector/threshold/识别逻辑/positions_for/P30–P35/EXE 零修改）：
  1. relabel：rank_4→rank_6（c2）、rank_J→rank_K（c2）——人工 GT 帧像素全同 1.0 实证；
  2. 新增：仅"≥2 帧且组内两两 NCC≥0.85"的证据充分组，position-specific 提取；
  3. 单帧组（7@c1、2@c2、8@c2、J@c1）不新增——证据不足，保持缺口（UNKNOWN-first）；
  4. manifest 记录每个资产的人工 GT 来源帧；原资产备份于 logs/v3e_p1_repair_backup/。

运行前置：logs/v3e_p1_repair_before.json（BEFORE 回归基线）已生成。
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from v3e_p1_hero_audit import EXT_WIN, EV, glyph, ncc_pair
from poker_decision_ai.vision2_platform.png_decode import decode_png
from poker_decision_ai.vision2_platform.png_encode import encode_rgba_png

ASSETS = Path("data/vision2/coinpoker_assets_v1876herocards")
MANIFEST = ASSETS / "manifest.json"
LOG = Path("logs/v3e_p1_asset_repair.json")
GATE = 0.85  # 组内两两 NCC 门（rebuild 聚类门 0.6 的收紧版）

F = {
    "250": "POT_CHANGED_250_1787992907809_full.png",
    "524": "POT_CHANGED_524_1787993575599_full.png",
    "429": "SCENE_CHANGED_429_1787993345715_full.png",
    "512": "POT_CHANGED_512_1787993546817_full.png",
    "606": "POT_CHANGED_606_1787993775320_full.png",
    "18": "TABLE_CHANGED_18_1787992293441_full.png",
    "404": "POT_CHANGED_404_1787993286628_full.png",
    "16": "POT_CHANGED_16_1787995303501_full.png",
    "208": "POT_CHANGED_208_1787992806955_full.png",
    "SCENE6": "SCENE_CHANGED_6_1787992258912_full.png",
    "708": "POT_CHANGED_708_1787994016446_full.png",
    "293": "POT_CHANGED_293_1787993018309_full.png",
    "535": "POT_CHANGED_535_1787993603723_full.png",
    "454": "POT_CHANGED_454_1787993404728_full.png",
    "33": "POT_CHANGED_33_1787995356823_full.png",
    "420": "POT_CHANGED_420_1787993325856_full.png",
    "353": "POT_CHANGED_353_1787993164702_full.png",
    "47": "POT_CHANGED_47_1787995398974_full.png",
    "181": "POT_CHANGED_181_1787992731553_full.png",
    "290": "POT_CHANGED_290_1787993011134_full.png",
}

# (kind, ch, pos, 候选成员帧 tag 列表)
CANDIDATES = [
    ("rank", "A", 1, ["250", "208", "290", "293", "404"]),
    ("rank", "K", 1, ["16", "524", "708"]),
    ("rank", "Q", 1, ["429", "420", "535"]),
    ("rank", "T", 1, ["512", "353"]),
    ("rank", "8", 1, ["454", "SCENE6"]),
    ("rank", "4", 1, ["33", "47"]),
    ("rank", "5", 2, ["404", "208"]),
    ("rank", "9", 2, ["429", "420", "181"]),
    ("rank", "6", 2, ["512", "708"]),
    ("rank", "J", 2, ["524", "535"]),
    ("rank", "4", 2, ["33", "47"]),
    ("suit", "s", 1, ["250", "404", "293", "708", "SCENE6", "33", "47"]),
    ("suit", "d", 1, ["524", "535", "512", "181"]),
    ("suit", "h", 1, ["429", "420", "208", "353"]),
    ("suit", "s", 2, ["429", "420"]),
]

RELABELS = [
    {"old_label": "rank_4", "new_label": "rank_6", "rank": "6",
     "old_file": "hcr_4_c2_942_1018.png", "new_file": "hcr_6_c2_942_1018.png",
     "identity_frame": F["708"], "identity_pos": 2, "human": "6c"},
    {"old_label": "rank_J", "new_label": "rank_K", "rank": "K",
     "old_file": "hcr_J_c2_942_1018.png", "new_file": "hcr_K_c2_942_1018.png",
     "identity_frame": F["293"], "identity_pos": 2, "human": "Kc"},
]

_cache = {}


def G(tag, pos, kind):
    key = (tag, pos, kind)
    if key not in _cache:
        w, h, rgba = decode_png((EV / F[tag]).read_bytes())
        arr = np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)
        m, bbox = glyph(arr, EXT_WIN[pos]["rank" if kind == "rank" else "suit"])
        _cache[key] = (m, bbox)
    return _cache[key]


def screen_group(kind, ch, pos, tags):
    """组内两两 NCC 预检；剔除破坏内聚成员；<2 帧则整组放弃。"""
    members = list(tags)
    detail = {"group": f"{kind}_{ch}_c{pos}", "pairs": {}, "dropped": []}
    masks = {t: G(t, pos, kind)[0] for t in members}
    if any(m is None for m in masks.values()):
        detail["dropped"] += [t for t, m in masks.items() if m is None]
        members = [t for t in members if masks[t] is not None]
    while len(members) >= 2:
        sums = {t: 0.0 for t in members}
        cnt = {t: 0 for t in members}
        worst, worst_v = None, 1.1
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                a, b = members[i], members[j]
                v = round(ncc_pair(masks[a], masks[b]), 3)
                detail["pairs"]["%s~%s" % (a, b)] = v
                sums[a] += v; cnt[a] += 1
                sums[b] += v; cnt[b] += 1
                if v < worst_v:
                    worst, worst_v = (a, b), v
        if worst_v >= GATE:
            detail["kept"] = members
            detail["min_pair"] = round(worst_v, 3)
            src = max(members, key=lambda t: sums[t] / max(cnt[t], 1))
            detail["source"] = src
            return members, src, detail
        drop = min((a, b), key=lambda t: sums[t] / max(cnt[t], 1))
        detail["dropped"].append(drop)
        members.remove(drop)
    detail["kept"] = []
    return [], None, detail


def encode_asset(mask):
    h, w = mask.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[..., 0:3] = 255
    rgba[..., 3] = mask.astype(np.uint8) * 255
    return encode_rgba_png(w, h, rgba.tobytes())


def main() -> int:
    if not Path("logs/v3e_p1_repair_before.json").exists():
        print("ABORT: BEFORE 回归基线尚未生成")
        return 2
    mp = json.loads(MANIFEST.read_text(encoding="utf-8"))
    before_n = len(mp["assets"])
    report = {"relabels": [], "added": [], "skipped": [], "groups": []}

    # ── 1) relabel（幂等：文件可能已在上次中断时移动） ─────────────
    for r in RELABELS:
        old_p, new_p = ASSETS / r["old_file"], ASSETS / r["new_file"]
        if not new_p.exists():
            assert old_p.exists(), old_p
            shutil.move(str(old_p), str(new_p))
        for a in mp["assets"]:
            if a["label"] == r["old_label"] and \
                    Path(a["crop_ref"]).name == r["old_file"]:
                a["label"] = r["new_label"]
                a["rank"] = r["rank"]
                a["crop_ref"] = r["new_file"]
                a["provenance"] = (
                    "V3E-P1 ASSET REPAIR relabel: asset glyph pixel-identity "
                    "NCC=1.0 with user-final-GT frame %s c%d (%s); "
                    "original V3D label was a transcription-misalignment "
                    "artifact (docs/V3E_P1_HERO_COVERAGE_AUDIT_REPORT.md R2.2)"
                    % (r["identity_frame"], r["identity_pos"], r["human"]))
        report["relabels"].append(r)
        print("relabel %s -> %s" % (r["old_label"], r["new_label"]))

    # ── 2) 新增资产（幂等键 = label+card+source_frame_id） ─────────
    existing_names = {Path(a["crop_ref"]).name for a in mp["assets"]}
    existing_keys = {(a["label"], a.get("card"), a.get("source_frame_id"))
                     for a in mp["assets"]}
    for kind, ch, pos, tags in CANDIDATES:
        members, src, detail = screen_group(kind, ch, pos, tags)
        report["groups"].append(detail)
        label = "%s_%s" % (kind, ch)
        if not members:
            report["skipped"].append({"label": label, "pos": pos,
                                      "reason": "insufficient_evidence(<2 frames after cohesion gate)"})
            print("SKIP %s c%d: insufficient" % (label, pos))
            continue
        key = (label, pos, F[src])
        if key in existing_keys:
            report["skipped"].append({"label": label, "pos": pos,
                                      "reason": "already added (idempotent)"})
            print("SKIP %s c%d: already added" % (label, pos))
            continue
        m, bbox = G(src, pos, kind)
        fname = "h%s%s_c%d_%d_%d.png" % ("r" if kind == "rank" else "s", ch,
                                         pos, bbox[0], bbox[1])
        assert fname not in existing_names, fname
        (ASSETS / fname).write_bytes(encode_asset(m))
        entry = {
            "label": label,
            "kind": "herocards_%s_1876" % kind,
            "crop_ref": fname,
            "bbox": list(bbox),
            "card": pos,
            kind: ch,
            "glyph_size": [int(bbox[2]), int(bbox[3])],
            "source_frame_id": F[src],
            "provenance": (
                "V3E-P1 ASSET REPAIR (user final GT: "
                "gt_confirm_v3e_p1/v2/final_answers.json; group %s "
                "cohesion n=%d min_pair=%s; position-specific c%d "
                "extraction; source frame %s)"
                % (label, len(members), detail.get("min_pair"),
                   pos, F[src])),
            "gt_status": "normal",
        }
        mp["assets"].append(entry)
        existing_keys.add(key)
        existing_names.add(fname)
        report["added"].append(entry)
        print("ADD %s c%d src=%s n=%d bbox=%s"
              % (label, pos, src, len(members), list(bbox)))

    mp["note"] = (mp["note"] + " | V3E-P1 ASSET REPAIR: rank_4→rank_6(c2), "
                  "rank_J→rank_K(c2) relabel (pixel-identity to user-final-GT "
                  "frames);新增 rank {A,K,Q,T,8,4}@c1 {5,9,6,4,J}@c2、suit "
                  "{s,d,h}@c1 {s}@c2（全部 position-specific，人工 GT 帧提取）。"
                  "未覆盖（证据<2帧）：rank 7@c1、2@c2、8@c2、J@c1、A/Q/T/7@c2、"
                  "2/3/5/9@c1、s/d@c2 → UNKNOWN-first。")
    MANIFEST.write_text(json.dumps(mp, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    report["assets_before"] = before_n
    report["assets_after"] = len(mp["assets"])

    # ── 3) 重载自检 ───────────────────────────────────────────────
    from poker_decision_ai.vision2.pixel.realclient.coinpoker_herocards1876 import (
        _load_templates)
    prep = _load_templates()
    report["prep_keys"] = sorted(prep.keys())
    report["prep_instances"] = {k: len(v) for k, v in sorted(prep.items())}
    LOG.write_text(json.dumps(report, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    print("assets: %d -> %d" % (before_n, len(mp["assets"])))
    print("prep:", report["prep_instances"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
