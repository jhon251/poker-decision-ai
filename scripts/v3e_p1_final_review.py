"""V3E-P1 终审汇总（dormant）——仅在用户终审答案存在时运行。

输入：gt_confirm_v3e_p1/v2/final_answers.json
  {
    "V2-04": {"c1": "Q", "c2": "6h"},      // 牌面 / "Q"(仅rank) / "unsure" / "not_visible"
    "V2-06": "✓",                           // 整帧确认与 Round-2 初读一致
    ...
  }

行为（只读 + 仅写 logs/v3e_p1_final_review.json）：
  1) 人工 GT 汇总： definite 标签入 GT；unsure/not_visible 如实排除；
  2) 冻结模板身份判定：每个 manifest 资产 glyph 与全部"人工已确认卡"
     的同_kind glyph 算 NCC；仅当与某人工帧像素全同(≥0.999)时，
     以该帧人工标签确立模板身份 → CORRECT / WRONG(asset=X 实为=Y) /
     UNVERIFIED（无人工全同帧，如实标注）。
  3) 不修改任何模板/manifest/detector；不做任何机器反推标签。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from v3e_p1_hero_audit import EXT_WIN, EV, glyph, ncc_pair
from poker_decision_ai.vision2_platform.png_decode import decode_png

ANSWERS = Path("gt_confirm_v3e_p1/v2/final_answers.json")
OUT = Path("logs/v3e_p1_final_review.json")
ASSETS = Path("data/vision2/coinpoker_assets_v1876herocards")

ITEM_FRAME = {
    "V2-01": "POT_CHANGED_250_1787992907809_full.png",
    "V2-02": "POT_CHANGED_524_1787993575599_full.png",
    "V2-03": "SCENE_CHANGED_429_1787993345715_full.png",
    "V2-04": "POT_CHANGED_512_1787993546817_full.png",
    "V2-05": "POT_CHANGED_606_1787993775320_full.png",
    "V2-06": "TABLE_CHANGED_18_1787992293441_full.png",
    "V2-07": "POT_CHANGED_404_1787993286628_full.png",
    "V2-08": "POT_CHANGED_16_1787995303501_full.png",
    "V2-09": "POT_CHANGED_208_1787992806955_full.png",
    "V2-10": "SCENE_CHANGED_6_1787992258912_full.png",
    "V2-11": "POT_CHANGED_708_1787994016446_full.png",
    "V2-12": "POT_CHANGED_293_1787993018309_full.png",
    "V2-13": "POT_CHANGED_535_1787993603723_full.png",
    "V2-14": "POT_CHANGED_454_1787993404728_full.png",
    "V2-15": "POT_CHANGED_33_1787995356823_full.png",
    "V2-16": "POT_CHANGED_420_1787993325856_full.png",
    "V2-17": "POT_CHANGED_353_1787993164702_full.png",
    "V2-18": "POT_CHANGED_47_1787995398974_full.png",
    "V2-19": "POT_CHANGED_181_1787992731553_full.png",
    "V2-20": "POT_CHANGED_290_1787993011134_full.png",
}
ROUND2 = {
    "POT_CHANGED_250_1787992907809_full.png": ("As", "3c"),
    "POT_CHANGED_524_1787993575599_full.png": ("Kd", "Jc"),
    "SCENE_CHANGED_429_1787993345715_full.png": ("Qh", "9s"),
    "POT_CHANGED_512_1787993546817_full.png": ("Td", "6h"),
    "POT_CHANGED_606_1787993775320_full.png": ("7c", "2c"),
    "TABLE_CHANGED_18_1787992293441_full.png": ("6c", "3h"),
    "POT_CHANGED_404_1787993286628_full.png": ("As", "5h"),
    "POT_CHANGED_16_1787995303501_full.png": ("Kc", "3h"),
    "POT_CHANGED_208_1787992806955_full.png": ("Ah", "5h"),
    "SCENE_CHANGED_6_1787992258912_full.png": ("8s", "8h"),
    "POT_CHANGED_708_1787994016446_full.png": ("Ks", "6c"),
    "POT_CHANGED_293_1787993018309_full.png": ("As", "Kc"),
    "POT_CHANGED_535_1787993603723_full.png": ("Qd", "Jc"),
    "POT_CHANGED_454_1787993404728_full.png": ("8c", "3d"),
    "POT_CHANGED_33_1787995356823_full.png": ("4s", "4h"),
    "POT_CHANGED_420_1787993325856_full.png": ("Qh", "9s"),
    "POT_CHANGED_353_1787993164702_full.png": ("Th", "5h"),
    "POT_CHANGED_47_1787995398974_full.png": ("4s", "4h"),
    "POT_CHANGED_181_1787992731553_full.png": ("Jd", "9h"),
    "POT_CHANGED_290_1787993011134_full.png": ("As", "Kc"),
}
R_SUITS = set("cdhs")


def norm_card(tok):
    """'10d'/'Qd'/'Q'/'unsure'/None → (rank|None, suit|None, status)。"""
    if tok is None:
        return None, None, "missing"
    t = str(tok).strip()
    if t in ("unsure", "not_visible", "?", ""):
        return None, None, t or "missing"
    t = t.replace("10", "T").upper()
    if len(t) == 2 and t[0] in "23456789TJQKA" and t[1] in "CDHS":
        return t[0], t[1].lower(), "ok"
    if len(t) == 1 and t in "23456789TJQKA":
        return t, None, "ok"          # 仅 rank 确认
    if len(t) == 1 and t in "CDHS":
        return None, t.lower(), "ok"  # 仅 suit 确认
    return None, None, "invalid"


def main() -> int:
    if not ANSWERS.exists():
        print("DORMANT: 未找到 final_answers.json —— 等待用户终审结果。")
        print("回填格式见 gt_confirm_v3e_p1/v2/FINAL_REVIEW_SHEET.md")
        return 0
    raw = json.loads(ANSWERS.read_text(encoding="utf-8"))

    # ── 1) 人工 GT 汇总 ────────────────────────────────────────────
    gt = {}
    for item, ans in raw.items():
        frame = ITEM_FRAME.get(item)
        if frame is None:
            gt[item] = {"error": "unknown item"}
            continue
        r2 = ROUND2[frame]
        if isinstance(ans, str) and ans.strip() in ("✓", "ok", "same", "一致"):
            ans = {"c1": r2[0], "c2": r2[1]}
        entry = {"frame": frame}
        for ci_key, ci in (("c1", 1), ("c2", 2)):
            rk, st, status = norm_card(ans.get(ci_key) if isinstance(ans, dict) else None)
            if status == "missing":
                # 未作答的卡：不继承、如实标 missing（用户终审为唯一来源）
                entry[ci_key] = {"status": "missing"}
                continue
            if status != "ok":
                entry[ci_key] = {"status": status}
                continue
            rec = {"status": "ok", "rank": rk, "suit": st}
            if rk is None or st is None:
                inh_r, inh_s = (r2[ci - 1][0], r2[ci - 1][1]) if False else (None, None)
                rec["partial"] = True
            entry[ci_key] = rec
        gt[item] = entry

    # 收集全部已确认卡（rank 与 suit 分开：partial 卡只入已确认侧）
    cards = []   # (frame, pos, kind, label_char, glyph_mask)
    glyph_cache = {}
    for item, entry in gt.items():
        frame = entry.get("frame")
        if not frame:
            continue
        for ci_key, ci in (("c1", 1), ("c2", 2)):
            c = entry.get(ci_key) or {}
            if c.get("status") != "ok":
                continue
            for kind, ch_key in (("rank", "rank"), ("suit", "suit")):
                ch = c.get(ch_key)
                if not ch:
                    continue
                key = (frame, ci, kind)
                if key not in glyph_cache:
                    w, h, rgba = decode_png((EV / frame).read_bytes())
                    arr = np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)
                    m, _bbox = glyph(arr, EXT_WIN[ci][kind])
                    glyph_cache[key] = m
                m = glyph_cache[key]
                if m is not None:
                    cards.append({"item": item, "frame": frame, "pos": ci,
                                  "kind": kind, "label": ch, "glyph": m})

    # ── 2) 冻结模板身份判定 ────────────────────────────────────────
    mp = json.loads((ASSETS / "manifest.json").read_text(encoding="utf-8"))
    verdicts = []
    for a in mp.get("assets", []):
        lab = a.get("label", "")
        kind = "rank" if lab.startswith("rank_") else \
            "suit" if lab.startswith("suit_") else None
        ch = lab.split("_", 1)[-1] if lab else ""
        p = ASSETS / Path(a["crop_ref"]).name
        if kind is None or not p.exists():
            verdicts.append({"asset": lab, "verdict": "SKIP"})
            continue
        w, h, rgba = decode_png(p.read_bytes())
        arr = np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)
        g = (arr[..., 3] > 128).astype(np.uint8)
        ys, xs = np.where(g > 0)
        if len(xs) < 4:
            verdicts.append({"asset": lab, "verdict": "EMPTY"})
            continue
        tg = g[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        best, best_card = 0.0, None
        for c in cards:
            if c["kind"] != kind:
                continue
            v = ncc_pair(tg, c["glyph"])
            if v > best:
                best, best_card = v, c
        if best_card is None or best < 0.999:
            verdicts.append({"asset": lab, "verdict": "UNVERIFIED",
                             "best_ncc": round(best, 3),
                             "best_card": None if not best_card else
                             (best_card["frame"], best_card["pos"],
                              best_card["label"])})
            continue
        ok = (best_card["label"] == ch)
        verdicts.append({
            "asset": lab, "verdict": "CORRECT" if ok else "WRONG",
            "identity_frame": best_card["frame"], "identity_pos": best_card["pos"],
            "human_label": best_card["label"], "ncc": round(best, 3),
            "note": "" if ok else f"asset 标为 '{ch}'，人工 GT 帧实为 '{best_card['label']}'"})

    summary = {
        "human_gt": gt,
        "confirmed_cards": [{"item": c["item"], "frame": c["frame"],
                             "pos": c["pos"], "kind": c["kind"],
                             "label": c["label"]} for c in cards],
        "template_verdicts": verdicts,
        "wrong_templates": [v["asset"] for v in verdicts
                            if v["verdict"] == "WRONG"],
        "note": "汇总仅为报告；模板/manifest 修改等待用户明确授权。",
    }
    OUT.write_text(json.dumps(summary, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    print(json.dumps({"wrong_templates": summary["wrong_templates"],
                      "verdicts": verdicts, "n_confirmed_cards": len(cards)},
                     ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
