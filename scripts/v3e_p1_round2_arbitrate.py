"""V3E-P1 round-2 — 今日人工判读的像素诊断与冲突仲裁（只读；仅写 logs/）。

背景：V3D 轮 CSV 标注被证实系统性错位（多帧共享同一标注、
CSV(X)=今日(Y)），故本轮以用户今日 pair-sheet 判读为标注源，
用冻结模板 + 同位置跨帧 glyph NCC 做机器诊断（仅辅助，不产生 GT）。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from v3e_p1_hero_audit import (EXT_WIN, EV, SUIT_WIN, glyph, ncc_pair,
                               _red_frac)
from poker_decision_ai.vision2.pixel.realclient.coinpoker_herocards1876 import (
    HeroCardsRecognizer1876, _load_templates, _match_glyph)
from poker_decision_ai.vision2_platform.png_decode import decode_png

OUT = Path("logs/v3e_p1_round2_arbitration.json")

# 今日人工判读（sheet 顺序 = 清单顺序；CROP_ERROR = 476）
READINGS = {
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

R_SUITS = "cdhs"
RED_SUITS = set("hd")


def load_all():
    frames = {}
    for f in READINGS:
        w, h, rgba = decode_png((EV / f).read_bytes())
        arr = np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)
        d = {"arr": arr, "cards": {}}
        for ci in (1, 2):
            cd = {"red_frac": round(float(_red_frac(arr, SUIT_WIN[ci])), 3)}
            for kind, key in (("rank", "r"), ("suit", "s")):
                m, bbox = glyph(arr, EXT_WIN[ci][kind])
                cd[key] = m
                cd[key + "_bbox"] = bbox
            d["cards"][ci] = cd
        frames[f] = d
    return frames


def main() -> int:
    frames = load_all()
    prep = _load_templates()
    rec = HeroCardsRecognizer1876()
    out = {"readings": {}, "arbitration": {}}

    # ── 1) 每卡机器诊断：当前 detector 输出 + glyph↔模板得分 + 颜色 ──
    for f, (c1, c2) in READINGS.items():
        arr = frames[f]["arr"]
        det = rec.recognize(arr)
        entry = {"user_now": [c1, c2], "detector": det.get("value"),
                 "det_reason": det.get("reason"), "cards": {}}
        for ci, lab in ((1, c1), (2, c2)):
            cd = frames[f]["cards"][ci]
            info = {"user": lab, "red_frac": cd["red_frac"],
                    "color_ok": (cd["red_frac"] >= 0.15)
                    if lab[1] in RED_SUITS else (cd["red_frac"] <= 0.6)}
            for kind, key, alphabet in (("rank", "r", None),
                                        ("suit", "s", R_SUITS)):
                m = cd[key]
                if m is None:
                    info[kind] = {"no_glyph": True}
                    continue
                ch, sc, mg = _match_glyph(m, prep, alphabet)
                info[kind] = {"pixel_says": ch, "score": round(sc, 3) if sc else None,
                              "margin": round(mg, 3) if mg else None}
            entry["cards"][ci] = info
        out["readings"][f] = entry

    def g(f, ci, kind):
        m = frames[f]["cards"][ci][kind]
        return m

    # ── 2) 冲突仲裁 NCC（同位置跨帧 / 同帧模板） ─────────────────────
    arb = out["arbitration"]

    def cross(name, a, b):
        if a is None or b is None:
            arb[name] = "missing_glyph"
        else:
            arb[name] = round(ncc_pair(a, b), 3)

    # 290 c1：今日 As vs V3D 4s —— 与 A 锚(250/404)和 4 锚(33/47)比
    for anchor in ("POT_CHANGED_250_1787992907809_full.png",
                   "POT_CHANGED_404_1787993286628_full.png"):
        cross(f"290c1_vs_A_{anchor[:18]}", g("POT_CHANGED_290_1787993011134_full.png", 1, "r"), g(anchor, 1, "r"))
    for anchor in ("POT_CHANGED_33_1787995356823_full.png",
                   "POT_CHANGED_47_1787995398974_full.png"):
        cross(f"290c1_vs_4_{anchor[:18]}", g("POT_CHANGED_290_1787993011134_full.png", 1, "r"), g(anchor, 1, "r"))

    # 708 c2：今日 6c vs V3D rank_4 模板 —— 与 4 模板 / 6 锚 / 3 锚 / K 锚比
    g708 = g("POT_CHANGED_708_1787994016446_full.png", 2, "r")
    g606c1 = g("POT_CHANGED_606_1787993775320_full.png", 1, "r")   # 今日 7c
    # 6 锚：TABLE_CHANGED_18 c1 = 6c（回归 GT）
    cross("708c2_vs_6anchor_18c1", g708, g("TABLE_CHANGED_18_1787992293441_full.png", 1, "r"))
    cross("708c2_vs_3_250c2", g708, g("POT_CHANGED_250_1787992907809_full.png", 2, "r"))
    cross("708c2_vs_3_454c2", g708, g("POT_CHANGED_454_1787993404728_full.png", 2, "r"))
    cross("708c2_vs_K_293c2", g708, g("POT_CHANGED_293_1787993018309_full.png", 2, "r"))
    # 4 模板 vs 6 锚（模板本身是否像 6）
    t4 = [p[0] for p in prep.get("4", [])]
    g18c1 = g("TABLE_CHANGED_18_1787992293441_full.png", 1, "r")
    arb["tmpl4_vs_6anchor"] = round(max(ncc_pair(t, g18c1) for t in t4), 3) if t4 else None
    arb["708c2_vs_tmpl4"] = round(max(ncc_pair(g708, t) for t in t4), 3) if t4 else None

    # SCENE6 c2：今日 8h vs V3D K —— 与 K 锚(293c2)比
    cross("scene6c2_vs_K_293c2", g("SCENE_CHANGED_6_1787992258912_full.png", 2, "r"),
          g("POT_CHANGED_293_1787993018309_full.png", 2, "r"))

    # 512 c1：今日 Td(10) vs V3D Qd —— 与 Q 锚(429/420 c1)和 T 锚(353 c1)比
    g512 = g("POT_CHANGED_512_1787993546817_full.png", 1, "r")
    for anchor, tag in (("SCENE_CHANGED_429_1787993345715_full.png", "Q_429"),
                        ("POT_CHANGED_420_1787993325856_full.png", "Q_420"),
                        ("POT_CHANGED_353_1787993164702_full.png", "T_353")):
        cross(f"512c1_vs_{tag}", g512, g(anchor, 1, "r"))

    # 429 c1 vs 420 c1（今日都读 Qh → 应≈1.0）
    cross("429c1_vs_420c1", g("SCENE_CHANGED_429_1787993345715_full.png", 1, "r"),
          g("POT_CHANGED_420_1787993325856_full.png", 1, "r"))

    # 16/208 互换检查（今日 16=Kc, 208=Ah；V3D 反转）
    cross("16c1_vs_208c1", g("POT_CHANGED_16_1787995303501_full.png", 1, "r"),
          g("POT_CHANGED_208_1787992806955_full.png", 1, "r"))

    # 606 c1：今日 7c —— 与 A 锚（250/404）与 6 模板与 6 锚(18c1)比
    for anchor in ("POT_CHANGED_250_1787992907809_full.png",
                   "POT_CHANGED_404_1787993286628_full.png"):
        cross(f"606c1_vs_A_{anchor[:18]}", g606c1, g(anchor, 1, "r"))
    t6 = [p[0] for p in prep.get("6", [])]
    arb["606c1_vs_tmpl6"] = round(max(ncc_pair(g606c1, t) for t in t6), 3) if t6 else None
    cross("606c1_vs_6anchor_18c1", g606c1, g18c1)

    # suit 侧：今日 ♠ 锚(250c1 s) vs ♦ 锚(524c1 d) vs ♥ 锚(429c1 h) 互距
    s_250 = g("POT_CHANGED_250_1787992907809_full.png", 1, "s")
    s_524 = g("POT_CHANGED_524_1787993575599_full.png", 1, "s")
    s_429 = g("SCENE_CHANGED_429_1787993345715_full.png", 1, "s")
    s_606 = g("POT_CHANGED_606_1787993775320_full.png", 1, "s")
    arb["suit_spade250_vs_diam524"] = round(ncc_pair(s_250, s_524), 3)
    arb["suit_spade250_vs_heart429"] = round(ncc_pair(s_250, s_429), 3)
    arb["suit_diam524_vs_heart429"] = round(ncc_pair(s_524, s_429), 3)
    arb["suit_club606_vs_spade250"] = round(ncc_pair(s_606, s_250), 3)
    tc = [p[0] for p in prep.get("c", [])]
    arb["suit_club606_vs_tmplc"] = round(max(ncc_pair(s_606, t) for t in tc), 3)
    arb["suit_spade250_vs_tmplc"] = round(max(ncc_pair(s_250, t) for t in tc), 3)
    arb["suit_diam524_vs_tmplc"] = round(max(ncc_pair(s_524, t) for t in tc), 3)
    th = [p[0] for p in prep.get("h", [])]
    arb["suit_heart429_vs_tmplh"] = round(max(ncc_pair(s_429, t) for t in th), 3) if th else None

    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(out["arbitration"], ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
