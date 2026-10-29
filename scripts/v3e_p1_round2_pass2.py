"""V3E-P1 round-2 pass2 — 扩展像素仲裁（只读；仅写 logs/）。

验证：a) "7s"聚类其余成员是否实为 A；b) 冻结 rank_4/rank_J 模板与
今日锚的一致性；c) Q/T 锚区分度；d) K/8/♦/♠ 锚内聚；e) c2 '3' 模板。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from v3e_p1_hero_audit import EXT_WIN, EV, glyph, ncc_pair
from poker_decision_ai.vision2.pixel.realclient.coinpoker_herocards1876 import (
    _load_templates)
from poker_decision_ai.vision2_platform.png_decode import decode_png

OUT = Path("logs/v3e_p1_round2_pass2.json")

FRAMES = [
    "POT_CHANGED_250_1787992907809_full.png", "POT_CHANGED_524_1787993575599_full.png",
    "SCENE_CHANGED_429_1787993345715_full.png", "POT_CHANGED_512_1787993546817_full.png",
    "POT_CHANGED_606_1787993775320_full.png", "TABLE_CHANGED_18_1787992293441_full.png",
    "POT_CHANGED_404_1787993286628_full.png", "POT_CHANGED_16_1787995303501_full.png",
    "POT_CHANGED_208_1787992806955_full.png", "SCENE_CHANGED_6_1787992258912_full.png",
    "POT_CHANGED_708_1787994016446_full.png", "POT_CHANGED_293_1787993018309_full.png",
    "POT_CHANGED_535_1787993603723_full.png", "POT_CHANGED_454_1787993404728_full.png",
    "POT_CHANGED_33_1787995356823_full.png", "POT_CHANGED_420_1787993325856_full.png",
    "POT_CHANGED_353_1787993164702_full.png", "POT_CHANGED_47_1787995398974_full.png",
    "POT_CHANGED_181_1787992731553_full.png", "POT_CHANGED_290_1787993011134_full.png",
    # "7s" 聚类其余成员（V3D rank_7_c1）
    "POT_CHANGED_295_1787993023273_full.png", "POT_CHANGED_312_1787993063870_full.png",
    "SCENE_CHANGED_405_1787993289258_full.png", "TABLE_CHANGED_298_1787993031027_full.png",
    "TABLE_CHANGED_318_1787993079007_full.png", "TABLE_CHANGED_406_1787993291442_full.png",
]

CACHE = {}
PREP = _load_templates()


def G(f, ci, kind):
    key = (f, ci, kind)
    if key not in CACHE:
        w, h, rgba = decode_png((EV / f).read_bytes())
        arr = np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)
        win = EXT_WIN[ci]["rank" if kind == "r" else "suit"]
        m, bbox = glyph(arr, win)
        CACHE[key] = m
    return CACHE[key]


def T(ch):
    return [p[0] for p in PREP.get(ch, [])]


def main() -> int:
    out = {}

    def x(name, a, b):
        out[name] = None if a is None or b is None else round(ncc_pair(a, b), 3)

    def xt(name, a, ch):
        ts = T(ch)
        out[name] = None if a is None or not ts else \
            round(max(ncc_pair(a, t) for t in ts), 3)

    A_250 = G(FRAMES[0], 1, "r")     # 今日 As
    # a) "7s" 聚类成员 vs A 锚
    for f in FRAMES[20:]:
        x("7cluster[%s]_vs_A250" % f.split("_")[2], G(f, 1, "r"), A_250)
    # b) 冻结模板一致性
    J = T("J")
    for f, tag in ((FRAMES[12], "535c2"), (FRAMES[1], "524c2")):
        x("tmplJ_vs_%s" % tag, max(J, key=lambda t: ncc_pair(G(f, 2, "r"), t))
          if J else None, G(f, 2, "r")) if False else None
        out["tmplJ_vs_%s" % tag] = round(max(ncc_pair(t, G(f, 2, "r"))
                                             for t in J), 3) if J else None
    F4 = T("4")
    g708 = G(FRAMES[10], 2, "r")
    out["tmpl4_vs_708c2"] = round(max(ncc_pair(t, g708) for t in F4), 3) if F4 else None
    for f, tag in ((FRAMES[14], "33c1"), (FRAMES[17], "47c1")):
        out["tmpl4_vs_%s(4s today)" % tag] = round(
            max(ncc_pair(t, G(f, 1, "r")) for t in F4), 3) if F4 else None
    for f, tag in ((FRAMES[15], "420c2"), (FRAMES[2], "429c2")):
        out["tmpl4_vs_%s(9s today)" % tag] = round(
            max(ncc_pair(t, G(f, 2, "r")) for t in F4), 3) if F4 else None
    # c) Q/T 锚区分度
    x("429c1(Q)_vs_353c1(T)", G(FRAMES[2], 1, "r"), G(FRAMES[16], 1, "r"))
    x("512c1_vs_535c1(Qd)", G(FRAMES[3], 1, "r"), G(FRAMES[12], 1, "r"))
    # d) K / 8 / suit 内聚
    x("16c1(Kc)_vs_524c1(Kd)", G(FRAMES[7], 1, "r"), G(FRAMES[1], 1, "r"))
    x("16c1(Kc)_vs_708c1(Ks)", G(FRAMES[7], 1, "r"), G(FRAMES[10], 1, "r"))
    x("208c1(Ah)_vs_250c1(As)", G(FRAMES[8], 1, "r"), A_250)
    x("293c1(As)_vs_250c1", G(FRAMES[11], 1, "r"), A_250)
    x("293c1_vs_353c1(T)", G(FRAMES[11], 1, "r"), G(FRAMES[16], 1, "r"))
    x("454c1(8c)_vs_scene6c1(8s)", G(FRAMES[13], 1, "r"), G(FRAMES[9], 1, "r"))
    x("scene6c1(8s)_vs_353c1(T)", G(FRAMES[9], 1, "r"), G(FRAMES[16], 1, "r"))
    # e) c2 '3' 模板 vs 今日 3 锚
    for f, tag in ((FRAMES[0], "250c2(3c)"), (FRAMES[7], "16c2(3h)"),
                   (FRAMES[13], "454c2(3d)")):
        xt("tmpl3_vs_%s" % tag, G(f, 2, "r"), "3")
    # c2 '9' 一致性（429/420 今日都 9s）
    x("429c2(9s)_vs_420c2(9s)", G(FRAMES[2], 2, "r"), G(FRAMES[15], 2, "r"))
    # f) suit 锚内聚
    x("diam524s_vs_diam535s(Qd)", G(FRAMES[1], 1, "s"), G(FRAMES[12], 1, "s"))
    x("heart429s_vs_heart420s", G(FRAMES[2], 1, "s"), G(FRAMES[15], 1, "s"))
    x("spade708s(Ks)_vs_spade250s", G(FRAMES[10], 1, "s"), G(FRAMES[0], 1, "s"))
    x("spade_scene6s(8s)_vs_spade250s", G(FRAMES[9], 1, "s"), G(FRAMES[0], 1, "s"))
    x("diam181s(Jd)_vs_diam524s", G(FRAMES[18], 1, "s"), G(FRAMES[1], 1, "s"))
    x("spade_404s_vs_spade250s", G(FRAMES[6], 1, "s"), G(FRAMES[0], 1, "s"))
    # c2 '5'/'6'/'2'/'8'/'K' 锚自检
    x("5h_404c2_vs_5h_208c2", G(FRAMES[6], 2, "r"), G(FRAMES[8], 2, "r"))
    x("6h_512c2_vs_6c_708c2(xpos)", G(FRAMES[3], 2, "r"), g708)
    x("2c_606c2_vs_3c_250c2", G(FRAMES[4], 2, "r"), G(FRAMES[0], 2, "r"))
    xt("tmpl5_vs_404c2(5h)", G(FRAMES[6], 2, "r"), "5")
    xt("tmpl6_vs_512c2(6h)", G(FRAMES[3], 2, "r"), "6")
    xt("tmpl9_vs_429c2(9s)", G(FRAMES[2], 2, "r"), "9")
    xt("tmplh_vs_512c2(6h)", G(FRAMES[3], 2, "s"), "h")

    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
