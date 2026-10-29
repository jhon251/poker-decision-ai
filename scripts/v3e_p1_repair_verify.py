"""V3E-P1 ASSET REPAIR — 验证①manifest/provenance 检查 + ②GT round-trip。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from poker_decision_ai.vision2.pixel.realclient.coinpoker_herocards1876 import (
    HeroCardsRecognizer1876, _load_templates)
from poker_decision_ai.vision2_platform.png_decode import decode_png

ASSETS = Path("data/vision2/coinpoker_assets_v1876herocards")
EV = Path("data/vision2/coinpoker_evidence")
OUT = Path("logs/v3e_p1_repair_verify.json")

GT_FRAMES = {
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


def check_manifest():
    mp = json.loads((ASSETS / "manifest.json").read_text(encoding="utf-8"))
    problems = []
    seen = set()
    for a in mp["assets"]:
        lab = a.get("label", "")
        kind, _, ch = lab.partition("_")
        p = ASSETS / Path(a["crop_ref"]).name
        if not p.exists():
            problems.append("%s: missing crop %s" % (lab, a["crop_ref"]))
            continue
        if not a.get("source_frame_id"):
            problems.append("%s: no source_frame_id" % lab)
        if not a.get("provenance"):
            problems.append("%s: no provenance" % lab)
        if kind == "rank" and a.get("rank") != ch:
            problems.append("%s: rank field mismatch" % lab)
        if kind == "suit" and a.get("suit") != ch:
            problems.append("%s: suit field mismatch" % lab)
        cpos = "_c%d_" % a.get("card", 0)
        if cpos not in a["crop_ref"]:
            problems.append("%s: card/position mismatch in %s"
                            % (lab, a["crop_ref"]))
        w, h, rgba = decode_png(p.read_bytes())
        arr = np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)
        ys, xs = np.where(arr[..., 3] > 128)
        gs = a.get("glyph_size") or []
        if gs and [int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)] != list(gs):
            problems.append("%s: glyph_size %s != tight %s"
                            % (lab, gs, [int(xs.max() - xs.min() + 1),
                                         int(ys.max() - ys.min() + 1)]))
        key = (lab, a.get("card"), a.get("source_frame_id"))
        if key in seen:
            problems.append("duplicate entry %s" % (key,))
        seen.add(key)
    return {"n_assets": len(mp["assets"]), "problems": problems}, mp


def round_trip():
    rec = HeroCardsRecognizer1876()
    prep = _load_templates()
    rows = []
    n_ok = n_unknown = n_fail = 0
    for f, gt in GT_FRAMES.items():
        w, h, rgba = decode_png((EV / f).read_bytes())
        arr = np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)
        out = rec.recognize(arr)
        if out["status"] == "OK":
            if out["value"] == list(gt):
                n_ok += 1
                rows.append({"frame": f, "gt": list(gt), "got": out["value"],
                             "result": "PASS"})
            else:
                n_fail += 1
                rows.append({"frame": f, "gt": list(gt), "got": out["value"],
                             "result": "FAIL_VALUE_MISMATCH",
                             "evidence": out.get("cards")})
        else:
            n_unknown += 1
            rows.append({"frame": f, "gt": list(gt), "got": None,
                         "status": "UNKNOWN", "reason": out.get("reason"),
                         "cards": out.get("cards")})
    return {"n_ok": n_ok, "n_unknown": n_unknown, "n_fail": n_fail,
            "rows": rows}


def main() -> int:
    man, mp = check_manifest()
    rt = round_trip()
    out = {"manifest_check": man, "round_trip": rt,
           "prep": {k: len(v) for k, v in sorted(_load_templates().items())}}
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    print("manifest assets:", man["n_assets"],
          "problems:", man["problems"] or "none")
    print("round-trip: OK=%d UNKNOWN=%d FAIL=%d"
          % (rt["n_ok"], rt["n_unknown"], rt["n_fail"]))
    for r in rt["rows"]:
        if r.get("result") != "PASS":
            print("  %-46s gt=%s got=%s %s" % (
                r["frame"][:46], r.get("gt"), r.get("got"),
                r.get("reason") or ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
