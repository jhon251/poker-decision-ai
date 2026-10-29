"""P25-C — allin 精确：每座卡区（anchor±150,140）白卡面 + equity pill 带状几何。

pill = 饱和红 (r>150,g<95,b<95) 或 teal (g>110,g>1.6r,g>1.3b,l>90)
      宽>=80 高 20-90 的圆角带 + 带内白字>30px。
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))
from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402

EV = Path("data/vision2/coinpoker_evidence")

ANCH = {"S1": (255, 310), "S2": (918, 210), "S3": (1578, 320),
        "S4": (218, 868), "S5": (1618, 868), "S6": (945, 1160)}
FRAMES = {
    "f01": "BUTTON_CHANGED_167_1787992682749_full.png",
    "f02": "POT_CHANGED_12_1787995290794_full.png",
    "f03": "POT_CHANGED_102_1787992505794_full.png",
    "f04": "POT_CHANGED_13_1787992279195_full.png",
    "f05": "POT_CHANGED_139_1787992592538_full.png",
    "f06": "BUTTON_CHANGED_598_1787993753587_full.png",
}
GT = {"f01": {"S3": "away", "S5": "folded", "S6": "folded"},
      "f02": {"S2": "folded", "S3": "away", "S4": "folded"},
      "f03": {"S1": "folded", "S2": "allin", "S3": "folded", "S4": "folded",
              "S5": "folded"},
      "f04": {"S1": "allin", "S2": "allin", "S3": "folded", "S4": "folded",
              "S5": "folded", "S6": "folded"},
      "f05": {"S2": "folded", "S3": "disconnected", "S6": "folded"},
      "f06": {"S3": "folded", "S6": "folded"}}


def load(fp):
    w, h, rgba = decode_png((EV / fp).read_bytes())
    return np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)


def probe(arr, ax, ay):
    x0, y0 = max(0, ax - 150), max(0, ay - 140)
    x1, y1 = min(arr.shape[1], ax + 150), min(arr.shape[0], ay + 140)
    roi = arr[y0:y1, x0:x1]
    r = roi[..., 0].astype(np.int16); g = roi[..., 1].astype(np.int16)
    b = roi[..., 2].astype(np.int16)
    l = r * 0.299 + g * 0.587 + b * 0.114
    white = ((l > 190) & (r > 180) & (g > 180) & (b > 180)).astype(np.uint8)
    red = ((r > 150) & (g < 95) & (b < 95)).astype(np.uint8)
    teal = ((g > 110) & (g > r * 1.6) & (g > b * 1.3)
            & (l > 90)).astype(np.uint8)
    # 白卡面：white 最大行带 >= 80px 宽
    rows = white.sum(axis=1)
    band_ok = int((rows >= 80).sum())
    white_n = int(white.sum())
    # pill：对 red+teal 找宽带（配合白字）
    pill = None
    for nm, m in (("red", red), ("teal", teal)):
        rowd = m.sum(axis=1)
        cand_rows = np.where(rowd >= 40)[0]
        if not len(cand_rows):
            continue
        # 取连续最长
        runs, run = [], None
        prev = None
        for rw in cand_rows:
            if run is None:
                run = [rw, rw]
            elif rw - run[1] <= 3:
                run[1] = rw
            else:
                runs.append(run); run = [rw, rw]
        if run:
            runs.append(run)
        for (a, bb) in runs:
            bh = bb - a + 1
            if not (20 <= bh <= 90):
                continue
            band = m[a:bb + 1]
            cold = band.sum(axis=0)
            bw2 = int((cold >= 8).sum())
            if bw2 < 80:
                continue
            # 白字 inside
            seg = white[a:bb + 1]
            wtxt = int(seg.sum())
            if wtxt < 30:
                continue
            if pill is None or bh * bw2 > pill[0] * pill[1]:
                pill = (bw2, bh, nm, wtxt)
    return {"white_n": white_n, "band_ok": band_ok, "pill": pill}


print(f"{'':5s} {'seat':4s} {'GT':8s} {'whiteN':>7s} {'band':>5s} {'pill'}")
rows = []
for fr, fp in FRAMES.items():
    arr = load(fp)
    for seat, (ax, ay) in ANCH.items():
        gt = GT.get(fr, {}).get(seat, "-")
        d = probe(arr, ax, ay)
        p = d["pill"]
        ps = f"{p[0]}x{p[1]}{p[2]}/w{p[3]}" if p else "-"
        rows.append((fr, seat, gt, d["white_n"], d["band_ok"], ps))
        print(f"{fr:5s} {seat:4s} {gt:8s} {d['white_n']:7d} "
              f"{d['band_ok']:5d} {ps}")
