"""P13-Action Foundation PRE-AUDIT v2 — 逐帧按钮签名 + 颜色直方图。

1) 每个 occurrence 内取按钮可见帧的均值签名（fold/cc/raise 分区）；
2) 加注按钮区域实际像素颜色直方图；
3) CC 区白字量：call vs check 的可区分性（全标注实例）。
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402

SINCE5 = 134324658401327108
SINCE6 = 134324688462724951
FOLD_REG = (1116, 1263, 1354, 1392)
CC_REG = (1354, 1263, 1629, 1392)
RAISE_REG = (1629, 1263, 1860, 1392)
ANNOT5 = {2: "fold", 3: "check", 4: "call", 6: "fold", 8: "fold", 9: "fold",
          10: "raise_pre", 11: "call", 12: "raise", 13: "allin", 14: "fold",
          15: "fold", 16: "fold", 17: "fold", 19: "call", 20: "fold",
          21: "fold", 22: "call", 23: "fold", 25: "fold"}
ANNOT6 = {1: "check", 2: "raise", 3: "raise", 4: "check"}


def load(p: Path) -> np.ndarray:
    w, h, rgba = decode_png(p.read_bytes())
    return np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)


def reg_sig(arr, reg):
    x0, y0, x1, y1 = reg
    r = arr[..., 0].astype(np.int16)
    g = arr[..., 1].astype(np.int16)
    b = arr[..., 2].astype(np.int16)
    lum = r * 0.299 + g * 0.587 + b * 0.114
    red = ((r > g + 50) & (r > b + 50) & (r > 110))[y0:y1, x0:x1].sum()
    orange = ((r > 140) & (g > 80) & (g < 190) & (b < 100))[y0:y1, x0:x1].sum()
    teal = ((g > r + 15) & (g >= b - 12) & (g > 50) & (g <= 135)
            )[y0:y1, x0:x1].sum()
    white = ((lum > 150) & (np.abs(r - g) < 40) & (np.abs(g - b) < 40)
             )[y0:y1, x0:x1].sum()
    return {"red": int(red), "orange": int(orange), "teal": int(teal),
            "white": int(white)}


def main() -> int:
    since_s = SINCE5 / 1e7 - 11644473600
    new = [p for p in sorted(Path("data/vision2/coinpoker_evidence")
                             .glob("*_full.png"))
           if p.stat().st_mtime >= since_s]

    def ts(p):
        return int(re.search(r"_(\d+)_full$", p.stem).group(1)) / 1000.0
    new.sort(key=ts)
    # 按钮可见帧（fold red>12000）→ 逐帧签名
    btn_frames = []
    for p in new:
        arr = load(p)
        r = arr[..., 0].astype(np.int16)
        g = arr[..., 1].astype(np.int16)
        b = arr[..., 2].astype(np.int16)
        m = ((r[1150:1400, :] > g[1150:1400, :] + 50)
             & (r[1150:1400, :] > b[1150:1400, :] + 50)
             & (r[1150:1400, :] > 110))
        if int(m.sum()) > 12000:
            btn_frames.append((ts(p), p))
    print("按钮可见帧总数:", len(btn_frames))
    # 加注按钮颜色直方图（第一帧按钮帧）
    if btn_frames:
        arr = load(btn_frames[0][1])
        sub = arr[1263:1392, 1629:1860]
        rs = sub[..., 0].astype(np.int16)
        gs = sub[..., 1].astype(np.int16)
        bs = sub[..., 2].astype(np.int16)
        lum = (rs * 0.299 + gs * 0.587 + bs * 0.114)
        # 高亮像素（按钮面，排除暗底）
        m2 = (lum > 70) & (np.abs(rs - gs) > 20)
        ys, xs = np.where(m2)
        vals = list(zip(rs[ys, xs], gs[ys, xs], bs[ys, xs]))
        from collections import Counter
        cnt = Counter((r // 20 * 20, g // 20 * 20, b // 20 * 20)
                      for r, g, b in vals)
        print("加注按钮面颜色直方图 top8:", cnt.most_common(8))
    # 逐 occurrence：按钮可见帧内 CC 区白字均值
    occ = []
    cur = None
    for t, p in btn_frames:
        if cur is None:
            cur = [t, t]
        else:
            if t - cur[1] < 20:
                cur[1] = t
            else:
                occ.append(cur)
                cur = [t, t]
    if cur:
        occ.append(cur)
    rows = []
    for n, (a, b) in enumerate(occ, 1):
        act = ANNOT5.get(n, ANNOT6.get(n, "?")) if n <= 27 else ANNOT6.get(n, "?")
        sigs = []
        for t, p in btn_frames:
            if a <= t <= b:
                sigs.append(reg_sig(load(p), CC_REG))
        if not sigs:
            continue
        avg_w = int(np.mean([s["white"] for s in sigs]))
        avg_t = int(np.mean([s["teal"] for s in sigs]))
        fr = reg_sig(load(btn_frames[0][1]), FOLD_REG)  # placeholder
        fold_sigs = [reg_sig(load(p), FOLD_REG)["red"]
                     for t, p in btn_frames if a <= t <= b]
        raise_sigs = [reg_sig(load(p), RAISE_REG)["orange"]
                      for t, p in btn_frames if a <= t <= b]
        rows.append({"occ": n, "action": act,
                     "cc_white": avg_w, "cc_teal": avg_t,
                     "fold_red": int(np.mean(fold_sigs)) if fold_sigs else 0,
                     "raise_orange": int(np.mean(raise_sigs))
                     if raise_sigs else 0})
    print("=== 每 occurrence（按钮可见帧均值）===")
    for r in rows:
        print(f"occ{r['occ']:02d} {r['action']:10s} fold_red={r['fold_red']:6d} "
              f"cc_teal={r['cc_teal']:6d} cc_white={r['cc_white']:5d} "
              f"raise_orange={r['raise_orange']:6d}")
    Path("data/vision2/action_foundation_preaudit2.json").write_text(
        json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
