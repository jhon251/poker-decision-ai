"""P13-Action Foundation PRE-AUDIT — 按钮分区测量（只读）。

对第5轮+补采的已标注 occurrence：分别测量三个按钮区（fold/callcheck/raise）
的颜色与白字签名；重点：call vs check 的视觉区分性。
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
# 按钮分区（1876x1400 窗口；用户截图实测+第5轮实测）
FOLD_REG = (1116, 1263, 1354, 1392)      # 盖牌 红
CALLCHK_REG = (1354, 1263, 1629, 1392)   # 跟注/过牌 深青
RAISE_REG = (1629, 1263, 1860, 1392)     # 加注 橙黄
# 第5轮标注
ANNOT5 = {2: "fold", 3: "check", 4: "call", 6: "fold", 8: "fold", 9: "fold",
          10: "raise_pre", 11: "call", 12: "raise", 13: "allin", 14: "fold",
          15: "fold", 16: "fold", 17: "fold", 19: "call", 20: "fold",
          21: "fold", 22: "call", 23: "fold", 25: "fold"}


def load(p: Path) -> np.ndarray:
    w, h, rgba = decode_png(p.read_bytes())
    return np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)


def reg_sig(arr: np.ndarray, reg: tuple) -> dict:
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
    gold = ((r > 150) & (g > 110) & (b < 90) & (np.abs(r - g) < 90)
            )[y0:y1, x0:x1].sum()
    return {"red": int(red), "orange": int(orange), "teal": int(teal),
            "white": int(white), "gold": int(gold)}


def occ_sigs(since_ft: int, annot: dict, label: str):
    since_s = since_ft / 1e7 - 11644473600
    new = [p for p in sorted(Path("data/vision2/coinpoker_evidence")
                             .glob("*_full.png"))
           if p.stat().st_mtime >= since_s]

    def ts(p):
        return int(re.search(r"_(\d+)_full$", p.stem).group(1)) / 1000.0
    new.sort(key=ts)
    occ = []
    cur = None
    for i, p in enumerate(new):
        arr = load(p)
        r = arr[..., 0].astype(np.int16)
        g = arr[..., 1].astype(np.int16)
        b = arr[..., 2].astype(np.int16)
        m = ((r[1150:1400, :] > g[1150:1400, :] + 50)
             & (r[1150:1400, :] > b[1150:1400, :] + 50)
             & (r[1150:1400, :] > 110))
        if int(m.sum()) > 12000:
            if cur is None:
                cur = [i, i]
            elif ts(new[i]) - ts(new[cur[1]]) < 20:
                cur[1] = i
            else:
                occ.append(cur)
                cur = [i, i]
    if cur:
        occ.append(cur)
    out = []
    for n, (s, e) in enumerate(occ, 1):
        act = annot.get(n, "?")
        arr = load(new[(s + e) // 2])
        out.append({
            "label": label, "occ": n, "action": act,
            "fold": reg_sig(arr, FOLD_REG),
            "callcheck": reg_sig(arr, CALLCHK_REG),
            "raise": reg_sig(arr, RAISE_REG),
        })
    return out


def main() -> int:
    rows = occ_sigs(SINCE5, ANNOT5, "R5") + occ_sigs(
        SINCE6, {1: "check", 2: "raise", 3: "raise", 4: "check"}, "R6")
    for rw in rows:
        f, c, ra = rw["fold"], rw["callcheck"], rw["raise"]
        print(f"{rw['label']} occ{rw['occ']:02d} {rw['action']:10s} "
              f"| FOLD r={f['red']:6d} w={f['white']:5d} "
              f"| CC teal={c['teal']:6d} w={c['white']:5d} gold={c['gold']:4d} "
              f"| RAISE o={ra['orange']:6d} w={ra['white']:5d} r={ra['red']:5d}")
    Path("data/vision2/action_foundation_preaudit.json").write_text(
        json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
