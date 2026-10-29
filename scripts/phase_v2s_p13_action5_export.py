"""P13-Action5 — 按钮出现时刻代表帧导出（只复制，不动原始证据）。"""
from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402

SINCE = 134324658401327108
OUT = Path("data/vision2/action5_check")


def main() -> int:
    since_s = SINCE / 1e7 - 11644473600
    new = [p for p in sorted(Path("data/vision2/coinpoker_evidence")
                             .glob("*_full.png"))
           if p.stat().st_mtime >= since_s]

    def ts(p):
        return int(re.search(r"_(\d+)_full$", p.stem).group(1)) / 1000.0
    new.sort(key=ts)

    def red_buttons(p):
        w, h, rgba = decode_png(p.read_bytes())
        arr = np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)
        r = arr[..., 0].astype(np.int16)
        g = arr[..., 1].astype(np.int16)
        b = arr[..., 2].astype(np.int16)
        m = ((r[1150:1400, :] > g[1150:1400, :] + 50)
             & (r[1150:1400, :] > b[1150:1400, :] + 50)
             & (r[1150:1400, :] > 110))
        return int(m.sum())
    OUT.mkdir(exist_ok=True)
    for f in OUT.glob("*.png"):
        f.unlink()
    # 聚类按钮帧 → 每 occurrence 取中帧
    marks = [(i, p, red_buttons(p)) for i, p in enumerate(new)]
    occ = []
    cur = None
    for i, p, m in marks:
        if m > 12000:
            if cur is None:
                cur = [i, i]
            else:
                if ts(new[i]) - ts(new[cur[1]]) < 20:
                    cur[1] = i
                else:
                    occ.append(cur)
                    cur = [i, i]
    if cur:
        occ.append(cur)
    import datetime as dt
    for n, (s, e) in enumerate(occ, 1):
        mid = (s + e) // 2
        p = new[mid]
        t = dt.datetime.fromtimestamp(ts(p)).strftime("%H%M%S")
        dst = OUT / f"occ{n:02d}_t{t}.png"
        shutil.copy2(p, dst)
        print(f"occ{n:02d} {dt.datetime.fromtimestamp(ts(new[s])).strftime('%H:%M:%S')}"
              f"-{dt.datetime.fromtimestamp(ts(new[e])).strftime('%H:%M:%S')} "
              f"({e - s + 1}帧) -> {dst.name}")
    print("输出:", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
