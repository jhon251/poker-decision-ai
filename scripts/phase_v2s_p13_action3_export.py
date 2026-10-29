"""P13-Action3 — 窗口代表帧导出（只复制，不移动/修改原始证据帧）。

对 16 个手牌窗口各复制 4 张均匀间隔帧到 data/vision2/action3_windows/，
文件名含窗口号+时间+原帧名——供用户目视回忆动作日志。
"""
from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

SINCE = 134324519952050945
OUT = Path("data/vision2/action3_windows")


def main() -> int:
    since_s = SINCE / 1e7 - 11644473600
    new = [p for p in sorted(Path("data/vision2/coinpoker_evidence")
                             .glob("*_full.png"))
           if p.stat().st_mtime >= since_s]

    def ts_of(p: Path) -> float:
        return int(re.search(r"_(\d+)_full$", p.stem).group(1)) / 1000.0
    new.sort(key=ts_of)
    import datetime as dt

    def hhmmss(t):
        return dt.datetime.fromtimestamp(t).strftime("%H%M%S")
    OUT.mkdir(parents=True, exist_ok=True)
    for f in OUT.glob("*.png"):
        f.unlink()
    # 需要红底手牌判定
    import numpy as np
    from poker_decision_ai.vision2_platform.png_decode import decode_png

    def hero_cards(p: Path) -> int:
        """dualang 手牌 = 右下白底红点牌面 (880-1100, 995-1066)。"""
        w, h, rgba = decode_png(p.read_bytes())
        arr = np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)
        r = arr[..., 0].astype(np.int16)
        g = arr[..., 1].astype(np.int16)
        b = arr[..., 2].astype(np.int16)
        lum = (r * 0.299 + g * 0.587 + b * 0.114)
        sub = (lum > 150) & (np.abs(r - g) < 40) & (np.abs(g - b) < 40)
        return int(sub[995:1066, 880:1100].sum())
    windows = []
    start = None
    prev = None
    for i, p in enumerate(new):
        if hero_cards(p) > 2000:
            if start is None or i - prev > 3:
                if start is not None:
                    windows.append((start, prev))
                start = i
            prev = i
    if start is not None:
        windows.append((start, prev))
    for wi, (s, e) in enumerate(windows, 1):
        t0 = hhmmss(ts_of(new[s]))
        t1 = hhmmss(ts_of(new[e]))
        picks = [s, (2 * s + e) // 3, (s + 2 * e) // 3, e]
        picks = sorted(set(max(s, min(e, k)) for k in picks))
        for pi, k in enumerate(picks, 1):
            p = new[k]
            dst = OUT / f"win{wi:02d}_{t0}_{t1}_f{pi}_{p.stem[:26]}.png"
            shutil.copy2(p, dst)
        print(f"win{wi:02d} {t0}-{t1} 帧数={e - s + 1} -> {len(picks)} 帧")
    print("输出目录:", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
