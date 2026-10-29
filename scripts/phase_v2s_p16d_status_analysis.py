"""P16-D — Status 1876 READ-ONLY：badge 检测 + 尺度实测。

用现有 1419 badge 模板（folded/away/allin）在 1876 帧上做 NCC（原尺度 +
1.32× 尺度），判定 1876 badge 是否存在 + 尺度关系。
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))
from poker_decision_ai.vision2.pixel.realclient.cashgame import (  # noqa: E402
    PlayerStatusRecognizer)
from poker_decision_ai.vision2.realtime.image import ImageData, to_gray  # noqa: E402
from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402

EV = Path("data/vision2/coinpoker_evidence")


def _ncc(g_full, t):
    th, tw = t.shape
    if th >= g_full.shape[0] or tw >= g_full.shape[1]:
        return -1.0
    wins = np.lib.stride_tricks.sliding_window_view(g_full, (th, tw))
    wm = wins.mean(axis=(2, 3), keepdims=True)
    wc = wins - wm
    wn = np.sqrt((wc * wc).sum(axis=(2, 3)))
    t0 = t - t.mean()
    tn = float(np.sqrt((t0 * t0).sum()))
    scores = np.abs((wc * t0).sum(axis=(2, 3)) / np.maximum(wn * tn, 1e-9))
    return float(scores.max())


def _resize(t, scale):
    import math
    h, w = t.shape
    nh, nw = max(1, int(h * scale)), max(1, int(w * scale))
    # 最近邻缩放（近似）
    ys = (np.arange(nh) * h / nh).astype(int)
    xs = (np.arange(nw) * w / nw).astype(int)
    return t[np.ix_(ys, xs)]


def main() -> int:
    rec = PlayerStatusRecognizer()
    names = [
        "BUTTON_CHANGED_167_1787992682749_full.png",
        "POT_CHANGED_102_1787992505794_full.png",
        "POT_CHANGED_12_1787995290794_full.png",
        "BUTTON_CHANGED_598_1787993753587_full.png",
    ]
    for name in names:
        p = EV / name
        w, h, rgba = decode_png(p.read_bytes())
        if (w, h) != (1876, 1400):
            continue
        arr = np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)
        g_full = np.asarray(to_gray(ImageData.from_rgba_bytes(
            w, h, arr.tobytes())).data, dtype=np.float32)
        # 4× downscale（同 recognizer）
        g = g_full[::4, ::4]
        print(f"\n===== {name} =====")
        for label, tpl in rec.badges.items():
            t_full = np.asarray(to_gray(tpl).data, dtype=np.float32)
            print(f"  [{label}] 模板 {t_full.shape[0]}x{t_full.shape[1]}")
            for scale in (1.0, 1.32, 1.5):
                t = _resize(t_full, scale)
                sc = _ncc(g, t)
                print(f"    scale={scale} (模板 {t.shape}) -> NCC={sc:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
