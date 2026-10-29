"""P16-B — Pot 1876 中文标签 vs 数字分离验证（READ-ONLY）。"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))
from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402

EV = Path("data/vision2/coinpoker_evidence")


def main() -> int:
    names = [
        "BUTTON_CHANGED_167_1787992682749_full.png",
        "POT_CHANGED_102_1787992505794_full.png",
        "BUTTON_CHANGED_598_1787993753587_full.png",
    ]
    for name in names:
        p = EV / name
        w, h, rgba = decode_png(p.read_bytes())
        if (w, h) != (1876, 1400):
            continue
        arr = np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)
        print(f"\n===== {name} =====")
        # 左区（标签候选 x 780-880） vs 右区（金额候选 x 880-1040）
        for (x0, x1, label) in [(780, 880, "左区(标签?)"),
                                (880, 1040, "右区(金额?)")]:
            roi = arr[70:180, x0:x1]
            r = roi[..., 0].astype(np.int16)
            g = roi[..., 1].astype(np.int16)
            b = roi[..., 2].astype(np.int16)
            luma = r * 0.299 + g * 0.587 + b * 0.114
            gold = (r > 120) & (g > 85) & (r >= g) & (g >= b - 30) & (luma > 90)
            white = luma > 200
            print(f"  [{label}] 金={int(gold.sum())} 白={int(white.sum())} "
                  f"R={r.mean():.0f} G={g.mean():.0f} luma={luma.mean():.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
