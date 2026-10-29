"""P15 — Stack 1876 尺度/识别探针（只读）。"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))
from poker_decision_ai.vision2.pixel.realclient.coinpoker_stack import (  # noqa: E402
    StackRecognizer)
from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402


def main() -> int:
    rec = StackRecognizer(window="1848")
    frames = sorted(Path("data/vision2/coinpoker_evidence").glob("*_full.png"))
    n = 0
    rec_n = 0
    samples = []
    for p in frames:
        w, h, rgba = decode_png(p.read_bytes())
        if (w, h) != (1876, 1400):
            continue
        n += 1
        arr = np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)
        res = rec.recognize(arr)
        ok = {k: v for k, v in res.items() if v["status"] == "RECOGNIZED"}
        if ok:
            rec_n += 1
            if len(samples) < 6:
                samples.append((p.name, {k: v["raw_text"] for k, v in ok.items()}))
    print(f"1876 帧: {n}, 有 stack 识别帧: {rec_n}")
    for s in samples:
        print("  ", s)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
