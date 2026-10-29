"""P28-B — HeroSeatDetector1876 验证：7 帧 GT + 负例 + dispatch。"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))
from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402
from poker_decision_ai.vision2.pixel.realclient.hero_seat import (  # noqa: E402
    HeroSeatDetector1876)
from poker_decision_ai.vision2.pixel.realclient.seat_position import (  # noqa: E402
    PositionObserver)

EV = Path("data/vision2/coinpoker_evidence")
REC = HeroSeatDetector1876()

FRAMES = {
    "f01": "BUTTON_CHANGED_167_1787992682749_full.png",
    "f02": "POT_CHANGED_12_1787995290794_full.png",
    "f03": "POT_CHANGED_102_1787992505794_full.png",
    "f04": "POT_CHANGED_13_1787992279195_full.png",
    "f05": "POT_CHANGED_139_1787992592538_full.png",
    "f06": "BUTTON_CHANGED_598_1787993753587_full.png",
    "u01": "42b553c9468d1d1e55b9801bebad7b4f.png",
}


def load(fp):
    w, h, rgba = decode_png((EV / fp).read_bytes())
    return np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)


print("=== 7 帧 hero 检测（GT: 全 S6）===")
exact = 0
for fr, fp in FRAMES.items():
    out = REC.detect(load(fp))
    hero = out["hero_seat"]
    ok = hero == "S6"
    exact += int(ok)
    maxw = max((v['white'] for v in out['evidence'].values()), default=0)
    print(f"  {fr}: hero={hero} {'OK' if ok else 'XX'} ({out['reason']}) "
          f"max_white={maxw}")
print(f"\nexact={exact}/7")

# 负例：blank / crop
print("\n=== 负例 ===")
fb = np.zeros((1400, 1876, 4), dtype=np.uint8)
fb[..., :3] = (6, 35, 27)
fb[..., 3] = 255
print("blank:", REC.detect(fb)["hero_seat"], REC.detect(fb)["reason"])
fc = np.zeros((400, 400, 4), dtype=np.uint8)
fc[..., :3] = (6, 35, 27)
fc[..., 3] = 255
print("crop:", REC.detect(fc)["hero_seat"], REC.detect(fc)["reason"])

# dispatch
print("\n=== PositionObserver dispatch ===")
w, h, rgba = decode_png((EV / FRAMES["f01"]).read_bytes())
out = PositionObserver().observe(rgba, w, h)
rr = out["runtime_result"]
print(f"hero_seat={rr.get('hero_seat')} uncertainty={rr.get('uncertainty')}")
