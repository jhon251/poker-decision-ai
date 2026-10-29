"""P30-2B — (a) stake 尾部 (0) 区域高倍目视确认；(b) 语料 stake 一致性扫描。"""
from __future__ import annotations
import random
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))
from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402
from poker_decision_ai.vision2_platform.png_encode import encode_rgba_png  # noqa: E402

EV = Path("data/vision2/coinpoker_evidence")
OUT = Path("data/vision2/p30_2a_crops")


def load(fp):
    w, h, rgba = decode_png((EV / fp).read_bytes())
    return np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)


# (a) 尾部放大（x150-200, y4-25）
arr = load("BUTTON_CHANGED_167_1787992682749_full.png")
sub = arr[4:26, 150:200]
big = np.repeat(np.repeat(sub, 10, axis=0), 10, axis=1)
(OUT / "z_f01_stake_tail.png").write_bytes(encode_rgba_png(
    big.shape[1], big.shape[0], big.tobytes()))
print("saved z_f01_stake_tail.png", big.shape[1], "x", big.shape[0])

# (b) 30 帧 stake 掩码 NCC vs f01 参考
ref_mask = (arr[0:26, 0:520][..., 0] * 0.299 + arr[0:26, 0:520][..., 1] * 0.587
            + arr[0:26, 0:520][..., 2] * 0.114) > 120
ref = ref_mask[0:26, 100:200].astype(np.float64)
ref0 = ref - ref.mean()
refn = float(np.sqrt((ref0 * ref0).sum()))

w, h, rgba = decode_png((EV /
    "8bf82f134651db0add626be0304a2ff1.png").read_bytes())
arr2 = np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)
mask2 = (arr2[0:26, 0:520][..., 0] * 0.299 + arr2[0:26, 0:520][..., 1] * 0.587
         + arr2[0:26, 0:520][..., 2] * 0.114) > 120
t = mask2[0:26, 100:200].astype(np.float64)
t0 = t - t.mean()
tn = float(np.sqrt((t0 * t0).sum()))
print(f"f01 vs 8bf8 stake NCC = {float((t0*ref0).sum()/max(tn*refn,1e-9)):.3f}")

files = sorted(EV.glob("*.png"))
random.seed(7)
sample = random.sample([f for f in files if f.name.endswith("_full.png")], 40)
scores = []
for fp in sample:
    try:
        w, h, rgba = decode_png(fp.read_bytes())
    except Exception:
        continue
    if (w, h) != (1876, 1400):
        continue
    a = np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)
    m = (a[0:26, 0:520][..., 0] * 0.299 + a[0:26, 0:520][..., 1] * 0.587
         + a[0:26, 0:520][..., 2] * 0.114) > 120
    tt = m[0:26, 100:200].astype(np.float64)
    t0 = tt - tt.mean()
    tnn = float(np.sqrt((t0 * t0).sum()))
    if tnn < 1e-9:
        scores.append((fp.name, None))
        continue
    sc = float((t0 * ref0).sum() / max(tnn * refn, 1e-9))
    scores.append((fp.name, round(sc, 3)))
present = [(n, s) for n, s in scores if s is not None]
print(f"有 header 帧: {len(present)}/{len(scores)}")
low = [x for x in present if x[1] < 0.70]
print(f"NCC<0.70: {len(low)}")
for n, s in sorted(present, key=lambda x: x[1])[:6]:
    print(f"  {n[:40]}: {s}")
