"""P26-A — 单字形 NCC 深度诊断：S1 f01 '1'(第一字符) vs pot '1' 模板。

打印：候选 tight 灰度形状、模板形状、NCC 分、尺寸对比。"""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))
from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402
from poker_decision_ai.vision2.pixel.realclient.cashgame import (  # noqa: E402
    PotRecognizer)
from poker_decision_ai.vision2.pixel.realclient.coinpoker_board import (  # noqa: E402
    _bright_tight, _pad_tight)
from poker_decision_ai.vision2_platform.png_encode import encode_rgba_png  # noqa: E402

EV = Path("data/vision2/coinpoker_evidence")
OUT = Path("data/vision2/p26a_crops")

POT = PotRecognizer(window="1876")


def load(fp):
    w, h, rgba = decode_png((EV / fp).read_bytes())
    return np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)


arr = load("BUTTON_CHANGED_167_1787992682749_full.png")
# S1 第一字符 '1'：tight x 188-198 附近（real comp x≈188-198, y 428-470）
roi = arr[425:475, 180:210]
lc = (roi[..., 0].astype(np.int16) * 0.299
      + roi[..., 1].astype(np.int16) * 0.587
      + roi[..., 2].astype(np.int16) * 0.114)
ys, xs = np.where((lc > 60))
print("gold luma>60 bbox rel:", xs.min(), ys.min(), xs.max() - xs.min() + 1,
      ys.max() - ys.min() + 1)
# 组件 tight
sub = roi[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
img_tight = _bright_tight_from_arr(sub)
print("candidate tight bbox:", img_tight.shape if hasattr(img_tight, "shape")
      else (img_tight.height, img_tight.width))

# 直接检查模板 '1' 的 prep
prep = POT._prep.get("1", [])
print("template '1' instances:", len(prep))
for (t, t0, tn) in prep:
    print("  tpl shape:", t.shape, "mean=%.2f" % t.mean())


def _bright_tight_from_arr(a):
    from poker_decision_ai.vision2.pixel.realclient.coinpoker_board import (
        _bright_tight)
    from poker_decision_ai.vision2.pixel.realclient.coinpoker_stack import (
        ImageData)
    return None
