"""P30-2A — integrity 门：read_image 是否实际送达图像？
用已知内容帧（f03 hero cards 白卡）验证。"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))
from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402
from poker_decision_ai.vision2_platform.png_encode import encode_rgba_png  # noqa: E402

EV = Path("data/vision2/coinpoker_evidence")
OUT = Path("data/vision2/p30_2a_crops")
OUT.mkdir(parents=True, exist_ok=True)

# 裁剪已知内容（f03 hero cards 5c3c 区域）——若 read_image 能看到牌面，
# 则图像能力可用；若返回 text-only/error → BLOCKED。
w, h, rgba = decode_png((EV / "8bf82f134651db0add626be0304a2ff1.png")
                        .read_bytes())
arr = np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)
sub = arr[1000:1200, 790:1080]
big = np.repeat(np.repeat(sub, 3, axis=0), 3, axis=1)
(OUT / "z_integrity_cards.png").write_bytes(encode_rgba_png(
    big.shape[1], big.shape[0], big.tobytes()))
print("saved z_integrity_cards.png", big.shape[1], "x", big.shape[0],
      "（已知内容：5c 3c 双黑梅花）")
