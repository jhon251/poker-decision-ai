"""P22-C — 双 D 模板提取 + NCC 实测（White-D 帧09 / Dark-D 帧05）。"""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))
from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402

EV = Path("data/vision2/coinpoker_evidence")
OUT = Path("data/vision2/coinpoker_assets_v1876dealer")


def _tight(mask):
    ys, xs = np.where(mask)
    if len(ys) < 5:
        return None
    return mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    # White-D：帧09 (772,984) 附近白 D
    w, h, rgba = decode_png((EV /
        "POT_CHANGED_139_1787992592538_full.png").read_bytes())
    arr = np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)
    l = arr[..., 0].astype(np.int16) * 0.299 + arr[..., 1].astype(np.int16) \
        * 0.587 + arr[..., 2].astype(np.int16) * 0.114
    roi = l[964:1005, 750:800]
    wm = roi > 150
    # 白 D（近红包围）：先粗找白簇
    dw = _tight(wm)
    print(f"White-D 提取: {dw.shape if dw is not None else None}")
    if dw is not None:
        rgba4 = np.stack([(dw * 255).astype(np.uint8)] * 4, axis=-1)
        from poker_decision_ai.vision2_platform import encode_rgba_png
        (OUT / "d_white.png").write_bytes(encode_rgba_png(
            dw.shape[1], dw.shape[0], rgba4.tobytes()))
        with open(OUT / "d_white.txt", "w") as f:
            f.write("label=D_white source_frame=POT_CHANGED_139 "
                    "dealer_seat=S6 provenance=gt_dealer_button\n")

    # Dark-D：帧05 按钮内暗 D（l<120 的非红暗区）
    w, h, rgba = decode_png((EV /
        "POT_CHANGED_102_1787992505794_full.png").read_bytes())
    arr = np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)
    r = arr[..., 0].astype(np.int16)
    g = arr[..., 1].astype(np.int16)
    b = arr[..., 2].astype(np.int16)
    l = r * 0.299 + g * 0.587 + b * 0.114
    # 按钮区 (1491,737,50,58)
    bx, by, bw, bh = 1491, 737, 50, 58
    roi = l[by:by + bh, bx:bx + bw]
    rroi = r[by:by + bh, bx:bx + bw]
    groi = g[by:by + bh, bx:bx + bw]
    broi = b[by:by + bh, bx:bx + bw]
    red = (rroi > 40) & (groi < 60) & (broi < 60) & (rroi > groi * 2)
    # 暗 D = 红圆圈内（被红包住）的暗区：整体暗 + 非红
    dark = (roi < 120) & (~red)
    # 只在红圈核心区找（排除边缘）
    safe = np.zeros_like(dark)
    safe[8:50, 8:44] = True       # 内部区域（避开圆外暗背景）
    dd = _tight(dark & safe)
    print(f"Dark-D 提取: {dd.shape if dd is not None else None}")
    if dd is not None and dd.sum() > 30:
        rgba4 = np.stack([(dd * 255).astype(np.uint8)] * 4, axis=-1)
        from poker_decision_ai.vision2_platform import encode_rgba_png
        (OUT / "d_dark.png").write_bytes(encode_rgba_png(
            dd.shape[1], dd.shape[0], rgba4.tobytes()))
        with open(OUT / "d_dark.txt", "w") as f:
            f.write("label=D_dark source_frame=POT_CHANGED_102 "
                    "dealer_seat=S5 provenance=gt_dealer_button\n")
        print("  Dark-D 保存成功, area =", int(dd.sum()))
    else:
        print("  Dark-D 质量不足（area<=30）→ 不强制")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
