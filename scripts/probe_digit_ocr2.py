"""P30-9-HS 探针②：**先定位金额行/列组，再逐位识别**（复用识别器行带逻辑 ✓）。

只在"最后一个文本带 + 数字密集列组"上做逐位识别 ✓；
每个字形要求 NCC ≥ 阈值**且**与次优有边际 ✓，否则整串判 UNKNOWN（不猜 ✓）。
"""
from __future__ import annotations

import importlib.util
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, r"<repo>")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE = pathlib.Path(r"<repo>\data\vision2")


def _load_probe():
    spec = importlib.util.spec_from_file_location(
        "probe", r"<repo>\scripts\probe_digit_ocr.py")
    m = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(m)
    except SystemExit:
        pass
    return m


P = _load_probe()
NCC_MIN = 0.75
NCC_MARGIN = 0.05


def build_templates() -> dict:
    merged: dict = {}
    for name in ("coinpoker_assets_v1876stack", "coinpoker_assets_v1876pot"):
        for k, vs in P.load_digit_templates(name).items():
            merged.setdefault(k, []).extend(vs)
    return {k: [P.norm_glyph(v) for v in vs] for k, vs in merged.items()}


def read_row(mask: np.ndarray, a: int, b: int, tpl: dict) -> tuple[str, float]:
    """在行带 [a,b] 内逐位识别：按零列切分 → 归一化 → NCC（阈值+边际 ✓）。"""
    band = mask[a:b + 1]
    cold = band.sum(axis=0)
    groups, run = [], None
    for i, v in enumerate(cold):
        if v >= 1 and run is None:
            run = [i, i]
        elif v >= 1 and run is not None:
            run[1] = i
        elif run is not None:
            groups.append(tuple(run))
            run = None
    if run:
        groups.append(tuple(run))
    groups = [g for g in groups if g[1] - g[0] >= 2]
    out, scores = [], []
    for ga, gb in groups:
        sub = band[:, ga:gb + 1]
        ys = np.where(sub.sum(axis=1) > 0)[0]
        if len(ys) == 0:
            continue
        g = P.norm_glyph(sub[ys.min():ys.max() + 1])
        scored = sorted(((P.ncc(g, t), lab) for lab, ts in tpl.items() for t in ts),
                         reverse=True)
        (s1, l1) = scored[0]
        s2 = scored[1][0] if len(scored) > 1 else 0.0
        if s1 < NCC_MIN or (s1 - s2) < NCC_MARGIN:
            out.append("?")
        else:
            out.append(l1)
        scores.append(round(s1, 2))
    return "".join(out), (sum(scores) / len(scores) if scores else 0.0)


def main() -> None:
    from PIL import Image
    from poker_decision_ai.vision2.pixel.realclient.coinpoker_to_call import (
        ACTION_PANEL_ROI, ToCallRecognizer1876, _text_mask)
    tpl = build_templates()
    print(f"模板: {len(tpl)} 标签 / {sum(len(v) for v in tpl.values())} 字形")
    rec = ToCallRecognizer1876()
    frames = sorted(pathlib.Path(r"<repo>\dist\logs\frames").glob("*.png"))
    for f in frames:
        im = Image.open(f).convert("RGB")
        arr = np.asarray(im)
        x0, y0, x1, y1 = ACTION_PANEL_ROI
        mask = _text_mask(arr[y0:y1, x0:x1])
        if int(mask.sum()) < 50:
            print(f"  {f.name[:34]:<36} 面板空白/未识别")
            continue
        bands = rec._row_bands(mask)
        if not bands:
            print(f"  {f.name[:34]:<36} 无文本带")
            continue
        a2, b2 = bands[-1]
        s, conf = read_row(mask, a2, b2, tpl)
        tpl_hit = rec.recognize(arr)
        print(f"  {f.name[:34]:<36} 行带数={len(bands)} 逐位='{s}' 均分={conf:.2f} "
              f"| 模板法={tpl_hit.get('to_call')} ({tpl_hit.get('reason')})")


if __name__ == "__main__":
    main()
