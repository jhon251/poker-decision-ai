"""P30-9-HR 探针：用 pot/stack 数字字形做**逐位识别**，先在**已知底池**上验证方法 ✓。

思路：pot 区域的文本（`₮0.55` 之类）其数值在日志里已知 ⇒ 可用作**真值**检验字形流水线 ✓；
若该流水线能复现底池数值，才把它用于动作面板的 `to_call`（无真值 ✗，需谨慎 ✓）。
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, r"<repo>")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE = pathlib.Path(r"<repo>\data\vision2")


def load_digit_templates(dir_name: str) -> dict:
    """从资产目录加载**逐位字形**（PNG ⇒ 二值掩码 ✓；标签即数字/分隔符 ✓）。"""
    import numpy as _np
    from PIL import Image as _Image
    d = BASE / dir_name
    mf = d / "manifest.json"
    if not mf.is_file():
        return {}
    m = json.loads(mf.read_text(encoding="utf-8"))
    root = BASE.parent
    out: dict = {}
    for a in m.get("assets", []):
        lab = str(a.get("label") or "")
        if lab not in list("0123456789") + [".", ","]:
            continue
        ref = str(a.get("crop_ref") or "")
        cands = [pathlib.Path(ref), d / pathlib.Path(ref).name, root / ref]
        f = next((c for c in cands if c.is_file()), None)
        if f is None:
            continue
        try:
            # P30-9-HR：字形可能是**透明背景（alpha 为墨迹）** ⇒ 不能只用灰度阈值 ✗
            im = _Image.open(f)
            a3 = _np.asarray(im)
            if a3.ndim == 3 and a3.shape[2] == 4:
                alpha = a3[..., 3].astype(_np.float64)
                rgb = a3[..., :3].astype(_np.float64).mean(axis=2)
                if alpha.max() > 0 and float((alpha < alpha.max() * 0.5).mean()) > 0.30:
                    arr = (alpha > alpha.max() * 0.5).astype(_np.float64)     # 墨迹 = 不透明 ✓
                else:
                    arr = (rgb < max(rgb.max() * 0.6, 1.0)).astype(_np.float64)
            else:
                g = a3.astype(_np.float64)
                if g.ndim == 3:
                    g = g.mean(axis=2)
                # 墨迹可能是深色（浅底）或浅色（深底）⇒ 取**少数派**为墨迹 ✓
                dark = float((g < g.max() * 0.6).mean())
                arr = ((g < g.max() * 0.6) if dark <= 0.5
                        else (g > g.min() + (g.max() - g.min()) * 0.4)).astype(_np.float64)
        except Exception:                                       # noqa: BLE001
            continue
        if arr.sum() <= 0:
            continue
        out.setdefault(lab, []).append(arr)
    return out


def norm_glyph(a: np.ndarray, box=(20, 28)) -> np.ndarray:
    """裁剪到墨迹包围盒 → 等比缩放进固定框（居中）⇒ 可跨字号比较 ✓。"""
    ys, xs = np.where(a > 0)
    if len(ys) == 0:
        return np.zeros(box, dtype=np.float64)
    crop = a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = crop.shape
    bh, bw = box
    sc = min(bh / max(h, 1), bw / max(w, 1))
    nh, nw = max(1, int(round(h * sc))), max(1, int(round(w * sc)))
    yi = (np.arange(nh) / sc).astype(int).clip(0, h - 1)
    xi = (np.arange(nw) / sc).astype(int).clip(0, w - 1)
    small = crop[yi][:, xi]
    out = np.zeros((bh, bw), dtype=np.float64)
    y0, x0 = (bh - nh) // 2, (bw - nw) // 2
    out[y0:y0 + nh, x0:x0 + nw] = small
    return out


def ncc(a: np.ndarray, b: np.ndarray) -> float:
    a0 = a - a.mean(); b0 = b - b.mean()
    an = float(np.sqrt((a0 * a0).sum())); bn = float(np.sqrt((b0 * b0).sum()))
    if an < 1e-9 or bn < 1e-9:
        return 0.0
    return float((a0 * b0).sum() / (an * bn))


def main() -> None:
    for name in ("coinpoker_assets_v1876stack", "coinpoker_assets_v1876pot"):
        t = load_digit_templates(name)
        print(f"{name}: 标签 {sorted(t.keys())} | 模板总数 {sum(len(v) for v in t.values())}")
    from poker_decision_ai.vision2.pixel.realclient.coinpoker_to_call import (
        ACTION_PANEL_ROI, _text_mask)
    from PIL import Image
    frames = sorted((pathlib.Path(r"<repo>\dist\logs\frames")).glob("*.png"))
    st = load_digit_templates("coinpoker_assets_v1876stack")
    st2 = load_digit_templates("coinpoker_assets_v1876pot")
    merged = {k: list(v) for k, v in st.items()}
    for k, vs in st2.items():
        merged.setdefault(k, []).extend(vs)
    prepped = {k: [norm_glyph(v) for v in vs] for k, vs in merged.items()}
    print("合并模板:", {k: len(v) for k, v in merged.items()})
    print(f"\n面板 ROI={ACTION_PANEL_ROI} | 帧数 {len(frames)}")
    for f in frames[:4]:
        im = Image.open(f).convert("RGB")
        arr = np.asarray(im)
        x0, y0, x1, y1 = ACTION_PANEL_ROI
        panel = arr[y0:y1, x0:x1]
        mask = _text_mask(panel)
        # 逐列切分字形（零列分隔 ✓）
        cold = mask.sum(axis=0)
        groups, run = [], None
        for i, v in enumerate(cold):
            if v >= 1 and run is None:
                run = [i, i]
            elif v >= 1 and run is not None:
                run[1] = i
            elif run is not None:
                groups.append(tuple(run)); run = None
        if run:
            groups.append(tuple(run))
        groups = [g for g in groups if g[1] - g[0] >= 2]
        read = []
        for ga, gb in groups:
            sub = mask[:, ga:gb + 1]
            ys = np.where(sub.sum(axis=1) > 0)[0]
            if len(ys) == 0:
                continue
            g = norm_glyph(sub[ys.min():ys.max() + 1])
            best, bs = None, -1.0
            for lab, tpls in prepped.items():
                for tpl in tpls:
                    s = ncc(g, tpl)
                    if s > bs:
                        best, bs = lab, s
            read.append(f"{best}:{bs:.2f}")
        print(f"  {f.name[:34]:<36} 字形数={len(groups)} ⇒ {read[:10]}")


if __name__ == "__main__":
    main()
