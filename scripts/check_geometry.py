# -*- coding: utf-8 -*-
"""check_geometry.py — CoinPoker 窗口几何自检（V-3 前置工具）

用法（保持 CoinPoker 窗口在前台可见）：
    .venv\\Scripts\\python.exe scripts\\check_geometry.py

判定：
    >=1876 宽        → 1876 类 ✅（hero 牌模板齐备，采集/验证有效）
    1350–1875 宽     → 1419 类 ⚠️（hero 座 S2/S6 裁剪坏——旧语料的缩放问题）
    <1350            → 太小 ❌
"""
import sys, time

def main():
    try:
        import mss
    except ImportError:
        print("需要 mss：.venv\\Scripts\\python.exe -m pip install mss")
        return 1
    import numpy as np
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
    from scripts.phase_v2s_auto_capture import locate_coinpoker

    loc = locate_coinpoker()
    if not loc or not loc.get("window_rect"):
        print("CoinPoker 窗口未找到——请确认客户端已打开且未最小化")
        return 1
    rect = loc["window_rect"]
    w, h = int(rect["width"]), int(rect["height"])
    print(f"CoinPoker 窗口: {w}x{h} at ({rect['left']},{rect['top']})")

    if w >= 1876:
        cls = "1876 类 ✅"
        verdict = "几何正确——hero 牌模板齐备，可以直接采集/标注"
        detail = "hero 座位（底部中央）与全部 6 座在有效采集区"
    elif w >= 1350:
        cls = "1419 类 ⚠️"
        verdict = "这就是旧语料的缩放问题几何——hero 座 S2/S6 裁剪坏，采集无效"
        detail = "解决：放大 CoinPoker 窗口（最大化），或提高 Windows 显示缩放"
    else:
        cls = "太小 ❌"
        verdict = "窗口过小，任何识别都无效"
        detail = "最大化 CoinPoker 窗口"

    print(f"render class: {cls}")
    print(f"判定: {verdict}")
    print(f"提示: {detail}")
    return 0 if w >= 1876 else 1

if __name__ == "__main__":
    sys.exit(main())
