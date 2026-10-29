"""Phase V2P — 只读客户端定位（无任何输入注入/窗口控制）。

枚举可见窗口 → 标题关键字匹配（poker/德州/扑克/得州/赢家等）→
输出候选；无唯一匹配 → CALIBRATION_REQUIRED（禁止猜测窗口）。
"""
from __future__ import annotations

import ctypes
import json
import sys
from ctypes import wintypes
from pathlib import Path

POKER_KEYWORDS = ("poker", "德州", "扑克", "得州", "pokerstars", "赢家",
                  "wzpk", "hhpoker", "pkw", "wepoker", "gpk")

KEYWORDS_ANY = ("poker", "德州", "扑克", "得州")
KEYWORDS_STRICT = ("pokerstars", "hhpoker", "pkw", "wepoker", "gpk")


def enumerate_windows() -> list:
    u32 = ctypes.windll.user32
    out = []

    @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    def cb(hwnd, _lparam):
        if not u32.IsWindowVisible(hwnd):
            return True
        length = u32.GetWindowTextLengthW(hwnd)
        if length <= 0:
            return True
        buf = ctypes.create_unicode_buffer(length + 1)
        u32.GetWindowTextW(hwnd, buf, length + 1)
        r = wintypes.RECT()
        u32.GetWindowRect(hwnd, ctypes.byref(r))
        out.append({"hwnd": int(hwnd), "title": buf.value,
                    "rect": [r.left, r.top, r.right - r.left,
                             r.bottom - r.top]})
        return True

    u32.EnumWindows(cb, 0)
    return out


def classify(windows: list) -> dict:
    """候选判定：严格命中 / 宽松命中 / 无。"""
    strict, loose = [], []
    for w in windows:
        t = (w["title"] or "").lower()
        if any(k.lower() in t for k in KEYWORDS_STRICT):
            strict.append(w)
        elif any(k.lower() in t for k in KEYWORDS_ANY):
            loose.append(w)
    return {"strict": strict, "loose": loose}


def main() -> int:
    sys.argv = sys.argv[:1]
    windows = enumerate_windows()
    cls = classify(windows)
    Path("data/vision2").mkdir(exist_ok=True)
    json.dump({"windows": windows, "classified": cls},
              open("data/vision2/window_enum.json", "w", encoding="utf-8"),
              indent=1, ensure_ascii=False)
    print(f"可见窗口: {len(windows)}")
    for w in windows:
        print(f"  hwnd={w['hwnd']} {w['rect']} title={w['title']!r}")
    if cls["strict"]:
        print("POKER_CLIENT_FOUND(STRICT):", cls["strict"])
        return 0
    if cls["loose"]:
        print("POKER_CLIENT_CANDIDATE(LOOSE):", cls["loose"])
        return 2
    print("POKER_CLIENT_NOT_FOUND → CALIBRATION_REQUIRED（禁止猜测窗口）")
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
