"""play_gui 视觉验证驱动：按真实节奏回放模拟事件，PrintWindow 截图两张状态。

仅用于开发验证，不参与产品运行（mock 事件，不跑引擎）。
产出 assets/ui_preview_midhand.png 与 assets/ui_preview_showdown.png。
"""
import ctypes
import os
import sys
from ctypes import wintypes

sys.path.insert(0, r"<repo>")
import tkinter as tk

from PIL import Image

from cash_game_ui.play_gui import PlayGuiApp

root = tk.Tk()
app = PlayGuiApp(root)

lineup = [{"seat": s, "archetype": a, "bot_kind": "pool_bot"}
          for s, a in ((1, "loose_passive_fish"), (2, "fish"),
                       (3, "recreational"), (4, "semi_reg"), (5, "reg"))]


def snap(name):
    root.update_idletasks()
    root.update()
    u32, g32 = ctypes.windll.user32, ctypes.windll.gdi32
    hwnd = u32.GetParent(root.winfo_id()) or root.winfo_id()
    rect = wintypes.RECT()
    u32.GetWindowRect(hwnd, ctypes.byref(rect))
    w, h = rect.right - rect.left, rect.bottom - rect.top
    hdc = u32.GetWindowDC(hwnd)
    mem = g32.CreateCompatibleDC(hdc)
    bmp = g32.CreateCompatibleBitmap(hdc, w, h)
    g32.SelectObject(mem, bmp)
    u32.PrintWindow(hwnd, mem, 2)          # PW_RENDERFULLCONTENT

    class BIH(ctypes.Structure):
        _fields_ = [("biSize", wintypes.DWORD), ("biWidth", wintypes.LONG),
                    ("biHeight", wintypes.LONG), ("biPlanes", wintypes.WORD),
                    ("biBitCount", wintypes.WORD),
                    ("biCompression", wintypes.DWORD),
                    ("biSizeImage", wintypes.DWORD),
                    ("biXPelsPerMeter", wintypes.LONG),
                    ("biYPelsPerMeter", wintypes.LONG),
                    ("biClrUsed", wintypes.DWORD),
                    ("biClrImportant", wintypes.DWORD)]

    info = BIH()
    info.biSize = ctypes.sizeof(BIH)
    info.biWidth, info.biHeight = w, -h
    info.biPlanes, info.biBitCount, info.biCompression = 1, 32, 0
    buf = ctypes.create_string_buffer(w * h * 4)
    g32.GetDIBits(mem, bmp, 0, h, buf, ctypes.byref(info), 0)
    img = Image.frombuffer("RGBA", (w, h), buf.raw, "raw", "BGRA", 0, 1)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "assets", name)
    img.convert("RGB").save(out)
    g32.DeleteObject(bmp)
    g32.DeleteDC(mem)
    u32.ReleaseDC(hwnd, hdc)
    print("saved", out)


class Ctx:
    legal_actions = ["check", "bet"]
    to_call_chips = 0
    bb_chips = 20
    min_raise_to_chips = 60
    max_raise_to_chips = 2000
    position_label = "BTN"
    street = "flop"
    pot_chips = 90


# 按真实节奏回放（dt = 距上一事件的间隔，与引擎事件延时近似）
SEQ = [
    (0, {"type": "session_start", "stake_label": "$1/$2 娱乐重仓桌",
         "n_hands": 6, "lineup": lineup}),
    (300, {"type": "hand_start", "hand_index": 0, "n_hands": 6, "button": 0,
           "hero_hole": ["As", "Kh"], "stacks": [2000] * 6, "pot_chips": 30}),
    (600, {"type": "action", "seat": 1, "is_hero": False, "action": "call",
           "amount_chips": 20, "pot_chips": 50, "stacks": [2000] * 6,
           "street_bets": [0, 20, 20, 0, 0, 0]}),
    (600, {"type": "action", "seat": 2, "is_hero": False, "action": "call",
           "amount_chips": 20, "pot_chips": 70, "stacks": [2000] * 6,
           "street_bets": [0, 20, 20, 0, 0, 0]}),
    (600, {"type": "action", "seat": 3, "is_hero": False, "action": "fold",
           "amount_chips": None, "pot_chips": 70, "stacks": [2000] * 6,
           "street_bets": [0, 20, 20, 0, 0, 0]}),
    (600, {"type": "street", "street": "flop", "board": ["Ts", "9d", "2h"],
           "pot_chips": 90, "stacks": [2000] * 6, "street_bets": [0] * 6}),
    (600, {"type": "action", "seat": 1, "is_hero": False, "action": "bet",
           "amount_chips": 46, "pot_chips": 136, "stacks": [2000] * 6,
           "street_bets": [0, 46, 0, 0, 0, 0]}),
    (600, {"type": "action", "seat": 2, "is_hero": False, "action": "call",
           "amount_chips": None, "pot_chips": 182, "stacks": [2000] * 6,
           "street_bets": [0, 46, 46, 0, 0, 0]}),
    (600, {"type": "hero_turn", "context": Ctx(), "advisor": None}),
]
t = 0
for dt, e in SEQ:
    t += dt
    root.after(t, lambda e=e: app._dispatch(e))
root.after(t + 400, lambda: snap("ui_preview_midhand.png"))

# 摊牌（河牌圈三家亮牌，seat1 赢）
TAIL = [
    (600, {"type": "hand_end", "hand_index": 0, "net_bb": -4.2,
           "net_bb_raked": -4.1, "pot_chips": 260, "terminal_reason": "showdown",
           "board": ["Ts", "9d", "2h", "4c", "7s"], "winners": [1],
           "revealed": [{"seat": 1, "hole": ["Qd", "Qc"]},
                        {"seat": 2, "hole": ["8s", "8d"]},
                        {"seat": 5, "hole": ["Jh", "Jc"]}],
           "analysis": {"n_hero_decisions": 3, "n_advisor_available": 3,
                        "n_agree": 2}}),
]
t2 = t + 1500
for dt, e in TAIL:
    t2 += dt
    root.after(t2, lambda e=e: app._dispatch(e))
root.after(t2 + 1400, lambda: snap("ui_preview_showdown.png"))
root.after(t2 + 1600, root.destroy)
root.mainloop()
