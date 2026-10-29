"""缩放 + 回放 功能验证（开发用，一次性脚本）。"""
import ctypes
import os
import sys
import time
from ctypes import wintypes

sys.path.insert(0, r"<repo>")
os.chdir(r"<repo>")                      # data/play_history 相对路径基准
import tkinter as tk

from PIL import Image
from poker_decision_ai.play.history import PlayHistoryWriter

from cash_game_ui.play_gui import PlayGuiApp, ReplayBrowser

# ── 造一个含 timeline 的历史会话 ──────────────────────────────
hist = PlayHistoryWriter("data/play_history", "zoomtest", stake="nl2",
                         label="$1/$2 娱乐重仓桌", seed_base=1,
                         lineup=[{"seat": s, "archetype": a, "bot_kind": "pool_bot"}
                                 for s, a in ((1, "fish"), (2, "recreational"),
                                              (3, "semi_reg"), (4, "semi_reg"),
                                              (5, "reg"))],
                         deterministic=True)
hist.write_hand(
    hand_index=0, deal_seed=1, button=0, hero_hole=["As", "Kh"],
    board=["Ts", "9d", "2h", "4c", "7s"], street_reached="showdown",
    n_players_dealt=5,
    hero_decisions=[{"street": "flop", "action": "bet", "amount_chips": 46}],
    net_bb=4.2, net_bb_raked=4.1, net_bb_raked_full=4.0, rake_chips=2.0,
    pot_chips=260, terminal_reason="showdown", aborted=None, hand_wall_ms=100.0,
    timeline=[
        {"street": "preflop", "seat": 1, "action": "call", "amount_chips": None,
         "pot_chips": 50, "stacks": [2000] * 6, "street_bets": [0, 20, 20, 0, 0, 0]},
        {"street": "preflop", "seat": 3, "action": "fold", "amount_chips": None,
         "pot_chips": 50, "stacks": [2000] * 6, "street_bets": [0, 20, 20, 0, 0, 0]},
        {"street": "flop", "seat": 1, "action": "bet", "amount_chips": 46,
         "pot_chips": 136, "stacks": [2000] * 6, "street_bets": [0, 46, 0, 0, 0, 0]},
        {"street": "flop", "seat": 2, "action": "call", "amount_chips": None,
         "pot_chips": 182, "stacks": [2000] * 6, "street_bets": [0, 46, 46, 0, 0, 0]},
        {"street": "river", "seat": 2, "action": "all-in", "amount_chips": None,
         "pot_chips": 260, "stacks": [1954] * 6, "street_bets": [0, 0, 100, 0, 0, 0]},
    ],
    winners=[1],
    revealed=[{"seat": 1, "hole": ["Qd", "Qc"]},
              {"seat": 2, "hole": ["8s", "8d"]}])

root = tk.Tk()
app = PlayGuiApp(root)


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
    u32.PrintWindow(hwnd, mem, 2)

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
    info.biWidth, info.biHeight, info.biPlanes = w, -h, 1
    info.biBitCount, info.biCompression = 32, 0
    buf = ctypes.create_string_buffer(w * h * 4)
    g32.GetDIBits(mem, bmp, 0, h, buf, ctypes.byref(info), 0)
    img = Image.frombuffer("RGBA", (w, h), buf.raw, "raw", "BGRA", 0, 1)
    out = os.path.join("cash_game_ui", "assets", name)
    img.convert("RGB").save(out)
    g32.DeleteObject(bmp)
    g32.DeleteDC(mem)
    u32.ReleaseDC(hwnd, hdc)
    print("saved", out, f"{w}x{h}")


def settle(ms):
    end = time.time() + ms / 1000
    while time.time() < end:
        root.update()
        time.sleep(0.02)


lineup = [{"seat": s, "archetype": a, "bot_kind": "pool_bot"}
          for s, a in ((1, "loose_passive_fish"), (2, "fish"),
                       (3, "recreational"), (4, "semi_reg"), (5, "reg"))]
app._dispatch({"type": "session_start", "stake_label": "$1/$2 娱乐重仓桌",
               "n_hands": 6, "lineup": lineup})
app._dispatch({"type": "hand_start", "hand_index": 0, "n_hands": 6, "button": 0,
               "hero_hole": ["As", "Kh"], "stacks": [2000] * 6, "pot_chips": 30})
app._dispatch({"type": "street", "street": "flop", "board": ["Ts", "9d", "2h"],
               "pot_chips": 90, "stacks": [2000] * 6, "street_bets": [0] * 6})
root.attributes("-topmost", True)
settle(2000)
snap("ui_preview_zoom100.png")

# ── 缩放 0.8 ─────────────────────────────────────────────
app.zoom_var.set(0.8)
app._on_zoom_menu()
settle(800)
snap("ui_preview_zoom80.png")

# ── 缩放 1.2 ─────────────────────────────────────────────
app.zoom_var.set(1.2)
app._on_zoom_menu()
settle(800)
snap("ui_preview_zoom120.png")

# ── 回放：回到 1.0，打开复盘浏览器并播放 ────────────────────
app.zoom_var.set(1.0)
app._on_zoom_menu()
settle(800)
browser = ReplayBrowser(app)
root.update()
browser.sess_lb.selection_set(0)
browser._on_session_select()
browser.hand_lb.selection_set(0)
browser._play()
settle(9000)                              # 等 timeline 走完
snap("ui_preview_replay.png")
print("replay_stop:", app.replay_stop, "| actions queued done")
browser._on_close()
root.destroy()
