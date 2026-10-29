"""完整前端外观巡检截图：主界面 / 新会话对话框 / 复盘浏览器（开发用）。"""
import ctypes
import os
import sys
from ctypes import wintypes

sys.path.insert(0, r"<repo>")
import tkinter as tk

from PIL import Image

from cash_game_ui.play_gui import PlayGuiApp


def snap_hwnd(hwnd, name):
    u32, g32 = ctypes.windll.user32, ctypes.windll.gdi32
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
    return out


root = tk.Tk()
app = PlayGuiApp(root)
root.attributes("-topmost", True)


def settle(ms):
    import time
    end = time.time() + ms / 1000
    while time.time() < end:
        root.update()
        time.sleep(0.02)


settle(600)
snap_hwnd(ctypes.windll.user32.GetParent(root.winfo_id()) or root.winfo_id(),
          "ui_preview_main_idle.png")

# 新会话对话框
app._on_new_session()
settle(400)
dlg = next(w for w in root.winfo_children() if isinstance(w, tk.Toplevel))
dlg.attributes("-topmost", True)
settle(300)
snap_hwnd(ctypes.windll.user32.GetParent(dlg.winfo_id()) or dlg.winfo_id(),
          "ui_preview_dialog.png")
dlg.destroy()

# 复盘浏览器
app._open_replay()
settle(600)
rb = next(w for w in root.winfo_children() if isinstance(w, tk.Toplevel))
rb.attributes("-topmost", True)
settle(300)
snap_hwnd(ctypes.windll.user32.GetParent(rb.winfo_id()) or rb.winfo_id(),
          "ui_preview_browser.png")
rb.destroy()
root.destroy()
