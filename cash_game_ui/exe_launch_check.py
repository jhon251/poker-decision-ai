# 启动打包后的 GUI EXE 并截图（开发验证用）
import ctypes
import os
import subprocess
import sys
import time
from ctypes import wintypes

os.chdir(r"<repo>\dist\PokerAI-Product")
proc = subprocess.Popen([r"<repo>\dist\PokerAI-Product\poker-ai-gui.exe"])

u32 = ctypes.windll.user32
hwnd_found = None
end = time.time() + 30
while time.time() < end and hwnd_found is None:
    time.sleep(0.5)

    @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    def enum_cb(hwnd, _l):
        global hwnd_found
        buf = ctypes.create_unicode_buffer(256)
        u32.GetWindowTextW(hwnd, buf, 256)
        if buf.value.startswith("poker-ai 对打模式") and u32.IsWindowVisible(hwnd):
            hwnd_found = hwnd
            return False
        return True
    u32.EnumWindows(enum_cb, 0)

print("hwnd:", hwnd_found)
if hwnd_found:
    time.sleep(2.5)                     # 等界面绘制完成
    g32 = ctypes.windll.gdi32
    rect = wintypes.RECT()
    u32.GetWindowRect(hwnd_found, ctypes.byref(rect))
    w, h = rect.right - rect.left, rect.bottom - rect.top
    hdc = u32.GetWindowDC(hwnd_found)
    mem = g32.CreateCompatibleDC(hdc)
    bmp = g32.CreateCompatibleBitmap(hdc, w, h)
    g32.SelectObject(mem, bmp)
    u32.PrintWindow(hwnd_found, mem, 2)
    class BIH(ctypes.Structure):
        _fields_ = [("biSize", wintypes.DWORD), ("biWidth", wintypes.LONG),
                    ("biHeight", wintypes.LONG), ("biPlanes", wintypes.WORD),
                    ("biBitCount", wintypes.WORD), ("biCompression", wintypes.DWORD),
                    ("biSizeImage", wintypes.DWORD), ("biXPelsPerMeter", wintypes.LONG),
                    ("biYPelsPerMeter", wintypes.LONG), ("biClrUsed", wintypes.DWORD),
                    ("biClrImportant", wintypes.DWORD)]
    info = BIH(); info.biSize = ctypes.sizeof(BIH)
    info.biWidth, info.biHeight, info.biPlanes = w, -h, 1
    info.biBitCount, info.biCompression = 32, 0
    buf = ctypes.create_string_buffer(w * h * 4)
    g32.GetDIBits(mem, bmp, 0, h, buf, ctypes.byref(info), 0)
    from PIL import Image
    img = Image.frombuffer("RGBA", (w, h), buf.raw, "raw", "BGRA", 0, 1)
    img.convert("RGB").save(r"<repo>\cash_game_ui\assets\ui_preview_exe.png")
    print("saved exe screenshot", f"{w}x{h}")
    g32.DeleteObject(bmp); g32.DeleteDC(mem); u32.ReleaseDC(hwnd_found, hdc)

proc.terminate()
try:
    proc.wait(timeout=10)
except subprocess.TimeoutExpired:
    proc.kill()
print("exe terminated")
