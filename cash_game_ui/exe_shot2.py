# EXE 真实运行截图：①试用到期→激活弹窗（价格表） ②licensed→主界面 ③新会话对话框
import base64
import ctypes
import hashlib
import hmac
import os
import subprocess
import sys
import time
from ctypes import wintypes

sys.path.insert(0, r"<repo>")
EXE = r"<repo>\dist\PokerAI-Product\poker-ai-gui.exe"
OUT = r"<repo>\cash_game_ui\assets"
u32, g32 = ctypes.windll.user32, ctypes.windll.gdi32


def find_hwnd(title_prefix):
    found = [None]

    @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    def cb(hwnd, _l):
        buf = ctypes.create_unicode_buffer(256)
        u32.GetWindowTextW(hwnd, buf, 256)
        if buf.value.startswith(title_prefix) and u32.IsWindowVisible(hwnd):
            found[0] = hwnd
            return False
        return True
    u32.EnumWindows(cb, 0)
    return found[0]


def shot(hwnd, name):
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
    from PIL import Image
    img = Image.frombuffer("RGBA", (w, h), buf.raw, "raw", "BGRA", 0, 1)
    img.convert("RGB").save(os.path.join(OUT, name))
    g32.DeleteObject(bmp)
    g32.DeleteDC(mem)
    u32.ReleaseDC(hwnd, hdc)
    print("saved", name, f"{w}x{h}")


def launch_and_find(prefix, timeout=40):
    proc = subprocess.Popen([EXE])
    end = time.time() + timeout
    hwnd = None
    while time.time() < end and hwnd is None:
        time.sleep(0.5)
        hwnd = find_hwnd(prefix)
    time.sleep(2.0)                       # 等绘制
    return proc, hwnd


# ── 场景①：试用到期 → 激活弹窗（价格表） ──────────────────────
sys.path.insert(0, r"<repo>")
from cash_game_ui import license as lic

old_date = time.strftime("%Y-%m-%d", time.gmtime(time.time() - 15 * 86400))
lic.STATE_FILE.unlink(missing_ok=True)
lic._save_state({"first_run": old_date})
lic._persist_first_run(old_date)
st = lic.check_license()
print("预设状态:", st["mode"], st["days_left"], "天")
assert st["mode"] == "expired"

proc, hwnd = launch_and_find("激活 poker-ai")
if hwnd:
    shot(hwnd, "ui_exe_activation.png")
proc.terminate()
try:
    proc.wait(timeout=10)
except subprocess.TimeoutExpired:
    proc.kill()
time.sleep(1)

# ── 场景②③：licensed 大师版 → 主界面 + 新会话对话框 ─────────────
mid = lic.machine_id()
key = lic.make_license_key(mid, days=365, tier="max", ref="DIRECT")
lic.activate(key)
st = lic.check_license()
print("恢复状态:", st["mode"], st["tier"])

proc, hwnd = launch_and_find("poker-ai 对打模式")
if hwnd:
    u32.SetForegroundWindow(hwnd)
    time.sleep(1.0)
    shot(hwnd, "ui_exe_main.png")
proc.terminate()
try:
    proc.wait(timeout=10)
except subprocess.TimeoutExpired:
    proc.kill()
print("done")
