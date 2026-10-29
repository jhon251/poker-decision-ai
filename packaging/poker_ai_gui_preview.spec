# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec — poker-ai GUI Preview（最小常驻 GUI 层）。

打包命令（仓库根目录执行）：
    pyinstaller --clean --noconfirm packaging/poker_ai_gui_preview.spec

产出：dist/poker-ai-gui-preview.exe（onefile console + Tk 窗口）。
分发：EXE + configs/ + examples/hands/（与 console 版一致；
见 docs/EXE_PREVIEW_BUILD_REPORT.md 与 GUI 报告）。

与 console 版 spec 的差异：
  - 入口 packaging/gui_launch.py（Tk 常驻窗口）；
  - excludes 不含 tkinter（GUI 需要；tk/tk8.6 DLL 由 PyInstaller 自动捆绑）；
  - hiddenimports 增补 cash_game_ui.* 与 scripts.phase_v2s_auto_capture
    （runner.capture_frame 的惰性导入——实时捕获路径所需）。

红线：不改 P30/P31/P32/P33 语义；无自动下注/点击/鼠标键盘；无定时采集；
REAL Solver 不随包分发（诚实 mock/unavailable）。
"""
from pathlib import Path

APP_NAME = "poker-ai-gui-preview"

# PyInstaller 以 spec 所在目录解析相对路径——用 SPECPATH 锚定仓库根
ROOT = Path(SPECPATH).resolve().parent

block_cipher = None

a = Analysis(
    [str(ROOT / "packaging" / "gui_launch.py")],
    pathex=[str(ROOT)],
    binaries=[],
    datas=[
        # 资源外置（便携目录分发）：包内不塞数据副本——EXE 目录为根
    ],
    hiddenimports=[
        "yaml",
        "jsonschema",
        "dotenv",
        "httpx",
        "numpy",
        "poker_decision_ai.core.gamemode.plugin",
        "poker_decision_ai.domain.poker.gamemodes",
        # GUI 链
        "cash_game_ui",
        "cash_game_ui.app",
        "cash_game_ui.runner",
        "cash_game_ui.preview_gui",
        # 实时捕获（惰性导入，显式钉住）
        "scripts.phase_v2s_auto_capture",
        "poker_decision_ai.vision2.pixel.realclient.capture",
        "poker_decision_ai.vision2.pixel.realclient.production_bridge",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # 纯体积排除（GUI 不需要的重资源；注意：tkinter 必须保留）
        "matplotlib", "IPython", "jedi",
        "setuptools", "pyinstaller",
    ],
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name=APP_NAME,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,                        # 预览不压缩
    console=True,                     # console 保留：stderr 诚实横幅 + 异常可见
    icon=None,
)
