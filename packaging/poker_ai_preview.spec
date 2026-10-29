# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec — poker-ai EXE Preview Build（最小可运行）。

打包命令（仓库根目录执行）：
    pyinstaller --clean --noconfirm packaging/poker_ai_preview.spec

产出：dist/poker-ai-preview.exe（onefile console）。
分发：EXE + configs/ + examples/hands/（布局与验证见
docs/EXE_PREVIEW_BUILD_REPORT.md）。

红线：不改 P30/P31/P32/P33 核心语义；不打包自动下注/点击能力；
REAL Solver 不随包分发（诚实 mock/unavailable）。
"""
from pathlib import Path

APP_NAME = "poker-ai-preview"

# PyInstaller 以 spec 所在目录解析相对路径——用 SPECPATH 锚定仓库根
ROOT = Path(SPECPATH).resolve().parent

block_cipher = None

a = Analysis(
    [str(ROOT / "packaging" / "preview_launch.py")],
    pathex=[str(ROOT)],
    binaries=[],
    datas=[
        # 注意：本 Preview 采用"便携目录"分发——configs/ 与 examples/hands/
        # 外置于 EXE 旁边（用户可编辑）；启动器 chdir 到 EXE 目录后按相对
        # 路径读取。包内不塞数据副本，路径语义单一（EXE 目录为根）。
    ],
    hiddenimports=[
        # 显式钉住运行时依赖（PyInstaller 对以下包的静态扫描可达，
        # 这里显式声明以防 hook 版本差异）
        "yaml",
        "jsonschema",
        "dotenv",
        "httpx",
        "numpy",
        "poker_decision_ai.core.gamemode.plugin",
        "poker_decision_ai.domain.poker.gamemodes",
        # P40：P36 连续生产闭环（无 CLI import 链可达——显式钉住，
        # 保证冻结包内含 continuous_runner/production_decision_loop 语义）
        "poker_decision_ai.continuous_runner",
        "poker_decision_ai.production_decision_loop",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # 纯体积排除：预览 EXE 不需要的重资源子系统
        # （matplotlib 等并非依赖——显式列出防环境里多余包被拖入）
        "matplotlib", "tkinter", "IPython", "jedi",
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
    upx=False,                        # 预览不压缩——避免误报与解压时间
    console=True,                     # console 应用：stdout JSON 可管道
    icon=None,
)
