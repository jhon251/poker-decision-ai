# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec — EXE-1 产品构建（双 EXE onedir + 单 COLLECT）。

打包命令（仓库根目录执行）：
    pyinstaller --clean --noconfirm packaging/poker_ai_product.spec

产出：dist/poker-ai-product/
    poker-ai.exe          CLI（analyze/play/replay 全子命令；console）
    poker-ai-gui.exe      对打界面（windowed）
    _internal/            共享运行时（两 EXE 单 COLLECT，体积只算一份）

分发组装（scripts/assemble_product.py 把外置数据拷到 EXE 旁）：
    configs/               配置（用户可编辑）
    data/solver/           spot 库（真实 CFR 查表；缺失时 solver 诚实降级）
    data/cache/frequency_prior_portable.json   频率先验便携索引（population 级）
    tools/bin/postflop-bench-cli.exe           真实求解器 CLI

红线：非实时辅助——不打包任何屏幕捕捉/自动操作模块；provider_status
如实（mock 标 mock）；诚实性守恒由验证脚本核验（verify_product_clean_env.py）。
"""
from pathlib import Path

APP_NAME = "poker-ai-product"

# PyInstaller 以 spec 所在目录解析相对路径——用 SPECPATH 锚定仓库根
ROOT = Path(SPECPATH).resolve().parent

block_cipher = None

COMMON_EXCLUDES = [
    "matplotlib", "IPython", "jedi", "setuptools", "pyinstaller",
    "pytest", "torch", "tensorflow", "pandas", "scipy",
]

# ---- CLI：全子命令（analyze/play/replay/agents/schemas/research/metrics/modes）
a_cli = Analysis(
    [str(ROOT / "packaging" / "product_launch.py")],
    pathex=[str(ROOT)],
    binaries=[],
    datas=[],
    hiddenimports=[
        # 运行时依赖
        "yaml", "jsonschema", "dotenv", "httpx", "numpy",
        # 决策引擎（EXE-1 全量进包）
        "poker_decision_ai.core.gamemode.plugin",
        "poker_decision_ai.domain.poker.gamemodes",
        "poker_decision_ai.continuous_runner",
        "poker_decision_ai.production_decision_loop",
        # R13-EXE-2：一键识别链（惰性导入——显式钉住；只读 BitBlt 捕捉，
        # 无自动点击/下注）
        "poker_decision_ai.frame_decision",
        "poker_decision_ai.vision2.pixel.realclient.capture",
        "poker_decision_ai.vision2.pixel.realclient.production_bridge",
        "scripts.phase_v2s_auto_capture",
        # 对打模式（引擎复用 external_harness——不复制语义）
        "poker_decision_ai.play.match",
        "poker_decision_ai.play.console",
        "poker_decision_ai.play.history",
        "external_harness.engine",
        "external_harness.view",
        "external_harness.adapter.judge_adapter",
        "external_harness.adapter.core_adapter",
        "external_harness.protocol.cash_match_6max",
        "external_harness.protocol.bench_match",
        "external_harness.protocol.bench_bots",
        "external_harness.protocol.pool_bots",
        "external_harness.protocol.baseline",
        "external_harness.protocol.rake_model",
        "external_harness.vendor.cfr_poker_bot_engine.engine",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=COMMON_EXCLUDES + ["tkinter"],
    cipher=block_cipher,
    noarchive=False,
)

# ---- GUI：对打界面（windowed；tkinter 进包）
a_gui = Analysis(
    [str(ROOT / "packaging" / "play_gui_launch.py")],
    pathex=[str(ROOT)],
    binaries=[],
    datas=[
        # 牌面/毛毡 PNG（7 套缩放档 × 55 + 3 毛毡）—— _ASSET_DIR 相对
        # cash_game_ui/__file__ 解析，目标路径必须保持 cash_game_ui/assets
        (str(ROOT / "cash_game_ui" / "assets"), "cash_game_ui/assets"),
    ],
    hiddenimports=[
        "yaml", "jsonschema", "dotenv", "httpx", "numpy",
        "poker_decision_ai.core.gamemode.plugin",
        "poker_decision_ai.domain.poker.gamemodes",
        "poker_decision_ai.continuous_runner",
        "poker_decision_ai.production_decision_loop",
        "poker_decision_ai.play.match",
        "poker_decision_ai.play.console",
        "poker_decision_ai.play.history",
        "cash_game_ui.play_gui",
        "cash_game_ui.weakness_report",
        "poker_decision_ai.realtime",
        "poker_decision_ai.replay",
        "poker_decision_ai.core.orchestration.runtime",
        "cash_game_ui.i18n",
        "cash_game_ui.license",
        "external_harness.engine",
        "external_harness.view",
        "external_harness.adapter.judge_adapter",
        "external_harness.adapter.core_adapter",
        "external_harness.protocol.cash_match_6max",
        "external_harness.protocol.bench_match",
        "external_harness.protocol.bench_bots",
        "external_harness.protocol.pool_bots",
        "external_harness.protocol.baseline",
        "external_harness.protocol.rake_model",
        "external_harness.vendor.cfr_poker_bot_engine.engine",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=COMMON_EXCLUDES,
    cipher=block_cipher,
    noarchive=False,
)

pyz_cli = PYZ(a_cli.pure, a_cli.zipped_data, cipher=block_cipher)
pyz_gui = PYZ(a_gui.pure, a_gui.zipped_data, cipher=block_cipher)

exe_cli = EXE(
    pyz_cli, a_cli.scripts, [], exclude_binaries=True,
    name="poker-ai", debug=False, bootloader_ignore_signals=False,
    strip=False, upx=False, console=True,
    icon=str(ROOT / "packaging" / "poker_ai.ico"),
)
exe_gui = EXE(
    pyz_gui, a_gui.scripts, [], exclude_binaries=True,
    name="poker-ai-gui", debug=False, bootloader_ignore_signals=False,
    strip=False, upx=False, console=False,
    icon=str(ROOT / "packaging" / "poker_ai.ico"),
)

coll = COLLECT(
    exe_cli, a_cli.binaries, a_cli.zipfiles, a_cli.datas,
    exe_gui, a_gui.binaries, a_gui.zipfiles, a_gui.datas,
    strip=False, upx=False, name=APP_NAME,
)
