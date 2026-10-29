"""EXE-1 产品构建启动器 — GUI（poker-ai-gui.exe；windowed）。

对打界面入口：chdir 到 EXE 目录（便携布局）→ 清环境变量 → 起 Tk 对打 UI
（cash_game_ui.play_gui，纯本地对打——无屏幕捕捉/自动操作路径）。
注意：旧的 packaging/gui_launch.py（Preview 实时分析壳，含捕捉链）不在
本产品 spec 的入口——产品 GUI 不打包任何捕捉路径。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path


def _app_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


def _selfcheck() -> int:
    """打包自检：冻结环境下验证资产/引擎/新功能模块全部命中。
    结果写 logs/selfcheck.txt；exit 0 = 全部通过。"""
    from pathlib import Path
    if not getattr(sys, "frozen", False):
        sys.path.insert(0, str(_app_dir()))   # 未冻结时补仓库根到 sys.path
    lines: list[str] = []
    ok = True

    def check(name: str, cond: bool, detail: str = "") -> None:
        nonlocal ok
        lines.append(f"{'PASS' if cond else 'FAIL'}  {name}  {detail}")
        ok = ok and cond

    try:
        import cash_game_ui  # noqa: F401
        pkg_dir = Path(cash_game_ui.__file__).resolve().parent
        assets = pkg_dir / "assets"
        for sub, want in (("cards", 55), ("cards_80", 55), ("cards_120", 55),
                          ("cards_small", 55), ("cards_small_80", 55),
                          ("cards_small_120", 55)):
            n = len(list((assets / sub).glob("*.png"))) if (assets / sub).is_dir() else 0
            check(f"assets/{sub}", n == want, f"{n}/{want}")
        felts = len(list(assets.glob("table_felt*.png"))) if assets.is_dir() else 0
        check("assets/table_felt*", felts >= 3, f"{felts}/3")

        from external_harness.engine.nseat_engine import position_labels
        check("engine.position_labels", position_labels(6)[0] == "BTN")
        from external_harness.protocol.cash_match_6max import STAKE_TABLES
        check("stakes", set(STAKE_TABLES) == {"nl2", "nl5", "nl10", "nl25", "nl50"},
              str(sorted(STAKE_TABLES)))
        from poker_decision_ai.play.match import DIFFICULTY_TIERS
        check("difficulty_tiers",
              set(DIFFICULTY_TIERS) == {"easy", "normal", "hard", "max"},
              str(sorted(DIFFICULTY_TIERS)))
        from poker_decision_ai.play.history import PlayHistoryWriter  # noqa: F401
        from cash_game_ui.weakness_report import build_report  # noqa: F401
        from cash_game_ui.i18n import tr  # noqa: F401
        check("modules", True)
        from cash_game_ui import license as lic
        st = lic.check_license()
        check("license", st["mode"] in ("licensed", "trial", "expired"),
              str(st))
    except Exception as exc:                      # noqa: BLE001
        ok = False
        lines.append(f"FAIL  exception  {type(exc).__name__}: {exc}")

    Path("logs").mkdir(exist_ok=True)
    (Path("logs") / "selfcheck.txt").write_text(
        "SELFCHECK " + ("OK" if ok else "FAILED") + "\n"
        + "\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0 if ok else 1


def main() -> int:
    if "--selfcheck" in sys.argv:
        if getattr(sys, "frozen", False):
            os.chdir(_app_dir())
        return _selfcheck()
    if getattr(sys, "frozen", False):
        os.chdir(_app_dir())
        for var in ("PYTHONPATH", "PYTHONHOME"):
            os.environ.pop(var, None)
    try:
        from cash_game_ui.play_gui import main as gui_main
    except Exception as exc:                              # noqa: BLE001
        # windowed 模式无 stderr——错误落文件，保证可诊断
        try:
            Path("logs").mkdir(exist_ok=True)
            Path("logs/gui_startup_error.txt").write_text(
                f"{type(exc).__name__}: {exc}", encoding="utf-8")
        except OSError:
            pass
        raise
    return gui_main()


if __name__ == "__main__":
    raise SystemExit(main())
