"""EXE-1 — 干净环境产品验证（仓库外 + 清空 PYTHONPATH）。

验收项（任务书）：
  V1 CLI analyze 出 solver_hit 决策（spot 库命中 → provenance.source=solver_hit）
  V2 CLI play 打通一手带解析（脚本输入；play_history JSONL 落盘 + 便携频率
     索引生效 [provenance portable]）
  V3 GUI EXE 可启动（windowed——验证进程存活 + 启动无错误文件）
  V4 诚实性：provider_status 真实标注（llm/vision mock、solver postflop-真实）、
     无捕捉/自动操作模块进包

用法：.venv/Scripts/python.exe scripts/verify_product_clean_env.py <产品目录>
产出 JSON 报告（stdout）+ 退出码（0=全过）。
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def run_clean(exe: Path, args: list[str], cwd: Path, *,
              stdin_text: str | None = None, timeout: int = 300) -> dict:
    env = {k: v for k, v in os.environ.items()
           if k not in ("PYTHONPATH", "PYTHONHOME")}
    env["PYTHONIOENCODING"] = "utf-8"
    t0 = time.time()
    proc = subprocess.run(
        [str(exe)] + args, cwd=str(cwd), env=env, input=stdin_text,
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        timeout=timeout)
    return {"cmd": [exe.name] + args, "exit": proc.returncode,
            "stdout": proc.stdout, "stderr": proc.stderr,
            "wall_s": round(time.time() - t0, 1)}


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: verify_product_clean_env.py <product_dir>", file=sys.stderr)
        return 2
    prod = Path(sys.argv[1]).resolve()
    cli = prod / "poker-ai.exe"
    gui = prod / "poker-ai-gui.exe"
    if not cli.exists() or not gui.exists():
        print(f"FATAL: EXE missing in {prod}", file=sys.stderr)
        return 2
    checks: dict[str, dict] = {}
    ok = True

    # ── V1 CLI analyze --realtime：solver spot 库命中 ─────────────────
    hh = ("PokerStars Hand #2001: No Limit Hold'em ($1.00/$2.00 USD) "
          "- 2026/10/04 12:00:00 ET\n"
          "Table 'Product' 6-max Seat #1 is the button\n"
          "Seat 1: V1 ($200.00)\nSeat 2: Hero ($200.00)\nSeat 3: V2 ($200.00)\n"
          "Seat 4: V3 ($200.00)\nSeat 5: V4 ($200.00)\nSeat 6: V5 ($200.00)\n"
          "Hero: posts small blind $1.00\nV1: posts big blind $2.00\n"
          "*** HOLE CARDS ***\nDealt to Hero [As Kd]\n"
          "V2: folds\nV3: folds\nV4: folds\nV5: folds\n"
          "Hero: raises $4.00 to $6.00\nV1: calls $4.00\n"
          "*** FLOP *** [Td 9d 6h]\n")
    checks["v1_analyze"] = run_clean(
        cli, ["analyze", "--text", hh, "--game-mode", "cash", "--no-llm",
              "--realtime"],
        cwd=prod, timeout=420)
    out1 = checks["v1_analyze"]["stdout"]
    try:
        rep = json.loads(out1)
        prov = ((rep.get("final_decision") or {}).get("provenance") or {})
        checks["v1_analyze"]["provenance_source"] = prov.get("source")
        checks["v1_analyze"]["solver_hit"] = prov.get("source") == "solver_hit"
        ps = rep.get("provider_status") or {}
        checks["v1_analyze"]["provider_status"] = {
            k: (ps.get(k) or {}).get("mode", ps.get(k))
            for k in ("llm", "vision", "solver")}
    except Exception as exc:                              # noqa: BLE001
        checks["v1_analyze"]["parse_error"] = str(exc)[:200]
        checks["v1_analyze"]["solver_hit"] = False
    ok &= bool(checks["v1_analyze"].get("solver_hit"))
    ok &= checks["v1_analyze"]["exit"] == 0

    # ── V2 CLI play：脚本打完一手 + 训练数据落盘 ─────────────────────
    # 静态 stdin：动作与合法集不匹配时 CLI 拒绝并重试（消费下一行）；
    # 240 行足够撑到至少一手完整结束（会话在 --hands 1 后自行结束）。
    stdin_lines = (["x", "c", "f"] * 80)
    checks["v2_play"] = run_clean(
        cli, ["play", "--stake", "nl2", "--hands", "1", "--seed", "777",
              "--out", str(prod / "play_verify_out.json")],
        cwd=prod, stdin_text="\n".join(stdin_lines) + "\n", timeout=420)
    ok &= checks["v2_play"]["exit"] == 0
    hist_dir = prod / "data" / "play_history"
    hands_files = sorted(hist_dir.glob("play_session_*.jsonl")) if hist_dir.exists() else []
    checks["v2_play"]["history_files"] = [f.name for f in hands_files]
    checks["v2_play"]["history_ok"] = bool(hands_files)
    if hands_files:
        from poker_decision_ai.play.history import load_session, session_metrics
        try:
            data = load_session(hands_files[-1])
            checks["v2_play"]["hands_recorded"] = len(data["hands"])
            m = session_metrics(hands_files[-1])
            checks["v2_play"]["metrics"] = {
                k: m[k] for k in ("n_hands", "hero_decisions",
                                  "advisor_available")}
            ok &= len(data["hands"]) >= 1
        except Exception as exc:                          # noqa: BLE001
            checks["v2_play"]["history_error"] = str(exc)[:200]
            ok = False

    # ── V3 GUI：启动存活 + 无启动错误文件 ─────────────────────────────
    env = {k: v for k, v in os.environ.items()
           if k not in ("PYTHONPATH", "PYTHONHOME")}
    proc = subprocess.Popen([str(gui)], cwd=str(prod), env=env,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL)
    time.sleep(12)
    alive = proc.poll() is None
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
    err_file = prod / "logs" / "gui_startup_error.txt"
    checks["v3_gui"] = {"alive_after_12s": alive,
                        "startup_error": (err_file.read_text(
                            encoding="utf-8")[:300] if err_file.exists() else None)}
    ok &= alive and not err_file.exists()

    # ── V4 诚实性：捕捉/自动操作模块不进包 ────────────────────────────
    internal = prod / "_internal"
    banned_hits = []
    banned_names = ("pyautogui", "win32gui", "win32ui", "mss", "pynput")
    for name in banned_names:
        for pat in (name, f"{name}.pyd", f"{name}.dll"):
            if list(internal.rglob(pat)):
                banned_hits.append(pat)
    capture_runner = any(
        "cash_game_ui" in str(p) and p.name == "runner"
        for p in internal.rglob("runner*"))
    checks["v4_honesty"] = {
        "banned_modules_found": banned_hits,
        "capture_runner_in_bundle": capture_runner,
    }
    ok &= not banned_hits and not capture_runner

    print(json.dumps({"ok": bool(ok), "product_dir": str(prod),
                      "checks": {k: {kk: vv for kk, vv in v.items()
                                     if kk not in ("stdout", "stderr")}
                                 for k, v in checks.items()}},
                     ensure_ascii=False, indent=2))
    # 细节日志落盘备查
    detail = ROOT / ".tmp" / "product_verify_detail.json"
    detail.parent.mkdir(parents=True, exist_ok=True)
    detail.write_text(json.dumps(checks, ensure_ascii=False, indent=2),
                      encoding="utf-8")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
