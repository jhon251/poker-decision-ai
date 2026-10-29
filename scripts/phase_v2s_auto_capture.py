"""Phase V2S — CoinPoker 自动捕获脚本（只读，Observe→Capture→Annotate）。

启动后自动：定位 CoinPoker → 持续捕获 → FrameStream 去重 → ROI digest
触发（pot/button/SB 徽章/table）+ 场景兜底 → 保存证据帧 → annotation queue。
用户正常玩牌，无需任何手动指令。

V3F-P30-12-C：可选 --decide 接入 ProductionDecisionLoop（安全接线——
每帧 tick → decision（幂等/去重）；输出 DecisionResult 供人/服务消费；
不自动点击/下注——仅当手动指定 --decide 时启用）。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from poker_decision_ai.vision2.pixel.realclient.annotate import (  # noqa: E402
    AutoCaptureEngine)
from poker_decision_ai.vision2.pixel.realclient.capture import (  # noqa: E402
    CAPTURE_OK, CtypesBitBltBackend)
from poker_decision_ai.vision2.pixel.realclient.runtime import (  # noqa: E402
    FrameStream)
from poker_decision_ai.vision2.pixel.realclient.seat_position import (  # noqa: E402
    BoardObserver, HeroCardsObserver, PositionObserver, PotObserver,
    StackObserver, StatusObserver, StreetObserver)
from poker_decision_ai.vision2.pixel.realclient.bet_observer import (  # noqa: E402
    BetObserver)
from poker_decision_ai.vision2.pixel.realclient.action_observer import (  # noqa: E402
    ActionObserver)
from poker_decision_ai.vision2.pixel.realclient.runtime_snapshot import (  # noqa: E402
    build_snapshot)

COIN_KEYWORD = "CoinPoker"


def locate_coinpoker() -> dict | None:
    import ctypes
    from ctypes import wintypes
    u32 = ctypes.windll.user32
    found = []

    @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    def cb(hwnd, _):
        if not u32.IsWindowVisible(hwnd):
            return True
        ln = u32.GetWindowTextLengthW(hwnd)
        if ln <= 0:
            return True
        b = ctypes.create_unicode_buffer(ln + 1)
        u32.GetWindowTextW(hwnd, b, ln + 1)
        if COIN_KEYWORD in b.value:
            r = wintypes.RECT()
            u32.GetWindowRect(hwnd, ctypes.byref(r))
            found.append({"hwnd": int(hwnd),
                          "bbox": [r.left, r.top, r.right - r.left,
                                   r.bottom - r.top]})
        return True

    u32.EnumWindows(cb, 0)
    # 取最大的可见窗口（牌桌通常最大）
    return max(found, key=lambda w: w["bbox"][2] * w["bbox"][3]) if found \
        else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=300.0)
    ap.add_argument("--fps", type=float, default=2.0)
    ap.add_argument("--out-dir", default="data/vision2/coinpoker_evidence",
                    help="帧落盘目录（V-3 新语料用 coinpoker_v3 与旧库隔离）")
    ap.add_argument("--decide", action="store_true",
                    help="V3F-P30-12-C：接入 ProductionDecisionLoop（安全"
                         "接线——幂等 tick；仅输出 DecisionResult，不自动"
                         "点击/下注）")
    args = ap.parse_args()
    # V3F-P30-12-C：决策循环（可选——仅在 --decide 时构建；复用既有 runtime）
    decision_loop = None
    decision_log: list = []
    if args.decide:
        try:
            from poker_decision_ai.core.orchestration.runtime import (
                build_runtime)
            from poker_decision_ai.production_decision_loop import (
                ProductionDecisionLoop)
            decision_loop = ProductionDecisionLoop(
                build_runtime(), hero_id="Hero", throttle_s=0.5)
            print("decision_loop: enabled (output-only; no auto-click)")
        except Exception as e:  # pragma: no cover - defensive
            print(f"decision_loop init failed: {e} → disabled")
            decision_loop = None
    win = locate_coinpoker()
    if win is None:
        print("COINPOKER_NOT_FOUND → BLOCK")
        return 3
    print("CoinPoker window:", win)
    x0, y0, cw, chh = win["bbox"]
    profile = json.loads(Path("data/vision2/reallayout_coinpoker_v1.json")
                         .read_text(encoding="utf-8"))
    engine = AutoCaptureEngine(
        profile, args.out_dir, cooldown_s=0.5,
        fps=args.fps)
    engine.window_ref = win
    backend = CtypesBitBltBackend()
    stream = FrameStream(fps=args.fps)
    # V3F-P15：观察者按帧几何 (cw,chh) 自动派发 render class（不再硬编码
    # window="1419"——1876 帧自动切 1848 锚点，消除跨 render class 污染）。
    position_observer = PositionObserver()                # V3E-P3 hook
    pot_observer = PotObserver()                          # V3F-P1 hook
    board_observer = BoardObserver()                      # V3F-P2 hook
    street_observer = StreetObserver()                    # V3F-P3 hook
    hero_cards_observer = HeroCardsObserver()             # V3F-P4 hook
    status_observer = StatusObserver()                    # V3F-P7 hook
    stack_observer = StackObserver()                      # V3F-P8 hook
    bet_observer = BetObserver()                          # V3F-P13-C hook
    action_observer = ActionObserver()                    # V3F-P13-ACTION hook
    frame_id = 0
    t_end = time.time() + args.seconds
    run_stats = {"frames_captured": 0, "frames_processed": 0,
                 "frames_dropped": 0, "duplicated": 0,
                 "capture_failures": 0}
    while time.time() < t_end:
        r = backend.capture()
        run_stats["frames_captured"] += 1
        if r["status"] != CAPTURE_OK:
            run_stats["capture_failures"] += 1
            continue
        arr = np.frombuffer(r["image_bytes"], dtype=np.uint8).reshape(
            r["height"], r["width"], 4)
        if x0 + cw > r["width"] or y0 + chh > r["height"]:
            cw = min(cw, r["width"] - x0)
            chh = min(chh, r["height"] - y0)
        sub = arr[y0:y0 + chh, x0:x0 + cw]
        acc = stream.accept(frame_id + 1, r["captured_at"],
                            ImageDataShim(sub.tobytes()))
        if not acc["accepted"]:
            if acc["reason"] == "duplicate_frame":
                run_stats["duplicated"] += 1
            else:
                run_stats["frames_dropped"] += 1
            continue
        frame_id += 1
        run_stats["frames_processed"] += 1
        engine.process_frame(frame_id, sub.tobytes(), cw, chh)
        # V3E-P3：Position Runtime 消费者（新增，零 capture 行为改变——
        # 结果不反馈 triggers/ROI/保存；仅消费当前帧）
        position_observer.observe(sub.tobytes(), cw, chh)
        # V3F-P1：Pot Runtime 消费者（同上，纯消费者——零 capture 行为改变）
        pot_observer.observe(sub.tobytes(), cw, chh)
        # V3F-P2：Board Runtime 消费者（同上，纯消费者——零 capture 行为改变；
        # 不推导 Street）
        board_observer.observe(sub.tobytes(), cw, chh)
        # V3F-P3：Street Runtime 消费者（冻结状态机；仅消费证据——零 capture
        # 行为改变；不修改 Board）
        street_observer.observe(sub.tobytes(), cw, chh)
        # V3F-P4：Hero Cards Runtime 消费者（纯消费者——零 capture 行为改变）
        hero_cards_observer.observe(sub.tobytes(), cw, chh)
        # V3F-P7：Player Status Runtime 消费者（纯消费者——零 capture 行为改变）
        status_observer.observe(sub.tobytes(), cw, chh)
        # V3F-P8：Stack Runtime 消费者（纯消费者——零 capture 行为改变）
        stack_observer.observe(sub.tobytes(), cw, chh)
        # V3F-P13-C：Bet Runtime 消费者（纯消费者——零 capture 行为改变；
        # render-class 按帧几何 (cw,chh) 选择，非 frame id / scene / 值）
        bet_observer.observe(sub.tobytes(), cw, chh)
        # V3F-P13-ACTION：Action Runtime 消费者（纯消费者——零 capture 行为
        # 改变；只透传 ActionRecognizer 当前帧结果，不推导已执行动作）
        action_observer.observe(sub.tobytes(), cw, chh)
        # V3F-P30-12-C：--decide 时接入 ProductionDecisionLoop（安全接线——
        # 幂等 tick；输出 DecisionResult 记录；不自动点击/下注）
        if decision_loop is not None:
            dres = decision_loop.tick(frame_id, sub.tobytes(), cw, chh)
            if dres["decision_status"] in ("READY", "UNAVAILABLE"):
                decision_log.append({**dres, "captured_at":
                                     r["captured_at"]})
        time.sleep(max(0.0, 1.0 / args.fps - 0.03))
    report = {"run_stats": run_stats, **engine.to_dict(),
              "position_runtime": position_observer.summary(),
              "pot_runtime": pot_observer.summary(),
              "board_runtime": board_observer.summary(),
              "street_runtime": street_observer.summary(),
              "hero_cards_runtime": hero_cards_observer.summary(),
              "status_runtime": status_observer.summary(),
              "stack_runtime": stack_observer.summary(),
              "bet_runtime": bet_observer.summary(),
              "action_runtime": action_observer.summary(),
              "decision_log": decision_log,       # V3F-P30-12-C（--decide 时）
              "runtime_snapshot": build_snapshot({
                  "position_runtime": position_observer,
                  "pot_runtime": pot_observer,
                  "board_runtime": board_observer,
                  "street_runtime": street_observer,
                  "hero_cards_runtime": hero_cards_observer,
                  "status_runtime": status_observer,
                  "stack_runtime": stack_observer,
                  "bet_runtime": bet_observer,
                  "action_runtime": action_observer}),
              "window": win, "fps": args.fps}
    Path("data/vision2/coinpoker_auto_capture.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=1, ensure_ascii=False)[:2500])
    return 0


class ImageDataShim:
    def __init__(self, data):
        self.data = data


if __name__ == "__main__":
    raise SystemExit(main())
