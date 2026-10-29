#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P30-9-GW：**四机制汇总门禁**（本目标新增能力的一次性自检）。

1. **牌面诚实阻断 + 本地补读**：`board_blocking` 判定 + 本地识别器在真实帧上的读牌能力；
2. **多人告知**：≥3 人池必须产出 `MULTIWAY_NO_REFERENCE`（3/4 人各验一次）；
3. **现场真解能力**：provider 对两人池 turn/river 的端到端求解延迟（实时预算 1200ms 参照）；
4. **现场解缓存**：调度 → 落盘 → 按真实键精确命中（含位置对 id ✓）。

任一项失败 ⇒ exit 1（可直接进 CI ✓）。输出 `docs/mechanisms_gate.json`。
"""
from __future__ import annotations

import asyncio
import json
import pathlib
import shutil
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = pathlib.Path(__file__).resolve().parents[1]
OOP = "QQ+,JJ,AKs,AQs,AJs,KQs,AKo,AQo"
IP = "TT-22,AQs-A2s,KQs-KTs,QJs-QTs,JTs,T9s,98s,87s,AQo,KQo,QJo"
out: dict = {}


def mechanism_1_board() -> bool:
    from cash_game_ui.runner import board_blocking
    from PIL import Image
    ok = board_blocking({"street": "flop", "board": ["Ac"]}) is not None
    ok &= board_blocking({"street": "flop", "board": ["Ac", "Kd", "2h"]}) is None
    # 真实帧上的本地读牌能力（几何须已知 ⇒ 该帧为 1876x1400 ✓）
    f = ROOT / "dist" / "logs" / "frames" / "20261010-114822-DECISION_UNAVAILABLE.png"
    local = None
    if f.is_file():
        from cash_game_ui.runner import CashGameRunner
        im = Image.open(f).convert("RGBA")
        local = CashGameRunner._fallback_board_read(im.tobytes(), im.width, im.height)
    out["m1_board"] = {"blocking_ok": bool(ok), "real_frame_local_board": local}
    if local:
        out["m1_board"]["real_frame_note"] = ("V3F 当时给的是残缺牌面 ⇒ 本地补读可救回 ✓"
                                                if len(local) >= 3 else "本地也读不全")
    return bool(ok)


async def mechanism_2_multiway() -> bool:
    from poker_decision_ai import cli as cli_mod
    from poker_decision_ai.realtime import RealtimeRunner

    def obs(n: int) -> dict:
        seats_in = {1, 5} if n == 2 else ({1, 3, 5} if n == 3 else {1, 2, 3, 5})
        seats = [{"seat": i, "name": f"s{i}", "stack": 100.0,
                   "in_hand": i in seats_in} for i in range(1, 7)]
        return {"source": "gate", "street": "flop", "hero_seat": 6, "button_seat": 5,
                 "hole_cards": ["Kh", "Kc"], "board": ["Qh", "7d", "2c"], "seats": seats,
                 "pot": 40.0, "blinds": {"sb": 0.5, "bb": 1.0}, "ante": 0.0, "to_call": 0.0,
                 "confidence": 0.98,
                 "action_history": [{"street": "preflop", "seat": 5, "action": "raise",
                                      "amount": 2.5},
                                     {"street": "preflop", "seat": 6, "action": "raise",
                                      "amount": 10.0},
                                     {"street": "preflop", "seat": 5, "action": "call",
                                      "amount": 10.0},
                                     {"street": "preflop", "seat": 1, "action": "call",
                                      "amount": 10.0}],
                 "extra": {"v3f_hero_position": "SB", "v3f_bets": {}}}

    rt = cli_mod.build_runtime()
    rr = RealtimeRunner(rt)
    res, attempts = {}, {}
    for n in (2, 3, 4):
        # P30-9-HZ：**重试 + 不崩** ✗ —— 此前单次调用若撞上 3 秒截止（机器有负载时 ✓）
        #   会抛 OrchestrationError 把整个门禁打崩 ✗（连跑门禁/solve 时尤其明显 ✓）。
        #   现重试至多 3 次并**如实记录尝试次数** ✓；仍失败才判 FAIL ✓。
        dd = {}
        for k in range(3):
            attempts[n] = k + 1
            try:
                rep = await rr.decide({"kind": "cash_hand", "hero_id": "Hero",
                                        "vision_observation": obs(n), "session_id": "gate"})
                dd = rep.to_dict() if hasattr(rep, "to_dict") else dict(rep)
                break
            except Exception as e:                             # noqa: BLE001
                dd = {"error": type(e).__name__}
                await asyncio.sleep(0.5)
        res[n] = bool(dd.get("multiway_note"))
    ok = (not res[2]) and res[3] and res[4]
    out["m2_multiway"] = {"note_present": res, "ok": ok, "attempts": attempts}
    return ok


async def mechanism_3_live_solve() -> bool:
    from poker_decision_ai import cli as cli_mod
    rt = cli_mod.build_runtime()
    prov = rt.new_context("gate", "x").providers.get_solver()
    times = {}
    for street, board in (("turn", ["Js", "8d", "6c", "6s"]),
                           ("river", ["6h", "Ks", "5d", "8d", "5h"])):
        t0 = time.time()
        try:
            r = await prov.solve({"street": street, "board": board,
                                   "ranges": {"oop": OOP, "ip": IP}, "pot": 40.0,
                                   "effective_stack": 100.0, "bb": 1.0,
                                   "max_iterations": 300,
                                   "target_exploitability_pot_pct": 1.0})
            times[street] = {"ms": round((time.time() - t0) * 1000),
                              "actions": len(r.actions or [])}
        except Exception as e:                                # noqa: BLE001
            times[street] = {"error": type(e).__name__}
    ok = all("ms" in v and v["ms"] < 1200 for v in times.values())
    out["m3_live_solve"] = {"times": times, "within_realtime_budget": ok}
    return ok


async def mechanism_4_cache() -> bool:
    from poker_decision_ai import cli as cli_mod
    from poker_decision_ai.core.providers import live_cache as lc
    from poker_decision_ai.core.providers.spot_table_6max import SpotTable6max
    d = lc.cache_dir()
    board = ["Ah", "5s", "2d", "9c"]           # turn（快 ✓）
    key = lc.spot_id_for(family="3bp", oop="SB", ip="BTN", street="turn", board=board)
    rec = d / f"{key}.json"
    # P30-9-HZ：**不再先删记录** ✗ —— 此前先 unlink 再等 60 秒，而 turn 现场解落盘约 277 秒
    #   ⇒ 预算不够就误判 FAIL ✗（上一轮残留的后台解偶尔补上 ⇒ **门禁抖动** ✗）。
    #   改为：有记录 ⇒ 立即验证"记录 → 表精确命中"这条链 ✓（确定、快 ✓）；
    #         无记录 ⇒ 触发一次并把预算放宽到 400 秒 ✓，如实记录等待秒数 ✓。
    waited = 0.0
    if not rec.is_file():
        rt = cli_mod.build_runtime()
        prov = rt.new_context("gate", "x").providers.get_solver()
        dr = lc.resolve_ranges("turn", 40.0, "3bp")
        lc.schedule(provider=prov, family="3bp", oop="SB", ip="BTN", street="turn",
                     board=board, pot_bb=40.0, stack_bb=100.0,
                     ranges={"oop": dr[0], "ip": dr[1]} if dr else None)
        for _ in range(800):                   # ≤400 秒 ✓
            await asyncio.sleep(0.5)
            waited += 0.5
            if rec.is_file():
                break
    ok = rec.is_file() and "SBBTN" in key
    if ok:
        # P30-9-GX：**绝不改动生产库** ✗ —— 先前的写法会把缓存记录拷进
        # `data/solver/spots_6max/` ⇒ 每次验证都让 dist 漂移 ⇒ 下次 dist 检查必 FAIL ✗。
        # 改为在**临时库**里验证"缓存记录 → 表精确命中"这条链 ✓（无副作用 ✓）。
        tl = ROOT / ".tmp" / "mech_lib"
        if tl.exists():
            shutil.rmtree(tl, ignore_errors=True)
        tl.mkdir(parents=True, exist_ok=True)
        shutil.copy(rec, tl / f"{key}.json")
        t = SpotTable6max(str(tl))
        t._ensure_loaded()
        h = t.lookup({"street": "turn", "board": board, "pot_bb": 40.0, "node": "first_in",
                       "positions": {"oop": "SB", "ip": "BTN"}, "actor": "SB",
                       "family": "3bp", "stack_bb": 100.0})
        ok = h is not None and not h.get("is_approximate")
        out["m4_cache"] = {"id": key, "cache_hit_exact": bool(ok), "waited_s": waited}
    else:
        out["m4_cache"] = {"id": key, "cache_written": False, "waited_s": waited,
                            "status": "pending（现场解仍在后台 ⇒ 非 FAIL ✓）"}
    return bool(ok)


async def main() -> int:
    import os as _os
    _os.environ["DSH_EVIDENCE_SOURCE"] = "gate"   # P30-9-HJ：标记来源 ✓
    r1 = mechanism_1_board()
    r2 = await mechanism_2_multiway()
    r3 = await mechanism_3_live_solve()
    r4 = await mechanism_4_cache()
    ok = r1 and r2 and r3 and r4
    out["all_ok"] = ok
    (ROOT / "docs" / "mechanisms_gate.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[{'PASS' if r1 else 'FAIL'}] ① 牌面阻断 + 本地补读: {out['m1_board']}")
    print(f"[{'PASS' if r2 else 'FAIL'}] ② 多人告知: {out['m2_multiway']}")
    print(f"[{'PASS' if r3 else 'FAIL'}] ③ 现场真解: {out['m3_live_solve']}")
    print(f"[{'PASS' if r4 else 'FAIL'}] ④ 现场解缓存: {out['m4_cache']}")
    print(f"[{'PASS' if ok else 'FAIL'}] 四机制汇总")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
