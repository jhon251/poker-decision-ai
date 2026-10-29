"""GATE 战役——单段调度入口（schtasks 每 20 分钟触发；锁防重叠）。

设计（R6 抗收割协议）：
  * 每次Invocation 只跑**一个**缺失段（~25-33 min < ~45-48 min 外杀窗口）；
  * 锁文件（.tmp/m4_gate_<campaign>_supervisor.lock，mtime 新鲜度 45 min）
    防重叠；
  * 全部段落盘后自删计划任务并退出；
  * worker 数（布局）每战役冻结——bot rng 消费 worker 内相对 hand_index，
    跨布局段不可比（见 M4 报告 §3.2）；任何续跑必须保持同布局：
      - gate_min：8（R6 后期回到验证过的 8 路）；
      - gate_gate1：8（R10 冻结；段 ~25-33 min，外杀窗口内）；
  * 段工件 = 驱动同构（run_m4_gate_campaign 的 _seat_configs/_run_segment/
    _merge 原样复用），落 docs/m4_gate/<campaign>/s{NNN}.{json,records.jsonl}。

用法（由计划任务调用；也可手动/驱动循环触发一段）：
  <repo>/.venv/Scripts/python.exe <repo>/scripts/run_m4_gate_segment_once.py \
      [--campaign gate_gate1]
"""
import argparse
import json
import pathlib
import sys
import time
from concurrent.futures import ProcessPoolExecutor

ROOT = pathlib.Path(r"<repo>")
sys.path.insert(0, str(ROOT))

#: 每战役冻结的 worker 布局（改动 = 跨布局不可比，禁止在战役中途改）
CAMPAIGN_WORKERS = {"gate_min": 8, "gate_gate1": 8}
DEFAULT_CAMPAIGN = "gate_min"   # 向后兼容（gate_min 已 50/50 完结，休眠路径）

LOCK_FRESH_SEC = 45 * 60


def log(LOG: pathlib.Path, msg: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} | {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def next_missing(spec: dict, out_dir: pathlib.Path) -> int | None:
    n_segs = spec["n_hands"] // spec["segment_hands"]
    for i in range(n_segs):
        if not ((out_dir / f"s{i:03d}.json").is_file()
                and (out_dir / f"s{i:03d}.records.jsonl").is_file()):
            return i
    return None


def main() -> int:
    import os
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", default=DEFAULT_CAMPAIGN,
                    choices=sorted(CAMPAIGN_WORKERS))
    args = ap.parse_args()
    campaign = args.campaign
    os.chdir(str(ROOT))          # 计划任务默认 cwd=System32；config/storage 全相对路径
    OUT_DIR = ROOT / "docs" / "m4_gate" / campaign
    LOG = ROOT / ".tmp" / f"{campaign}_supervisor.log"
    LOCK = ROOT / ".tmp" / f"{campaign}_supervisor.lock"
    TASK_NAME = f"M4{campaign.capitalize()}SegmentRunner"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # ── 并行驱动分区：M4_CELL_MOD/M4_CELL_RES 声明本实例只认 段号%MOD==RES
    #    的格子；多实例各管一个剩余类互不争抢（单实例时 MOD=0 全管）──
    _mod = int(os.environ.get("M4_CELL_MOD", "0") or 0)
    _res = int(os.environ.get("M4_CELL_RES", "0") or 0)
    if _mod > 1:
        LOCK = ROOT / ".tmp" / f"{campaign}_supervisor_r{_res}.lock"
    # ── 锁检查（防重叠：段长 ~25-33 min > 触发间隔 20 min）──
    if LOCK.is_file():
        age = time.time() - LOCK.stat().st_mtime
        if age < LOCK_FRESH_SEC:
            print(f"locked (age={int(age)}s) — another segment running",
                  flush=True)
            return 0
        log(LOG, f"stale lock (age={int(age)}s) — reclaiming")
    LOCK.write_text(time.strftime("%Y-%m-%d %H:%M:%S"), encoding="utf-8")
    try:
        from scripts.run_m4_gate_campaign import (
            CAMPAIGNS, _merge, _run_segment, _seat_configs)
        spec = CAMPAIGNS[campaign]
        n_segs = spec["n_hands"] // spec["segment_hands"]
        all_missing = [i for i in range(n_segs)
                       if not ((OUT_DIR / f"s{i:03d}.json").is_file()
                               and (OUT_DIR / f"s{i:03d}.records.jsonl")
                               .is_file())]
        if _mod > 1:
            all_missing = [i for i in all_missing if i % _mod == _res]
        seg_idx = all_missing[0] if all_missing else None
        if seg_idx is None:
            log(LOG, "all segments complete — self-removing scheduled task")
            LOCK.unlink(missing_ok=True)
            import subprocess
            subprocess.run(["schtasks", "/Delete", "/TN", TASK_NAME, "/F"],
                           capture_output=True)
            return 0
        seg = f"s{seg_idx:03d}"
        off = seg_idx * spec["segment_hands"]
        n = spec["segment_hands"]
        seats = _seat_configs(campaign)
        WORKERS = CAMPAIGN_WORKERS[campaign]
        chunk = n // WORKERS
        jobs = [{"campaign": campaign, "seg": seg, "w": w,
                 "hand_lo": off + w * chunk,
                 "n_hands": (off + n) - (off + w * chunk) if w == WORKERS - 1
                 else chunk,
                 "seed_base": spec["seed_base"], "seats": seats}
                for w in range(WORKERS)]
        log(LOG, f"{seg} start off={off} n={n} workers={WORKERS}")
        t0 = time.perf_counter()
        with ProcessPoolExecutor(max_workers=WORKERS) as ex:
            parts = list(ex.map(_run_segment, jobs))
        # ── 编码修复（合并前）：部分 worker 以 Windows 默认编码写决策仓库，
        #    GBK 字节会让 _merge 的 UTF-8 读取崩溃。逐行 GBK→UTF-8 转码。──
        import glob as _glob
        for _rf in _glob.glob(str(ROOT / "data" / f"m4_gate_{campaign}"
                                      / "repository_*" / "decisions.jsonl")):
            _raw = open(_rf, "rb").read()
            try:
                _raw.decode("utf-8")
                continue
            except UnicodeDecodeError:
                pass
            _out, _fixed = [], 0
            for _ln in _raw.split(b"\n"):
                try:
                    _ln.decode("utf-8"); _out.append(_ln); continue
                except UnicodeDecodeError:
                    pass
                try:
                    _out.append(_ln.decode("gbk").encode("utf-8"))
                except UnicodeDecodeError:
                    _out.append(_ln.decode("utf-8", errors="replace")
                                .encode("utf-8"))
                _fixed += 1
            open(_rf, "wb").write(b"\n".join(_out))
            log(LOG, f"encoding repair: {_rf.rsplit(chr(92),1)[-1]} "
                     f"lines={_fixed}")
        agg = _merge(parts, campaign)
        agg["seat_configs_public"] = {
            str(s): {"stake": c["bot_params"]["stake"],
                     "archetype": c["bot_params"]["archetype"]}
            for s, c in sorted(seats.items())}
        agg["composition_note"] = spec["composition_note"]
        agg["seed_derivation"] = ("deal_seed = (seed_base + "
                                  "hand_index*0x9E3779B1) mod 2^64; "
                                  "hand_index 绝对（= hand_lo + i）；"
                                  "bot rng = f(bot_seed, seat, "
                                  "hand_index, decision_index)")
        if spec.get("pool_params_provenance"):
            agg["pool_params_provenance"] = spec["pool_params_provenance"]
        (OUT_DIR / f"{seg}.json").write_text(
            json.dumps(agg, ensure_ascii=False, indent=1), encoding="utf-8")
        (OUT_DIR / f"{seg}.records.jsonl").write_text(
            "\n".join(json.dumps(r, ensure_ascii=False)
                      for p in parts for r in p["records"]) + "\n",
            encoding="utf-8")
        missing = next_missing(spec, OUT_DIR)
        # missing = 下一缺失段索引 → 已完成段数 = missing（索引前全体齐）
        done = n_segs if missing is None else missing
        log(LOG, f"{seg} done rakefree={agg['bb_per_100_rakefree']} "
                 f"net={agg['bb_per_100_net']} "
                 f"({round((time.perf_counter() - t0) / 60, 1)}min) "
                 f"[{done}/{n_segs} segments]")
        return 0
    except Exception as exc:                     # noqa: BLE001
        log(LOG, f"!! segment failed: {exc!r} (next invocation retries)")
        return 1
    finally:
        LOCK.unlink(missing_ok=True)


if __name__ == "__main__":
    sys.exit(main())
