"""M4 Phase B3 — B 组解封（POOL_BOT_PLAN 防自欺防线 2 的 GATE 动作）。

协议：
  1. 复算封存文件 SHA-256，必须与 M3 登记的封存哈希一致
     （docs/m3_opponent/group_b_seal.json，前缀 5b040e47…）；
  2. 抽取每原型 {seed, bet_cbb}（M3 封存的隐藏内容 = OS 熵种子 +
     bet_cbb ±0.05 扰动）；行为模型基础参数取**当前 FACTION_SPECS v2**
     （M4 postflop 强度条件化是结构升级——两档原型共用同一 v2 模型与
     同一声明包络；B 组的随机化内容原样解封使用，报告如实注明）；
  3. 落盘 `configs/bench/pool_factions_group_b.unsealed.json`（解封时间
     + 核验哈希 + 包络声明），GATE 战役驱动只消费该文件；
  4. 解封动作记录进进度档案（由调用方执行）。

用法：
  <repo>/.venv/Scripts/python.exe scripts/unseal_group_b_m4.py
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SEALED = ROOT / "configs" / "bench" / "pool_factions_group_b.sealed.json"
SEAL_REGISTRY = ROOT / "docs" / "m3_opponent" / "group_b_seal.json"
UNSEALED = ROOT / "configs" / "bench" / "pool_factions_group_b.unsealed.json"
EXPECTED_PREFIX = "5b040e47"          # 任务书/M3 报告登记的封存哈希前缀


def main() -> int:
    if not SEALED.is_file() or not SEAL_REGISTRY.is_file():
        print("sealed file or registry missing", file=sys.stderr)
        return 2
    digest = hashlib.sha256(json.dumps(
        json.loads(SEALED.read_text(encoding="utf-8")),
        ensure_ascii=False, sort_keys=True,
        separators=(",", ":")).encode("utf-8")).hexdigest()
    registry = json.loads(SEAL_REGISTRY.read_text(encoding="utf-8"))
    registered = str(registry.get("sha256") or "")
    if digest != registered or not digest.startswith(EXPECTED_PREFIX):
        print(f"HASH MISMATCH: computed={digest[:16]}… "
              f"registered={registered[:16]}…", file=sys.stderr)
        return 3
    sealed = json.loads(SEALED.read_text(encoding="utf-8"))
    from external_harness.protocol.pool_bots import (
        FACTION_SPECS, declared_envelope, declared_post_envelope)
    bots_out: dict[str, dict] = {}
    for stake in ("nl2", "nl5"):
        for arch in FACTION_SPECS[stake]["archetypes"]:
            key = f"{stake}/{arch}"
            b = (sealed.get("bots") or {}).get(key)
            if b is None:
                print(f"sealed file missing {key}", file=sys.stderr)
                return 4
            # 解封验证：包络与 FACTION_SPECS 声明一致（封存时登记的包络
            # 没有被改动——独立抽查，M3 报告 §7 的解封要求）
            env = declared_envelope(stake, arch)
            assert b.get("envelope", {}).get("vpip") == env["vpip"], key
            bots_out[key] = {
                "seed": int(b["seed"]),
                "bet_cbb": float(b["bet_cbb"]),
                "envelope": env,
                "post_envelope": declared_post_envelope(stake, arch),
                "model_base": "FACTION_SPECS v2 (m4-pool-bots-v2-strength)",
            }
    out = {
        "unsealed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "verified_sha256": digest,
        "registry": str(SEAL_REGISTRY.relative_to(ROOT)),
        "perturb_rules": sealed.get("perturb_rules"),
        "note": ("M4 GATE 专用。隐藏内容 = seed + bet_cbb（M3 封存协议）；"
                 "postflop 行为模型为 M4 v2（结构升级，非封存内容；两档"
                 "原型同一模型同一包络——诚实声明见 M4 报告）。"),
        "bots": bots_out,
    }
    UNSEALED.write_text(json.dumps(out, ensure_ascii=False, indent=1),
                        encoding="utf-8")
    print(f"unsealed OK: sha256 verified {digest[:16]}… "
          f"({len(bots_out)} bots) -> {UNSEALED.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
