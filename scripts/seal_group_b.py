"""M3 — B 组池 bot 参数封存（POOL_BOT_PLAN 防自欺防线 2 的"哈希冻结"）。

机制（诚实声明，一次性执行）：
  * 本脚本用 **OS 熵**（secrets）现场抽取 B 组扰动种子/尺寸扰动（同
    POOL_BOT_PLAN 声明包络内的扰动），把明文写入
    `configs/bench/pool_factions_group_b.sealed.json`（文件头声明 SEALED
    FOR M4 GATE），并只把 **SHA-256** 写入 `docs/m3_opponent/group_b_seal.json`；
  * 盐/熵源**不落盘、不打印**——封存后明文即封存物本身，M3 的开发/评测
    链路禁止读取（契约测试保证 M3 代码无引用；解封是 M4 GATE 的动作）；
  * 完整性：seal 的 sha256 可随时复算（`--verify`），明文被改动即暴露。

用法：
  <repo>/.venv/Scripts/python.exe scripts/seal_group_b.py           # 封存
  <repo>/.venv/Scripts/python.exe scripts/seal_group_b.py --verify  # 只验哈希
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import secrets
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SEALED_PATH = ROOT / "configs" / "bench" / "pool_factions_group_b.sealed.json"
SEAL_PATH = ROOT / "docs" / "m3_opponent" / "group_b_seal.json"

#: B 组扰动规则（声明式；与 A 组同包络，只扰动种子与尺寸档）
PERTURB_RULES = {
    "seed_space": "64-bit",          # 每原型独立 64-bit rng 种子
    "bet_cbb_jitter": [-0.05, 0.05],  # 下注尺寸档扰动（pot 份额）
}


def _canonical(blob: dict) -> bytes:
    return json.dumps(blob, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":")).encode("utf-8")


def seal() -> int:
    from external_harness.protocol.pool_bots import FACTION_SPECS
    rng = secrets.SystemRandom()
    bots: dict[str, dict] = {}
    for stake, spec in FACTION_SPECS.items():
        for arch, asp in spec["archetypes"].items():
            base = dict(asp["params"])
            jitter = rng.uniform(*PERTURB_RULES["bet_cbb_jitter"])
            bots[f"{stake}/{arch}"] = {
                "seed": rng.getrandbits(64),
                "bet_cbb": round(
                    min(0.9, max(0.3, base["bet_cbb"] + jitter)), 3),
                "envelope": dict(asp["declared"]),
            }
    blob = {
        "classification": "SEALED FOR M4 GATE — M3 DEV MUST NOT READ/USE",
        "sealed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "perturb_rules": PERTURB_RULES,
        "bots": bots,
    }
    SEALED_PATH.parent.mkdir(parents=True, exist_ok=True)
    SEALED_PATH.write_text(json.dumps(blob, ensure_ascii=False, indent=1),
                           encoding="utf-8")
    digest = hashlib.sha256(_canonical(blob)).hexdigest()
    seal = {
        "sealed_at": blob["sealed_at"],
        "sha256": digest,
        "canonical_form": "json.dumps(sealed_json, sort_keys, separators)",
        "seled_plaintext_path": "configs/bench/pool_factions_group_b.sealed.json",
        "note": ("B 组参数由 OS 熵一次性生成并封存；熵源不落盘。M3 开发/评测"
                 "链路不得读取明文（POOL_BOT_PLAN 防线 2）；解封与使用是 "
                 "M4 GATE 的动作，解封前须复算本哈希。"),
    }
    SEAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    SEAL_PATH.write_text(json.dumps(seal, ensure_ascii=False, indent=1),
                         encoding="utf-8")
    # 只打印哈希（不回显任何参数）
    print("sealed:", SEALED_PATH.name)
    print("sha256:", digest)
    return 0


def verify() -> int:
    blob = json.loads(SEALED_PATH.read_text(encoding="utf-8"))
    digest = hashlib.sha256(_canonical(blob)).hexdigest()
    seal = json.loads(SEAL_PATH.read_text(encoding="utf-8"))
    ok = digest == seal.get("sha256")
    print("sha256:", digest)
    print("match :", ok)
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()
    if args.verify:
        return verify()
    return seal()


if __name__ == "__main__":
    sys.exit(main())
