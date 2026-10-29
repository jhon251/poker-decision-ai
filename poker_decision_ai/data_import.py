"""真实数据导入接口（V2 Phase 11）。

三类入口（全部落到 Repository，Agent 不直接碰文件/数据库）：
  1. 手牌历史文本 → 归一化 HAND_RECORD（hand_to_records，确定性近似，逐字段注释）
  2. 归一化记录 JSON 列表 → repos.hands（严格校验，坏记录拒绝并计数）
  3. CSV（表头映射）→ repos.hands
"""
from __future__ import annotations

import csv
import io
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .core.config import AppConfig
from .core.orchestration.runtime import Runtime
from .domain.poker.hand_history import ParsedHand, parse_hand_history
from .domain.poker.stats import HAND_RECORD_FIELDS


@dataclass
class ImportReport:
    source: str = ""
    hands_imported: int = 0
    records_imported: int = 0
    rejected: int = 0
    warnings: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "source": self.source, "hands_imported": self.hands_imported,
            "records_imported": self.records_imported, "rejected": self.rejected,
            "warnings": list(self.warnings),
        }


def hand_to_records(hand: ParsedHand, config: AppConfig) -> list[dict]:
    """一手牌历史 → 每个玩家的归一化统计记录（确定性近似，逐字段注释）。

    近似假设：从行动序列推导（vpip=翻前自愿入池、pfr=翻前加注、cbet=翻牌下注
    且是翻前进攻者…）；精确到逐手牌的 tracker 数据请用 normalized 记录导入。
    """
    now = time.time()
    stakes = _stakes_name(config, hand.sb, hand.bb)
    game = hand.game
    players = {s["name"]: s for s in hand.seats}
    records: list[dict] = []

    preflop_actions = [a for a in hand.actions if a["street"] == "preflop"]
    flop_actions = [a for a in hand.actions if a["street"] == "flop"]
    aggressor = None
    raises = 0
    for a in preflop_actions:
        if a["action"] == "raise":
            raises += 1
            if raises == 1:
                aggressor = a["player"]
            elif raises == 2 and a["player"] != aggressor:
                pass  # 3bet 由 faced_3bet 推导
    first_raiser = aggressor

    winner_names = {w["player"] for w in hand.winners}

    for name, seat in players.items():
        acts = [a for a in hand.actions if a["player"] == name]
        pf = [a for a in preflop_actions if a["player"] == name]
        fl = [a for a in flop_actions if a["player"] == name]
        is_blind = any(a["action"] == "post" for a in pf)

        vpip = any(a["action"] in ("call", "raise") for a in pf) and not all(
            a["action"] == "post" for a in pf)
        pfr = any(a["action"] == "raise" for a in pf)
        three_bet = any(a["action"] == "raise" for a in pf) and name != first_raiser and pfr

        faced_raise = any(a["action"] == "raise" and a["player"] != name for a in pf)
        folded_to_raise = faced_raise and any(a["action"] == "fold" for a in pf)

        cbet_opp = name == first_raiser and any(a["action"] in ("bet", "raise") for a in fl)
        cbet = cbet_opp
        faced_flop_bet = any(a["action"] in ("bet", "raise") and a["player"] != name
                             for a in flop_actions)
        folded_to_flop_bet = faced_flop_bet and any(a["action"] == "fold" for a in fl)

        saw_flop = bool(fl)
        showdown_names = {s["player"] for s in hand.shows}
        went_to_sd = name in showdown_names
        won_sd = name in winner_names if went_to_sd else None
        won = name in winner_names

        invested = sum(float(a.get("amount") or 0.0) for a in acts if a["action"] in ("call", "raise", "bet", "post"))
        won_amount = sum(float(w["amount"]) for w in hand.winners if w["player"] == name)
        net_bb = (won_amount - invested) / hand.bb if hand.bb > 0 else 0.0

        records.append({
            "hand_id": hand.hand_id, "player_id": name, "game": game,
            "stakes": stakes, "n_players": len(players), "position": "",
            "vpip": vpip, "pfr": pfr, "three_bet": three_bet,
            "faced_3bet": faced_raise, "folded_to_3bet": folded_to_raise,
            "four_bet": False,
            "cbet_opportunity": cbet_opp, "cbet": cbet,
            "faced_cbet": faced_flop_bet, "folded_to_cbet": folded_to_flop_bet,
            "saw_flop": saw_flop, "went_to_showdown": went_to_sd,
            "won_at_showdown": won_sd, "won_hand": won, "net_bb": round(net_bb, 4),
            "bets": sum(1 for a in acts if a["action"] in ("bet", "raise")),
            "calls": sum(1 for a in acts if a["action"] == "call"),
            "folds": sum(1 for a in acts if a["action"] == "fold"),
            "ts": now, "is_blind": is_blind,
        })
    return records


def _stakes_name(config: AppConfig, sb: float, bb: float) -> str:
    for name, info in config.stakes_table().items():
        if isinstance(info, dict) and float(info.get("sb", -1) or -1) == float(sb) \
                and float(info.get("bb", -1) or -1) == float(bb):
            return str(name)
    return f"NL{100.0 * float(bb):g}"


def _validate_record(rec: dict) -> list[str]:
    problems = []
    if not rec.get("player_id"):
        problems.append("missing player_id")
    if not rec.get("hand_id"):
        problems.append("missing hand_id")
    if rec.get("game") not in ("cash", "mtt", "satellite", None, ""):
        problems.append(f"bad game={rec.get('game')}")
    return problems


class DataImporter:
    """真实数据导入（文本 / 归一化 JSON / CSV → Repository）。"""

    def __init__(self, runtime: Runtime) -> None:
        self.runtime = runtime

    # ---------------------------------------------------------------- 文本
    def import_hand_text(self, text: str) -> ImportReport:
        hand = parse_hand_history(text)
        records = hand_to_records(hand, self.runtime.config)
        self.runtime.repos.hands.save({"hand_id": hand.hand_id, "parsed": hand.to_dict()})
        for rec in records:
            # 同一手牌的多个玩家记录需要唯一键：record_id = hand:player
            rec["record_id"] = f"{hand.hand_id}:{rec['player_id']}"
            self.runtime.repos.hands.save(rec)
        return ImportReport(source="hand_history", hands_imported=1,
                            records_imported=len(records))

    def import_file(self, path: str) -> ImportReport:
        p = Path(path)
        if not p.exists():
            return ImportReport(source=path, warnings=[f"file not found: {path}"])
        return self.import_hand_text(p.read_text(encoding="utf-8"))

    def import_directory(self, directory: str, ext: str = ".txt") -> ImportReport:
        files = sorted(str(p) for p in Path(directory).glob(f"*{ext}"))
        total = ImportReport(source=directory)
        for f in files:
            r = self.import_file(f)
            total.hands_imported += r.hands_imported
            total.records_imported += r.records_imported
            total.rejected += r.rejected
            total.warnings.extend(r.warnings)
        return total

    # ---------------------------------------------------------------- 归一化记录
    def import_normalized_records(self, records: list[dict]) -> ImportReport:
        report = ImportReport(source="normalized_records")
        for rec in records:
            problems = _validate_record(rec)
            if problems:
                report.rejected += 1
                report.warnings.append(f"rejected {rec.get('hand_id', '?')}: {problems}")
                continue
            clean = {k: v for k, v in rec.items() if k in HAND_RECORD_FIELDS or k in
                     ("ts", "is_blind", "chased_bounty", "bounty_overcall",
                      "bounty_overfold", "equity_at_decision", "allin", "bad_beat")}
            clean["record_id"] = f"{clean.get('hand_id', '?')}:{clean.get('player_id', '?')}"
            self.runtime.repos.hands.save(clean)
            report.records_imported += 1
        return report

    def import_normalized_json(self, json_text: str) -> ImportReport:
        data = json.loads(json_text)
        records = data if isinstance(data, list) else data.get("records", [])
        return self.import_normalized_records(records)

    def import_csv(self, csv_text: str) -> ImportReport:
        """CSV 导入：首行表头 = 记录字段名；bool 列支持 1/0/true/false。"""
        reader = csv.DictReader(io.StringIO(csv_text))
        records = []
        for row in reader:
            rec: dict[str, Any] = {}
            for k, v in row.items():
                k = k.strip()
                v = (v or "").strip()
                if v == "":
                    continue
                if v.lower() in ("true", "1"):
                    rec[k] = True
                elif v.lower() in ("false", "0"):
                    rec[k] = False
                else:
                    try:
                        rec[k] = float(v) if ("." in v) else int(v)
                    except ValueError:
                        rec[k] = v
            records.append(rec)
        return self.import_normalized_records(records)
