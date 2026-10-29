"""V3G-11 — V3G 路线回归、治理核验与**封存契约**测试。

定位：**只**覆盖 V3G-11 的封存/回归契约，**不**重新测试已完成的策略逻辑
（各层内部行为已由 V3G-1～V3G-10 自身测试覆盖）。

本文件把 V3G 全部工件的 SHA-256 **钉死**（freeze pins），使此后任何对 V3G 工件
的静默改动都会被检出；并对生产边界、mode 隔离、fail-closed、Solver/profit 状态
做**契约级**核验。

**不修改**任何既有测试定义；**不触碰**冻结文件。
"""
from __future__ import annotations

import ast
import hashlib
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PKG = ROOT / "poker_decision_ai"


SELF_TEST = "test_v3g11_freeze_contract.py"
SELF_TEST_SENTINEL = "__SELF__"


def _sha(rel: str) -> str:
    p = ROOT / rel
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else "MISSING"


# ── V3G 工件封存基线（V3G-11 冻结）─────────────────────────────────────
V3G_MODULES: dict[str, str] = {
    "cash_state_integrity": "f8dca76a2fa3fe9c1231d39c4dfee12ccb05d346ce66c4fcf1391ffd71983c83",
    "cash_integrity_context": "4f7fe96ea43ceff662143b9d5eff1364ffd2b575bc19005a96d4fb1137b32ce8",
    "cash_preflop_decision": "5c2a511d2e59b4859a8fabaf5b0da3b66d72f372a131ea3394730bac492bf965",
    "cash_multistreet": "e896c978b2763108760f0648c12dbd35e31a6ba44f7fd6be539341f04ee92b9a",
    "cash_opponent_model": "e7a7d7af68c0f5ea4eab14487a2c126b70b6930f5a4616435d4f540943889179",
    "cash_range_model": "552155721104d39c9835b13a4511cf5c3e39286a16a90d6bb78232d4354ae9b5",
    "cash_action_tree": "b22b702db5e4fca94b9bddad180ba621bbf4759287ce84c60fab883ec9131027",
    "cash_unknown_e2e": "6c69941198fb4b1e5783ca77599a2b9dbd111ba23871afcca00fbf94606874e7",
    "cash_solver_interface": "a3d55431669e25c60424cd709105bf93abde32c5442a88d2dae2f940d6eb2669",
    "cash_profit_interface": "5007491c5b472a84db591ee683c374ee681a2e1701cc2063c10fa2ff9e10015e",
}

V3G_TESTS: dict[str, str] = {
    "test_v3g1_cash_state_integrity.py": "0f537785f5f8c9ec",
    "test_v3g2_cash_integrity_bypass.py": "f2719fe6d4f9e1f9",
    "test_v3g3_cash_preflop_layer.py": "7acf4a04ec158548",
    "test_v3g4_cash_multistreet_layer.py": "15931a1e3e1031f3",
    "test_v3g5_cash_opponent_model.py": "0000883f6ab4ca90",
    "test_v3g6_cash_range_model.py": "5b4fb5c75c703ce0",
    "test_v3g7_cash_action_tree.py": "b1ea32bcba46cb5a",
    "test_v3g8_cash_unknown_e2e.py": "2989dd706ebf066d",
    "test_v3g9_cash_solver_interface.py": "7ae8fcabd0f451a1",
    "test_v3g10_cash_profit_interface.py": "142bae45aecf5cb9",
    "test_v3g11_freeze_contract.py": "__SELF__",
}

#: V3G-11 封存清单（JSON，非 markdown closeout；单独处理）
V3G_MANIFEST_FILE = "V3G_11_FREEZE_MANIFEST.json"

#: 封存清单**自身**的 SHA-256（最后封存时记录）。
#: 自引用说明：清单在 `frozen_tests` 中记录**本测试文件**的真实哈希，故本
#: 常量必须在清单重生成**之后**回填；回填会改变本文件哈希，从而使清单中记录
#: 的旧值失效 —— 此时 `test_freeze_manifest_hash_is_pinned` 会报错，提示必须
#: "重生成清单 → 回填常量" 直至**定点收敛**（本次交付已完成收敛）。
MANIFEST_SHA256 = "e2449e8302b8c59693da055a054d6c6c8579161924a50c1bff40062ad8b5b3b7"

V3G_CLOSEOUTS: dict[str, str] = {
    "V3G_0_CASH_ROADMAP_AND_GAP_AUDIT.md": "f29c0415a3d0a928",
    "V3G_1_CASH_STATE_INTEGRITY_CLOSEOUT.md": "8f100cd42ba2e4d8",
    "V3G_2_CASH_CONTEXT_BYPASS_CLOSEOUT.md": "0ee55e615ac3d5a9",
    "V3G_3_CASH_PREFLOP_LAYER_CLOSEOUT.md": "58713db200b0baf5",
    "V3G_4_CASH_MULTISTREET_LAYER_CLOSEOUT.md": "df5c51402eb088a5",
    "V3G_5_CASH_OPPONENT_MODEL_CLOSEOUT.md": "df030f338ba2396a",
    "V3G_6_CASH_RANGE_MODEL_CLOSEOUT.md": "4a7fd90a384a8e98",
    "V3G_7_CASH_ACTION_TREE_CLOSEOUT.md": "864824760bd3655d",
    "V3G_8_CASH_UNKNOWN_E2E_CLOSEOUT.md": "d7b62c6691e25875",
    "V3G_9_CASH_SOLVER_INTERFACE_CLOSEOUT.md": "5ce9d21b50ec72c0",
    "V3G_10_CASH_PROFIT_INTERFACE_CLOSEOUT.md": "b8d2850ab50f7cf8",
    "V3G_11_EVALUATION_FREEZE_CLOSEOUT.md": "__SELF__",
    "V3G_11_FREEZE_MANIFEST.json": "__SELF__",
}

#: 生产链关键文件（**未经授权不得变化**；V3G-2/V3G-8 已授权的 metadata 变更后基线）
PRODUCTION_BASELINE: dict[str, str] = {
    "poker_decision_ai/domain/poker/strategy_core.py": "378a68671ff09e3f",
    "poker_decision_ai/agents/judge/agent.py": "239436fae7299c47",
    "poker_decision_ai/domain/poker/models.py": "6775c06050319a85",
    "poker_decision_ai/domain/poker/conditional_model.py": "686c9f58b71d9408",
    "poker_decision_ai/realtime.py": "3400a20ff857880d",
    "poker_decision_ai/research_unfreeze/pi_ev_contract.py": "52e6267db49a2e11",
    "poker_decision_ai/research_unfreeze/unfreeze_core.py": "17236d4ed8070d01",
}

#: 只有这两个（已授权）生产文件可以引用 V3G 层
SANCTIONED_WIRING: dict[str, tuple[str, ...]] = {
    "realtime.py": ("cash_integrity_context", "cash_unknown_e2e"),
}
#: 本阶段不得被任何生产模块引用的层
UNWIRED_LAYERS: tuple[str, ...] = (
    "cash_preflop_decision", "cash_multistreet", "cash_opponent_model",
    "cash_range_model", "cash_action_tree", "cash_solver_interface",
    "cash_profit_interface",
)


# ══ 1. 工件完整性 / 封存 pin ══════════════════════════════════════════
@pytest.mark.parametrize("mod,digest", sorted(V3G_MODULES.items()))
def test_v3g_module_frozen(mod, digest):
    assert _sha(f"poker_decision_ai/{mod}.py") == digest


@pytest.mark.parametrize("name,prefix",
                         sorted((k, v) for k, v in V3G_TESTS.items()
                                if v != SELF_TEST_SENTINEL))
def test_v3g_test_file_frozen(name, prefix):
    assert _sha(f"tests/unit/{name}").startswith(prefix)


def test_self_freeze_placeholder_is_the_only_self_reference():
    """自引用说明：本文件不能钉死自身哈希（写入即改变哈希）。

    因此 V3G_TESTS 中本文件登记为哨兵值 __SELF__；清单完整性由
    test_freezes_cover_every_v3g_artifact_on_disk 保证，运行期 hash 由
    docs/V3G_11_FREEZE_MANIFEST.json（V3G-11 封存清单）记录。
    """
    assert V3G_TESTS[SELF_TEST] == SELF_TEST_SENTINEL
    assert list(V3G_TESTS.values()).count(SELF_TEST_SENTINEL) == 1


@pytest.mark.parametrize("name,prefix",
                         sorted((k, v) for k, v in V3G_CLOSEOUTS.items()
                                if v != SELF_TEST_SENTINEL))
def test_v3g_closeout_frozen(name, prefix):
    assert _sha(f"docs/{name}").startswith(prefix)


def test_freeze_manifest_hash_is_pinned():
    """封存清单自身亦被钉死（自引用链的最后一环）。

    链条：本常量记录`docs/V3G_11_FREEZE_MANIFEST.json`的哈希，而清单记录
    其余**全部**工件的哈希（本测试文件在清单中登记哨兵，避免不可收敛的环）。
    重生成清单后若忘记回填本常量，此测试立即报错。
    """
    assert MANIFEST_SHA256 != "__MANIFEST_SHA__", "常量尚未回填（清单未封存）"
    assert len(MANIFEST_SHA256) == 64, "必须是完整 SHA-256"
    assert _sha(f"docs/{V3G_MANIFEST_FILE}") == MANIFEST_SHA256


def test_self_referential_artifacts_are_exactly_two():
    """自引用链如实登记：共 **3** 个自引用工件。

    1. 本测试文件 —— 在 `V3G_TESTS` 中为哨兵 `__SELF__`；
    2. `V3G_11_EVALUATION_FREEZE_CLOSEOUT.md` —— 在 `V3G_CLOSEOUTS` 中为哨兵；
    3. `V3G_11_FREEZE_MANIFEST.json` —— 在 `V3G_CLOSEOUTS` 中为哨兵。

    三者的**真实哈希**在清单生成时刻被写入 `docs/V3G_11_FREEZE_MANIFEST.json`
    （`frozen_tests` / `frozen_closeouts`）；清单自己的哈希由
    `test_freeze_manifest_hash_is_pinned` 钉死 —— 链条闭合。
    """
    sentinels = [k for k, v in V3G_CLOSEOUTS.items()
                 if v == SELF_TEST_SENTINEL]
    assert sorted(sentinels) == ["V3G_11_EVALUATION_FREEZE_CLOSEOUT.md",
                                "V3G_11_FREEZE_MANIFEST.json"]


def test_all_ten_layers_present_and_importable():
    import importlib
    for mod in V3G_MODULES:
        m = importlib.import_module(f"poker_decision_ai.{mod}")
        assert m is not None, mod


def test_every_layer_declares_a_version_constant():
    import importlib
    for mod in V3G_MODULES:
        m = importlib.import_module(f"poker_decision_ai.{mod}")
        assert any(k.endswith("_VERSION") for k in dir(m)), mod


def test_freezes_cover_every_v3g_artifact_on_disk():
    """封存清单必须**完整**：磁盘上不得存在未登记的 V3G 工件。"""
    on_disk_mods = {p.stem for p in PKG.glob("cash_*.py")}
    assert set(V3G_MODULES) <= on_disk_mods
    on_disk_tests = {p.name for p in (ROOT / "tests" / "unit").glob("test_v3g*.py")}
    assert on_disk_tests == set(V3G_TESTS), on_disk_tests ^ set(V3G_TESTS)
    on_disk_docs = {p.name for p in (ROOT / "docs").glob("V3G_*.md")}
    declared_docs = {k for k, v in V3G_CLOSEOUTS.items()
                     if v != SELF_TEST_SENTINEL
                     or k == "V3G_11_EVALUATION_FREEZE_CLOSEOUT.md"}
    assert on_disk_docs == declared_docs, on_disk_docs ^ declared_docs
    # JSON 封存清单单独核验存在性
    assert (ROOT / "docs" / V3G_MANIFEST_FILE).exists()
    import json as _json
    man = _json.loads((ROOT / "docs" / V3G_MANIFEST_FILE).read_text(
        encoding="utf-8"))
    assert man["counts"]["modules"] == len(V3G_MODULES)
    assert man["counts"]["tests"] == len(V3G_TESTS)


# ══ 2. 生产链关键文件无未授权变化 ═════════════════════════════════════
@pytest.mark.parametrize("rel,prefix", sorted(PRODUCTION_BASELINE.items()))
def test_production_files_match_baseline(rel, prefix):
    assert _sha(rel).startswith(prefix), rel


def test_strategy_core_and_judge_untouched_by_v3g():
    """strategy_core / Judge 的完整哈希必须与 V3F 封存一致（零漂移）。"""
    assert _sha("poker_decision_ai/domain/poker/strategy_core.py") == \
        "378a68671ff09e3f3c86349e5517a725e11459bdd8e96155d79e5e1429213de7"
    # Judge 基线为 V3G-2 接线后的哈希（V3G-11 未再改动）
    assert _sha("poker_decision_ai/agents/judge/agent.py") == \
        "239436fae7299c47a66fbc272e8caa9c566983ba59d142dd9a8df347c5bbb104"


def test_positions_for_module_untouched():
    assert _sha("poker_decision_ai/domain/poker/models.py") == \
        "6775c06050319a85cca0b4a33afb78103cc3524f7feb4b6918240f6ce8853327"


def test_realtime_only_gained_metadata_fields():
    """`realtime.py` 的变更必须**仅**是 metadata：两个旁路字段 + 事后计算。"""
    src = (PKG / "realtime.py").read_text(encoding="utf-8")
    assert "state_integrity" in src and "unknown_report" in src
    # 不得出现 V3G 未授权层
    for layer in UNWIRED_LAYERS:
        assert layer not in src, layer
    # 不得触碰决策计算
    for banned in ("compute_ev", "evaluate_intel"):
        assert banned not in src, banned


# ══ 3. 生产边界：只有已授权的两个旁路 ════════════════════════════════
def _imports_of(path: Path) -> set:
    tree = ast.parse(path.read_text(encoding="utf-8-sig", errors="ignore"))
    out = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            out |= {a.name for a in n.names}
        elif isinstance(n, ast.ImportFrom):
            base = n.module or ""
            out.add(base)
            out |= {f"{base}.{a.name}" for a in n.names}
    return out


@pytest.mark.parametrize("layer", UNWIRED_LAYERS)
def test_unwired_layers_have_zero_real_imports_in_production(layer):
    """未授权层不得被任何生产模块**真实 import**（docstring 提及不算）。"""
    offenders = []
    for p in sorted(PKG.rglob("*.py")):
        if p.name == f"{layer}.py":
            continue
        hit = [m for m in _imports_of(p)
               if m == layer or m.endswith("." + layer)]
        if hit:
            offenders.append(str(p.relative_to(ROOT)))
    assert offenders == [], f"{layer} 被生产模块 import: {offenders}"


def test_only_sanctioned_wiring_exists():
    """生产包内**真实 import** V3G 层的模块，只允许已授权的两个旁路点。

    `realtime.py`            → `cash_integrity_context`（V3G-2）、
                               `cash_unknown_e2e`（V3G-8）
    `agents/judge/agent.py`  → `cash_integrity_context`（V3G-2 evidence 旁路）

    注意：`agent.py` 中另有一处字符串标签 `"source": "cash_state_integrity"`
    与 docstring 提及——**不是** import，故由 AST 判定自然排除
    （V3G-1 门禁仍只能经适配层到达）。
    """
    all_layers = tuple(V3G_MODULES)
    offenders = {}
    for p in sorted(PKG.rglob("*.py")):
        if p.name in {f"{m}.py" for m in all_layers}:
            continue
        used = [m for m in all_layers
                if any(imp == m or imp.endswith("." + m)
                       for imp in _imports_of(p))]
        if used:
            offenders[p.name] = sorted(used)
    sanctioned = {
        "realtime.py": {"cash_integrity_context", "cash_unknown_e2e"},
        "agent.py": {"cash_integrity_context"},
    }
    assert set(offenders) <= set(sanctioned), offenders
    for name, allowed in sanctioned.items():
        assert set(offenders.get(name, [])) <= allowed, name


def test_no_research_unfreeze_import_from_cash_layers():
    for mod in V3G_MODULES:
        src = (PKG / f"{mod}.py").read_text(encoding="utf-8")
        tree = ast.parse(src)
        for n in ast.walk(tree):
            if isinstance(n, ast.ImportFrom):
                assert "research_unfreeze" not in (n.module or ""), mod
            elif isinstance(n, ast.Import):
                assert all("research_unfreeze" not in a.name for a in n.names), mod


def test_layers_do_not_call_frozen_decision_functions():
    banned = ("compute_ev", "evaluate_intel", "positions_for")
    for mod in V3G_MODULES:
        src = (PKG / f"{mod}.py").read_text(encoding="utf-8")
        tree = ast.parse(src)
        called = {n.func.id for n in ast.walk(tree)
                  if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
        attrs = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
        for b in banned:
            assert b not in called and b not in attrs, (mod, b)


# ══ 4. CASH / 非 CASH 隔离（契约级）══════════════════════════════════
def _cash_state(**over) -> dict:
    b = {
        "game_type": "cash", "street": "flop", "board": ["Ah", "7c", "2c"],
        "pot": 20.0, "to_call": 5.0, "bb": 1.0, "sb": 0.5,
        "min_raise_to": 10.0, "hero_id": "hero", "button_seat": 2,
        "table_size": 3, "synthetic_test_only": True,
        "players": [
            {"player_id": "v1", "seat": 1, "position": "BB", "status": "active",
             "stack": 100.0, "hole_cards": []},
            {"player_id": "hero", "seat": 2, "position": "BTN", "is_hero": True,
             "status": "active", "hole_cards": ["As", "Kd"], "stack": 100.0},
            {"player_id": "v2", "seat": 3, "position": "SB", "status": "active",
             "stack": 100.0, "hole_cards": []},
        ],
        "action_history": [],
    }
    b.update(over)
    return b


def _layer_functions() -> list:
    from poker_decision_ai.cash_action_tree import build_action_tree
    from poker_decision_ai.cash_integrity_context import cash_integrity_context
    from poker_decision_ai.cash_multistreet import build_multistreet_context
    from poker_decision_ai.cash_opponent_model import build_opponent_model
    from poker_decision_ai.cash_preflop_decision import build_preflop_decision
    from poker_decision_ai.cash_profit_interface import profit_experiment_interface
    from poker_decision_ai.cash_range_model import build_range_model
    from poker_decision_ai.cash_solver_interface import solver_status
    from poker_decision_ai.cash_unknown_e2e import build_unknown_report
    return [cash_integrity_context, build_unknown_report, build_preflop_decision,
            build_multistreet_context, build_opponent_model, build_range_model,
            build_action_tree, solver_status, profit_experiment_interface]


@pytest.mark.parametrize("mode", ["mtt", "pko", "ko", "mystery", "spin"])
def test_all_layers_fail_closed_for_non_cash(mode):
    s = _cash_state(game_type=mode)
    for fn in _layer_functions():
        r = fn(s)
        assert r.get("present") is False, (fn.__name__, mode)


@pytest.mark.parametrize("mode", [None, "", "   ", "holdem", "CASHX"])
def test_all_layers_fail_closed_for_illegal_mode(mode):
    s = _cash_state(game_type=mode)
    for fn in _layer_functions():
        assert fn(s).get("present") is False, (fn.__name__, mode)


def test_all_layers_fail_closed_for_non_dict():
    for fn in _layer_functions():
        for bad in (None, [], "cash", 3):
            assert fn(bad).get("present") is False, fn.__name__


def test_cash_produces_every_layer():
    """每个层在其**适用街**上产出（preflop 层只在 preflop 成立；其余在 flop）。"""
    flop = _cash_state()
    pre = _cash_state(street="preflop", board=[], to_call=2.0,
                      min_raise_to=4.0)
    from poker_decision_ai.cash_preflop_decision import build_preflop_decision
    assert build_preflop_decision(pre).get("present") is True
    for fn in _layer_functions():
        if fn is build_preflop_decision:
            continue
        assert fn(flop).get("present") is True, fn.__name__


def test_no_tournament_terms_leak_into_cash_outputs():
    import json
    s = _cash_state()
    for fn in _layer_functions():
        blob = json.dumps(fn(s), ensure_ascii=False).lower()
        for banned in ("icm", "bounty", "satellite", "payout"):
            assert banned not in blob, (fn.__name__, banned)


# ══ 5. V3G-2 / V3G-8 旁路不改变决策 ══════════════════════════════════
def test_bypasses_are_computed_after_decision_and_never_scored():
    src = (PKG / "realtime.py").read_text(encoding="utf-8")
    # 旁路必须在 extract_final_decision 之后
    i_dec = src.index("extract_final_decision(result)")
    i_int = src.index("cash_integrity_context")
    i_unk = src.index("cash_unknown_e2e")
    assert i_dec < i_int and i_dec < i_unk
    # 旁路不得进入决策计算
    tail = src[i_dec:]
    for banned in ("scores[", "weights_used[", "compute_ev"):
        assert banned not in tail, banned


def test_bypass_failures_cannot_break_the_decision():
    src = (PKG / "realtime.py").read_text(encoding="utf-8")
    assert src.count("except Exception") >= 2
    assert "integrity = None" in src and "unknown_report = None" in src


# ══ 6. Solver 状态契约（V3G-9）═══════════════════════════════════════
def test_solver_still_blocked_with_no_equilibrium_or_gto():
    from poker_decision_ai.cash_solver_interface import solver_status
    r = solver_status(_cash_state())
    assert r["solver_status"] == "BLOCKED"
    assert r["discovery_state"] in ("UNAVAILABLE", "UNKNOWN")
    assert r["equilibrium"] is False and r["gto_exact"] is False
    assert r["executed"] is False and r["valid"] is False
    assert r["provenance"]["valid_result_reachable"] is False
    assert r["policy"]["valid_result_reachable"] is False


def test_solver_layer_has_no_execution_capability():
    """AST：不得 import 执行相关模块（docstring 提及不算）。"""
    tree = ast.parse((PKG / "cash_solver_interface.py").read_text(encoding="utf-8"))
    imported = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            imported |= {a.name for a in n.names}
        elif isinstance(n, ast.ImportFrom):
            imported.add(n.module or "")
    for banned in ("subprocess", "asyncio", "multiprocessing"):
        assert not any(m == banned or m.startswith(banned + ".")
                       for m in imported), banned


# ══ 7. profit 实验契约（V3G-10）══════════════════════════════════════
def test_profit_interface_still_blocked_and_unidentifiable():
    from poker_decision_ai.cash_profit_interface import (
        NEVER_EMITTED_FIELDS, profit_experiment_interface)
    r = profit_experiment_interface(_cash_state())
    assert r["status"] == "BLOCKED"
    assert r["result_status"] == "EXECUTION_ENVIRONMENT_REQUIRED"
    assert r["profit_identifiable"] is False
    assert r["executed"] is False
    assert r["valid_result_reachable"] is False

    def keys(o, out=None):
        if out is None:
            out = set()
        if isinstance(o, dict):
            for k, v in o.items():
                out.add(k)
                keys(v, out)
        elif isinstance(o, list):
            for v in o:
                keys(v, out)
        return out

    assert not (set(NEVER_EMITTED_FIELDS) & keys(r))


def test_profit_layer_has_no_execution_or_randomness_capability():
    """AST：不得 import 执行/随机/网络模块（docstring 提及不算）。"""
    tree = ast.parse((PKG / "cash_profit_interface.py").read_text(encoding="utf-8"))
    imported = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            imported |= {a.name for a in n.names}
        elif isinstance(n, ast.ImportFrom):
            imported.add(n.module or "")
    for banned in ("subprocess", "asyncio", "socket", "random",
                   "multiprocessing"):
        assert not any(m == banned or m.startswith(banned + ".")
                       for m in imported), banned


# ══ 8. 无越界声明（跨全部 V3G 文档）══════════════════════════════════
def test_no_overreach_claims_in_any_v3g_document():
    """V3G 文档不得出现**肯定式**越界断言。

    注意："**不**得宣布 CASH 最终完成"这类**要求性/否定式**语句是必须存在的
    （各阶段免责声明），故此处只检查**肯定式**表述，避免自相矛盾。
    引用规则：被引号包裹的片段一律视为对禁项清单的**引用**（例如报告项 10 的
    检查机制说明），不作为断言本身；判定只针对引号之外的裸文本，前文窗口
    跨行取上一行尾部，避免把否定式表述误判为肯定式越界。
    """
    positives = ("CASH 最终完成", "GTO 已实现", "达到均衡", "盈利已验证",
                 "全量 PASS", "Solver 已接入", "已接入 GTO")
    negations = ("不", "未", "无", "禁止", "不得")
    bt = chr(96)
    quoted = re.compile(bt + "[^" + bt + "]*" + bt + "|\"[^\"]*\"|\u201c[^\u201d]*\u201d|\u300c[^\u300d]*\u300d")

    def bare(line):
        """把引号片段替换为中性单字符占位，只留下裸断言文本。"""
        pieces, last = [], 0
        for m in quoted.finditer(line):
            pieces.append(line[last:m.start()])
            pieces.append("Z")
            last = m.end()
        pieces.append(line[last:])
        return "".join(pieces)

    for name in V3G_CLOSEOUTS:
        lines = (ROOT / "docs" / name).read_text(encoding="utf-8").splitlines()
        for lineno, raw in enumerate(lines, 1):
            line = bare(raw)
            prev = bare(lines[lineno - 2])[-12:] if lineno >= 2 else ""
            for ph in positives:
                i = line.find(ph)
                while i >= 0:
                    window = prev + line[max(0, i - 12):i]
                    assert any(neg in window for neg in negations), \
                        f"{name}:{lineno}: 肯定式越界表述 {ph!r}: {line.strip()[:80]}"
                    i = line.find(ph, i + 1)


def test_every_closeout_records_test_counts_and_environment_errors():
    """每份 closeout 必须如实记录测试计数与环境性 ERROR。"""
    for name in V3G_CLOSEOUTS:
        if name.startswith("V3G_0") or name == V3G_MANIFEST_FILE:
            continue            # V3G-0 为纯审计（无测试）；清单为 JSON
        text = (ROOT / "docs" / name).read_text(encoding="utf-8")
        assert "passed" in text, name
        assert "WinError 5" in text or "环境性" in text, name


def test_roadmap_stage_count_matches_delivered_layers():
    """路线图声明的阶段数必须与实际交付的层数一致（防止漏做/多做）。"""
    doc = (ROOT / "docs" / "V3G_0_CASH_ROADMAP_AND_GAP_AUDIT.md").read_text(
        encoding="utf-8")
    for i in range(1, 12):
        assert f"V3G-{i}" in doc, i
    assert len(V3G_MODULES) == 10        # V3G-1～V3G-10 各一层
    assert len(V3G_TESTS) == 11          # 10 层 + V3G-11 封存契约
    assert len(V3G_CLOSEOUTS) == 13      # V3G-0 审计 + V3G-1～11 + 封存清单
