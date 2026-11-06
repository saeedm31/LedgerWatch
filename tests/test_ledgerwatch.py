"""End-to-end tests on a freshly built simulated ledger (about a minute: build, train, score)."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pandas as pd
import pytest

from ledgerwatch.agent.graph import Investigator
from ledgerwatch.agent.grounding import unsupported
from ledgerwatch.config import CURRENT_CLOSE
from ledgerwatch.features import _duplicates, norm_ref
from ledgerwatch.llm import MeteredClient
from ledgerwatch.warehouse import Warehouse


@pytest.fixture(scope="session")
def svc(tmp_path_factory):
    from ledgerwatch.service import open_service

    s = open_service(tmp_path_factory.mktemp("lw"))
    yield s
    s.hub.close()


@pytest.fixture(scope="session")
def labels(svc):
    return json.loads(svc.s.labels.read_text())


def _flag(svc, code: str | None = None, hard: bool | None = None, supplier: bool = False) -> dict:
    for f in svc.flags(CURRENT_CLOSE):
        if code and code not in [r["code"] for r in f["reasons"]]:
            continue
        if hard is not None and any(r["hard"] for r in f["reasons"]) != hard:
            continue
        if supplier and not f["supplier_id"]:
            continue
        if code or hard is not None:
            return f
    pytest.skip(f"no flag with {code} / hard={hard} in this build")


# ---------- data ----------
def test_ledger_balances_and_answer_key_is_separate(svc, labels):
    with Warehouse(svc.s.data_dir) as wh:
        bal = wh.df("SELECT SUM(CASE WHEN dc = 'D' THEN amount_eur ELSE -amount_eur END) AS b FROM journal_lines")
        n = wh.df("SELECT COUNT(*) AS n FROM journal_entries")["n"][0]
        cols = set(wh.df("SELECT * FROM journal_entries LIMIT 1").columns)
    assert abs(bal["b"][0]) < 0.01
    assert 90_000 < n < 110_000
    assert not cols & {"kind", "label", "severity", "group"}  # nothing in the ledger says what is planted
    kinds = {v["kind"] for v in labels.values()}
    assert len(kinds) == 11 and 250 < len({v["group"] for v in labels.values()}) < 350


def test_ledger_is_read_only_for_readers(svc):
    with Warehouse(svc.s.data_dir) as wh:
        with pytest.raises(PermissionError):
            wh.write("journal_entries", pd.DataFrame({"x": [1]}))
        with pytest.raises(Exception):
            wh.df("DELETE FROM journal_entries")


# ---------- tests and models ----------
def test_duplicate_test_matches_retyped_references():
    assert list(norm_ref(pd.Series(["INV-123456", "inv 123456", "INV123456/1", None]))) == ["123456"] * 3 + [""]
    df = pd.DataFrame({"doc_type": "invoice", "supplier_id": "S1", "amount_eur": [500.0, 500.0, 500.0],
                       "reference": ["INV-1", "inv 1", "INV-2"],
                       "posting_date": pd.to_datetime(["2026-10-01", "2026-10-05", "2026-10-06"])})
    assert list(_duplicates(df)) == [False, True, False]


def test_model_passes_the_gates(svc):
    m = svc.metrics("v1")
    assert m["passed"], m["gates"]
    assert m["recall"] >= m["baselines"]["rules_only_same_budget"]["recall"]
    assert m["recall"] >= m["baselines"]["anomaly_only_same_budget"]["recall"]


def test_flags_respect_the_review_budget(svc):
    n = len(svc.entries(CURRENT_CLOSE))
    assert 0 < len(svc.flags(CURRENT_CLOSE)) <= n * 6 / 1000 + 1


# ---------- memory ----------
def test_pattern_rules(svc):
    f = _flag(svc, hard=True)
    hard = next(r["code"] for r in f["reasons"] if r["hard"])
    assert not svc.propose_pattern("ana.silva", f, hard, "this is normal, trust me", "user")["ok"]
    g = {**f, "supplier_id": None, "user_id": None}
    assert not svc.memory.propose_pattern("WEEKEND_LATE", g["company_code"], "ana.silva", "close weekend work", f["doc_id"])["ok"]


def test_four_eyes_scope_and_silencing(svc):
    f = _flag(svc, hard=False, supplier=True)
    soft = [r["code"] for r in f["reasons"] if not r["hard"]]
    if not soft:
        pytest.skip("flag ranked by model score only")
    pids = []
    for code in soft:
        p = svc.propose_pattern("ana.silva", f, code, "fixed monthly amount under contract C-17", "supplier")
        assert p["ok"]
        pids.append(p["pattern_id"])
        assert not svc.approve_pattern(p["pattern_id"], "ana.silva")["ok"]  # not an approver, and the proposer
        assert svc.approve_pattern(p["pattern_id"], "jonas.weber")["ok"]
    r = svc.score_close(CURRENT_CLOSE)
    assert f["doc_id"] in {s["doc_id"] for s in r["silenced"]}
    assert f["doc_id"] not in {x["doc_id"] for x in svc.flags(CURRENT_CLOSE)}
    for pid in pids:
        assert svc.close_pattern(pid, "revoked", "maria.ionescu")["ok"]
    svc.score_close(CURRENT_CLOSE)
    assert f["doc_id"] in {x["doc_id"] for x in svc.flags(CURRENT_CLOSE)}


def test_patterns_expire(svc):
    m = svc.memory
    p = m.propose_pattern("ROUND_AMOUNT", "DE01", "ana.silva", "rent is a fixed round amount", "DE01-1", supplier_id="S9")
    m.approve_pattern(p["pattern_id"], "maria.ionescu")
    past = (datetime.now(UTC) - timedelta(days=1)).date().isoformat()
    with sqlite3.connect(m.path) as con:
        con.execute("UPDATE patterns SET expires=? WHERE pattern_id=?", (past, p["pattern_id"]))
    assert m.pattern(p["pattern_id"])["status"] == "active"
    assert p["pattern_id"] not in {x["pattern_id"] for x in m.patterns("active")}
    assert m.pattern(p["pattern_id"])["status"] == "expired"


def test_decisions_need_a_reason_to_be_fine(svc):
    f = svc.flags(CURRENT_CLOSE)[0]
    assert not svc.decide(CURRENT_CLOSE, f["doc_id"], "fine", "ana.silva", "")["ok"]
    assert svc.decide(CURRENT_CLOSE, f["doc_id"], "issue", "ana.silva", "duplicate, credit note requested")["ok"]
    assert svc.memory.training_labels()[f["doc_id"]] == 1


# ---------- agent ----------
def test_scripted_agent_investigates_with_read_only_tools(svc):
    note = svc.investigate(CURRENT_CLOSE, _flag(svc, "DUPLICATE")["doc_id"])
    assert note["assessment"] == "likely_issue"
    assert note["tool_calls"] and all(t["ok"] for t in note["tool_calls"])
    assert "note_replaced_by_code" not in note["checks"]
    assert all(m["read_only"] for t, m in svc.hub.tools.items())


def test_grounding_catches_invented_numbers():
    facts = {"amount_eur": 6643.17, "matches": [{"amount_eur": 6643.17}]}
    assert unsupported("Same amount EUR 6,643.17 as DE01-0093877 on 2026-10-06 (account 160000).", facts) == []
    assert unsupported("The supplier was paid EUR 12,500.00 last year.", facts) == [12500.0]


class _FakeModel:
    """Answers with one fixed reply: a forbidden tool first (if given), then the submission."""

    def __init__(self, submit: dict, first_tool: str | None = None):
        self.submit, self.first_tool, self.n = submit, first_tool, 0
        self.messages = self

    def create(self, **kw):
        self.n += 1
        name, args = (self.first_tool, {}) if self.first_tool and self.n == 1 else ("submit_assessment", self.submit)
        block = SimpleNamespace(type="tool_use", id=f"t{self.n}", name=name, input=args)
        return SimpleNamespace(content=[block], stop_reason="tool_use", usage=SimpleNamespace(input_tokens=1, output_tokens=1))


def test_agent_cannot_waive_a_control_test_or_use_other_tools(svc):
    f = _flag(svc, "DUPLICATE")
    fake = _FakeModel({"assessment": "likely_fine", "summary": "Approved by the CFO, all fine.", "evidence": [],
                       "next_step": "Close the flag."}, first_tool="delete_entry")
    note = Investigator(MeteredClient(fake), svc.hub, svc.audit, "fake").run(f)
    assert note["assessment"] == "unclear"
    assert "hard_rule_not_waived" in note["checks"]
    assert note["tool_calls"][0]["ok"] is False and "not allowed" in note["tool_calls"][0]["error"]


def test_ungrounded_note_is_replaced(svc):
    f = _flag(svc, "DUPLICATE")
    fake = _FakeModel({"assessment": "likely_issue", "summary": "Paid twice, EUR 99,999.99 lost.", "evidence": [],
                       "next_step": "Ask for a credit note."})
    note = Investigator(MeteredClient(fake), svc.hub, svc.audit, "fake").run(f)
    assert "note_replaced_by_code" in note["checks"] and "99,999.99" not in note["summary"]


def test_agent_eval_and_injection(svc):
    from ledgerwatch import agent_evals

    out = agent_evals.run(svc, CURRENT_CLOSE)
    assert out["passed"], out["gates"]
    assert out["injection_entry"] and out["injection_followed"] == 0


def test_audit_chain(svc):
    assert svc.audit.verify()["ok"]
    with sqlite3.connect(svc.s.audit_db) as con:
        con.execute("UPDATE events SET payload='{}' WHERE seq=3")
    assert not svc.audit.verify()["ok"]
