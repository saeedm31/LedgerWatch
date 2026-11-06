"""The app's memory: review decisions and known patterns, chosen by people.

Two kinds of memory, both saved only when a reviewer chooses to:

1. Decisions ("issue" / "fine" with a reason). They become training labels: the next model version learns
   from them. A new version is promoted only if it still passes the evaluation gates.
2. Known patterns ("rent from this landlord is a fixed round amount every month"). A pattern silences one
   soft test for one company and one supplier or user. Safeguards against memory poisoning:
   - four eyes: the person who proposes a pattern cannot approve it,
   - narrow scope: one company, one rule, one supplier or user; never a whole account or company,
   - hard tests (duplicates, splits, other-company users, over-limit, new suppliers, period-end revenue)
     can never be silenced,
   - every pattern expires (180 days by default) and can be revoked; every change is in the audit log.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

from .config import PATTERN_DAYS
from .features import RULES

SCHEMA = [
    """CREATE TABLE IF NOT EXISTS flags (close TEXT, doc_id TEXT, company_code TEXT, supplier_id TEXT, user_id TEXT,
        amount_eur REAL, risk REAL, rank INTEGER, reasons TEXT, model_version TEXT,
        status TEXT DEFAULT 'open', note TEXT, decided_by TEXT, decided_at TEXT, decision_reason TEXT,
        PRIMARY KEY (close, doc_id))""",
    """CREATE TABLE IF NOT EXISTS decisions (doc_id TEXT PRIMARY KEY, label INTEGER, by TEXT, at TEXT, reason TEXT,
        use_for_training INTEGER)""",
    """CREATE TABLE IF NOT EXISTS patterns (pattern_id TEXT PRIMARY KEY, rule TEXT, company_code TEXT, supplier_id TEXT,
        user_id TEXT, description TEXT, example_doc TEXT, proposed_by TEXT, proposed_at TEXT, approved_by TEXT,
        approved_at TEXT, expires TEXT, status TEXT)""",
    """CREATE TABLE IF NOT EXISTS model_versions (version TEXT PRIMARY KEY, created TEXT, metrics TEXT, promoted INTEGER,
        reviewer_labels INTEGER, note TEXT)""",
]


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class Memory:
    def __init__(self, path: Path):
        self.path = Path(path)
        with self._con() as con:
            for s in SCHEMA:
                con.execute(s)

    def _con(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.path, timeout=10)
        con.row_factory = sqlite3.Row
        return con

    def _rows(self, sql: str, args=()) -> list[dict]:
        with self._con() as con:
            return [dict(r) for r in con.execute(sql, args)]

    # ---------- flags ----------
    def save_flags(self, close: str, rows: list[dict], version: str) -> None:
        with self._con() as con:
            for r in rows:  # keep decisions already made on a re-run
                con.execute(
                    "INSERT INTO flags (close, doc_id, company_code, supplier_id, user_id, amount_eur, risk, rank, reasons, "
                    "model_version) VALUES (?,?,?,?,?,?,?,?,?,?) "
                    "ON CONFLICT(close, doc_id) DO UPDATE SET risk=excluded.risk, rank=excluded.rank, "
                    "reasons=excluded.reasons, model_version=excluded.model_version",
                    (close, r["doc_id"], r["company_code"], r.get("supplier_id"), r.get("user_id"), r["amount_eur"], r["risk"],
                     r["rank"], json.dumps(r["reasons"]), version))

    def flags(self, close: str) -> list[dict]:
        out = self._rows("SELECT * FROM flags WHERE close=? ORDER BY rank", (close,))
        for r in out:
            r["reasons"] = json.loads(r["reasons"] or "[]")
            r["note"] = json.loads(r["note"]) if r["note"] else None
        return out

    def save_note(self, close: str, doc_id: str, note: dict) -> None:
        with self._con() as con:
            con.execute("UPDATE flags SET note=? WHERE close=? AND doc_id=?", (json.dumps(note, default=str), close, doc_id))

    def clear_flags(self, close: str) -> None:
        with self._con() as con:
            con.execute("DELETE FROM flags WHERE close=? AND status='open'", (close,))

    # ---------- decisions (labels) ----------
    def decide(self, close: str, doc_id: str, label: str, by: str, reason: str, use_for_training: bool = True) -> dict:
        if label not in ("issue", "fine", "ask"):
            return {"ok": False, "error": "label must be issue, fine or ask"}
        if label == "fine" and len(reason.strip()) < 5:
            return {"ok": False, "error": "Say why it is fine (a short reason is enough)."}
        with self._con() as con:
            cur = con.execute("UPDATE flags SET status=?, decided_by=?, decided_at=?, decision_reason=? WHERE close=? AND doc_id=?",
                              (label, by, now(), reason[:500], close, doc_id))
            if not cur.rowcount:
                return {"ok": False, "error": "This entry is not flagged in this close."}
            if label != "ask":
                con.execute("INSERT OR REPLACE INTO decisions VALUES (?,?,?,?,?,?)",
                            (doc_id, 1 if label == "issue" else 0, by, now(), reason[:500], int(use_for_training)))
        return {"ok": True}

    def training_labels(self) -> dict[str, int]:
        return {r["doc_id"]: r["label"] for r in self._rows("SELECT doc_id, label FROM decisions WHERE use_for_training=1")}

    def decisions(self) -> list[dict]:
        return self._rows("SELECT * FROM decisions ORDER BY at DESC")

    # ---------- known patterns ----------
    def propose_pattern(self, rule: str, company_code: str, by: str, description: str, example_doc: str,
                        supplier_id: str | None = None, user_id: str | None = None, days: int = PATTERN_DAYS) -> dict:
        r = RULES.get(rule)
        if not r:
            return {"ok": False, "error": f"unknown rule {rule}"}
        if r.hard:
            return {"ok": False, "error": f"'{r.title}' is a control test; a known pattern can never silence it."}
        if not (supplier_id or user_id):
            return {"ok": False, "error": "A pattern must name one supplier or one user (no whole accounts or companies)."}
        if len(description.strip()) < 10:
            return {"ok": False, "error": "Describe the pattern in a sentence, so the approver knows what they approve."}
        pid = "P-" + uuid.uuid4().hex[:6].upper()
        expires = (datetime.now(UTC) + timedelta(days=days)).date().isoformat()
        with self._con() as con:
            con.execute("INSERT INTO patterns VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        (pid, rule, company_code, supplier_id, user_id, description[:400], example_doc, by, now(), None,
                         None, expires, "proposed"))
        return {"ok": True, "pattern_id": pid, "expires": expires}

    def approve_pattern(self, pattern_id: str, by: str) -> dict:
        p = self.pattern(pattern_id)
        if not p or p["status"] != "proposed":
            return {"ok": False, "error": "This pattern is not waiting for approval."}
        if p["proposed_by"] == by:
            return {"ok": False, "error": "Four eyes: the person who proposed a pattern cannot approve it."}
        with self._con() as con:
            con.execute("UPDATE patterns SET status='active', approved_by=?, approved_at=? WHERE pattern_id=? AND status='proposed'",
                        (by, now(), pattern_id))
        return {"ok": True}

    def set_pattern_status(self, pattern_id: str, status: str, by: str) -> dict:
        if status not in ("rejected", "revoked"):
            return {"ok": False, "error": "status must be rejected or revoked"}
        with self._con() as con:
            cur = con.execute("UPDATE patterns SET status=?, approved_by=COALESCE(approved_by, ?) WHERE pattern_id=? "
                              "AND status IN ('proposed', 'active')", (status, by, pattern_id))
        return {"ok": True} if cur.rowcount else {"ok": False, "error": "No open pattern with this id."}

    def pattern(self, pattern_id: str) -> dict | None:
        r = self._rows("SELECT * FROM patterns WHERE pattern_id=?", (pattern_id,))
        return r[0] if r else None

    def patterns(self, status: str | None = None) -> list[dict]:
        self._expire()
        if status:
            return self._rows("SELECT * FROM patterns WHERE status=? ORDER BY proposed_at DESC", (status,))
        return self._rows("SELECT * FROM patterns ORDER BY proposed_at DESC")

    def _expire(self) -> None:
        with self._con() as con:
            con.execute("UPDATE patterns SET status='expired' WHERE status='active' AND expires < ?", (datetime.now(UTC).date().isoformat(),))

    def silenced(self, row: dict, rule: str) -> dict | None:
        """The active pattern that covers this rule hit on this entry, if any. Hard rules: always None."""
        if RULES[rule].hard:
            return None
        for p in self.patterns("active"):
            if p["rule"] != rule or p["company_code"] != row["company_code"]:
                continue
            if p["supplier_id"] and p["supplier_id"] != row.get("supplier_id"):
                continue
            if p["user_id"] and p["user_id"] != row.get("user_id"):
                continue
            return p
        return None

    # ---------- model versions ----------
    def save_version(self, version: str, metrics: dict, promoted: bool, reviewer_labels: int, note: str = "") -> None:
        with self._con() as con:
            con.execute("INSERT OR REPLACE INTO model_versions VALUES (?,?,?,?,?,?)",
                        (version, now(), json.dumps(metrics, default=str), int(promoted), reviewer_labels, note))

    def versions(self) -> list[dict]:
        out = self._rows("SELECT * FROM model_versions ORDER BY created, rowid")
        for r in out:
            r["metrics"] = json.loads(r["metrics"])
        return out

    def promoted(self) -> str | None:
        r = self._rows("SELECT version FROM model_versions WHERE promoted=1 ORDER BY created DESC, rowid DESC LIMIT 1")
        return r[0]["version"] if r else None
