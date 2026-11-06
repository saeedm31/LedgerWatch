"""What the app does, in one place: score a close, investigate flags, record decisions, manage memory, retrain.

The ledger is only read. Everything LedgerWatch writes (flags, notes, decisions, patterns, model versions)
goes to its own database and to the hash-chained audit log.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .agent.graph import Investigator
from .agent.scripted import ScriptedAnthropic
from .audit import AuditLog
from .config import CURRENT_CLOSE, FLAGS_PER_1000, INVESTIGATE_TOP, MONTHS, Settings
from .features import RULES, load_entries, reasons
from .llm import Meter, MeteredClient, make_client
from .mcp_hub import MCPHub
from .memory import Memory
from .models import Bundle, enrich, risk_scores, select_flags
from .warehouse import Warehouse

# Who may do what. In a company this comes from SSO groups.
PEOPLE = {
    "ana.silva": "Accountant (reviewer)",
    "jonas.weber": "Senior accountant (reviewer)",
    "maria.ionescu": "Group controller (approver)",
}
APPROVERS = {"jonas.weber", "maria.ionescu"}  # may approve known patterns (never their own)


class Service:
    def __init__(self, settings: Settings, hub: MCPHub):
        self.s, self.hub = settings, hub
        self.audit = AuditLog(settings.audit_db)
        self.memory = Memory(settings.app_db)
        self._bundles: dict[str, Bundle] = {}

    # ---------- models ----------
    @property
    def version(self) -> str:
        return self.memory.promoted() or "v1"

    def bundle(self, version: str | None = None) -> Bundle:
        v = version or self.version
        if v not in self._bundles:
            self._bundles[v] = Bundle.load(self.s.models / f"bundle-{v}.joblib")
        return self._bundles[v]

    def metrics(self, version: str | None = None) -> dict:
        p = self.s.models / f"metrics-{version or self.version}.json"
        return json.loads(p.read_text()) if p.exists() else {}

    def retrain(self, by: str) -> dict:
        """Train a new version with the reviewers' decisions. Promote it only if it passes the gates and does not
        lose recall against the current version."""
        from .evals import run
        from .tracking import log_run

        labels = self.memory.training_labels()
        if not labels:
            return {"ok": False, "error": "No reviewer decisions yet. Mark some flags as issue or fine first."}
        version = f"v{len(self.memory.versions()) + 1}"
        _, m = run(self.s.data_dir, self.s.models, extra_labels=labels, version=version, compare_with=self.bundle())
        tracked = log_run(m, self.s.models)
        old = m["previous_same_entries"]
        better = m["passed"] and m["recall"] >= old["recall"] and m["precision"] >= old["precision"] - 0.02
        why = ("passed the gates and is at least as good as the current version on the same entries" if better
               else "failed a gate" if not m["passed"] else "worse than the current version on the same entries")
        self.memory.save_version(version, m, better, len(labels), why)
        self.audit.write("model_trained", dict(version=version, by=by, promoted=better, reason=why, reviewer_labels=len(labels),
                                               recall=m["recall"], precision=m["precision"], tracked_in=tracked), actor=by)
        return {"ok": True, "version": version, "promoted": better, "reason": why, "metrics": m}

    # ---------- a close ----------
    def entries(self, close: str) -> pd.DataFrame:
        with Warehouse(self.s.data_dir) as wh:
            return load_entries(wh, [close])

    def score_close(self, close: str = CURRENT_CLOSE, per_1000: float = FLAGS_PER_1000) -> dict:
        """Score every entry of a close, apply approved known patterns, flag the riskiest up to the review budget."""
        b = self.bundle()
        f = enrich(self.entries(close), b)
        f["risk"] = risk_scores(f, b)
        silenced = []
        for i in f.index[f[[f"r_{c}" for c in RULES]].any(axis=1)]:
            row = f.loc[i]
            hits = [c for c in RULES if row[f"r_{c}"]]
            covered = {c: self.memory.silenced(row, c) for c in hits}
            if all(covered.values()):  # every hit is a soft rule covered by an approved pattern
                f.at[i, "risk"] = row["risk"] * 0.05
                for c in hits:
                    f.at[i, f"r_{c}"] = False
                silenced.append({"doc_id": row["doc_id"], "patterns": sorted({p["pattern_id"] for p in covered.values()})})
        flagged = select_flags(f, f["risk"].to_numpy(), per_1000)
        top = f[flagged].sort_values("risk", ascending=False)
        rows = [dict(doc_id=r.doc_id, company_code=r.company_code, supplier_id=r.supplier_id, user_id=r.user_id,
                     amount_eur=float(r.amount_eur), risk=float(r.risk), rank=k + 1, reasons=reasons(r))
                for k, (_, r) in enumerate(top.iterrows())]
        self.memory.clear_flags(close)
        self.memory.save_flags(close, rows, self.version)
        self.audit.write("close_scored", dict(close=close, entries=len(f), flags=len(rows), version=self.version,
                                              silenced=silenced[:50]), actor="detector")
        return {"close": close, "entries": len(f), "flags": len(rows), "silenced": silenced, "version": self.version,
                "frame": f}

    def flags(self, close: str = CURRENT_CLOSE) -> list[dict]:
        return self.memory.flags(close)

    # ---------- the agent ----------
    def investigator(self, mode: str = "scripted", api_key: str | None = None, model: str | None = None,
                     meter: Meter | None = None) -> Investigator:
        if mode == "scripted":
            return Investigator(MeteredClient(ScriptedAnthropic(), meter), self.hub, self.audit, "scripted")
        llm = make_client(self.s.llm_provider if mode != "bedrock" else "bedrock", api_key=api_key,
                          region=self.s.aws_region, meter=meter)
        return Investigator(llm, self.hub, self.audit, model or self.s.agent_model)

    def investigate(self, close: str, doc_id: str, mode: str = "scripted", **kw) -> dict:
        flag = next((f for f in self.flags(close) if f["doc_id"] == doc_id), None)
        if not flag:
            return {"assessment": "unclear", "summary": "This entry is not flagged in this close.", "checks": ["not_flagged"]}
        note = self.investigator(mode, **kw).run(flag)
        self.memory.save_note(close, doc_id, note)
        return note

    def investigate_top(self, close: str = CURRENT_CLOSE, n: int = INVESTIGATE_TOP, mode: str = "scripted", **kw) -> int:
        todo = [f for f in self.flags(close) if not f["note"]][:n]
        agent = self.investigator(mode, **kw)
        for f in todo:
            self.memory.save_note(close, f["doc_id"], agent.run(f))
        return len(todo)

    # ---------- people ----------
    def decide(self, close: str, doc_id: str, label: str, by: str, reason: str, learn: bool = True) -> dict:
        r = self.memory.decide(close, doc_id, label, by, reason, learn)
        self.audit.write("decision", dict(close=close, doc_id=doc_id, label=label, reason=reason, learn=learn, ok=r["ok"]),
                         run_id=doc_id, actor=by)
        return r

    def propose_pattern(self, by: str, flag: dict, rule: str, description: str, scope: str) -> dict:
        r = self.memory.propose_pattern(rule, flag["company_code"], by, description, flag["doc_id"],
                                        supplier_id=flag.get("supplier_id") if scope == "supplier" else None,
                                        user_id=flag.get("user_id") if scope == "user" else None)
        self.audit.write("pattern_proposed", dict(rule=rule, scope=scope, description=description, **r), run_id=flag["doc_id"], actor=by)
        return r

    def approve_pattern(self, pattern_id: str, by: str) -> dict:
        if by not in APPROVERS:
            r = {"ok": False, "error": f"{by} ({PEOPLE.get(by, 'unknown')}) cannot approve patterns."}
        else:
            r = self.memory.approve_pattern(pattern_id, by)
        self.audit.write("pattern_approval", dict(pattern_id=pattern_id, **r), actor=by)
        return r

    def close_pattern(self, pattern_id: str, status: str, by: str) -> dict:
        r = self.memory.set_pattern_status(pattern_id, status, by)
        self.audit.write(f"pattern_{status}", dict(pattern_id=pattern_id, **r), actor=by)
        return r

    def entry(self, doc_id: str) -> dict | None:
        r = self.hub.call("get_entry", {"doc_id": doc_id})
        return r.data if r.ok else None

    # ---------- setup ----------
    def reset(self) -> None:
        """Start the demo again: same ledger and model v1, no flags, decisions, patterns or later versions."""
        for p in (self.s.app_db, self.s.audit_db):
            Path(p).unlink(missing_ok=True)
        for p in self.s.models.glob("*"):
            if p.name not in ("bundle-v1.joblib", "metrics-v1.json"):
                p.unlink()
        self._bundles.clear()
        self.audit = AuditLog(self.s.audit_db)
        self.memory = Memory(self.s.app_db)
        self.memory.save_version("v1", self.metrics("v1"), True, 0, "trained on past review outcomes")
        self.score_close(CURRENT_CLOSE)
        self.investigate_top(CURRENT_CLOSE)


def ensure_ready(s: Settings) -> None:
    """Build the simulated group and train model v1 if they are missing (first start, fresh container)."""
    from .evals import run
    from .simulate import ensure_built
    from .tracking import log_run

    ensure_built(s.data_dir)
    if not (s.models / "bundle-v1.joblib").exists():
        _, m = run(s.data_dir, s.models, version="v1")
        log_run(m, s.models)


def open_service(data_dir: Path | None = None) -> Service:
    s = Settings() if data_dir is None else Settings(data_dir=Path(data_dir))
    ensure_ready(s)
    svc = Service(s, MCPHub(s.data_dir))
    if not svc.memory.versions():
        svc.memory.save_version("v1", svc.metrics("v1"), True, 0, "trained on past review outcomes")
    if not svc.flags(CURRENT_CLOSE):
        svc.score_close(CURRENT_CLOSE)
        svc.investigate_top(CURRENT_CLOSE)
    return svc


__all__ = ["Service", "open_service", "PEOPLE", "APPROVERS", "MONTHS"]
