"""MCP server: the casebook, read-only. What reviewers decided before and which known patterns are active.

The agent reads memory through this server; it cannot write to it. Decisions and patterns are saved only
by the app, when a person clicks.
"""

from __future__ import annotations

import sqlite3

from mcp.server.mcpserver import MCPServer

from .common import READ, data_dir, serve

server = MCPServer("ledgerwatch-casebook", instructions="Past review decisions and approved known patterns (read-only).")


def _q(sql: str, params: tuple = ()) -> list[dict]:
    path = data_dir() / "app.db"
    if not path.exists():
        return []
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=10)
    con.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in con.execute(sql, params)]
    except sqlite3.OperationalError:  # tables not created yet
        return []
    finally:
        con.close()


@server.tool(annotations=READ)
def known_patterns(company_code: str, supplier_id: str | None = None, user_id: str | None = None) -> dict:
    """Active known patterns (approved by a second person, not expired) for this company and supplier or user."""
    rows = _q("SELECT pattern_id, rule, supplier_id, user_id, description, proposed_by, approved_by, expires FROM patterns "
              "WHERE status = 'active' AND company_code = ? AND (supplier_id IS NULL OR supplier_id = ?) "
              "AND (user_id IS NULL OR user_id = ?) AND expires >= date('now')", (company_code, supplier_id or "", user_id or ""))
    return {"company_code": company_code, "patterns": rows}


@server.tool(annotations=READ)
def past_decisions(supplier_id: str | None = None, user_id: str | None = None) -> dict:
    """Earlier review decisions (issue or fine, with the reviewer's reason) for this supplier or user."""
    if not (supplier_id or user_id):
        return {"decisions": [], "note": "give a supplier_id or a user_id"}
    rows = _q("SELECT close, doc_id, status AS decision, decided_by, decision_reason, amount_eur FROM flags "
              "WHERE status IN ('issue', 'fine') AND ((? <> '' AND supplier_id = ?) OR (? <> '' AND user_id = ?)) "
              "ORDER BY decided_at DESC LIMIT 10", (supplier_id or "", supplier_id or "", user_id or "", user_id or ""))
    return {"supplier_id": supplier_id, "user_id": user_id, "decisions": rows}


if __name__ == "__main__":
    serve(server)
