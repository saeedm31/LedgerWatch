"""MCP server: the general ledger, read-only.

The agent investigates with these tools. They open the warehouse read-only (DuckDB/SQLite read-only mode,
or a Snowflake role with SELECT only), so nothing here can change the books. The answer key for the
planted problems is a separate file that this server never opens.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from ..warehouse import Warehouse
from .common import READ, data_dir, serve

server = MCPServer(
    "ledgerwatch-ledger",
    instructions="Read-only general ledger of Aurelia Energy Group: journal entries, users, suppliers, account history.",
)


def _q(sql: str, params: tuple = ()) -> list[dict]:
    with Warehouse(data_dir()) as wh:
        df = wh.df(sql, params)
    return [{k: (None if v != v else v) for k, v in r.items()} for r in df.to_dict("records")]  # NaN -> None


def _entry(doc_id: str) -> dict:
    r = _q("SELECT * FROM journal_entries WHERE doc_id = ?", (doc_id.strip(),))
    if not r:
        raise ToolError(f"no journal entry {doc_id}")
    return r[0]


@server.tool(annotations=READ)
def get_entry(doc_id: str) -> dict:
    """One journal entry: header, debit and credit lines, the user who posted it and the supplier."""
    e = _entry(doc_id)
    e["lines"] = _q("SELECT line_no, account, dc, amount_eur, cost_center FROM journal_lines WHERE doc_id = ? ORDER BY line_no",
                    (e["doc_id"],))
    u = _q("SELECT user_id, name, company_code, role, approval_limit_eur, is_batch FROM users WHERE user_id = ?", (e["user_id"],))
    e["user"] = u[0] if u else None
    if e.get("supplier_id"):
        s = _q("SELECT supplier_id, name, country, bank_country, created_date, main_account FROM suppliers WHERE supplier_id = ?",
               (e["supplier_id"],))
        e["supplier"] = s[0] if s else None
    return e


@server.tool(annotations=READ)
def similar_invoices(doc_id: str, days: int = 90) -> dict:
    """Other invoices from the same supplier within +/- `days` days with the same amount or the same reference
    (references compared without spaces, dashes and the INV prefix)."""
    e = _entry(doc_id)
    if not e.get("supplier_id"):
        return {"doc_id": doc_id, "matches": [], "note": "entry has no supplier"}
    d = date.fromisoformat(str(e["posting_date"])[:10])
    rows = _q("SELECT doc_id, posting_date, amount_eur, reference, user_id, description FROM journal_entries "
              "WHERE supplier_id = ? AND doc_id <> ? AND posting_date BETWEEN ? AND ? AND doc_type = 'invoice'",
              (e["supplier_id"], e["doc_id"], (d - timedelta(days=days)).isoformat(), (d + timedelta(days=days)).isoformat()))

    def norm(r):
        r = (r or "").upper().split("/")[0]
        r = "".join(ch for ch in r if ch.isalnum())
        return r[3:] if r.startswith("INV") else r

    out = []
    for r in rows:
        same_amount = abs(r["amount_eur"] - e["amount_eur"]) < 0.005
        same_ref = bool(e.get("reference")) and norm(r["reference"]) == norm(e["reference"])
        if same_amount or same_ref:
            out.append({**r, "same_amount": same_amount, "same_reference": same_ref})
    return {"doc_id": doc_id, "supplier_id": e["supplier_id"], "amount_eur": e["amount_eur"], "reference": e.get("reference"),
            "window_days": days, "matches": out[:10]}


@server.tool(annotations=READ)
def user_activity(user_id: str, period: str) -> dict:
    """What one user posted in one period (YYYY-MM): count and totals by account (with the user's average month on
    each account over the 6 months before), weekend or late manual postings, and postings in other companies."""
    u = _q("SELECT user_id, name, company_code, role, approval_limit_eur, is_batch FROM users WHERE user_id = ?", (user_id,))
    if not u:
        raise ToolError(f"no user {user_id}")
    by_acc = _q("SELECT l.account, COUNT(*) AS entries, ROUND(SUM(l.amount_eur), 2) AS total_eur FROM journal_entries e "
                "JOIN journal_lines l ON l.doc_id = e.doc_id AND l.dc = 'D' WHERE e.user_id = ? AND e.period = ? "
                "GROUP BY l.account ORDER BY total_eur DESC LIMIT 8", (user_id, period))
    y, m = map(int, period.split("-"))
    before = [f"{(y * 12 + m - 1 - k) // 12}-{(y * 12 + m - 1 - k) % 12 + 1:02d}" for k in range(1, 7)]
    hist = _q(f"SELECT period, COUNT(*) AS entries, ROUND(SUM(amount_eur), 2) AS total_eur FROM journal_entries "
              f"WHERE user_id = ? AND period IN ({','.join('?' * 6)}) GROUP BY period", (user_id, *before))
    acc_before = _q(f"SELECT l.account, ROUND(SUM(l.amount_eur) / {len(before)}, 2) AS avg_month_eur FROM journal_entries e "
                    f"JOIN journal_lines l ON l.doc_id = e.doc_id AND l.dc = 'D' WHERE e.user_id = ? "
                    f"AND e.period IN ({','.join('?' * 6)}) GROUP BY l.account", (user_id, *before))
    avg = {r["account"]: r["avg_month_eur"] for r in acc_before}
    for r in by_acc:
        r["avg_month_before_eur"] = avg.get(r["account"], 0.0)
    odd = _q("SELECT doc_id, entry_ts, amount_eur, company_code FROM journal_entries WHERE user_id = ? AND period = ? "
             "AND source = 'manual'", (user_id, period))
    late = [r for r in odd if datetime.fromisoformat(str(r["entry_ts"])).weekday() >= 5
            or not 6 <= datetime.fromisoformat(str(r["entry_ts"])).hour < 22]
    other_co = sorted({r["company_code"] for r in odd if r["company_code"] != u[0]["company_code"]})
    n_hist = max(len(hist), 1)
    return {
        "user": u[0], "period": period, "by_account": by_acc,
        "entries": sum(r["entries"] for r in by_acc), "total_eur": round(sum(r["total_eur"] for r in by_acc), 2),
        "avg_month_before": {"entries": round(sum(h["entries"] for h in hist) / n_hist, 1),
                             "total_eur": round(sum(h["total_eur"] for h in hist) / n_hist, 2), "months": len(hist)},
        "weekend_or_late_manual": len(late), "posted_in_other_companies": other_co,
    }


@server.tool(annotations=READ)
def supplier_profile(supplier_id: str) -> dict:
    """Supplier master data and invoice history: created date, country, bank country, invoices per month."""
    s = _q("SELECT * FROM suppliers WHERE supplier_id = ?", (supplier_id,))
    if not s:
        raise ToolError(f"no supplier {supplier_id}")
    hist = _q("SELECT period, COUNT(*) AS invoices, ROUND(SUM(amount_eur), 2) AS total_eur FROM journal_entries "
              "WHERE supplier_id = ? AND doc_type = 'invoice' GROUP BY period ORDER BY period", (supplier_id,))
    first = _q("SELECT MIN(posting_date) AS first_invoice FROM journal_entries WHERE supplier_id = ?", (supplier_id,))
    return {**s[0], "bank_in_other_country": s[0]["bank_country"] != s[0]["country"],
            "first_invoice": str(first[0]["first_invoice"])[:10] if first and first[0]["first_invoice"] else None,
            "invoices_by_month": hist[-12:]}


@server.tool(annotations=READ)
def account_history(company_code: str, account: str) -> dict:
    """Monthly debit totals of one account in one company (all months), to compare this month with the past."""
    rows = _q("SELECT e.period, COUNT(*) AS entries, ROUND(SUM(l.amount_eur), 2) AS total_eur FROM journal_entries e "
              "JOIN journal_lines l ON l.doc_id = e.doc_id AND l.dc = 'D' WHERE e.company_code = ? AND l.account = ? "
              "GROUP BY e.period ORDER BY e.period", (company_code, account))
    name = _q("SELECT name FROM accounts WHERE account = ?", (account,))
    return {"company_code": company_code, "account": account, "account_name": name[0]["name"] if name else None,
            "months": rows}


@server.tool(annotations=READ)
def account_pair_usage(debit_account: str, credit_account: str) -> dict:
    """How often this debit/credit combination is used across the group, and by whom (top 5 users)."""
    rows = _q("SELECT e.user_id, COUNT(*) AS n FROM journal_entries e JOIN journal_lines d ON d.doc_id = e.doc_id AND d.dc = 'D' "
              "JOIN journal_lines c ON c.doc_id = e.doc_id AND c.dc = 'C' WHERE d.account = ? AND c.account = ? "
              "GROUP BY e.user_id ORDER BY n DESC", (debit_account, credit_account))
    return {"debit_account": debit_account, "credit_account": credit_account, "times_used": sum(r["n"] for r in rows),
            "top_users": rows[:5]}


if __name__ == "__main__":
    serve(server)
