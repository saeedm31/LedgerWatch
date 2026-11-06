"""A scripted stand-in for Claude, for the free demo and for CI. Not AI.

It follows a fixed playbook per reason code (which tools to call, how to read the answers) through the same
message format and tool calls as the real model, so the MCP servers, grounding check, audit log, memory and
evals run exactly as they do with Claude. Its text is a template; Claude writes better notes.
"""

from __future__ import annotations

import json
import uuid
from types import SimpleNamespace

PLAYBOOK = {  # reason code -> tools to call (in order)
    "DUPLICATE": ["similar_invoices"],
    "REPEAT_AMOUNT": ["similar_invoices"],
    "SPLIT": ["user_activity", "similar_invoices"],
    "OTHER_COMPANY_USER": ["user_activity"],
    "OVER_LIMIT": ["user_activity"],
    "NEW_SUPPLIER_LARGE": ["supplier_profile"],
    "PERIOD_END_REVENUE": ["account_history", "user_activity"],
    "WEEKEND_LATE": ["user_activity"],
    "RARE_PAIR": ["account_pair_usage"],
    "ROUND_AMOUNT": ["similar_invoices"],
    "TEXT_MISMATCH": ["account_history"],
    "UNUSUAL_TOTAL": ["user_activity", "account_history"],
}


def _reply(name: str, args: dict, text: str = ""):
    content = ([{"type": "text", "text": text}] if text else []) + [
        {"type": "tool_use", "id": "toolu_" + uuid.uuid4().hex[:12], "name": name, "input": args}]
    return SimpleNamespace(content=content, stop_reason="tool_use",
                           usage=SimpleNamespace(input_tokens=900, output_tokens=len(json.dumps(args)) // 4))


def _args(tool: str, e: dict) -> dict:
    debit = next((ln["account"] for ln in e["lines"] if ln["dc"] == "D"), None)
    credit = next((ln["account"] for ln in e["lines"] if ln["dc"] == "C"), None)
    return {
        "similar_invoices": {"doc_id": e["doc_id"]},
        "user_activity": {"user_id": e["user_id"], "period": e["period"]},
        "supplier_profile": {"supplier_id": e.get("supplier_id") or ""},
        "account_history": {"company_code": e["company_code"],
                            "account": credit if debit in ("120000", "113000") else debit},
        "account_pair_usage": {"debit_account": debit, "credit_account": credit},
    }[tool]


def _debit(e: dict) -> str | None:
    return next((ln["account"] for ln in e["lines"] if ln["dc"] == "D"), None)


def _eur(v) -> str:
    return f"EUR {float(v):,.2f}"


def _assess(e: dict, codes: list[str], done: dict) -> dict:
    ev, issue, fine = [], [], []
    sim = done.get("similar_invoices", {})
    for m in sim.get("matches", [])[:2]:
        what = "same amount and reference" if m["same_amount"] and m["same_reference"] else \
            "same amount" if m["same_amount"] else "same reference"
        ev.append(f"{m['doc_id']} on {str(m['posting_date'])[:10]}: {_eur(m['amount_eur'])}, {what}.")
    if any(m["same_amount"] and m["same_reference"] for m in sim.get("matches", [])):
        issue.append("an earlier invoice has the same amount and reference")
    elif {"DUPLICATE", "REPEAT_AMOUNT"} & set(codes) and sim.get("matches"):
        issue.append("the supplier billed the same amount shortly before")
    if "ROUND_AMOUNT" in codes and len([m for m in sim.get("matches", []) if m["same_amount"]]) >= 2:
        fine.append("the same round amount is billed regularly, which looks like a fixed contract")
    ua = done.get("user_activity")
    if ua:
        u = ua["user"]
        ev.append(f"{u['user_id']} ({u['role']}, {u['company_code']}, limit {_eur(u['approval_limit_eur'])}) posted "
                  f"{ua['entries']} entries for {_eur(ua['total_eur'])} in {ua['period']}; average month before "
                  f"{_eur(ua['avg_month_before']['total_eur'])}.")
        if ua["posted_in_other_companies"]:
            issue.append(f"the user also posted in {', '.join(ua['posted_in_other_companies'])}")
        if "SPLIT" in codes:
            issue.append("several invoices sit just under the user's approval limit")
        acc = next((a for a in ua["by_account"] if a["account"] == _debit(e)), None)
        if acc:
            ev.append(f"On account {acc['account']} this month: {_eur(acc['total_eur'])} in {acc['entries']} entries; "
                      f"average month before {_eur(acc['avg_month_before_eur'])}.")
            if "UNUSUAL_TOTAL" in codes and acc["total_eur"] > 2 * max(acc["avg_month_before_eur"], 1):
                issue.append(f"the user's total on account {acc['account']} is far above their usual month")
    sp = done.get("supplier_profile")
    if sp:
        ev.append(f"Supplier {sp['supplier_id']} created {sp['created_date']}, bank country {sp['bank_country']}, "
                  f"country {sp['country']}.")
        if sp["bank_in_other_country"]:
            issue.append("the new supplier's bank account is in another country")
        else:
            issue.append("a large first payment to a new supplier needs the onboarding file")
    ah = done.get("account_history")
    if ah and ah["months"]:
        cur = next((m for m in ah["months"] if m["period"] == e["period"]), ah["months"][-1])
        prev = [m for m in ah["months"] if m["period"] < cur["period"]]
        ev.append(f"{ah['account_name'] or ah['account']} in {ah['company_code']}: {_eur(cur['total_eur'])} in "
                  f"{cur['period']}" + (f", {_eur(prev[-1]['total_eur'])} in {prev[-1]['period']}." if prev else "."))
    ap = done.get("account_pair_usage")
    if ap:
        ev.append(f"Debit {ap['debit_account']} / credit {ap['credit_account']} used {ap['times_used']} times in the group.")
        if ap["times_used"] <= 10:
            issue.append("the account combination is almost never used")
    pats = done.get("known_patterns", {}).get("patterns", [])
    if pats:
        fine.append(f"known pattern {pats[0]['pattern_id']} applies: {pats[0]['description']}")
    past = done.get("past_decisions", {}).get("decisions", [])
    if past:
        ev.append(f"Earlier decision on {past[0]['doc_id']}: {past[0]['decision']} ({past[0]['decision_reason']}).")
    if "TEXT_MISMATCH" in codes:
        issue.append("the description does not fit the account")
    if "PERIOD_END_REVENUE" in codes:
        issue.append("manual revenue at the very end of the period needs the contract and delivery proof")
    if "WEEKEND_LATE" in codes and not issue:
        issue.append("a manual posting outside working hours needs a reason")
    if issue and not (fine and not {"DUPLICATE", "SPLIT"} & set(codes)):
        a, why = "likely_issue", issue
    elif fine:
        a, why = "likely_fine", fine
    else:
        a, why = "unclear", ["the ledger alone does not explain it"]
    text = "; ".join(why)
    summary = f"{e['doc_id']}: {_eur(e['amount_eur'])} posted by {e['user_id']}. " + text[:1].upper() + text[1:] + "."
    nxt = {"likely_issue": "Ask the preparer for the supporting document before the close.",
           "likely_fine": "Mark as fine if the supporting document matches.",
           "unclear": "Ask the preparer."}[a]
    q = f"Can you send the supporting document for {e['doc_id']} ({_eur(e['amount_eur'])}) and the reason for this posting?"
    return {"assessment": a, "summary": summary, "evidence": ev[:5], "next_step": nxt,
            "question_for_preparer": q if a != "likely_fine" else ""}


class _Messages:
    def create(self, **kw):
        msgs = kw["messages"]
        ctx = json.loads(msgs[0]["content"].split("\n", 1)[1].split("\n<untrusted", 1)[0])
        e, codes = ctx["entry"], [r["code"] for r in ctx["reasons"]]
        calls: dict[str, str] = {}
        done: dict[str, dict] = {}
        for m in msgs[1:]:
            for b in m["content"] if isinstance(m["content"], list) else []:
                if b.get("type") == "tool_use":
                    calls[b["id"]] = b["name"]
                elif b.get("type") == "tool_result":
                    done[calls.get(b["tool_use_id"], "?")] = {} if b.get("is_error") else json.loads(b["content"])
        forced = (kw.get("tool_choice") or {}).get("name") == "submit_assessment"
        if not forced:
            plan = [t for c in codes for t in PLAYBOOK.get(c, [])] + ["known_patterns", "past_decisions"]
            for tool in dict.fromkeys(plan):
                if tool in done or (tool == "supplier_profile" and not e.get("supplier_id")):
                    continue
                if tool == "known_patterns":
                    return _reply(tool, {"company_code": e["company_code"], "supplier_id": e.get("supplier_id"),
                                         "user_id": e["user_id"]})
                if tool == "past_decisions":
                    return _reply(tool, {"supplier_id": e.get("supplier_id"), "user_id": e["user_id"]})
                return _reply(tool, _args(tool, e))
        return _reply("submit_assessment", _assess(e, codes, done))


class ScriptedAnthropic:
    """Drop-in for anthropic.Anthropic() in this app: same messages.create() shape."""

    def __init__(self):
        self.messages = _Messages()
