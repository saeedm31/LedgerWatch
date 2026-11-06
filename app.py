"""LedgerWatch: continuous journal-entry monitoring for the month-end close.

Classic ML finds the risky entries, an AI agent investigates them with read-only tools, and accountants decide.
What they decide becomes memory: labels that retrain the model, and known patterns (approved by a second person)
that stop the same false alarm next month."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime

import pandas as pd
import streamlit as st

from ledgerwatch import config as C
from ledgerwatch.features import RULES
from ledgerwatch.llm import AGENT_MODELS, PRICES_CHECKED, Meter, eur
from ledgerwatch.service import APPROVERS, PEOPLE, open_service

st.set_page_config(page_title="LedgerWatch", page_icon="⚡", layout="wide")
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@500;700&family=Inter:wght@400;500;600&display=swap');
html, body, [data-testid="stAppViewContainer"], [data-testid="stSidebar"] { font-family: 'Inter', sans-serif; }
h1, h2, h3, [data-testid="stMetricValue"] { font-family: 'Sora', sans-serif !important; letter-spacing: -0.01em; }
[data-testid="stAppViewContainer"] {
  background: radial-gradient(1000px 500px at 90% -10%, #134A4F 0%, rgba(8,24,28,0) 60%),
              linear-gradient(140deg, #08181C 0%, #0E2A30 100%);
}
[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 2.2rem; max-width: 1400px; }
#MainMenu, footer { visibility: hidden; }
[data-testid="stSidebar"] { border-right: 1px solid #23474F; }
[data-testid="stMetric"] { background: rgba(18,48,56,0.85); border: 1px solid #23474F; border-radius: 14px;
  padding: 14px 18px; box-shadow: 0 8px 24px rgba(0,0,0,0.25); }
[data-testid="stMetricLabel"] p { color: #8FB3B5; font-size: 0.9rem; }
.stTabs [data-baseweb="tab-list"] { gap: 4px; border-bottom: 1px solid #23474F; }
.stTabs [data-baseweb="tab"] { padding: 10px 16px; border-radius: 10px 10px 0 0; color: #B5CACB; }
.stTabs [aria-selected="true"] { background: #123038; color: #F2F4EE; }
.stTabs [data-baseweb="tab-highlight"] { background-color: #F5C542; }
[data-testid="stDataFrame"] { border: 1px solid #23474F; border-radius: 12px; overflow: hidden; }
.lw-kicker { color: #F5C542; font-weight: 600; letter-spacing: 0.18em; text-transform: uppercase; font-size: 0.8rem; }
.lw-title { font-family: 'Sora', sans-serif; font-size: 2.1rem; font-weight: 700; line-height: 1.15;
            margin: 0.35rem 0 0.5rem 0; color: #F2F4EE; }
.lw-sub { color: #B5CACB; font-size: 1.05rem; line-height: 1.5; max-width: 66rem; }
.lw-steps { display: flex; gap: 0.6rem; flex-wrap: wrap; margin: 0.9rem 0 1.2rem 0; }
.lw-step { background: rgba(18,48,56,0.85); border: 1px solid #23474F; border-radius: 999px;
           padding: 0.35rem 0.95rem; color: #DDE8E6; font-size: 0.92rem; }
.lw-step b { color: #F5C542; margin-right: 0.35rem; }
.lw-pill { display: inline-block; padding: 0.1rem 0.55rem; border-radius: 999px; font-size: 0.8rem; margin: 0 0.25rem 0.25rem 0;
           border: 1px solid #23474F; background: #123038; color: #DDE8E6; }
.lw-hard { border-color: #E8735A; color: #FFB4A2; }
"""
HERO = """
<div class="lw-kicker">LedgerWatch · journal-entry monitoring for the month-end close</div>
<div class="lw-title">Machine learning finds the risky entries. An agent investigates. Accountants decide.</div>
<div class="lw-sub">Every journal entry of a 20-company energy-technology group is scored by four trained models and twelve
audit tests. The riskiest few per thousand go to review, with an AI investigation attached. Reviewer decisions become
memory: labels that retrain the model and known patterns, approved by a second person, that stop repeat false alarms.</div>
<div class="lw-steps">
  <span class="lw-step"><b>1</b>📊 Close: what was flagged and why</span>
  <span class="lw-step"><b>2</b>🔍 Review: read the agent's investigation, decide</span>
  <span class="lw-step"><b>3</b>🧠 Memory: approve patterns, retrain the model</span>
</div>
"""
st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)
ASSESS = {"likely_issue": "🔴 likely issue", "likely_fine": "🟢 likely fine", "unclear": "🟠 unclear"}
STATUS = {"open": "⏳ open", "issue": "🔴 issue", "fine": "🟢 fine", "ask": "✉️ asked preparer"}
SESSION_EUR = float(os.getenv("DEMO_SESSION_BUDGET_EUR", "0.5"))
DAILY_EUR = float(os.getenv("DEMO_DAILY_BUDGET_EUR", "3"))


def _shared_key() -> str | None:
    try:
        key = st.secrets.get("ANTHROPIC_API_KEY")
    except Exception:  # no secrets file
        key = None
    return key or os.getenv("ANTHROPIC_API_KEY") or None


SHARED_KEY = _shared_key()
BEDROCK = os.getenv("LLM_PROVIDER", "anthropic") == "bedrock"  # company AWS account: the task role, no key


def _code_version() -> str:
    """Changes when the package code changes, so a redeploy starts a fresh service and fresh MCP servers."""
    return str(max(f.stat().st_mtime_ns for f in (C.ROOT / "ledgerwatch").rglob("*.py")))


@st.cache_resource(show_spinner="First start: building the simulated ledger (≈100k entries), training the models and "
                                "starting the MCP servers. About a minute...", max_entries=1)
def service(code_version: str):
    return open_service(C.Settings().data_dir)


@st.cache_resource
def shared_spend() -> dict:
    return {"day": "", "eur": 0.0}


svc = service(_code_version())
ss = st.session_state
ss.setdefault("meter", Meter())
ss.setdefault("session_eur", 0.0)
labels = json.loads(svc.s.labels.read_text())

# ---------- sidebar ----------
with st.sidebar:
    st.title("⚡ LedgerWatch")
    st.caption("Journal-entry monitoring for the month-end close, built the way a finance team could run it.")
    st.warning("**Simulated group** (Aurelia Energy, 20 companies, 12 months). All names and numbers are fake.", icon="🧪")
    me = st.selectbox("Acting as", list(PEOPLE), format_func=lambda p: f"{p} · {PEOPLE[p]}",
                      help="In a company this comes from single sign-on. Try approving your own pattern: it is refused.")
    close = st.selectbox("Close", C.TEST_MONTHS[::-1], help="Months the models never saw in training.")
    show_key = st.toggle("Show the answer key", value=False,
                         help="The planted problems. Only for trying the demo: no model, agent or tool can read it.")
    st.divider()
    st.markdown("**AI settings** (for the investigating agent)")
    modes = (["AI (Claude)"] if SHARED_KEY or BEDROCK else []) + ["Demo (scripted, no AI)"] \
        + ([] if BEDROCK else ["Live (your Anthropic key)"])
    mode = st.radio("Mode", modes, index=0)
    api_key = SHARED_KEY if mode.startswith("AI") else None
    if mode.startswith("Live"):
        api_key = st.text_input("Anthropic API key", type="password", help="Used only for this session. Never stored or logged.")
    if mode.startswith("AI") and BEDROCK:
        st.caption(f"Claude on Amazon Bedrock ({svc.s.aws_region}), with this app's AWS role. No API key.")
    elif mode.startswith("AI"):
        st.caption(f"Free to try: up to €{SESSION_EUR:.2f} per visitor and €{DAILY_EUR:.2f} per day.")
    model = st.selectbox("Model", list(AGENT_MODELS), format_func=AGENT_MODELS.get,
                         index=list(AGENT_MODELS).index(svc.s.agent_model) if svc.s.agent_model in AGENT_MODELS else 0,
                         disabled=mode.startswith("Demo"))
    if mode.startswith("Demo"):
        st.caption("The scripted model is not AI: a fixed playbook per test, so the tools, grounding check, memory "
                   "and audit log can be tried for free. The first 25 flags already have its notes.")
    today = datetime.now(UTC).strftime("%Y-%m-%d")
    spend = shared_spend()
    if spend["day"] != today:
        spend.update(day=today, eur=0.0)
    st.progress(min(ss.session_eur / SESSION_EUR, 1.0), text=f"Your AI spend {eur(ss.session_eur)} of {eur(SESSION_EUR)}")
    st.caption(f"Prices checked {PRICES_CHECKED}.")
    st.divider()
    if st.button("↺ Reset demo", help="Back to model v1, no decisions or patterns. Everyone shares one demo."):
        with st.spinner("Resetting..."):
            svc.reset()
        st.rerun()

st.markdown(HERO, unsafe_allow_html=True)
tab_close, tab_review, tab_mem, tab_model, tab_how, tab_audit = st.tabs(
    ["📊 Close", "🔍 Review", "🧠 Memory", "🧪 Model & evals", "🧭 How it works", "🧾 Audit log"])


def can_run() -> str | None:
    if mode.startswith("Demo"):
        return None
    if not api_key and not (BEDROCK and mode.startswith("AI")):
        return "Enter your Anthropic API key in the sidebar, or pick Demo mode."
    if mode.startswith("AI") and spend["eur"] >= DAILY_EUR:
        return "Today's free AI budget is used up. Try Demo mode, or your own key in Live mode."
    if mode.startswith("AI") and ss.session_eur >= SESSION_EUR:
        return f"You have used your free {eur(SESSION_EUR)} of AI. Use Demo mode or your own key."
    return None


def investigate(doc_id: str) -> dict | None:
    if mode.startswith("Demo"):
        kw, m = {}, "scripted"
    else:
        kw, m = {"api_key": api_key, "model": model, "meter": ss.meter}, ("bedrock" if BEDROCK else "live")
    try:
        note = svc.investigate(close, doc_id, mode=m, **kw)
    except Exception as exc:
        st.error(f"The agent failed: {type(exc).__name__}: {exc}")
        return None
    cost = (note.get("cost") or {}).get("eur") or 0.0
    ss.session_eur += cost
    if mode.startswith("AI"):
        spend["eur"] += cost
    return note


def pills(reasons: list[dict]) -> str:
    return "".join(f'<span class="lw-pill{" lw-hard" if r["hard"] else ""}">{"⛔ " if r["hard"] else ""}{r["title"]}</span>'
                   for r in reasons) or '<span class="lw-pill">model score only</span>'


flags = svc.flags(close)
if not flags:
    with st.spinner(f"Scoring the {close} close..."):
        svc.score_close(close)
        svc.investigate_top(close)
    flags = svc.flags(close)

# ---------- close ----------
with tab_close:
    n_entries = len(svc.entries(close))
    decided = sum(f["status"] != "open" for f in flags)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Journal entries", f"{n_entries:,}")
    c2.metric("Flagged for review", len(flags), help=f"Review budget: {C.FLAGS_PER_1000:g} per 1,000 entries")
    c3.metric("Control tests hit", sum(any(r["hard"] for r in f["reasons"]) for f in flags))
    c4.metric("Decided", f"{decided} / {len(flags)}")
    c5.metric("Model", svc.version)
    rows = []
    for f in flags:
        n = f["note"] or {}
        row = {"#": f["rank"], "Entry": f["doc_id"], "Company": f["company_code"], "Amount (EUR)": f["amount_eur"],
               "Risk": f["risk"], "Why flagged": ", ".join(r["title"] for r in f["reasons"]) or "model score",
               "Agent": ASSESS.get(n.get("assessment"), "—"), "Status": STATUS.get(f["status"], f["status"])}
        if show_key:
            row["Answer key"] = labels.get(f["doc_id"], {}).get("kind", "— (normal entry)")
        rows.append(row)
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", height=560,
                 column_config={"Amount (EUR)": st.column_config.NumberColumn(format="%.2f"),
                                "Risk": st.column_config.ProgressColumn(min_value=0.0, max_value=1.0, format="%.2f")})
    st.caption("⛔ marks control tests (duplicates, splits, other-company users, over-limit, new suppliers, period-end "
               "revenue): always reviewed, never silenced by memory. Open an entry in the 🔍 Review tab.")
    left, right = st.columns(2)
    with left:
        st.markdown("**Flags by company**")
        st.bar_chart(pd.Series([f["company_code"] for f in flags]).value_counts().sort_index(), height=260)
    with right:
        st.markdown("**Flags by test**")
        hits = pd.Series([r["title"] for f in flags for r in f["reasons"]] or ["(none)"]).value_counts()
        st.bar_chart(hits, horizontal=True, height=260)
    if show_key:
        in_close = set(svc.entries(close).doc_id)
        planted = {g["group"] for d, g in labels.items() if d in in_close}
        found = {labels[f["doc_id"]]["group"] for f in flags if f["doc_id"] in labels}
        st.info(f"Answer key for {close}: {len(found)} of {len(planted)} planted problems are in the flag list.")

# ---------- review ----------
with tab_review:
    if not flags:
        st.info("No flags in this close.")
    else:
        by_id = {f["doc_id"]: f for f in flags}
        ids = list(by_id)
        pick = st.selectbox(
            "Flagged entry", ids, index=ids.index(ss["pick"]) if ss.get("pick") in ids else 0,
            format_func=lambda d: f"#{by_id[d]['rank']} · {d} · EUR {by_id[d]['amount_eur']:,.2f} · "
                                  f"{', '.join(r['title'] for r in by_id[d]['reasons']) or 'model score'} · "
                                  f"{STATUS.get(by_id[d]['status'])}")
        ss["pick"] = pick
        f = by_id[pick]
        e = svc.entry(pick) or {}
        top_l, top_r = st.columns([3, 2])
        with top_l:
            st.markdown(f"### {pick} · EUR {f['amount_eur']:,.2f}")
            st.markdown(pills(f["reasons"]), unsafe_allow_html=True)
            for r in f["reasons"]:
                st.markdown(f"- **{r['title']}.** {r['detail']}")
            if not f["reasons"]:
                st.markdown("- No single test fired; the risk model ranks it high from its combination of features.")
            if show_key:
                k = labels.get(pick)
                st.caption(f"Answer key: {k['kind'] if k else 'normal entry (a false alarm)'}")
        with top_r:
            if e:
                u, s = e.get("user") or {}, e.get("supplier") or {}
                st.markdown(
                    f"**{e.get('description', '')}**  \n"
                    f"{e['company_code']} · {e['doc_type']} · {e['source']} · posted {str(e['entry_ts'])[:16]}  \n"
                    f"By **{u.get('user_id')}** ({u.get('role')}, {u.get('company_code')}, limit EUR "
                    f"{u.get('approval_limit_eur') or 0:,.0f})  \n"
                    + (f"Supplier **{s.get('name')}** ({s.get('supplier_id')}), created {s.get('created_date')}, "
                       f"bank in {s.get('bank_country')}  \n" if s else "")
                    + f"Reference {e.get('reference') or '—'} · {e['amount_local']:,.2f} {e['currency']}")
                st.dataframe(pd.DataFrame(e.get("lines", [])), hide_index=True, width="stretch")
        st.divider()
        note_col, decide_col = st.columns([3, 2])
        with note_col:
            st.markdown("#### 🤖 Agent investigation")
            n = f["note"]
            blocked = can_run()
            label = "▶ Investigate with " + ("the scripted playbook" if mode.startswith("Demo") else "Claude")
            if n:
                label = label.replace("Investigate", "Investigate again")
            if blocked:
                st.info(blocked)
            if st.button(label, type="primary" if not n else "secondary", disabled=bool(blocked)):
                with st.spinner("The agent is reading the ledger and the casebook (read-only tools)..."):
                    n = investigate(pick) or n
            if n:
                st.markdown(f"**{ASSESS.get(n['assessment'], n['assessment'])}** · {n.get('model')} · "
                            f"{n.get('steps', 0)} steps · {n.get('seconds', 0)} s · cost {eur((n.get('cost') or {}).get('eur'))}")
                (st.error if n["assessment"] == "likely_issue" else st.success if n["assessment"] == "likely_fine"
                 else st.warning)(n["summary"])
                for ev in n.get("evidence") or []:
                    st.markdown(f"- {ev}")
                st.markdown(f"**Next step:** {n.get('next_step', '')}")
                if n.get("question_for_preparer"):
                    st.text_area("Draft question to the preparer", n["question_for_preparer"], height=80, key=f"q-{pick}")
                for c in n.get("checks") or []:
                    msg = {"note_replaced_by_code": "The model's note used numbers that are not in the evidence, so code "
                           "wrote this note instead.", "hard_rule_not_waived": "The model said likely fine, but a control "
                           "test fired: a person must decide."}.get(c, c)
                    st.caption(f"🛡️ Check: {msg}")
                with st.expander(f"Tool calls ({len(n.get('tool_calls') or [])})"):
                    st.dataframe(pd.DataFrame([{"tool": t["tool"], "args": json.dumps(t["args"])[:120], "ok": t["ok"],
                                                "ms": t["ms"]} for t in n.get("tool_calls") or []]),
                                 hide_index=True, width="stretch")
            else:
                st.caption("Not investigated yet.")
        with decide_col:
            st.markdown("#### ✅ Your decision")
            if f["status"] != "open":
                st.success(f"{STATUS[f['status']]} by {f['decided_by']}: {f['decision_reason']}")
            with st.form(f"decide-{pick}"):
                choice = st.radio("Decision", ["issue", "fine", "ask"], horizontal=True,
                                  format_func={"issue": "🔴 Issue", "fine": "🟢 Fine", "ask": "✉️ Ask preparer"}.get)
                reason = st.text_input("Reason", placeholder="e.g. duplicate of DE01-0093877; credit note requested")
                learn = st.checkbox("Use this decision to train the next model version", value=True)
                if st.form_submit_button("Save decision"):
                    r = svc.decide(close, pick, choice, me, reason, learn)
                    (st.success("Saved. It is in the audit log.") if r["ok"] else st.error(r["error"]))
                    if r["ok"]:
                        st.rerun()
            soft = [r for r in f["reasons"] if not r["hard"]]
            st.markdown("#### 🧠 Remember as a known pattern")
            if not soft:
                st.caption("Only soft tests can be remembered. "
                           + ("This entry hit a control test, which memory can never silence." if f["reasons"] else
                              "This entry was ranked by the model score alone; mark it fine to teach the model instead."))
            else:
                with st.form(f"pattern-{pick}"):
                    rule = st.selectbox("Test to silence", [r["code"] for r in soft], format_func=lambda c: RULES[c].title)
                    scope_opts = (["supplier"] if f.get("supplier_id") else []) + ["user"]
                    scope = st.radio("Only for", scope_opts, horizontal=True, format_func=lambda x: {
                        "supplier": f"supplier {f.get('supplier_id')} in {f['company_code']}",
                        "user": f"user {f.get('user_id')} in {f['company_code']}"}[x])
                    desc = st.text_input("Why this is normal", placeholder="e.g. fixed monthly freight tariff, contract FR-2024-17")
                    if st.form_submit_button("Propose pattern (needs a second person)"):
                        r = svc.propose_pattern(me, f, rule, desc, scope)
                        (st.success(f"Proposed {r['pattern_id']}, expires {r['expires']}. A controller must approve it in "
                                    "the 🧠 Memory tab.") if r["ok"] else st.error(r["error"]))

# ---------- memory ----------
with tab_mem:
    st.markdown("Memory is only what people choose to save, and it has guard rails: **four eyes** (who proposes a "
                "pattern cannot approve it), **narrow scope** (one company, one test, one supplier or user), **control "
                f"tests are never silenced**, and every pattern **expires** after {C.PATTERN_DAYS} days.")
    pats = svc.memory.patterns()
    st.markdown("#### Known patterns")
    if not pats:
        st.caption("None yet. Propose one from the 🔍 Review tab (a soft test, e.g. a fixed freight tariff that repeats "
                   "the same amount).")
    for p in pats:
        icon = {"proposed": "🟠", "active": "🟢", "rejected": "⛔", "revoked": "⚪", "expired": "⚪"}[p["status"]]
        with st.container(border=True):
            st.markdown(f"{icon} **{p['pattern_id']}** · {p['status']} · {RULES[p['rule']].title} · {p['company_code']} · "
                        f"{'supplier ' + p['supplier_id'] if p['supplier_id'] else 'user ' + str(p['user_id'])}  \n"
                        f"“{p['description']}” · proposed by {p['proposed_by']} from {p['example_doc']}"
                        + (f" · approved by {p['approved_by']}" if p["approved_by"] else "") + f" · expires {p['expires']}")
            b1, b2, _ = st.columns([1, 1, 4])
            if p["status"] == "proposed":
                if b1.button("Approve", key=f"ap-{p['pattern_id']}", disabled=me not in APPROVERS):
                    r = svc.approve_pattern(p["pattern_id"], me)
                    if r["ok"]:
                        svc.score_close(close)
                        st.rerun()
                    st.error(r["error"])
                if b2.button("Reject", key=f"rj-{p['pattern_id']}", disabled=me not in APPROVERS):
                    svc.close_pattern(p["pattern_id"], "rejected", me)
                    st.rerun()
            elif p["status"] == "active" and b1.button("Revoke", key=f"rv-{p['pattern_id']}"):
                svc.close_pattern(p["pattern_id"], "revoked", me)
                svc.score_close(close)
                st.rerun()
    if me not in APPROVERS:
        st.caption(f"{me} is a reviewer: approving needs a senior accountant or the group controller.")
    st.markdown("#### Decisions → model versions")
    dec = svc.memory.decisions()
    learn = [d for d in dec if d["use_for_training"]]
    st.caption(f"{len(dec)} decisions saved, {len(learn)} marked for training. Retraining fits a new version on the "
               "9 training months plus these decisions, measures it on the test months (without the entries you "
               "decided, so nothing leaks), and promotes it only if it passes the gates and is at least as good as "
               "the current version on the same entries.")
    if st.button("🔁 Retrain with the reviewers' decisions (~15 s)", disabled=not learn):
        with st.spinner("Training and evaluating..."):
            out = svc.retrain(me)
        if not out["ok"]:
            st.error(out["error"])
        else:
            m, old = out["metrics"], out["metrics"]["previous_same_entries"]
            (st.success if out["promoted"] else st.warning)(
                f"{out['version']}: {'promoted' if out['promoted'] else 'not promoted'} ({out['reason']}). "
                f"Recall {m['recall']:.0%} (was {old['recall']:.0%}), precision {m['precision']:.0%} "
                f"(was {old['precision']:.0%}).")
            if out["promoted"]:
                svc.score_close(close)
    vs = svc.memory.versions()
    st.dataframe(pd.DataFrame([{"Version": v["version"], "Promoted": "✅" if v["promoted"] else "—",
                                "Reviewer labels": v["reviewer_labels"], "Recall": v["metrics"].get("recall"),
                                "Precision": v["metrics"].get("precision"), "Flags / 1,000": v["metrics"].get("flags_per_1000"),
                                "Note": v["note"], "Created": v["created"]} for v in vs]), hide_index=True, width="stretch")
    if dec:
        with st.expander(f"All decisions ({len(dec)})"):
            st.dataframe(pd.DataFrame(dec), hide_index=True, width="stretch")

# ---------- model & evals ----------
with tab_model:
    m = svc.metrics()
    st.markdown(f"Model **{svc.version}**, trained on **{m.get('train_months')}**, measured on **{m.get('test_months')}** "
                f"({m.get('entries', 0):,} entries, {m.get('problems')} planted problems the models never saw).")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Recall", f"{m.get('recall', 0):.0%}", help="Planted problems with at least one entry flagged")
    c2.metric("High-severity recall", f"{m.get('high_severity_recall', 0):.0%}")
    c3.metric("Precision", f"{m.get('precision', 0):.0%}", help="Flags that are real problems")
    c4.metric("Flags per 1,000 entries", m.get("flags_per_1000"), help="The review workload")
    gates = m.get("gates", {})
    st.caption("CI gates: " + " · ".join(f"{'✅' if ok else '❌'} {k}" for k, ok in gates.items())
               + "  (recall ≥ 90%, high-severity recall ≥ 95%, at most 8 flags per 1,000)")
    left, right = st.columns([3, 2])
    with left:
        st.markdown("**Recall at each review budget** (same number of flags for each method)")
        curve = pd.DataFrame(m.get("recall_by_budget", []))
        if not curve.empty:
            st.line_chart(curve.set_index("flags_per_1000").rename(columns={
                "risk_model": "Risk model (all signals)", "rules_count": "Audit tests only", "anomaly_only": "Anomaly model only"}),
                height=300, x_label="flags per 1,000 entries", y_label="recall")
    with right:
        st.markdown("**Found per problem kind**")
        st.dataframe(pd.DataFrame([{"Kind": k.replace("_", " "), "Found": v} for k, v in m.get("per_kind", {}).items()]),
                     hide_index=True, width="stretch")
        base = m.get("baselines", {})
        st.markdown("**At the same budget**")
        st.dataframe(pd.DataFrame([{"Method": k.replace("_", " "), **v} for k, v in base.items()]), hide_index=True,
                     width="stretch")
    with st.expander("📄 Model card", expanded=False):
        st.markdown(f"""
**Purpose.** Rank a month's journal entries so accountants review the few that matter, with reasons. Advisory only:
nothing is posted, reversed or blocked automatically.

**Models.** (1) Text model: TF-IDF character n-grams + logistic regression, description → account, works across
English, German, Portuguese, Romanian and Spanish. (2) Isolation Forest on amount, time, user, supplier and
account-pair features. (3) Monthly forecast per company and expense account (gradient boosting on lags, walk-forward).
(4) Risk model: gradient boosting classifier on the twelve test results and the three scores, trained on past review
outcomes ({m.get('n_positive')} confirmed problems in {m.get('n_train', 0):,} entries) plus reviewer decisions
({m.get('n_reviewer_labels', 0)} so far).

**Data.** Simulated, SAP-like journal entries (header + debit/credit lines) for 20 company codes, 12 months, about
100,000 entries and 300 planted problems of 11 kinds. Training: the first 9 months. Test: the last 3.

**Limits.** Planted problems are easier than real fraud, so real recall will be lower: the review budget, the
reviewer feedback loop and the control tests are there for that. The model knows only what is in the ledger, not
contracts or emails. Thresholds are set for a demo and must be tuned with the controllers who use it.

**Monitoring.** Flags per 1,000 by company and test, the share of flags decided "fine" (false alarms), and recall on a
monthly sample of seeded test entries. A drop in any of them opens a ticket; retraining is gated as above.
""")

# ---------- how it works ----------
with tab_how:
    st.graphviz_chart("""
digraph G {
  rankdir=LR; bgcolor="transparent"; node [shape=box, style="rounded,filled", fillcolor="#123038", color="#23474F",
  fontcolor="#F2F4EE", fontname="Inter"]; edge [color="#8FB3B5"];
  sap [label="SAP S/4HANA\\njournal entries"]; wh [label="Snowflake / DuckDB\\nledger (read-only)"];
  det [label="Detector\\n12 audit tests\\n+ text, anomaly,\\nforecast, risk models"];
  q [label="Review queue\\n6 per 1,000"]; ag [label="Agent (Claude)\\nLangGraph"];
  mcp [label="MCP servers\\nledger + casebook\\nread-only tools"]; ppl [label="Accountants\\nissue / fine / ask", fillcolor="#3A3415", color="#F5C542"];
  mem [label="Memory\\nlabels + known patterns\\n(four eyes, expiry)"]; tr [label="Retrain\\n+ eval gates"];
  sap -> wh -> det -> q -> ag; ag -> mcp [dir=both]; mcp -> wh; q -> ppl; ag -> ppl [label=" note"];
  ppl -> mem -> tr -> det; mem -> det [label=" silence soft tests"]; mem -> mcp [style=dashed];
}""", width="stretch")
    st.markdown("""
- **Read-only by design.** The agent's tools open the ledger read-only; on Snowflake it runs with a SELECT-only role.
  LedgerWatch writes only to its own database (flags, notes, decisions, patterns) and to a hash-chained audit log.
- **Classic ML does the finding, the LLM does the reading.** Scoring 8,000 entries costs nothing per run; Claude is
  used only on the ~50 flagged entries, where its time saves an accountant's.
- **Guard rails on the agent.** Allow-listed read-only tools, a step budget, descriptions treated as untrusted text,
  every number in its note checked against the tool results, and it can never call a control-test hit "fine".
- **Memory with guard rails.** Only people write memory. Patterns need a second person, have a narrow scope and an
  expiry, and cannot touch control tests. Retraining is gated by the same evals as CI.
- **Path to production.** Data from SAP via Snowflake (ACDOCA/BKPF tables), the app on AWS (ECS Fargate, single
  sign-on, Claude on Amazon Bedrock in the EU), models retrained after each close with MLflow tracking.
""")

# ---------- audit ----------
with tab_audit:
    v = svc.audit.verify()
    (st.success if v["ok"] else st.error)(
        f"Audit chain {'intact' if v['ok'] else 'BROKEN'}: {v['rows']} events, each holding the hash of the one before.")
    ev = svc.audit.read(limit=300)  # newest first
    st.dataframe(pd.DataFrame([{"time": e["ts"][:19], "actor": e["actor"], "event": e["kind"], "run": e["run_id"],
                                "details": json.dumps(e["payload"], default=str)[:200]} for e in ev]),
                 hide_index=True, width="stretch", height=520)
