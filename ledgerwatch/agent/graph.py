"""The investigation of one flagged entry, as a LangGraph state machine.

    gather (code) -> agent (LLM) <-> tools (read-only MCP) -> check (code)

- gather: code reads the entry through the ledger MCP server and adds the detector's reasons and scores.
- agent:  the model chooses read-only tools (ledger and casebook) until it submits an assessment, or the step
          budget runs out and it must submit.
- check:  code checks the note. Every number must appear in the facts the tools returned (grounding); a note
          that fails is replaced by a plain code-written one. On a control test (a hard rule) the model cannot
          say "likely fine": that becomes "unclear, a person decides".

The agent advises only. It has no tool that writes anything; the reviewer decides in the app.
Nodes are plain methods, so the same steps run without LangGraph (a small runner follows the same edges).
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any, TypedDict

from ..audit import AuditLog
from ..config import MAX_AGENT_STEPS
from ..llm import MeteredClient, total
from ..mcp_hub import MCPHub
from . import prompts
from .grounding import unsupported

try:
    from langgraph.graph import END, StateGraph

    HAS_LANGGRAPH = True
except ImportError:  # pragma: no cover - the plain runner below takes over
    END, HAS_LANGGRAPH = "__end__", False

UNTRUSTED = ("description", "reference")


class State(TypedDict, total=False):
    run_id: str
    doc_id: str
    flag: dict
    entry: dict | None
    messages: list
    pending: bool
    steps: int
    tool_calls: list
    facts: list
    submitted: dict | None
    note: dict
    status: str


def _block(b: Any) -> dict:
    if isinstance(b, dict):
        return b
    if b.type == "text":
        return {"type": "text", "text": b.text}
    if b.type == "tool_use":
        return {"type": "tool_use", "id": b.id, "name": b.name, "input": b.input}
    return {"type": b.type}


class Investigator:
    def __init__(self, llm: MeteredClient, hub: MCPHub, audit: AuditLog, model: str, actor: str = "lw-agent"):
        self.llm, self.hub, self.audit, self.model, self.actor = llm, hub, audit, model, actor
        self.graph = self._compile() if HAS_LANGGRAPH else None

    def _log(self, state: State, kind: str, payload: dict) -> None:
        self.audit.write(kind, payload, run_id=state.get("run_id"), actor=self.actor)

    # ---------- nodes ----------
    def gather(self, state: State) -> State:
        r = self.hub.call("get_entry", {"doc_id": state["doc_id"]})
        self._log(state, "tool_call", dict(by="code", tool="get_entry", ok=r.ok, ms=r.ms, error=r.error))
        if not r.ok:
            return {"entry": None, "status": "failed"}
        return {"entry": r.data, "facts": [r.data]}

    def agent(self, state: State) -> State:
        e = state["entry"]
        trusted = {k: v for k, v in e.items() if k not in UNTRUSTED}
        ctx = {"entry": trusted, "reasons": state["flag"]["reasons"], "risk": round(state["flag"]["risk"], 3),
               "rank_in_close": state["flag"]["rank"]}
        text = "\n".join(f'<untrusted field="{k}">{json.dumps(e.get(k))}</untrusted>' for k in UNTRUSTED)
        messages = state.get("messages") or [
            {"role": "user", "content": "Flagged entry:\n" + json.dumps(ctx, default=str) + "\n" + text}]
        steps = state.get("steps", 0)
        force = steps >= MAX_AGENT_STEPS
        tools = [prompts.SUBMIT] if force else self.hub.agent_tool_specs() + [prompts.SUBMIT]
        kw = {"tool_choice": {"type": "tool", "name": "submit_assessment"}} if force else {}
        mark = len(self.llm.meter.calls)
        t0 = time.time()
        resp = self.llm.create(step="investigate", model=self.model, max_tokens=1200, system=prompts.SYSTEM,
                               messages=messages, tools=tools, **kw)
        content = [_block(b) for b in resp.content]
        uses = [b for b in content if b["type"] == "tool_use"]
        self._log(state, "llm_call", dict(model=self.model, step=steps, tools=[b["name"] for b in uses],
                                          usage=[u.to_dict() for u in self.llm.meter.since(mark)], ms=int(1000 * (time.time() - t0))))
        messages = messages + [{"role": "assistant", "content": content}]
        sub = next((b for b in uses if b["name"] == "submit_assessment"), None)
        if sub:
            return {"messages": messages, "submitted": sub["input"], "pending": False, "steps": steps + 1}
        if not uses:  # the model answered in text: ask it to submit
            messages.append({"role": "user", "content": "Call submit_assessment now."})
            return {"messages": messages, "pending": True, "steps": steps + 1}
        return {"messages": messages, "pending": True, "steps": steps + 1}

    def tools(self, state: State) -> State:
        last = state["messages"][-1]["content"]
        results, calls, facts = [], list(state.get("tool_calls") or []), list(state.get("facts") or [])
        for b in [b for b in last if b["type"] == "tool_use"]:
            if not self.hub.agent_may_call(b["name"]):
                ok, data, err, ms = False, None, f"tool {b['name']} is not allowed", 0
            else:
                r = self.hub.call(b["name"], b["input"])
                ok, data, err, ms = r.ok, r.data, r.error, r.ms
            calls.append(dict(tool=b["name"], args=b["input"], ok=ok, ms=ms, error=err))
            self._log(state, "tool_call", dict(by="model", tool=b["name"], args=b["input"], ok=ok, ms=ms, error=err,
                                               result=json.dumps(data, default=str)[:500] if data else None))
            if ok:
                facts.append(data)
            results.append({"type": "tool_result", "tool_use_id": b["id"], "is_error": not ok,
                            "content": json.dumps(data, default=str) if ok else str(err)})
        if not results:
            results = [{"type": "text", "text": "Call submit_assessment now."}]
        return {"messages": state["messages"] + [{"role": "user", "content": results}], "tool_calls": calls, "facts": facts}

    def check(self, state: State) -> State:
        if not state.get("entry"):
            return {"note": {"assessment": "unclear", "summary": "The entry could not be read from the ledger.",
                             "evidence": [], "next_step": "Retry later.", "checks": ["ledger_unavailable"]}, "status": "failed"}
        sub = dict(state.get("submitted") or {})
        facts = state.get("facts", []) + [state["flag"]]
        checks = []
        if sub.get("assessment") not in ("likely_issue", "likely_fine", "unclear"):
            sub = {}
            checks.append("no_valid_assessment")
        text = " ".join([sub.get("summary", ""), *sub.get("evidence", []), sub.get("question_for_preparer") or ""])
        bad = unsupported(text, facts) if sub else []
        if bad:
            checks.append(f"ungrounded_numbers:{bad[:5]}")
            self._log(state, "grounding_failed", dict(numbers=bad[:10]))
        hard = [r["code"] for r in state["flag"]["reasons"] if r["hard"]]
        if sub.get("assessment") == "likely_fine" and hard:
            sub["assessment"] = "unclear"
            sub["next_step"] = f"Control test {', '.join(hard)} hit: a person must check the evidence and decide."
            checks.append("hard_rule_not_waived")
        if not sub or bad:
            e = state["entry"]
            sub = {"assessment": sub.get("assessment", "unclear") if sub else "unclear",
                   "summary": f"{e['doc_id']}: EUR {e['amount_eur']:,.2f} posted by {e['user_id']}. "
                              + " ".join(r["detail"] for r in state["flag"]["reasons"][:3]),
                   "evidence": [], "next_step": sub.get("next_step") or "Review the entry and its supporting document.",
                   "question_for_preparer": ""}
            checks.append("note_replaced_by_code")
        note = {**sub, "checks": checks, "tool_calls": state.get("tool_calls", []), "model": self.model,
                "steps": state.get("steps", 0)}
        self._log(state, "assessment", {k: v for k, v in note.items() if k != "tool_calls"})
        return {"note": note, "status": "done"}

    # ---------- wiring ----------
    @staticmethod
    def _after_gather(state: State) -> str:
        return "agent" if state.get("entry") else "check"

    @staticmethod
    def _after_agent(state: State) -> str:
        return "tools" if state.get("pending") else "check"

    def _compile(self):
        g = StateGraph(State)
        for name in ("gather", "agent", "tools", "check"):
            g.add_node(name, getattr(self, name))
        g.set_entry_point("gather")
        g.add_conditional_edges("gather", self._after_gather, {"agent": "agent", "check": "check"})
        g.add_conditional_edges("agent", self._after_agent, {"tools": "tools", "check": "check"})
        g.add_edge("tools", "agent")
        g.add_edge("check", END)
        return g.compile()

    def _run_plain(self, state: State) -> State:
        state = {**state, **self.gather(state)}
        if self._after_gather(state) == "agent":
            while True:
                state = {**state, **self.agent(state)}
                if self._after_agent(state) == "check":
                    break
                state = {**state, **self.tools(state)}
        return {**state, **self.check(state)}

    def run(self, flag: dict) -> dict:
        state: State = {"run_id": "RUN-" + uuid.uuid4().hex[:8].upper(), "doc_id": flag["doc_id"], "flag": flag, "steps": 0}
        mark = len(self.llm.meter.calls)
        t0 = time.time()
        out = self.graph.invoke(state, {"recursion_limit": 4 * MAX_AGENT_STEPS + 10}) if self.graph else self._run_plain(state)
        note = out["note"]
        note.update(run_id=state["run_id"], seconds=round(time.time() - t0, 2), cost=total(self.llm.meter.since(mark)))
        return note
