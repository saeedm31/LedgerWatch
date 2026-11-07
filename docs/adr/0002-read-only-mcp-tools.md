# 2. The agent reads through read-only MCP servers

**Status:** accepted

## Context
The agent needs ledger facts (similar invoices, a user's activity, a supplier's history) and past decisions. Giving
it SQL or write access would make prompt injection and mistakes dangerous.

## Decision
Two MCP servers, `ledger` and `casebook`, expose fixed read-only tools. The ledger connection is opened read-only; on
Snowflake the role has SELECT only. The agent may only call allow-listed tools and has no tool that writes.

## Consequences
- The worst a fooled agent can do is write a wrong note, which a person reads and the grounding check tests.
- The same servers can run as subprocesses locally or as separate services in production.
- New questions need a new tool, not a free-form query.
