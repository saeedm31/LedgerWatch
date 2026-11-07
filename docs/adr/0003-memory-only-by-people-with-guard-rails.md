# 3. Memory is written only by people, with guard rails

**Status:** accepted

## Context
Learning from reviewers is the point: the same false alarm should not come back every month. But memory that silences
alerts is also a way to hide a problem (memory poisoning).

## Decision
- Only people write memory. The agent can read it, never write it.
- Decisions become training labels; a retrained model is promoted only if it passes the evaluation gates and is no
  worse than the current one.
- A known pattern silences one soft test, for one supplier or user, in one company. The proposer cannot approve it,
  it expires after 180 days, and control tests can never be silenced.

## Consequences
- Every change to memory has two names on it and is in the audit log.
- Some repeat false alarms stay until a second person approves the pattern. That cost is accepted.
