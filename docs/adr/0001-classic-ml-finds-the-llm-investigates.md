# 1. Classic ML finds the entries, the LLM only investigates the flags

**Status:** accepted

## Context
A month has about 8,000 journal entries. Sending each one to an LLM would be slow, costly and hard to measure.

## Decision
Audit tests and scikit-learn models score every entry. Only the flagged entries, the riskiest few per 1,000, go to
the Claude agent.

## Consequences
- Scoring costs nothing per run and is measured with plain recall and precision at a fixed review budget.
- The LLM spends its time where it saves an accountant's time: gathering evidence on a short list.
- An entry the models do not flag is never investigated. Seeded test entries and the review budget are the check.
