# 4. One small SQL interface for DuckDB, SQLite and Snowflake

**Status:** accepted

## Context
The demo must run anywhere with no accounts, but the company setup uses Snowflake.

## Decision
`warehouse.py` hides the engine behind a few methods and plain SQL that all three engines accept. `LW_ENGINE`
chooses the engine; Snowflake uses key-pair login from a secret.

## Consequences
- Tests and the public demo run on DuckDB; the same code runs on Snowflake by changing settings.
- Engine-specific SQL features are avoided, which keeps queries simple.
