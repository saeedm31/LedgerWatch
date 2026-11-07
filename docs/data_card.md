# Data card: the simulated ledger

| | |
|---|---|
| **Source** | `ledgerwatch/simulate.py`, fixed seed, built on first start |
| **Group** | "Aurelia Energy", a fictional energy-technology group: 20 company codes in several countries and currencies |
| **Period** | 12 months, Nov 2025 to Oct 2026 |
| **Size** | About 8,300 journal entries a month, about 100,000 in total |
| **Personal data** | None. All people, suppliers and amounts are invented |

## Tables (SAP-style)

| Table | Like in SAP | One row per |
|---|---|---|
| `journal_entries` | BKPF (header) | document: company, type, source (manual or interface), user, supplier, reference, amount, posting time |
| `journal_lines` | ACDOCA / BSEG | debit or credit line: account, amount. Every debit equals its credit |
| `company_codes` | company code table (T001) | company: country, currency, fixed EUR rate |
| `users` | user master | user: company, role, approval limit |
| `suppliers` | vendor master | supplier: company, creation date, bank country |
| `accounts` | chart of accounts | account: type and name |

Descriptions are written in the company's language (English, German, Portuguese, Romanian, Spanish), as real
ledgers are.

## Planted problems

About 300 problems of 11 kinds are planted on purpose, about 24 a month, so recall can be measured:

| Kind | Severity |
|---|---|
| duplicate invoice (also with a retyped reference) | high |
| split payment to stay under an approval limit | high |
| large payment to a brand-new supplier | high |
| manual revenue at the end of a quarter | high |
| red-team entry whose description tries to instruct an AI reviewer (one, in the current close) | high |
| weekend or late-night manual posting | medium |
| description that does not fit the account | medium |
| unusual monthly total for a user | medium |
| rare account pair | medium |
| large round amount | medium |
| posting by a user of another company | medium |

The answer key is `labels.json`. Only the evaluation reads it. No model feature, MCP tool or agent prompt can reach it.

## Split

| Use | Months |
|---|---|
| Training | Nov 2025 to Jul 2026 (9 months) |
| Test | Aug to Oct 2026 (3 months, never seen in training) |
| Current close in the app | Oct 2026 |

## Limits
Simulated data shows the method, not real performance. Real problems are rarer, more varied and less clearly
labelled. Exchange rates are fixed and fictional.
