# Model card: journal-entry risk

| | |
|---|---|
| **Name** | LedgerWatch risk bundle, versions `v1`, `v2`, ... (saved in `data/models`, listed in the app's Memory tab) |
| **Task** | Rank a month's journal entries by how likely they are to need an accountant's review, with reasons |
| **Models** | Four scikit-learn models, trained together (below) |
| **Data** | Simulated SAP-style ledger, 12 months (see [data card](data_card.md)) |
| **Owner** | Saeed Hojjatpanah |

## The four models

| Model | Algorithm | Output used by the risk model |
|---|---|---|
| Text model | TF-IDF character n-grams (3-5) + logistic regression | probability that the description fits the booked account |
| Anomaly model | Isolation Forest (200 trees) on amount, time, user, supplier and account-pair features | anomaly percentile |
| Forecast model | Gradient boosting regressor on lags, per company and expense account, walk-forward | how far the month is above forecast |
| Risk model | Gradient boosting classifier (balanced classes) on the 12 audit-test results and the three scores above | risk score 0-1 |

The risk model learns from past review outcomes in the training months and from reviewer decisions in the app
(weighted 3x). The 12 audit tests give the reasons a person reads; the first six are control tests that are always
reviewed.

## Intended use
Choosing which entries an accountant reviews at month-end close, and explaining why. A person decides every flag.
Nothing is posted, reversed or blocked.

## Not intended for
Judging people's performance, automatic approval or rejection of entries, or any decision without a reviewer. Not
validated on real data.

## Evaluation
Trained on the first 9 months, measured on the last 3, which no model saw. Measured at the same review budget for
every method: recall per planted problem, recall on high-severity problems, precision per flag, flags per 1,000
entries and recall per problem kind. Baselines: audit tests only, anomaly model only, and all audit-test hits. Run
`python -m ledgerwatch eval`; the app's Model & evals tab shows the latest numbers.

**CI gates** (`evals.py`): recall at least 90%, high-severity recall at least 95%, at most 8 flags per 1,000 entries.
A retrained version must also be no worse than the current one on the same entries.

## Known limits
- Planted problems are easier than real fraud, so real recall will be lower.
- The models only know what is in the ledger, not contracts, emails or approvals outside SAP.
- Reviewer labels are a feedback loop: entries that are never flagged are never labelled. Seeded test entries are the
  check for that.
- Thresholds and the review budget are demo values; they must be set with the controllers who use the tool.

## Monitoring (production plan)
Per close: flags per 1,000 by company and test, the share of flags marked fine (false alarms), and recall on a small
monthly sample of seeded test entries. A drop opens a ticket; retraining is gated as above.
