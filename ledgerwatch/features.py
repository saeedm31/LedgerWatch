"""One row per journal entry with the facts a reviewer would look at, plus the journal-entry tests.

The tests follow the usual journal-entry testing list (ISA 240 / ISA 330 practice): duplicates, split
postings under an approval limit, postings by users of another company, new suppliers with large amounts,
weekend or late manual postings, rare account pairs, round amounts, period-end revenue, descriptions that do
not fit the account, and unusual totals. Each test gives a reason code and a plain sentence.

"Hard" tests are control breaches. A known pattern in memory can never silence them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .warehouse import Warehouse

ENTRY_SQL = """
SELECT e.*, d.account AS debit_account, c.account AS credit_account,
       u.company_code AS user_company, u.role AS user_role, u.approval_limit_eur AS user_limit, u.is_batch,
       s.created_date AS supplier_created, s.bank_country AS supplier_bank_country, s.country AS supplier_country,
       s.name AS supplier_name
FROM journal_entries e
JOIN journal_lines d ON d.doc_id = e.doc_id AND d.dc = 'D'
JOIN journal_lines c ON c.doc_id = e.doc_id AND c.dc = 'C'
LEFT JOIN users u ON u.user_id = e.user_id
LEFT JOIN suppliers s ON s.supplier_id = e.supplier_id
"""


@dataclass(frozen=True)
class Rule:
    code: str
    title: str
    hard: bool


RULES = {r.code: r for r in [
    Rule("DUPLICATE", "Possible duplicate invoice", True),
    Rule("SPLIT", "Split under approval limit", True),
    Rule("OTHER_COMPANY_USER", "Posted by a user of another company", True),
    Rule("OVER_LIMIT", "Above the user's approval limit", True),
    Rule("NEW_SUPPLIER_LARGE", "Large amount to a new supplier", True),
    Rule("PERIOD_END_REVENUE", "Manual revenue at period end", True),
    Rule("REPEAT_AMOUNT", "Same supplier and amount again, new reference", False),
    Rule("WEEKEND_LATE", "Manual posting on a weekend or late at night", False),
    Rule("RARE_PAIR", "Rare account combination", False),
    Rule("ROUND_AMOUNT", "Large round amount", False),
    Rule("TEXT_MISMATCH", "Description does not fit the account", False),
    Rule("UNUSUAL_TOTAL", "Unusual monthly total for this user and account", False),
]}

NUMERIC = ["log_amount", "amount_z", "hour", "weekend", "manual", "limit_ratio", "supplier_age_log", "foreign_bank",
           "pair_log_count", "days_to_period_end", "user_acc_ratio_log", "is_round", "is_revenue", "is_expense", "repeat_gap_log"]


def load_entries(wh: Warehouse, periods: list[str] | None = None) -> pd.DataFrame:
    sql = ENTRY_SQL
    params: list = []
    if periods:
        sql += f" WHERE e.period IN ({','.join('?' * len(periods))})"
        params = list(periods)
    df = wh.df(sql, params)
    df["posting_date"] = pd.to_datetime(df["posting_date"])
    df["entry_ts"] = pd.to_datetime(df["entry_ts"])
    df["supplier_created"] = pd.to_datetime(df["supplier_created"])
    return df.sort_values("entry_ts").reset_index(drop=True)


def add_features(df: pd.DataFrame, baseline: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """Add numeric features. `baseline` holds history statistics from the training months; when None it is
    computed from `df` itself (training)."""
    df = df.copy()
    amt = df["amount_eur"].astype(float)
    df["log_amount"] = np.log1p(amt)
    if baseline is None:
        g = df.groupby("debit_account")["log_amount"]
        user_acc = df[df.is_batch == 0].groupby(["user_id", "debit_account", "period"])["amount_eur"].sum()
        baseline = dict(
            acc_mean=g.mean().to_dict(), acc_std=g.std().fillna(1).clip(lower=0.2).to_dict(),
            pair_count=df.groupby(["debit_account", "credit_account"]).size().to_dict(),
            user_acc_typical=user_acc.groupby("debit_account").median().to_dict(),
            revenue_p99=df[df.credit_account.str.startswith("40")].groupby("credit_account")["amount_eur"]
            .quantile(0.99).to_dict(),
            n_months=int(df["period"].nunique()),
        )
    mu = df["debit_account"].map(baseline["acc_mean"]).fillna(df["log_amount"].mean())
    sd = df["debit_account"].map(baseline["acc_std"]).fillna(1.0)
    df["amount_z"] = (df["log_amount"] - mu) / sd
    df["hour"] = df["entry_ts"].dt.hour + df["entry_ts"].dt.minute / 60
    df["weekend"] = (df["posting_date"].dt.weekday >= 5).astype(int)
    df["manual"] = (df["source"] == "manual").astype(int)
    lim = df["user_limit"].where(df["is_batch"] == 0)
    df["limit_ratio"] = (amt / lim).where(lim > 0).fillna(0.0)
    age = (df["posting_date"] - df["supplier_created"]).dt.days
    df["supplier_age_days"] = age
    df["supplier_age_log"] = np.log1p(age.clip(lower=0)).fillna(10.0)
    df["foreign_bank"] = ((df["supplier_bank_country"].notna()) &
                          (df["supplier_bank_country"] != df["supplier_country"])).astype(int)
    pairs = list(zip(df["debit_account"], df["credit_account"]))
    df["pair_count"] = [baseline["pair_count"].get(p, 0) for p in pairs]
    df["pair_log_count"] = np.log1p(df["pair_count"])
    end = df["posting_date"] + pd.offsets.MonthEnd(0)
    df["days_to_period_end"] = (end - df["posting_date"]).dt.days
    tot = df[df.is_batch == 0].groupby(["user_id", "debit_account", "period"])["amount_eur"].transform("sum")
    df["user_acc_total"] = tot.reindex(df.index).fillna(0.0)
    typical = df["debit_account"].map(baseline["user_acc_typical"]).fillna(df["user_acc_total"])
    df["user_acc_ratio"] = (df["user_acc_total"] / typical.clip(lower=1.0)).fillna(0.0)
    df["user_acc_ratio_log"] = np.log1p(df["user_acc_ratio"])
    gap = _repeat_amount(df)
    df["repeat_gap_days"] = gap
    df["repeat_gap_log"] = np.log1p(gap.fillna(999))
    df["is_round"] = ((amt >= 10_000) & (np.isclose(amt % 1000, 0))).astype(int)
    df["is_revenue"] = df["credit_account"].str.startswith("40").astype(int)
    df["is_expense"] = df["debit_account"].str.startswith("6").astype(int)
    return df, baseline


def norm_ref(ref: pd.Series) -> pd.Series:
    """'INV-123456', 'inv 123456' and 'INV123456/1' are the same reference."""
    r = ref.fillna("").str.upper().str.replace(r"/\d{1,2}$", "", regex=True)
    return r.str.replace(r"[^A-Z0-9]", "", regex=True).str.replace(r"^INV", "", regex=True)


def _repeat_amount(df: pd.DataFrame) -> pd.Series:
    """Days since another invoice from the same supplier with exactly the same amount (any reference)."""
    inv = df[(df.doc_type == "invoice") & df.supplier_id.notna()].sort_values("posting_date")
    key = inv["supplier_id"].astype(str) + "|" + inv["amount_eur"].round(2).astype(str)
    gap = inv.groupby(key)["posting_date"].diff().dt.days
    return gap.reindex(df.index)


def _duplicates(df: pd.DataFrame) -> pd.Series:
    """Same supplier, amount and reference as an earlier invoice in the last 60 days."""
    inv = df[(df.doc_type == "invoice") & df.supplier_id.notna()]
    ref = norm_ref(inv["reference"])
    key = inv["supplier_id"].astype(str) + "|" + inv["amount_eur"].round(2).astype(str) + "|" + ref
    first = inv.groupby(key)["posting_date"].transform("min")
    rank = inv.groupby(key).cumcount()
    hit = (rank > 0) & ((inv["posting_date"] - first).dt.days <= 60)
    return hit.reindex(df.index, fill_value=False)


def _splits(df: pd.DataFrame) -> pd.Series:
    """Two or more invoices by one user to one supplier within 3 days, each at 85-100% of the user's limit."""
    near = df[(df.manual == 1) & df.supplier_id.notna() & df.limit_ratio.between(0.85, 1.0)]
    hit = pd.Series(False, index=df.index)
    for _, g in near.groupby(["user_id", "supplier_id"]):
        if len(g) < 2:
            continue
        d = g["posting_date"].sort_values()
        close = (d.diff().dt.days.fillna(99) <= 3) | (d.diff(-1).dt.days.abs().fillna(99) <= 3)
        hit.loc[close[close].index] = True
    return hit


def apply_rules(df: pd.DataFrame, baseline: dict) -> pd.DataFrame:
    """Add one boolean column per rule (`r_<CODE>`)."""
    df = df.copy()
    manual_user = (df.is_batch == 0) & (df.manual == 1)
    df["r_DUPLICATE"] = _duplicates(df)
    df["r_SPLIT"] = _splits(df)
    df["r_REPEAT_AMOUNT"] = ~df["r_DUPLICATE"] & (df.repeat_gap_days <= 21) & (df.amount_eur >= 1000)
    df["r_OTHER_COMPANY_USER"] = manual_user & (df.user_company != df.company_code)
    df["r_OVER_LIMIT"] = manual_user & (df.limit_ratio > 1.0)
    df["r_NEW_SUPPLIER_LARGE"] = df.supplier_age_days.between(0, 45) & (df.amount_eur >= 50_000)
    p99 = df["credit_account"].map(baseline["revenue_p99"])
    df["r_PERIOD_END_REVENUE"] = manual_user & (df.is_revenue == 1) & (df.days_to_period_end <= 1) & \
        ((df.hour >= 19) | (df.amount_eur > p99) | (df.weekend == 1))
    df["r_WEEKEND_LATE"] = manual_user & ((df.weekend == 1) | (df.hour >= 22) | (df.hour < 6))
    df["r_RARE_PAIR"] = manual_user & (df.pair_count < 10)
    df["r_ROUND_AMOUNT"] = manual_user & (df.is_round == 1)
    df["r_UNUSUAL_TOTAL"] = manual_user & (df.user_acc_ratio >= 6) & (df.amount_z > 1.0)
    if "text_fit" in df:
        df["r_TEXT_MISMATCH"] = (df.text_fit < 0.05) & (df.text_best >= 0.5)
    else:
        df["r_TEXT_MISMATCH"] = False
    return df


def reasons(row: pd.Series) -> list[dict]:
    """Plain sentences for the rules an entry hit. Built from data only (no model text)."""
    out = []
    amt = f"EUR {row.amount_eur:,.2f}"
    for code, rule in RULES.items():
        if not bool(row.get(f"r_{code}", False)):
            continue
        detail = {
            "DUPLICATE": lambda: f"Same supplier, amount ({amt}) and reference {row.reference} as an earlier invoice.",
            "REPEAT_AMOUNT": lambda: f"The same supplier billed exactly {amt} {int(row.repeat_gap_days)} days earlier "
                             f"under another reference.",
            "SPLIT": lambda: f"One of several invoices by {row.user_id} to this supplier within 3 days, each just under the "
                     f"user's EUR {row.user_limit:,.0f} limit.",
            "OTHER_COMPANY_USER": lambda: f"{row.user_id} belongs to company {row.user_company}, not {row.company_code}.",
            "OVER_LIMIT": lambda: f"{amt} is above {row.user_id}'s approval limit of EUR {row.user_limit:,.0f}.",
            "NEW_SUPPLIER_LARGE": lambda: f"{amt} to a supplier created {int(row.supplier_age_days)} days before"
                                  + (f", with a bank account in {row.supplier_bank_country}." if row.foreign_bank else "."),
            "PERIOD_END_REVENUE": lambda: f"Manual revenue of {amt} posted {row.entry_ts:%a %d %b %H:%M}, at the end of the period.",
            "WEEKEND_LATE": lambda: f"Manual posting on {row.entry_ts:%a %d %b at %H:%M}.",
            "RARE_PAIR": lambda: f"Debit {row.debit_account} / credit {row.credit_account} was used {int(row.pair_count)} times "
                         "in the 9 training months.",
            "ROUND_AMOUNT": lambda: f"Round amount {amt} entered by hand.",
            "TEXT_MISMATCH": lambda: f"The description reads like account {row.get('text_account', '?')}, not {row.debit_account} "
                             f"(text model: {100 * row.get('text_fit', 0):.0f}% fit for {row.debit_account}).",
            "UNUSUAL_TOTAL": lambda: f"{row.user_id}'s total on {row.debit_account} this month is "
                             f"{row.user_acc_ratio:.0f}x a typical month.",
        }[code]()
        out.append({"code": code, "title": rule.title, "hard": rule.hard, "detail": detail})
    return out
