"""The four trained models and how they work together.

1. Text model (document classification): reads an entry's description and says which account it fits.
   TF-IDF on character n-grams + logistic regression, so it works across English, German, Portuguese,
   Romanian and Spanish without a translation step.
2. Anomaly model: Isolation Forest on amount, time, user, supplier and account-pair features. Needs no labels.
3. Forecast model: predicts each company's monthly total per expense account from the months before
   (gradient boosting on lags, walk-forward). A month far above its forecast lifts the risk of its entries.
4. Risk model (feedback model): gradient boosting classifier that combines the rule hits and the three
   scores above. It learns from past review outcomes (training months only) and from reviewer decisions
   saved in memory, so it gets better with use.

All training uses months 1-9; months 10-12 are only for measuring.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor, IsolationForest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from .config import FLAGS_PER_1000, TRAIN_MONTHS
from .features import NUMERIC, RULES, add_features, apply_rules

TEXT_DOC_TYPES = ("invoice", "expense claim")
ANOMALY_COLS = ["log_amount", "amount_z", "hour", "weekend", "manual", "limit_ratio", "supplier_age_log",
                "foreign_bank", "pair_log_count", "days_to_period_end", "user_acc_ratio_log", "is_round"]
RULE_COLS = [f"r_{c}" for c in RULES]
RISK_COLS = NUMERIC + RULE_COLS + ["anomaly", "text_fit", "month_gap"]
SEED = 7


@dataclass
class Bundle:
    baseline: dict
    text: object
    anomaly: IsolationForest
    anomaly_ref: np.ndarray  # training scores, to turn a raw score into a percentile
    forecast_history: pd.DataFrame  # monthly totals used as lags
    risk: HistGradientBoostingClassifier
    version: str = "v1"
    info: dict = field(default_factory=dict)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: Path) -> Bundle:
        return joblib.load(path)


# ---------- 1. text ----------
def train_text(df: pd.DataFrame):
    d = df[df.doc_type.isin(TEXT_DOC_TYPES)]
    model = make_pipeline(TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, sublinear_tf=True, max_features=60_000),
                          LogisticRegression(max_iter=300, C=4.0))
    model.fit(d["description"].str.replace(r"[\dA-Z]{2,}-[\w-]+", " ", regex=True), d["debit_account"])
    return model


def text_scores(model, df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({"text_fit": 1.0, "text_best": 0.0, "text_account": None}, index=df.index)
    m = df.doc_type.isin(TEXT_DOC_TYPES)
    if not m.any():
        return out
    proba = model.predict_proba(df.loc[m, "description"].str.replace(r"[\dA-Z]{2,}-[\w-]+", " ", regex=True))
    classes = list(model.classes_)
    idx = np.array([classes.index(a) if a in classes else -1 for a in df.loc[m, "debit_account"]])
    fit = np.where(idx >= 0, proba[np.arange(len(idx)), np.clip(idx, 0, None)], 0.0)
    other = proba.copy()
    other[np.arange(len(idx))[idx >= 0], idx[idx >= 0]] = -1
    out.loc[m, "text_fit"] = fit
    out.loc[m, "text_best"] = other.max(axis=1)
    out.loc[m, "text_account"] = np.array(classes)[other.argmax(axis=1)]
    return out


# ---------- 3. forecast ----------
def monthly_totals(df: pd.DataFrame) -> pd.DataFrame:
    exp = df[df.debit_account.str.startswith("6")]
    return exp.groupby(["company_code", "debit_account", "period"], as_index=False)["amount_eur"].sum()


def _lag_frame(tot: pd.DataFrame) -> pd.DataFrame:
    tot = tot.sort_values("period").copy()
    g = tot.groupby(["company_code", "debit_account"])["amount_eur"]
    for k in (1, 2, 3):
        tot[f"lag{k}"] = g.shift(k)
    tot["lag_mean"] = tot[["lag1", "lag2", "lag3"]].mean(axis=1)
    tot["month"] = tot["period"].str[5:].astype(int)
    tot["acc"] = tot["debit_account"].astype(int) // 1000
    return tot


def _fit_forecast(rows: pd.DataFrame) -> HistGradientBoostingRegressor:
    X = np.log1p(rows[["lag1", "lag2", "lag3", "lag_mean"]]).assign(month=rows.month, acc=rows.acc)
    m = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, random_state=SEED,
                                      categorical_features=[False] * 4 + [True, True])
    return m.fit(X, np.log1p(rows["amount_eur"]))


def forecast_gaps(history: pd.DataFrame, current: pd.DataFrame) -> pd.DataFrame:
    """Walk forward: each month is forecast by a model fitted on earlier months only. Returns
    company, account, period, actual, forecast, gap (actual / forecast - 1)."""
    tot = _lag_frame(pd.concat([history, current]).drop_duplicates(["company_code", "debit_account", "period"], keep="last"))
    periods = sorted(tot.period.unique())
    want = set(current.period.unique())
    out = []
    for i, p in enumerate(periods):
        if p not in want:
            continue
        train = tot[(tot.period < p) & tot.lag3.notna()]
        rows = tot[(tot.period == p) & tot.lag3.notna()]
        if len(train) < 50 or rows.empty:
            continue
        m = _fit_forecast(train)
        X = np.log1p(rows[["lag1", "lag2", "lag3", "lag_mean"]]).assign(month=rows.month, acc=rows.acc)
        out.append(rows.assign(forecast=np.expm1(m.predict(X))))
    if not out:
        return pd.DataFrame(columns=["company_code", "debit_account", "period", "amount_eur", "forecast", "gap"])
    r = pd.concat(out)
    r["gap"] = r["amount_eur"] / r["forecast"].clip(lower=1.0) - 1
    return r[["company_code", "debit_account", "period", "amount_eur", "forecast", "gap"]]


# ---------- full pipeline ----------
def enrich(df: pd.DataFrame, b: Bundle) -> pd.DataFrame:
    """Features, text scores, rules, anomaly percentile and forecast gap for a set of entries."""
    f, _ = add_features(df, b.baseline)
    f = f.join(text_scores(b.text, f))
    f = apply_rules(f, b.baseline)
    raw = -b.anomaly.score_samples(f[ANOMALY_COLS].astype(float))
    f["anomaly"] = np.searchsorted(np.sort(b.anomaly_ref), raw) / len(b.anomaly_ref)
    gaps = forecast_gaps(b.forecast_history, monthly_totals(f))
    f = f.merge(gaps[["company_code", "debit_account", "period", "gap", "forecast"]].rename(
        columns={"gap": "month_gap", "forecast": "month_forecast"}), how="left", on=["company_code", "debit_account", "period"])
    f["month_gap"] = f["month_gap"].fillna(0.0)
    return f


def risk_scores(f: pd.DataFrame, b: Bundle) -> np.ndarray:
    return b.risk.predict_proba(f[RISK_COLS].astype(float))[:, 1]


def train(df_train: pd.DataFrame, labels: dict, extra_labels: dict | None = None, version: str = "v1") -> tuple[Bundle, pd.DataFrame]:
    """Fit all four models on the training months. `labels` are past review outcomes (doc_id -> info) for
    training months; `extra_labels` are reviewer decisions from memory (doc_id -> 1 issue / 0 fine)."""
    extra_labels = extra_labels or {}
    df_all = df_train
    df_train = df_all[df_all.period.isin(TRAIN_MONTHS)]
    f, baseline = add_features(df_train)
    text = train_text(f)
    iso = IsolationForest(n_estimators=200, max_samples=4096, random_state=SEED).fit(f[ANOMALY_COLS].astype(float))
    ref = -iso.score_samples(f[ANOMALY_COLS].astype(float))
    hist = monthly_totals(f)
    b = Bundle(baseline, text, iso, ref, hist, risk=None, version=version)
    f = enrich(df_train, b)
    # reviewer decisions on later closes: enrich their whole month (features need the month), keep the decided rows
    later = df_all[df_all.doc_id.isin(extra_labels) & ~df_all.period.isin(TRAIN_MONTHS)]
    for p in sorted(later.period.unique()):
        fm = enrich(df_all[df_all.period == p], b)
        f = pd.concat([f, fm[fm.doc_id.isin(extra_labels)]], ignore_index=True)
    y = f["doc_id"].map(lambda d: 1 if d in labels else 0).astype(int).to_numpy().copy()
    w = np.ones(len(f))
    for i, doc in enumerate(f["doc_id"]):  # a person's decision overrides the old outcome and counts more
        if doc in extra_labels:
            y[i] = int(extra_labels[doc])
            w[i] = 3.0
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, max_leaf_nodes=15, l2_regularization=1.0,
                                         class_weight="balanced", random_state=SEED)
    clf.fit(f[RISK_COLS].astype(float), y, sample_weight=w)
    b.risk = clf
    b.info = dict(train_months=TRAIN_MONTHS, n_train=len(f), n_positive=int(y.sum()), n_reviewer_labels=len(extra_labels))
    return b, f


def select_flags(f: pd.DataFrame, risk: np.ndarray, per_1000: float = FLAGS_PER_1000) -> pd.Series:
    """Per close, flag the highest-risk entries up to the review capacity. Hard rule hits always count first."""
    f = f.assign(_risk=risk)
    hard = f[[f"r_{c}" for c, r in RULES.items() if r.hard]].any(axis=1)
    flagged = pd.Series(False, index=f.index)
    for _, g in f.groupby("period"):
        k = math.ceil(per_1000 * len(g) / 1000)
        order = g.assign(_hard=hard[g.index]).sort_values(["_hard", "_risk"], ascending=False)
        flagged.loc[order.index[:k]] = True
    return flagged
