"""Measure the detector on the test months (Aug-Oct 2026), which no model saw in training.

Numbers that matter to a controller:
- recall: share of planted problems found (a problem counts as found when any of its entries is flagged),
- flags per 1,000 entries: the review workload,
- precision: share of flags that are real problems,
- recall per problem kind, and the same numbers for "rules only" and "anomaly only" for comparison.

The gates in GATES are checked in CI; a model that misses them is not shipped.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .config import FLAGS_PER_1000, TEST_MONTHS, TRAIN_MONTHS
from .features import RULES, load_entries
from .models import Bundle, enrich, risk_scores, select_flags, train
from .warehouse import Warehouse

GATES = {"recall": 0.90, "high_severity_recall": 0.95, "flags_per_1000_max": 8.0}


def _summary(f: pd.DataFrame, flagged: pd.Series, labels: dict) -> dict:
    lab = pd.DataFrame.from_dict(labels, orient="index")
    lab = lab[lab.index.isin(set(f.doc_id))]
    hit_docs = set(f.loc[flagged, "doc_id"])
    lab["hit"] = lab.index.isin(hit_docs)
    groups = lab.groupby("group").agg(kind=("kind", "first"), severity=("severity", "first"), found=("hit", "any"))
    per_kind = groups.groupby("kind")["found"].agg(["sum", "count"])
    n_flags = int(flagged.sum())
    true_flags = len(hit_docs & set(lab.index))
    return dict(
        entries=len(f), problems=len(groups), found=int(groups.found.sum()),
        recall=round(float(groups.found.mean()), 4) if len(groups) else 0.0,
        high_severity_recall=round(float(groups[groups.severity == "high"].found.mean()), 4),
        flags=n_flags, flags_per_1000=round(1000 * n_flags / max(len(f), 1), 2),
        precision=round(true_flags / max(n_flags, 1), 4),
        per_kind={k: f"{int(r['sum'])}/{int(r['count'])}" for k, r in per_kind.iterrows()},
    )


def evaluate(b: Bundle, f_test: pd.DataFrame, labels: dict) -> dict:
    risk = risk_scores(f_test, b)
    model = _summary(f_test, select_flags(f_test, risk), labels)
    # baselines with the same review budget
    rules = f_test[[f"r_{c}" for c in RULES]].sum(axis=1).to_numpy(dtype=float)
    rules = rules + 1e-6 * f_test["amount_eur"].rank(pct=True).to_numpy()  # more rule hits first, then larger amounts
    rules_only = _summary(f_test, select_flags(f_test, rules), labels)
    anomaly_only = _summary(f_test, select_flags(f_test.assign(**{f"r_{c}": False for c in RULES}), f_test["anomaly"].to_numpy()), labels)
    any_rule = f_test[[f"r_{c}" for c in RULES]].any(axis=1)
    model["baselines"] = {
        "rules_only_same_budget": {k: rules_only[k] for k in ("recall", "precision", "flags_per_1000")},
        "anomaly_only_same_budget": {k: anomaly_only[k] for k in ("recall", "precision", "flags_per_1000")},
        "all_rule_hits_no_budget": {k: v for k, v in _summary(f_test, any_rule, labels).items()
                                     if k in ("recall", "precision", "flags_per_1000")},
    }
    curve = []  # recall at different review budgets (no hard-rule priority, to compare the scores fairly)
    plain = f_test.assign(**{f"r_{c}": False for c in RULES})
    for k in (2, 3, 4, 6, 8, 10, 12):
        curve.append({"flags_per_1000": k,
                      "risk_model": _summary(f_test, select_flags(plain, risk, k), labels)["recall"],
                      "rules_count": _summary(f_test, select_flags(plain, rules, k), labels)["recall"],
                      "anomaly_only": _summary(f_test, select_flags(plain, f_test["anomaly"].to_numpy(), k), labels)["recall"]})
    model["recall_by_budget"] = curve
    model["gates"] = {k: (model[k] >= v if k != "flags_per_1000_max" else model["flags_per_1000"] <= v) for k, v in GATES.items()}
    model["passed"] = all(model["gates"].values())
    return model


def run(data_dir: Path, out_dir: Path | None = None, extra_labels: dict | None = None, version: str = "v1",
        compare_with: Bundle | None = None) -> tuple[Bundle, dict]:
    """Train on months 1-9, evaluate on 10-12, save the bundle and metrics."""
    data_dir = Path(data_dir)
    labels = json.loads((data_dir / "labels.json").read_text())
    with Warehouse(data_dir) as wh:
        df = load_entries(wh)
    b, _ = train(df, labels, extra_labels, version)  # train() only sees training-month entries, so only their labels
    f_test = enrich(df[df.period.isin(TEST_MONTHS)], b)
    # entries a reviewer decided are now training data: measure on the others only (no leak)
    seen = set(extra_labels or {})
    f_test = f_test[~f_test.doc_id.isin(seen)].reset_index(drop=True)
    metrics = evaluate(b, f_test, {d: v for d, v in labels.items() if d not in seen})
    metrics["excluded_reviewed_entries"] = len(seen)
    if compare_with is not None:  # the current version on exactly the same entries, for a fair comparison
        f_old = enrich(df[df.period.isin(TEST_MONTHS)], compare_with)
        f_old = f_old[~f_old.doc_id.isin(seen)].reset_index(drop=True)
        old = evaluate(compare_with, f_old, {d: v for d, v in labels.items() if d not in seen})
        metrics["previous_same_entries"] = {k: old[k] for k in ("recall", "high_severity_recall", "precision", "flags_per_1000")}
    metrics.update(version=version, train_months=f"{TRAIN_MONTHS[0]}..{TRAIN_MONTHS[-1]}",
                   test_months=f"{TEST_MONTHS[0]}..{TEST_MONTHS[-1]}", flags_budget_per_1000=FLAGS_PER_1000,
                   **{k: v for k, v in b.info.items() if k != "train_months"})
    out = Path(out_dir or data_dir / "models")
    b.save(out / f"bundle-{version}.joblib")
    (out / f"metrics-{version}.json").write_text(json.dumps(metrics, indent=1, default=str))
    return b, metrics


if __name__ == "__main__":
    import sys

    from .simulate import ensure_built

    d = Path(sys.argv[1] if len(sys.argv) > 1 else "data")
    ensure_built(d)
    _, m = run(d)
    print(json.dumps(m, indent=1, default=str))
    raise SystemExit(0 if m["passed"] else 1)
