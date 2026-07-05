"""Golden-set evaluation of the extraction step.

Workflow:
    python -m riskplatform.intel.evals template   write evals/golden_template.csv
    (hand-fill label_event_type / label_materiality, save as evals/golden_set.csv)
    python -m riskplatform.intel.evals score      compare labels vs model signals,
                                                  write data/intel_eval_metrics.json

The template deliberately excludes the model's predictions so labelling is blind.
Scoring reports per-class precision/recall/F1 for event type, plus precision and
recall on the decision that actually matters downstream: is this announcement
"high" materiality (the rows that feed risk commentary)?
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from .. import artifacts
from ..config import DEFAULT_DATA_DIR, PROJECT_ROOT

log = logging.getLogger(__name__)

EVALS_DIR = PROJECT_ROOT / "evals"
TEMPLATE = EVALS_DIR / "golden_template.csv"
GOLDEN = EVALS_DIR / "golden_set.csv"

TEMPLATE_COLUMNS = [
    "doc_key",
    "date",
    "ticker",
    "ann_type",
    "price_sensitive",
    "headline",
    "label_event_type",
    "label_materiality",
]


def write_template(data_dir: Path, n: int = 200) -> Path:
    """Sample stored announcements into a blind labelling template (existing labels kept)."""
    ann = artifacts.read(data_dir / artifacts.ANNOUNCEMENTS)
    if ann is None or ann.empty:
        raise SystemExit("no announcements ingested yet — run the intel pipeline first")
    sample = ann.sort_values("date", ascending=False).head(n).copy()
    sample["label_event_type"] = ""
    sample["label_materiality"] = ""
    out = sample[TEMPLATE_COLUMNS[:6] + ["label_event_type", "label_materiality"]]

    if GOLDEN.exists():  # don't ask for re-labels of rows already in the golden set
        labelled = set(pd.read_csv(GOLDEN)["doc_key"])
        out = out[~out["doc_key"].isin(labelled)]

    EVALS_DIR.mkdir(exist_ok=True)
    out.to_csv(TEMPLATE, index=False)
    log.info("wrote %d rows to %s — fill labels, save as %s", len(out), TEMPLATE, GOLDEN)
    return TEMPLATE


def _per_class_prf(labels: pd.Series, preds: pd.Series) -> dict:
    out = {}
    for cls in sorted(set(labels) | set(preds)):
        tp = int(((preds == cls) & (labels == cls)).sum())
        fp = int(((preds == cls) & (labels != cls)).sum())
        fn = int(((preds != cls) & (labels == cls)).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        out[cls] = {"precision": precision, "recall": recall, "f1": f1, "support": tp + fn}
    return out


def score(data_dir: Path, golden_path: Path = GOLDEN) -> dict:
    """Join golden labels with model signals and write the eval-metrics artifact."""
    if not golden_path.exists():
        raise SystemExit(f"{golden_path} not found — run `template` and label it first")
    golden = pd.read_csv(golden_path).dropna(subset=["label_event_type", "label_materiality"])
    golden = golden[(golden["label_event_type"] != "") & (golden["label_materiality"] != "")]
    signals = artifacts.read(data_dir / artifacts.ANNOUNCEMENT_SIGNALS)
    if signals is None or signals.empty:
        raise SystemExit("no signals extracted yet — run the intel pipeline first")

    joined = golden.merge(signals, on="doc_key")
    if joined.empty:
        raise SystemExit("no overlap between golden set and extracted signals")

    et_labels, et_preds = joined["label_event_type"], joined["event_type"]
    high_labels = joined["label_materiality"].eq("high")
    high_preds = joined["materiality"].eq("high")
    tp = int((high_preds & high_labels).sum())

    metrics = {
        "n_labelled": int(len(joined)),
        "event_type_accuracy": float((et_labels == et_preds).mean()),
        "event_type_per_class": _per_class_prf(et_labels, et_preds),
        "materiality_accuracy": float((joined["label_materiality"] == joined["materiality"]).mean()),
        "high_materiality_precision": tp / int(high_preds.sum()) if high_preds.sum() else 0.0,
        "high_materiality_recall": tp / int(high_labels.sum()) if high_labels.sum() else 0.0,
        "confusion": pd.crosstab(et_labels, et_preds).to_dict(),
        "model": signals["model"].iloc[-1],
        "scored_utc": pd.Timestamp.now("UTC").isoformat(),
    }
    artifacts.write_json(data_dir / artifacts.INTEL_EVAL, metrics)
    log.info(
        "scored %d labels — event-type accuracy %.0f%%, high-materiality P %.0f%% / R %.0f%%",
        metrics["n_labelled"],
        100 * metrics["event_type_accuracy"],
        100 * metrics["high_materiality_precision"],
        100 * metrics["high_materiality_recall"],
    )
    return metrics


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(prog="riskplatform.intel.evals")
    parser.add_argument("command", choices=["template", "score"])
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--n", type=int, default=200, help="template sample size")
    args = parser.parse_args(argv)
    if args.command == "template":
        write_template(args.data_dir, args.n)
    else:
        score(args.data_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
