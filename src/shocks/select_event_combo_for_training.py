#!/usr/bin/env python3
"""
select_event_combo_for_training.py

Search for the best training event combination by training a model on a subset
of shock events and evaluating MAE on the remaining events.
"""

from __future__ import annotations

import argparse
import copy
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd

from train_propagation_model import (
    LABEL_CLIP_ABS_DEFAULT,
    LABEL_MAX_ABS_DEFAULT,
    SHOCK_CLIP_ABS_DEFAULT,
    TARGET_COL,
    build_xy_groups,
    clip_shock_feature,
    event_balanced_weights,
    eval_regression,
    filter_and_clip_labels,
    make_models,
)


def _total_combos(n: int, max_k: int) -> int:
    return int(sum(math.comb(n, k) for k in range(1, max_k + 1)))


def _parse_list(value: Optional[str]) -> List[str]:
    if not value:
        return []
    return [v.strip() for v in value.split(",") if v.strip()]


def _apply_filters(events: Iterable[str], include: List[str], exclude: List[str]) -> List[str]:
    out = list(events)
    if include:
        include_set = set(include)
        out = [e for e in out if e in include_set]
    if exclude:
        exclude_set = set(exclude)
        out = [e for e in out if e not in exclude_set]
    return sorted(set(out))


def main() -> None:
    ap = argparse.ArgumentParser(description="Find best event combos for train/test MAE.")
    ap.add_argument("--data", default=str(Path(__file__).parent / "training_data_v4.csv"))
    ap.add_argument("--model", default="ridge", help="Model key from train_propagation_model.make_models().")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-combo-size", type=int, default=3)
    ap.add_argument("--max-combos", type=int, default=2000)
    ap.add_argument("--min-test-events", type=int, default=1)
    ap.add_argument("--max-train-mae", type=float, default=0.3)
    ap.add_argument("--event-balanced-weighting", action="store_true")
    ap.add_argument("--include-events", default="", help="Comma-separated shock_event allowlist.")
    ap.add_argument("--exclude-events", default="", help="Comma-separated shock_event blocklist.")
    ap.add_argument("--out-csv", default=str(Path(__file__).parent / "event_combo_train_test.csv"))
    args = ap.parse_args()

    data_path = Path(args.data)
    df = pd.read_csv(data_path)

    n0 = len(df)
    if "shock_expected" in df.columns:
        df = df[~(pd.to_numeric(df["shock_expected"], errors="coerce") < 0)].copy()
    if "prop_expected" in df.columns:
        df = df[~(pd.to_numeric(df["prop_expected"], errors="coerce") < 0)].copy()
    if len(df) != n0:
        print(f"Filtered negative expected rows: {n0:,} -> {len(df):,}")

    X, y, groups = build_xy_groups(df)
    X, y, groups = filter_and_clip_labels(
        X,
        y,
        groups,
        max_abs=float(LABEL_MAX_ABS_DEFAULT),
        clip_abs=float(LABEL_CLIP_ABS_DEFAULT),
    )
    X = clip_shock_feature(X, clip_abs=float(SHOCK_CLIP_ABS_DEFAULT))

    all_events = sorted(set(groups.astype(str)))
    include = _parse_list(args.include_events)
    exclude = _parse_list(args.exclude_events)
    events = _apply_filters(all_events, include, exclude)
    if not events:
        raise SystemExit("No events available after include/exclude filtering.")

    max_k = min(int(args.max_combo_size), len(events))
    total = _total_combos(len(events), max_k)
    if total > int(args.max_combos):
        raise SystemExit(
            f"Combination count {total} exceeds --max-combos={args.max_combos}. "
            "Reduce --max-combo-size or increase --max-combos."
        )

    models = make_models(random_state=int(args.seed))
    if args.model not in models:
        raise SystemExit(f"Unknown model '{args.model}'. Available: {sorted(models.keys())}")

    rows: List[Dict[str, object]] = []
    for k in range(1, max_k + 1):
        for combo in itertools.combinations(events, k):
            train_events = set(combo)
            test_events = set(events) - train_events
            if len(test_events) < int(args.min_test_events):
                continue

            train_mask = groups.astype(str).isin(train_events)
            test_mask = groups.astype(str).isin(test_events)
            if not train_mask.any() or not test_mask.any():
                continue

            X_train = X.loc[train_mask]
            y_train = y.loc[train_mask]
            X_test = X.loc[test_mask]
            y_test = y.loc[test_mask]

            model = copy.deepcopy(models[args.model])
            w_train = (
                event_balanced_weights(groups.loc[train_mask])
                if bool(args.event_balanced_weighting)
                else None
            )
            if w_train is not None:
                try:
                    model.fit(X_train, y_train, model__sample_weight=w_train.to_numpy())
                except TypeError:
                    model.fit(X_train, y_train)
            else:
                model.fit(X_train, y_train)

            y_train_pred = model.predict(X_train)
            train_metrics = eval_regression(y_train.to_numpy(), y_train_pred)
            if float(train_metrics["mae"]) > float(args.max_train_mae):
                continue

            y_test_pred = model.predict(X_test)
            test_metrics = eval_regression(y_test.to_numpy(), y_test_pred)

            rows.append(
                {
                    "events": "|".join(combo),
                    "n_train_events": int(len(train_events)),
                    "n_test_events": int(len(test_events)),
                    "n_train_rows": int(len(y_train)),
                    "n_test_rows": int(len(y_test)),
                    "train_mae": float(train_metrics["mae"]),
                    "test_mae": float(test_metrics["mae"]),
                    "test_rmse": float(test_metrics["rmse"]),
                    "test_r2": float(test_metrics["r2"]),
                    "test_spearman": float(test_metrics["spearman"]),
                }
            )

    if not rows:
        raise SystemExit("No combinations met the criteria (try increasing --max-train-mae).")

    out_df = pd.DataFrame(rows).sort_values(["test_mae", "train_mae"], ascending=[True, True])
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"Saved {len(out_df)} combos -> {out_path}")
    print(f"Best: events={out_df.iloc[0]['events']} | test_mae={out_df.iloc[0]['test_mae']:.4f}")


if __name__ == "__main__":
    import itertools

    main()

