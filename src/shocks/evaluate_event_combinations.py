#!/usr/bin/env python3
"""
evaluate_event_combinations.py

Run live_partner_impact for events from training_data_v4.csv filtered by
per-event LOEO MAE, then score every combination by relative MAE between
y_pred and y_observed.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))


def _safe_float(value: object) -> Optional[float]:
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


def _parse_shock_date(value: object) -> Tuple[int, int]:
    if value is None or pd.isna(value):
        raise ValueError("Missing shock_date")
    s = str(value)
    parts = s.split("-")
    if len(parts) < 2:
        raise ValueError(f"Invalid shock_date: {s}")
    return int(parts[0]), int(parts[1])


def _select_best_model(metrics: dict) -> str:
    models = metrics.get("models", {})
    best_name = None
    best_mae = float("inf")
    for name, data in models.items():
        loeo = data.get("leave_one_event_out", {})
        mae = loeo.get("summary", {}).get("mae", {}).get("mean")
        if mae is None or (isinstance(mae, float) and math.isnan(mae)):
            continue
        try:
            mae_f = float(mae)
        except Exception:
            continue
        if mae_f < best_mae:
            best_mae = mae_f
            best_name = str(name)
    if best_name is None:
        raise ValueError("No model with leave_one_event_out summary MAE found.")
    return best_name


def _load_event_mae(metrics_path: Path) -> Tuple[str, Dict[str, float]]:
    metrics = json.loads(metrics_path.read_text())
    model_name = _select_best_model(metrics)
    loeo = metrics["models"][model_name].get("leave_one_event_out", {})
    fold_details = loeo.get("fold_details", [])
    if not fold_details:
        raise ValueError("leave_one_event_out.fold_details missing in metrics.json.")
    event_mae: Dict[str, float] = {}
    for fold in fold_details:
        groups = fold.get("test_groups", [])
        if len(groups) != 1:
            continue
        event = str(groups[0])
        mae = fold.get("metrics", {}).get("mae")
        mae_f = _safe_float(mae)
        if mae_f is None:
            continue
        event_mae[event] = mae_f
    if not event_mae:
        raise ValueError("No per-event MAE extracted from leave_one_event_out fold_details.")
    return model_name, event_mae


def _first_non_null(series: pd.Series) -> Optional[object]:
    for v in series:
        if v is not None and not pd.isna(v):
            return v
    return None


def _load_event_metadata(training_csv: Path, events: Iterable[str]) -> Dict[str, dict]:
    df = pd.read_csv(training_csv)
    df = df[df["shock_event"].astype(str).isin(set(events))]
    meta: Dict[str, dict] = {}
    for event, g in df.groupby("shock_event"):
        shock_node = _first_non_null(g.get("shock_node"))
        shock_date = _first_non_null(g.get("shock_date"))
        shock_yoy_actual = _first_non_null(g.get("shock_yoy_actual"))
        shock_yoy_dev = _first_non_null(g.get("shock_yoy_dev"))
        if shock_node is None or shock_date is None:
            continue
        shock_year, shock_month = _parse_shock_date(shock_date)
        meta[str(event)] = {
            "shock_node": str(shock_node),
            "shock_year": int(shock_year),
            "shock_month": int(shock_month),
            "shock_yoy_actual": _safe_float(shock_yoy_actual),
            "shock_yoy_dev": _safe_float(shock_yoy_dev),
        }
    return meta


def _relative_mae(df: pd.DataFrame, epsilon: float) -> Tuple[Optional[float], int]:
    if df.empty:
        return None, 0
    mask = df["y_pred"].notna() & df["y_observed"].notna()
    if not mask.any():
        return None, 0
    sub = df.loc[mask].copy()
    denom = sub["y_observed"].abs().clip(lower=epsilon)
    rel_mae = (sub["y_pred"] - sub["y_observed"]).abs().div(denom).mean()
    return float(rel_mae), int(len(sub))


def _total_combos(n: int, max_k: int) -> int:
    return int(sum(math.comb(n, k) for k in range(1, max_k + 1)))


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate live_partner_impact across event combinations.")
    ap.add_argument("--training-csv", default=str(SCRIPT_DIR / "training_data_v4.csv"))
    ap.add_argument("--metrics-json", default=str(SCRIPT_DIR / "models" / "v4" / "metrics.json"))
    ap.add_argument("--model-dir", default=str(SCRIPT_DIR / "models" / "v4"))
    ap.add_argument(
        "--embeddings-dir",
        default=str((SCRIPT_DIR.parent / "embeddings").as_posix()),
        help="Directory containing ICIO graphs (graph_YYYY_labeled.pt)",
    )
    ap.add_argument("--mae-threshold", type=float, default=0.3)
    ap.add_argument("--months-after-shock", type=int, default=1)
    ap.add_argument("--history-months", type=int, default=24)
    ap.add_argument("--candidates", type=int, default=5)
    ap.add_argument("--hops", type=int, default=1)
    ap.add_argument("--epsilon", type=float, default=1e-9)
    ap.add_argument("--max-combo-size", type=int, default=3)
    ap.add_argument("--max-combos", type=int, default=1000)
    ap.add_argument("--limit-events", type=int, default=0)
    ap.add_argument("--no-network", action="store_true")
    ap.add_argument("--out-csv", default=str(SCRIPT_DIR / "event_combo_results.csv"))
    args = ap.parse_args()

    from live_partner_impact import ScenarioInputs, run_multihop_scenario, run_scenario

    metrics_path = Path(args.metrics_json)
    model_name, event_mae = _load_event_mae(metrics_path)
    keep_events = [e for e, mae in event_mae.items() if float(mae) <= float(args.mae_threshold)]
    keep_events = sorted(set(keep_events))
    if args.limit_events and args.limit_events > 0:
        keep_events = keep_events[: int(args.limit_events)]

    if not keep_events:
        raise SystemExit("No events meet the MAE threshold; adjust --mae-threshold or metrics JSON.")

    meta = _load_event_metadata(Path(args.training_csv), keep_events)
    keep_events = [e for e in keep_events if e in meta]
    if not keep_events:
        raise SystemExit("No events with valid metadata found in training CSV.")

    max_combo_size = int(args.max_combo_size)
    if max_combo_size <= 0:
        max_combo_size = len(keep_events)
    max_combo_size = min(max_combo_size, len(keep_events))
    total = _total_combos(len(keep_events), max_combo_size)
    if total > int(args.max_combos):
        raise SystemExit(
            f"Combination count {total} exceeds --max-combos={args.max_combos}. "
            "Reduce --max-combo-size or increase --max-combos."
        )

    event_results: Dict[str, pd.DataFrame] = {}
    for event in keep_events:
        info = meta[event]
        shock_yoy_actual = info.get("shock_yoy_actual")
        shock_yoy_dev = info.get("shock_yoy_dev")
        if int(args.hops) > 1:
            shock_value = shock_yoy_dev
        else:
            shock_value = shock_yoy_actual
        if shock_value is None:
            continue
        inputs = ScenarioInputs(
            shock_node=str(info["shock_node"]),
            shock_year=int(info["shock_year"]),
            shock_month=int(info["shock_month"]),
            shock_yoy_actual=float(shock_value),
            months_after_shock=int(args.months_after_shock),
            history_months=int(args.history_months),
            candidates=int(args.candidates),
        )
        if int(args.hops) <= 1:
            df = run_scenario(
                model_dir=Path(args.model_dir),
                training_csv=Path(args.training_csv),
                embeddings_dir=Path(args.embeddings_dir),
                inputs=inputs,
                use_network=not bool(args.no_network),
                hop=0,
                shock_is_dev=False,
            )
        else:
            df = run_multihop_scenario(
                model_dir=Path(args.model_dir),
                training_csv=Path(args.training_csv),
                embeddings_dir=Path(args.embeddings_dir),
                inputs=inputs,
                use_network=not bool(args.no_network),
                hops=int(args.hops),
            )
        event_results[event] = df

    rows: List[Dict[str, object]] = []
    for k in range(1, max_combo_size + 1):
        for combo in itertools.combinations(keep_events, k):
            dfs = [event_results[e] for e in combo if e in event_results]
            if not dfs:
                continue
            df_combo = pd.concat(dfs, ignore_index=True)
            rel_mae, n_rows = _relative_mae(df_combo, epsilon=float(args.epsilon))
            if rel_mae is None:
                continue
            rows.append(
                {
                    "events": "|".join(combo),
                    "n_events": int(len(combo)),
                    "n_rows": int(n_rows),
                    "relative_mae": float(rel_mae),
                }
            )

    if not rows:
        raise SystemExit("No combinations produced valid y_pred/y_observed rows.")

    out_df = pd.DataFrame(rows).sort_values(["relative_mae", "n_events"], ascending=[True, True])
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(
        f"Saved {len(out_df)} combinations to {out_path} "
        f"(model={model_name}, events={len(keep_events)}, max_k={max_combo_size})."
    )


if __name__ == "__main__":
    main()

