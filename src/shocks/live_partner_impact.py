#!/usr/bin/env python3
"""
live_partner_impact.py

Scenario runner on top of `propagation_inference.py`:
- user provides: shocked node, shock month (present year by default), shock magnitude (shock_yoy_change)
- we compute partner exposure features from ICIO graphs
- optionally query:
  - World Bank indicators (via `src/world_data.py`)
  - UN Comtrade (via `src/shocks/ComtradeAPI.py`)
- return top-k affected downstream partners with:
  - predicted prop_yoy_dev (deviation vs expected YoY export growth)
  - export value(s) derived from a simple 24-month "expected growth" model

Notes:
- This script is designed to be robust: it will still run if APIs fail, but some columns will be NA.
- Run from the repo with the same Python env you used to train (your venv recommended).
"""

from __future__ import annotations

import argparse
import math
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from propagation_inference import PropagationPredictor
from shock_helpers import (
    ICIOHelper,
    SupplierMetricsHelper,
    baseline_year_for,
    expected_from_pre_shock,
    month_keys,
)

try:
    # Local module (repo root adds `src/` in PYTHONPATH in typical usage; fallback below if needed).
    from ComtradeAPI import ComtradeAPI  # type: ignore
except Exception:  # pragma: no cover
    from shocks.ComtradeAPI import ComtradeAPI  # type: ignore


def _parse_country(node: str) -> str:
    return str(node).split("_", 1)[0]


def _parse_sector(node: str) -> str:
    parts = str(node).split("_", 1)
    return parts[1] if len(parts) == 2 else ""


def _ym_key(y: int, m: int) -> str:
    return f"{int(y)}-{int(m):02d}"


def _add_months(y: int, m: int, delta: int) -> Tuple[int, int]:
    """
    Add delta months to (y, m) where m is 1-12.
    """
    total = (y * 12 + (m - 1)) + int(delta)
    ny = total // 12
    nm = (total % 12) + 1
    return int(ny), int(nm)

def _months_between_inclusive(start_y: int, start_m: int, end_y: int, end_m: int) -> int:
    """
    Number of months in [start, end] inclusive (start <= end).
    """
    s = start_y * 12 + (start_m - 1)
    e = end_y * 12 + (end_m - 1)
    if e < s:
        raise ValueError("end must be >= start")
    return (e - s) + 1


def _month_range_ending(end_y: int, end_m: int, n_months: int) -> List[str]:
    """
    Ascending list of YYYY-MM keys ending at (end_y, end_m), length n_months.
    """
    keys: List[str] = []
    for i in range(n_months - 1, -1, -1):
        y, m = _add_months(end_y, end_m, -i)
        keys.append(_ym_key(y, m))
    return keys


def _latest_graph_year(icio_dir: Path) -> int:
    yrs: List[int] = []
    for p in icio_dir.glob("graph_*_labeled.pt"):
        stem = p.stem  # graph_YYYY_labeled
        parts = stem.split("_")
        for token in parts:
            if token.isdigit() and len(token) == 4:
                yrs.append(int(token))
    if not yrs:
        raise FileNotFoundError(f"No ICIO graphs found in: {icio_dir}")
    return max(yrs)


@dataclass(frozen=True)
class ScenarioInputs:
    shock_node: str
    shock_year: int
    shock_month: int
    shock_yoy_change: float
    months_after_shock: int
    history_months: int
    candidates: int


def _try_load_indicators(
    *,
    target_nodes: List[str],
    year: int,
) -> Optional[pd.DataFrame]:
    """
    Best-effort World Bank indicators load; returns df indexed by target_node or None.
    """
    try:
        # Import lazily so the script still runs if wbgapi isn't installed / network is down.
        # Note: world_data.py lives in `src/` (repo root); if running from src/shocks, parent is src.
        import logging
        import sys

        here = Path(__file__).resolve()
        src_dir = here.parents[1]
        if str(src_dir) not in sys.path:
            sys.path.insert(0, str(src_dir))
        from world_data import load_indicators  # type: ignore

        # world_data is intentionally chatty; keep scenario output readable.
        logging.getLogger("world_data").setLevel(logging.CRITICAL)

        needed = ["gdp_growth", "inflation", "unemployment_rate", "gdp_per_capita"]
        df = load_indicators(year, pd.Index(target_nodes), indicator_list=needed) 
        return df
    except Exception:
        return None


def _median_fallbacks_from_training(
    training_csv: Path,
) -> Dict[str, float]:
    """
    Get robust numeric fallbacks for macro columns (and any other numerics we need).
    """
    df = pd.read_csv(training_csv)
    out: Dict[str, float] = {}
    for col in [
        "target_log_gdp_per_capita",
        "target_gdp_growth",
        "target_inflation",
        "target_unemployment_rate",
    ]:
        if col in df.columns:
            out[col] = float(pd.to_numeric(df[col], errors="coerce").median())
    return out


def _value(series: Dict[str, float], key: str) -> Optional[float]:
    v = series.get(key)
    if v is None:
        return None
    try:
        fv = float(v)
        return fv
    except Exception:
        return None


def _safe_yoy(n: Optional[float], d: Optional[float]) -> Optional[float]:
    if n is None or d is None or d <= 0:
        return None
    return (n - d) / d


def _expected_yoy_from_history(
    *,
    series_all: Dict[str, float],
    obs_key: str,
    history_months: int,
) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float], Optional[float]]:
    """
    Build a simple expectation model using `expected_from_pre_shock` over the last `history_months`
    ending at obs_key-1 month, and compute expected_yoy for obs_key.

    Returns:
      expected_yoy, y_lag_12, y_expected, y_pred_stub, y_observed

    y_pred_stub is None here (we compute it after prop_yoy_dev prediction).
    """
    oy, om = map(int, obs_key.split("-"))
    lag_y, lag_m = _add_months(oy, om, -12)
    lag_key = _ym_key(lag_y, lag_m)

    # Pre window ends at obs_key-1 month.
    pre_end_y, pre_end_m = _add_months(oy, om, -1)
    pre_keys = _month_range_ending(pre_end_y, pre_end_m, history_months)
    forecast_keys = sorted({obs_key, lag_key})

    expected_map, _resid_std = expected_from_pre_shock(
        series_all=series_all,
        pre_keys=pre_keys,
        forecast_keys=forecast_keys,
    )

    exp_obs = expected_map.get(obs_key)
    exp_lag = expected_map.get(lag_key)
    expected_yoy = _safe_yoy(exp_obs, exp_lag)

    y_lag_12 = _value(series_all, lag_key)
    y_observed = _value(series_all, obs_key)
    y_expected = (y_lag_12 * (1.0 + expected_yoy)) if (y_lag_12 is not None and expected_yoy is not None) else None
    return expected_yoy, y_lag_12, y_expected, None, y_observed


def run_scenario(
    *,
    model_dir: Path,
    training_csv: Path,
    embeddings_dir: Path,
    inputs: ScenarioInputs,
    use_network: bool,
) -> pd.DataFrame:
    # Model
    predictor = PropagationPredictor(str(model_dir))

    # ICIO features
    icio_year = min(inputs.shock_year, _latest_graph_year(embeddings_dir))
    icio = ICIOHelper(embeddings_dir)
    graph = icio.load_graph(icio_year)
    supplier = SupplierMetricsHelper(icio)

    # Candidate partners from ICIO
    downstream = icio.get_downstream_partners(inputs.shock_node, icio_year, top_k=inputs.candidates)
    if not downstream:
        raise RuntimeError("No downstream partners found (check shocked node + available ICIO years).")

    # Macro indicators (best effort)
    indicators_year = min(inputs.shock_year - 1, date.today().year - 1)
    indicators_df = _try_load_indicators(target_nodes=[p["target_node"] for p in downstream], year=indicators_year)
    fallbacks = _median_fallbacks_from_training(training_csv)

    # Comtrade API (optional)
    api = ComtradeAPI(rate_limit_delay=1.0) if use_network else None

    shocked_country = _parse_country(inputs.shock_node)
    shocked_sector = _parse_sector(inputs.shock_node)

    # Observation date
    obs_year, obs_month = _add_months(inputs.shock_year, inputs.shock_month, inputs.months_after_shock)
    obs_key = _ym_key(obs_year, obs_month)
    if inputs.months_after_shock < 0 or inputs.months_after_shock > 11:
        raise ValueError("To match build_shock_dataset.py behavior, months_after_shock must be in [0, 11].")

    rows: List[Dict[str, object]] = []

    for partner in downstream:
        target_node = str(partner["target_node"])
        target_country = str(partner["target_country"])
        target_sector = str(partner.get("target_sector") or _parse_sector(target_node))
        icio_edge_value = float(partner.get("edge_value", 0.0) or 0.0)
        is_domestic = bool(target_country == shocked_country)

        # Supplier metrics
        sm = supplier.compute_supplier_metrics(target_node, inputs.shock_node, icio_year)
        supplier_hhi = float(sm.get("supplier_hhi", 1.0))
        shocked_supplier_share = float(sm.get("shocked_supplier_share", 0.0))

        # ICIO allocation weight (used when turning import delta into a target-sector shock_value)
        icio_weight = 1.0 if is_domestic else float(ICIOHelper.calculate_industry_weight(target_node, inputs.shock_node, graph))

        # Macros
        target_log_gdp_per_capita = fallbacks.get("target_log_gdp_per_capita")
        target_gdp_growth = fallbacks.get("target_gdp_growth")
        target_inflation = fallbacks.get("target_inflation")
        target_unemployment_rate = fallbacks.get("target_unemployment_rate")
        if indicators_df is not None and target_node in indicators_df.index:
            r = indicators_df.loc[target_node]
            if "log_gdp_per_capita" in r.index and pd.notna(r["log_gdp_per_capita"]):
                target_log_gdp_per_capita = float(r["log_gdp_per_capita"])
            if "gdp_growth" in r.index and pd.notna(r["gdp_growth"]):
                target_gdp_growth = float(r["gdp_growth"])
            if "inflation" in r.index and pd.notna(r["inflation"]):
                target_inflation = float(r["inflation"])
            if "unemployment_rate" in r.index and pd.notna(r["unemployment_rate"]):
                target_unemployment_rate = float(r["unemployment_rate"])

        # Shock value (absolute delta) from baseline import and user-provided shock_yoy_change
        shock_value: Optional[float] = None
        import_baseline: Optional[float] = None
        import_delta: Optional[float] = None
        if api is not None:
            # Need baseline for same month last year to convert user yoy->absolute delta.
            baseline_year = inputs.shock_year - 1
            baseline_key = f"{baseline_year}-{obs_month:02d}"
            base_series = api.get_trade_data(
                reporter=target_country,
                partner=shocked_country,
                sector_code=shocked_sector,
                flow_code="M",
                start_year=baseline_year,
                start_month=obs_month,
                duration_months=1,
                verbose=True,
            )
            import_baseline = _value(base_series, baseline_key)
            if import_baseline is not None:
                import_delta = import_baseline * float(inputs.shock_yoy_change)
                shock_value = float(import_delta) * float(icio_weight)
            else:
                print(f"      [FALLBACK] Missing import baseline for {target_country}<-{shocked_country} at {baseline_key}; shock_value=None")

        # Export history for target's TOTAL exports (sector-level) to the world
        export_series: Dict[str, float] = {}
        expected_yoy: Optional[float] = None
        y_lag_12: Optional[float] = None
        y_expected: Optional[float] = None
        y_observed: Optional[float] = None

        if api is not None:
            start_y = inputs.shock_year - 2
            start_m = inputs.shock_month
            # Ensure the window includes the observation month so y_observed is populated.
            duration_months = _months_between_inclusive(start_y, start_m, obs_year, obs_month)
            export_series = api.get_trade_data(
                reporter=target_country,
                partner="WLD",
                sector_code=target_sector,
                flow_code="X",
                start_year=start_y,
                start_month=start_m,
                duration_months=duration_months,
                verbose=True,
            )
            # Expectations: fit on 24 months pre-shock window, forecast for obs_key and its t-12 lag.
            pre_keys = month_keys(inputs.shock_year - 2, inputs.shock_month, 24)
            lag_key = f"{obs_year - 1}-{obs_month:02d}"
            forecast_keys_ext = sorted({obs_key, lag_key})
            export_expected_map, _export_resid_std = expected_from_pre_shock(
                series_all=export_series,
                pre_keys=pre_keys,
                forecast_keys=forecast_keys_ext,
            )
            export_expected = export_expected_map.get(obs_key)
            export_expected_lag = export_expected_map.get(lag_key)
            expected_yoy = (
                (export_expected - export_expected_lag) / export_expected_lag
                if (export_expected is not None and export_expected_lag is not None and export_expected_lag > 0)
                else None
            )
            y_lag_12 = _value(export_series, lag_key)
            y_observed = _value(export_series, obs_key)
            y_expected = (float(y_lag_12) * (1.0 + float(expected_yoy))) if (y_lag_12 is not None and expected_yoy is not None) else None

        # Predict propagation deviation
        pred_prop_yoy_dev = predictor.predict_prop_yoy_dev(
            shock_node=inputs.shock_node,
            target_node=target_node,
            target_country=target_country,
            months_after_shock=int(inputs.months_after_shock),
            observation_month=int(obs_month),
            is_domestic=is_domestic,
            shock_yoy_change=float(inputs.shock_yoy_change),
            shock_value=shock_value,
            icio_edge_value=float(icio_edge_value),
            supplier_hhi=float(supplier_hhi),
            shocked_supplier_share=float(shocked_supplier_share),
            target_log_gdp_per_capita=target_log_gdp_per_capita,
            target_gdp_growth=target_gdp_growth,
            target_inflation=target_inflation,
            target_unemployment_rate=target_unemployment_rate,
            shock_month=int(inputs.shock_month),
        )
        feature_row = predictor._build_row(  # type: ignore[attr-defined]
            shock_node=inputs.shock_node,
            target_node=target_node,
            target_country=target_country,
            months_after_shock=int(inputs.months_after_shock),
            observation_month=int(obs_month),
            is_domestic=is_domestic,
            shock_yoy_change=float(inputs.shock_yoy_change),
            shock_value=shock_value,
            icio_edge_value=float(icio_edge_value),
            supplier_hhi=float(supplier_hhi),
            shocked_supplier_share=float(shocked_supplier_share),
            target_log_gdp_per_capita=target_log_gdp_per_capita,
            target_gdp_growth=target_gdp_growth,
            target_inflation=target_inflation,
            target_unemployment_rate=target_unemployment_rate,
            shock_month=int(inputs.shock_month),
        ).iloc[0].to_dict()

        # Convert to absolute delta in export value (if we have a baseline + expectation)
        export_delta_pred: Optional[float] = None
        y_pred: Optional[float] = None
        if y_lag_12 is not None and expected_yoy is not None:
            export_delta_pred = PropagationPredictor.convert_dev_to_absolute_delta(
                y_lag_12=float(y_lag_12),
                expected_yoy=float(expected_yoy),
                pred_prop_yoy_dev=float(pred_prop_yoy_dev),
            )
            y_pred = (float(y_expected) + float(export_delta_pred)) if y_expected is not None else None

        rows.append(
            {
                "shock_node": inputs.shock_node,
                "shock_year": int(inputs.shock_year),
                "shock_month": int(inputs.shock_month),
                "observation_key": obs_key,
                "months_after_shock": int(inputs.months_after_shock),
                "target_node": target_node,
                "target_country": target_country,
                "target_sector": target_sector,
                "icio_edge_value": float(icio_edge_value),
                "supplier_hhi": float(supplier_hhi),
                "shocked_supplier_share": float(shocked_supplier_share),
                "icio_weight": float(icio_weight),
                "shock_yoy_change": float(inputs.shock_yoy_change),
                "import_baseline": import_baseline,
                "import_delta": import_delta,
                "shock_value": shock_value,
                "target_log_gdp_per_capita": target_log_gdp_per_capita,
                "target_gdp_growth": target_gdp_growth,
                "target_inflation": target_inflation,
                "target_unemployment_rate": target_unemployment_rate,
                "expected_yoy": expected_yoy,
                "y_lag_12": y_lag_12,
                "y_expected": y_expected,
                "y_pred": y_pred,
                "y_observed": y_observed,
                "pred_prop_yoy_dev": float(pred_prop_yoy_dev),
                "pred_prop_yoy_dev_pp": float(pred_prop_yoy_dev) * 100.0,
                "pred_export_delta": export_delta_pred,
                "model_features": feature_row,
            }
        )

    df_out = pd.DataFrame(rows)
    # Most negative deviation is "worst impact" (below expected growth)
    df_out = df_out.sort_values("pred_prop_yoy_dev", ascending=True).reset_index(drop=True)
    return df_out


def main() -> None:
    ap = argparse.ArgumentParser(description="Predict top-k affected partners for a shock scenario.")
    ap.add_argument("--model-dir", default="models/prop_yoy_dev", help="Directory containing model.joblib + feature_schema.json")
    ap.add_argument("--training-csv", default="training_data_clean.csv", help="CSV used to compute fallback medians")
    ap.add_argument(
        "--embeddings-dir",
        default=str((Path(__file__).resolve().parents[1] / "embeddings").as_posix()),
        help="Directory containing ICIO graphs (graph_YYYY_labeled.pt)",
    )
    ap.add_argument("--shock-node", required=True, help="Shocked node, e.g. JPN_C29")
    ap.add_argument("--shock-month", type=int, required=True, help="Shock month-of-year (1-12)")
    ap.add_argument(
        "--shock-year",
        type=int,
        default=2025,
        help="Shock year (defaults to 2025)",
    )
    ap.add_argument("--shock-yoy-change", type=float, required=True, help="Shock magnitude as YoY change (e.g. -0.25)")
    ap.add_argument("--months-after-shock", type=int, default=1, help="Forecast horizon in months after the shock month")
    ap.add_argument("--history-months", type=int, default=24, help="Months of export history used to estimate expected growth")
    ap.add_argument("--candidates", type=int, default=50, help="How many downstream partners to evaluate before ranking")

    ap.add_argument(
        "--no-network",
        action="store_true",
        help="Disable Comtrade/WorldBank calls (predictions still run but export/value columns will be NA).",
    )
    ap.add_argument(
        "--print-features",
        action="store_true",
        help="Print model feature rows for the displayed top-k results.",
    )
    ap.add_argument("--out-csv", default="", help="Optional: path to save full ranked table as CSV")
    args = ap.parse_args()

    if not (1 <= int(args.shock_month) <= 12):
        raise ValueError("--shock-month must be in [1,12]")

    inputs = ScenarioInputs(
        shock_node=str(args.shock_node),
        shock_year=int(args.shock_year),
        shock_month=int(args.shock_month),
        shock_yoy_change=float(args.shock_yoy_change),
        months_after_shock=int(args.months_after_shock),
        history_months=int(args.history_months),
        candidates=int(args.candidates),
    )

    model_dir = Path(args.model_dir)
    training_csv = Path(args.training_csv)
    embeddings_dir = Path(args.embeddings_dir)

    df = run_scenario(
        model_dir=model_dir,
        training_csv=training_csv,
        embeddings_dir=embeddings_dir,
        inputs=inputs,
        use_network=not bool(args.no_network),
    )

    # Print top-k with a stable set of columns
    cols = [
        "target_node",
        "target_country",
        "icio_edge_value",
        "shocked_supplier_share",
        "supplier_hhi",
        "pred_prop_yoy_dev_pp",
        "y_observed",
        "y_expected",
        "y_pred",
        "pred_export_delta",
    ]
    table = df[cols].copy()
    # readability
    pd.set_option("display.width", 180)
    pd.set_option("display.max_columns", 50)
    pd.set_option("display.float_format", lambda x: f"{x:,.4g}")
    print(table.to_string(index=False))

    if args.print_features:
        print("\nModel features for all rows:")
        feat_rows = df[["target_node", "target_country", "model_features"]]
        # Pretty-print as JSON per row for readability.
        for _, r in feat_rows.iterrows():
            print(f"- {r['target_node']} / {r['target_country']}: {json.dumps(r['model_features'], default=str)}")

    if args.out_csv:
        out_path = Path(args.out_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()


