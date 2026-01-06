#!/usr/bin/env python3
"""
live_partner_impact.py

Scenario runner on top of `propagation_inference.py`:
- user provides: shocked node, shock month (present year by default), shock magnitude as realized YoY
- we convert realized YoY into a shock-only deviation vs expected YoY per target, and propagate deviations across hops
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
import json
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from propagation_inference import PropagationPredictor
from ICIO.ICIO_parser import format_node_name 
from shock_helpers import (
    ICIOHelper,
    SupplierMetricsHelper,
    expected_from_pre_shock,
    month_keys,
)

from ComtradeAPI import ComtradeAPI  
from world_data import load_indicators  


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
    # User input at hop-0: realized YoY (vs observed t-12). At hop>=1 we pass deviations.
    shock_yoy_actual: float
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





def _summarize_impacts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate impacts across hops: sum shock-attributable level deltas per target,
    count occurrences, and collect hop levels.

    Summary intentionally excludes target_country (requested).
    """
    agg: Dict[str, Dict[str, object]] = {}
    for _, r in df.iterrows():
        # Only keep nodes/rows with negative predicted YoY deviation (below expected growth).
        dev = r.get("pred_prop_yoy_dev")
        if dev is None or pd.isna(dev) or float(dev) >= 0:
            continue

        node = r.get("target_node")
        if node is None:
            continue
        node = str(node)
        delta = r.get("shock_only_delta")
        hop = int(r.get("hop", 0)) if r.get("hop") is not None else 0
        if delta is None or pd.isna(delta):
            continue
        entry = agg.setdefault(node, {"total_shock_only_delta": 0.0, "count": 0, "hops": set()})
        entry["total_shock_only_delta"] = float(entry["total_shock_only_delta"]) + float(delta)
        entry["count"] = int(entry["count"]) + 1
        entry["hops"].add(hop)

    rows: List[Dict[str, object]] = []
    for node, info in agg.items():
        rows.append(
            {
                "target_name": format_node_name(node),
                "total_shock_only_delta": float(info["total_shock_only_delta"]),
                "occurrences": int(info["count"]),
                "hops": sorted(info["hops"]),
            }
        )
    if not rows:
        return pd.DataFrame()
    # Most negative total shock effect first
    return pd.DataFrame(rows).sort_values("total_shock_only_delta", ascending=True).reset_index(drop=True)

def run_scenario(
    *,
    model_dir: Path,
    training_csv: Path,
    embeddings_dir: Path,
    inputs: ScenarioInputs,
    use_network: bool,
    hop: int = 0,
    predictor: Optional[PropagationPredictor] = None,
    icio: Optional[ICIOHelper] = None,
    supplier: Optional[SupplierMetricsHelper] = None,
    graph=None,
    icio_year: Optional[int] = None,
    shock_is_dev: bool = False,
) -> pd.DataFrame:
    # Model
    predictor = predictor or PropagationPredictor(str(model_dir))

    # ICIO features
    icio = icio or ICIOHelper(embeddings_dir)
    icio_year = icio_year or min(inputs.shock_year, _latest_graph_year(embeddings_dir))
    graph = graph or icio.load_graph(icio_year)
    supplier = supplier or SupplierMetricsHelper(icio)

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
        # Supplier metrics
        sm = supplier.compute_supplier_metrics(target_node, inputs.shock_node, icio_year)
        supplier_hhi = float(sm.get("supplier_hhi", 1.0))
        shocked_supplier_share = float(sm.get("shocked_supplier_share", 0.0))

        # ICIO allocation weight (used when turning import delta into a target-sector shock_value)
        icio_weight = float(ICIOHelper.calculate_industry_weight(target_node, inputs.shock_node, graph))

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

        # Shock definition:
        # - hop 0: user provides realized YoY; we compute expected import YoY and convert to deviation (shock-only)
        # - hop>=1: we pass shock_yoy_dev directly (already a deviation)
        shock_value: Optional[float] = None
        import_baseline: Optional[float] = None
        import_delta: Optional[float] = None
        shock_yoy_expected: Optional[float] = None
        shock_yoy_dev: Optional[float] = None
        if api is not None:
            # Need baseline for same month last year to convert user yoy->absolute delta.
            # Baseline is t-12 relative to the observation month (handles year rollovers).
            baseline_year = obs_year - 1
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
            if import_baseline is None:
                # If we can't anchor the absolute shock value, skip this node entirely.
                print(f"      [SKIP] Missing import baseline for {target_country}<-{shocked_country} at {baseline_key}; skipping node")
                continue

            if shock_is_dev:
                shock_yoy_dev = float(inputs.shock_yoy_actual)
            else:
                # Compute expected import YoY (for this target importer / shocked exporter / sector) from a pre-shock window.
                # This lets us convert the user's realized YoY into a shock-only deviation, consistent with training.
                pre_keys_imp = month_keys(inputs.shock_year - 2, inputs.shock_month, 24)
                imp_hist = api.get_trade_data(
                    reporter=target_country,
                    partner=shocked_country,
                    sector_code=shocked_sector,
                    flow_code="M",
                    start_year=inputs.shock_year - 2,
                    start_month=inputs.shock_month,
                    duration_months=24,
                    verbose=False,
                )
                imp_expected_map, _ = expected_from_pre_shock(
                    series_all=imp_hist,
                    pre_keys=pre_keys_imp,
                    forecast_keys=[obs_key],
                )
                imp_expected = imp_expected_map.get(obs_key)
                shock_yoy_expected = (
                    (float(imp_expected) - float(import_baseline)) / float(import_baseline)
                    if (imp_expected is not None and float(import_baseline) > 0)
                    else None
                )
                shock_yoy_dev = (
                    float(inputs.shock_yoy_actual) - float(shock_yoy_expected)
                    if shock_yoy_expected is not None
                    else None
                )

            if shock_yoy_dev is not None and float(import_baseline) > 0:
                # Option A: shock_value is the shock-only level delta (vs expected), allocated by ICIO weight.
                import_delta = float(import_baseline) * float(shock_yoy_dev)
                shock_value = float(import_delta) * float(icio_weight)

        # Export history for target's TOTAL exports (sector-level) to the world
        export_series: Dict[str, float] = {}
        expected_yoy: Optional[float] = None
        pred_yoy: Optional[float] = None
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
            y_lag_12 = _value(export_series, lag_key)
            y_observed = _value(export_series, obs_key)
            export_expected = export_expected_map.get(obs_key)
            # Realized baseline anchoring: expected_yoy is computed vs the OBSERVED t-12 value (y_lag_12),
            # not vs an "expected lag" which may be inaccurate.
            expected_yoy = (
                (float(export_expected) - float(y_lag_12)) / float(y_lag_12)
                if (export_expected is not None and y_lag_12 is not None and float(y_lag_12) > 0)
                else None
            )
            y_expected = float(export_expected) if export_expected is not None else None

        # Predict propagation deviation
        pred_prop_yoy_dev = predictor.predict_prop_yoy_dev(
            shock_node=inputs.shock_node,
            target_node=target_node,
            target_country=target_country,
            months_after_shock=int(inputs.months_after_shock),
            observation_month=int(obs_month),
            shock_yoy_dev=shock_yoy_dev,
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
            shock_yoy_dev=shock_yoy_dev,
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
        shock_only_delta: Optional[float] = None
        # Realized impact focus:
        # - define predicted YoY vs observed t-12 baseline
        # - compute predicted level directly from y_lag_12 (no counterfactual delta vs "expected level" needed)
        if y_lag_12 is not None and expected_yoy is not None:
            pred_yoy = float(expected_yoy) + float(pred_prop_yoy_dev)
            y_pred = float(y_lag_12) * (1.0 + float(pred_yoy))
            export_delta_pred = float(y_pred) - float(y_lag_12)
        # Shock-attributable level delta: (y_pred - y_expected) = y_lag_12 * pred_prop_yoy_dev
        if y_lag_12 is not None and pred_prop_yoy_dev is not None:
            shock_only_delta = float(y_lag_12) * float(pred_prop_yoy_dev)

        rows.append(
            {
                "source_shock_node": inputs.shock_node,
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
                "shock_yoy_actual": float(inputs.shock_yoy_actual),
                "shock_yoy_expected": shock_yoy_expected,
                "shock_yoy_dev": shock_yoy_dev,
                "import_baseline": import_baseline,
                "import_delta": import_delta,
                "shock_value": shock_value,
                "target_log_gdp_per_capita": target_log_gdp_per_capita,
                "target_gdp_growth": target_gdp_growth,
                "target_inflation": target_inflation,
                "target_unemployment_rate": target_unemployment_rate,
                "expected_yoy": expected_yoy,
                "pred_yoy_change": pred_yoy,
                "y_lag_12": y_lag_12,
                "y_expected": y_expected,
                "y_pred": y_pred,
                "y_observed": y_observed,
                "pred_prop_yoy_dev": float(pred_prop_yoy_dev),
                "pred_prop_yoy_dev_pp": float(pred_prop_yoy_dev) * 100.0,
                "pred_export_delta": export_delta_pred,
                "shock_only_delta": shock_only_delta,
                "model_features": feature_row,
            }
        )

    df_out = pd.DataFrame(rows)
    df_out["hop"] = hop
    # Most negative deviation is "worst impact" (below expected growth)
    df_out = df_out.sort_values("pred_prop_yoy_dev", ascending=True).reset_index(drop=True)
    return df_out


def run_multihop_scenario(
    *,
    model_dir: Path,
    training_csv: Path,
    embeddings_dir: Path,
    inputs: ScenarioInputs,
    use_network: bool,
    hops: int,
) -> pd.DataFrame:
    """
    Propagate shocks up to `hops` levels.

    Gating rule (requested): only nodes with negative predicted YoY *deviation* spawn further propagation:
      pred_prop_yoy_dev < 0
    """
    predictor = PropagationPredictor(str(model_dir))
    icio = ICIOHelper(embeddings_dir)
    icio_year = min(inputs.shock_year, _latest_graph_year(embeddings_dir))
    graph = icio.load_graph(icio_year)
    supplier = SupplierMetricsHelper(icio)

    active: Dict[str, Dict[str, object]] = {
        inputs.shock_node: {
            "shock_node": inputs.shock_node,
            "shock_yoy_dev": inputs.shock_yoy_actual,
        }
    }
    all_rows: List[pd.DataFrame] = []

    for hop in range(max(1, int(hops))):
        if not active:
            break
        next_active: Dict[str, Dict[str, object]] = {}
        for shock in active.values():
            hop_inputs = ScenarioInputs(
                shock_node=str(shock["shock_node"]),
                shock_year=inputs.shock_year,
                shock_month=inputs.shock_month,
                shock_yoy_actual=float(shock["shock_yoy_dev"]),
                months_after_shock=inputs.months_after_shock,
                history_months=inputs.history_months,
                candidates=inputs.candidates,
            )
            df_hop = run_scenario(
                model_dir=model_dir,
                training_csv=training_csv,
                embeddings_dir=embeddings_dir,
                inputs=hop_inputs,
                use_network=use_network,
                hop=hop,
                predictor=predictor,
                icio=icio,
                supplier=supplier,
                graph=graph,
                icio_year=icio_year,
                shock_is_dev=True,
            )
            all_rows.append(df_hop)

            for _, r in df_hop.iterrows():
                dev = r.get("pred_prop_yoy_dev")
                if dev is None or pd.isna(dev) or float(dev) >= 0:
                    continue  # only propagate negative deviation

                target_node = str(r["target_node"])
                # Propagate the shock-attributable component (growth deviation), not the full predicted YoY.
                # This aligns the gating rule (dev < 0) with what we pass downstream.
                pred_yoy = float(dev)
                # If multiple upstream shocks point to same node, propagate the most negative YoY.
                existing = next_active.get(target_node)
                if existing is None or float(pred_yoy) < float(existing["shock_yoy_dev"]):
                    next_active[target_node] = {
                        "shock_node": target_node,
                        "shock_yoy_dev": float(pred_yoy),
                    }
        active = next_active

    if not all_rows:
        return pd.DataFrame()
    return pd.concat(all_rows, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Predict top-k affected partners for a shock scenario.")
    ap.add_argument("--model-dir", default="shocks/models/prop_yoy_dev", help="Directory containing model.joblib + feature_schema.json")
    ap.add_argument("--training-csv", default="shocks/training_data_clean.csv", help="CSV used to compute fallback medians")
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
    ap.add_argument("--shock-yoy-change", type=float, required=True, help="Shock magnitude as realized YoY change (e.g. -0.25). Converted to deviation internally.")
    ap.add_argument("--months-after-shock", type=int, default=1, help="Forecast horizon in months after the shock month")
    ap.add_argument("--history-months", type=int, default=24, help="Months of export history used to estimate expected growth")
    ap.add_argument("--candidates", type=int, default=50, help="How many downstream partners to evaluate before ranking")
    ap.add_argument("--hops", type=int, default=1, help="Number of propagation hops (>=1). Only negative shocks propagate.")

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
        shock_yoy_actual=float(args.shock_yoy_change),
        months_after_shock=int(args.months_after_shock),
        history_months=int(args.history_months),
        candidates=int(args.candidates),
    )

    model_dir = Path(args.model_dir)
    training_csv = Path(args.training_csv)
    embeddings_dir = Path(args.embeddings_dir)

    if int(args.hops) <= 1:
        df = run_scenario(
            model_dir=model_dir,
            training_csv=training_csv,
            embeddings_dir=embeddings_dir,
            inputs=inputs,
            use_network=not bool(args.no_network),
            hop=0,
            shock_is_dev=False,
        )
    else:
        df = run_multihop_scenario(
            model_dir=model_dir,
            training_csv=training_csv,
            embeddings_dir=embeddings_dir,
            inputs=inputs,
            use_network=not bool(args.no_network),
            hops=int(args.hops),
        )

    # Print top-k with a stable set of columns
    cols = [
        "target_name",
        "shocked_supplier_share",
        "supplier_hhi",
        "pred_prop_yoy_dev_pp",
        "y_observed",
        "y_pred",
        "y_lag_12",
        "pred_yoy_change",
        "shock_only_delta",
    ]
    display_cols = cols + (["hop"] if "hop" in df.columns else [])
    df = df.copy()
    df["target_name"] = df["target_node"].map(lambda n: format_node_name(str(n)) if (n is not None and pd.notna(n)) else "NA")
    table = df[display_cols].copy()
    table["pred_yoy_change"] = table["pred_yoy_change"].map(
        lambda v: f"{float(v)*100.0:+.2f}%" if (v is not None and pd.notna(v)) else "NA"
    )
    table["pred_prop_yoy_dev_pp"] = table["pred_prop_yoy_dev_pp"].map(
        lambda v: f"{float(v):.2f}%" if (v is not None and pd.notna(v)) else "NA"
    )
    table["shocked_supplier_share"] = table["shocked_supplier_share"].map(
        lambda v: f"{float(v)*100.0:+.2f}%" if (v is not None and pd.notna(v)) else "NA"
    )

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

    summary = _summarize_impacts(df)
    if not summary.empty:
        top_summary = summary.head(20)
        print("\nPropagation summary (top 20 most negative total impact):")
        print(top_summary.to_string(index=False))

    if args.out_csv:
        out_path = Path(args.out_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()


