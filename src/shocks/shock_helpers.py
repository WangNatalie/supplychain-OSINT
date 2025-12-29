from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from statsmodels.tsa.seasonal import STL


def baseline_year_for(shock_year: int) -> int:
    """Avoid COVID-contaminated baselines for recent shocks."""
    return 2019 if shock_year in (2021, 2022, 2023) else shock_year - 1


def month_keys(start_year: int, start_month: int, duration_months: int) -> List[str]:
    """Return a list of 'YYYY-MM' keys starting at (start_year, start_month) for duration_months."""
    keys: List[str] = []
    y = start_year
    m = start_month
    for _ in range(duration_months):
        keys.append(f"{y}-{m:02d}")
        m += 1
        if m > 12:
            m = 1
            y += 1
    return keys


def slice_shock_and_baseline(
    series_all: Dict[str, float],
    shock_keys: List[str],
    baseline_year: int,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Slice a full series into shock-year keys and baseline-year same-month keys."""
    baseline_keys = [f"{baseline_year}-{k.split('-')[1]}" for k in shock_keys]
    shock = {k: series_all[k] for k in shock_keys if k in series_all}
    base = {k: series_all[k] for k in baseline_keys if k in series_all}
    return shock, base


def get_import_shock_series(
    *,
    api,
    is_domestic: bool,
    target_country: str,
    target_node: str,
    shocked_country: str,
    shocked_sector: str,
    shocked_node: str,
    shock_year: int,
    shock_month: int,
    baseline_year: int,
    shock_graph,
    baseline_graph,
    lookback_years: int = 2,
    duration_months_total: int = 36,
    verbose: bool = True,
) -> Optional[Tuple[Dict[str, float], Dict[str, float], List[str], Dict[str, float]]]:
    """
    Build shock/baseline import series with identical shapes for domestic/foreign targets.

    Returns:
      (imports_shock, imports_baseline, shock_keys_12, imports_all)

    Domestic:
      - Uses ICIO annual edge values -> monthly proxy (annual/12 repeated)
      - Validates using annual YoY; returns None if annual_yoy >= 0

    Foreign:
      - Uses Comtrade mirror imports (reporter=target_country, partner=shocked_country) for shocked sector HS codes.
      - Queries a wide range once and slices out shock and baseline windows.
    """
    shock_keys_12 = month_keys(shock_year, shock_month, 12)
    baseline_keys_12 = [f"{baseline_year}-{k.split('-')[1]}" for k in shock_keys_12]

    if is_domestic:
        try:
            shocked_idx = shock_graph.node_id_to_idx[shocked_node]
            target_idx = shock_graph.node_id_to_idx[target_node]
            src_i, tgt_i = shock_graph.edge_index
            mask = (src_i == shocked_idx) & (tgt_i == target_idx)
            shock_edge = shock_graph.value_t[mask].item() if mask.any() else 0.0

            shocked_idx_b = baseline_graph.node_id_to_idx[shocked_node]
            target_idx_b = baseline_graph.node_id_to_idx[target_node]
            src_b, tgt_b = baseline_graph.edge_index
            mask_b = (src_b == shocked_idx_b) & (tgt_b == target_idx_b)
            base_edge = baseline_graph.value_t[mask_b].item() if mask_b.any() else 0.0

            if base_edge <= 0:
                return None

            annual_yoy = (shock_edge - base_edge) / base_edge
            print(f"      ICIO annual YoY: {annual_yoy:+.1%}")
            if annual_yoy >= 0.0:
                print(f"    ⚠️  Annual ICIO shows YoY change of {annual_yoy:+.1%} (recovery dominated), skipping")
                return None

            shock_m = shock_edge / 12.0
            base_m = base_edge / 12.0
            imports_shock = {k: shock_m for k in shock_keys_12}
            imports_base = {k: base_m for k in baseline_keys_12}

            # Synthetic "all series" (needed for downstream decomposition API shape).
            # Since ICIO is annual, this is just a flat proxy series.
            all_keys = month_keys(shock_year - lookback_years, shock_month, duration_months_total)
            imports_all = {k: base_m for k in all_keys}
            for k in shock_keys_12:
                imports_all[k] = shock_m

            return imports_shock, imports_base, shock_keys_12, imports_all
        except Exception as e:
            print(f"    ⚠️  Error processing domestic ICIO shock: {e}")
            return None

    hs_codes = api.get_hs_codes_for_sector(shocked_sector)
    imports_all = api.get_trade_data(
        reporter=target_country,
        partner=shocked_country,
        commodity_code=hs_codes,
        flow_code="M",
        start_year=shock_year - lookback_years,
        start_month=shock_month,
        duration_months=duration_months_total,
        verbose=verbose,
    )

    imports_shock, imports_base = slice_shock_and_baseline(imports_all, shock_keys_12, baseline_year)
    return imports_shock, imports_base, shock_keys_12, imports_all


def _month_diff(start_key: str, end_key: str) -> int:
    """Number of months between YYYY-MM keys: end - start."""
    sy, sm = map(int, start_key.split("-"))
    ey, em = map(int, end_key.split("-"))
    return (ey - sy) * 12 + (em - sm)


def expected_from_pre_shock(
    *,
    series_all: Dict[str, float],
    pre_keys: List[str],
    forecast_keys: List[str],
    seasonal_period: int = 12,
) -> Tuple[Dict[str, float], float]:
    """
    Fit STL on pre-shock window and produce expected values for forecast_keys.

    Returns:
      (expected_by_key, residual_std)
    """
    if not pre_keys:
        return {}, 0.0

    # Build pre-shock series (missing months -> NaN; we'll fill small gaps only)
    idx = pd.to_datetime([f"{k}-01" for k in pre_keys])
    y = np.array([float(series_all[k]) if k in series_all else np.nan for k in pre_keys], dtype=float)
    s = pd.Series(y, index=idx)

    missing_rate = float(s.isna().mean())
    # Too sparse => can't decompose reliably
    if missing_rate > 0.30:
        last_valid = float(s.dropna().iloc[-1]) if s.dropna().size else 0.0
        expected = {k: last_valid for k in forecast_keys}
        return expected, 0.0

    # Fill small gaps only (avoid treating "not reported" as true zeros)
    # - interpolate up to 2-month gaps
    # - then forward/back fill any remaining edge NaNs
    s = s.interpolate(limit=2, limit_direction="both")
    s = s.ffill().bfill()

    # If series is constant or too short, expected is flat
    if len(s) < seasonal_period * 2 or float(s.std()) == 0.0:
        expected = {k: float(s.iloc[-1]) for k in forecast_keys}
        return expected, 0.0

    fit = STL(s, period=seasonal_period, robust=True).fit()
    trend = fit.trend
    seasonal = fit.seasonal
    resid = fit.resid

    resid_std = float(pd.Series(resid).dropna().std()) if resid is not None else 0.0
    resid_std = 0.0 if np.isnan(resid_std) else resid_std

    # Guard against near-zero residual std (z-scores would explode and be meaningless).
    # Use a scale-based floor: if residual std is tiny relative to the series level, treat as 0 (disable z).
    scale = float(s.mean()) if len(s) else 0.0
    scale = abs(scale)
    min_std = max(1.0, scale) * 1e-6
    if resid_std < min_std:
        resid_std = 0.0

    # Fit linear trend on non-nan portion
    t = np.arange(len(trend), dtype=float)
    trend_vals = np.asarray(trend, dtype=float)
    mask = ~np.isnan(trend_vals)
    if mask.sum() >= 2:
        a, b = np.polyfit(t[mask], trend_vals[mask], 1)
    else:
        a, b = float(trend_vals[mask][0]) if mask.sum() == 1 else float(s.mean()), 0.0

    # Seasonal pattern by calendar month (1-12)
    seasonal_vals = pd.Series(np.asarray(seasonal, dtype=float), index=idx)
    seasonal_by_month = seasonal_vals.groupby(seasonal_vals.index.month).mean().to_dict()

    start_key = pre_keys[0]
    expected: Dict[str, float] = {}
    for k in forecast_keys:
        # position relative to pre window start
        pos = float(_month_diff(start_key, k))
        month_num = int(k.split("-")[1])
        seas = float(seasonal_by_month.get(month_num, 0.0))
        expected[k] = float(a + b * pos + seas)

    return expected, resid_std


class ICIOHelper:
    """Utilities for loading and querying ICIO graphs."""

    def __init__(self, icio_dir: Path):
        self.icio_dir = Path(icio_dir)
        self._graph_cache: Dict[int, object] = {}

    def load_graph(self, year: int):
        """Load ICIO graph for given year (cached)."""
        if year in self._graph_cache:
            return self._graph_cache[year]

        graph_path = self.icio_dir / f"graph_{year}_labeled.pt"
        if not graph_path.exists():
            raise FileNotFoundError(f"ICIO graph not found: {graph_path}")

        graph = torch.load(graph_path, map_location="cpu", weights_only=False)
        self._graph_cache[year] = graph
        return graph

    def get_downstream_partners(self, shocked_node: str, year: int, top_k: int = 10) -> List[Dict]:
        """
        Identify top downstream partners from ICIO table.

        NOTE: This returns FOREIGN partners only (targets in a different country than the shocked node).
        NOTE: This returns at most ONE partner per target country (largest edge per country).

        Returns list of dicts:
          - target_node
          - target_country
          - target_sector
          - edge_value
        """
        graph = self.load_graph(year)

        if shocked_node not in graph.node_id_to_idx:
            raise ValueError(f"Node {shocked_node} not found in {year} ICIO table")

        shocked_country = shocked_node.split("_", 1)[0]
        shocked_idx = graph.node_id_to_idx[shocked_node]
        src_indices, tgt_indices = graph.edge_index
        outgoing_mask = src_indices == shocked_idx

        # Keep only the largest edge per target country
        best_by_country: Dict[str, Dict] = {}
        for i in torch.where(outgoing_mask)[0]:
            tgt_idx = tgt_indices[i].item()
            tgt_node = graph.node_labels[tgt_idx]
            edge_value = graph.value_t[i].item()

            if "_" not in tgt_node:
                continue
            tgt_country, tgt_sector = tgt_node.split("_", 1)
            if tgt_country == "ROW":
                continue
            # Keep FOREIGN downstream partners only (skip domestic).
            if tgt_country == shocked_country:
                continue

            rec = {
                "target_node": tgt_node,
                "target_country": tgt_country,
                "target_sector": tgt_sector,
                "edge_value": edge_value,
            }
            prev = best_by_country.get(tgt_country)
            if prev is None or edge_value > float(prev["edge_value"]):
                best_by_country[tgt_country] = rec

        downstream = list(best_by_country.values())
        downstream.sort(key=lambda x: x["edge_value"], reverse=True)
        return downstream[:top_k]

    @staticmethod
    def calculate_industry_weight(
        target_node: str,
        shocked_node: str,
        graph,
    ):
        """
        Weight = ICIO(shocked_node -> target_node) / sum over ICIO(shocked_node -> all industries in target country)
        """
        src_indices, tgt_indices = graph.edge_index
        shocked_idx = graph.node_id_to_idx[shocked_node]

        target_country = target_node.split("_", 1)[0]
        outgoing_mask = src_indices == shocked_idx

        edge_value = 0.0
        total_to_country = 0.0
        for i in torch.where(outgoing_mask)[0]:
            tgt_idx = tgt_indices[i].item()
            tgt_node_temp = graph.node_labels[tgt_idx]
            if tgt_node_temp.startswith(f"{target_country}_"):
                v = graph.value_t[i].item()
                total_to_country += v
                if tgt_node_temp == target_node:
                    edge_value = v

        weight = edge_value / total_to_country if total_to_country > 0 else 0.0
        # print(f"      Industry weight: {weight:.1%} (${edge_value:,.0f} / ${total_to_country:,.0f})")
        return weight


class SupplierMetricsHelper:
    """Supplier concentration and dependency metrics computed from ICIO graphs."""

    def __init__(self, icio_helper: ICIOHelper):
        self.icio_helper = icio_helper
        self._cache: Dict[str, Dict[str, float]] = {}

    def compute_supplier_metrics(self, target_node: str, shocked_node: str, year: int) -> Dict[str, float]:
        cache_key = f"{year}_{target_node}_{shocked_node}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        graph = self.icio_helper.load_graph(year)
        if target_node not in graph.node_id_to_idx:
            result = {"supplier_hhi": 1.0, "shocked_supplier_share": 0.0}
            self._cache[cache_key] = result
            return result

        target_idx = graph.node_id_to_idx[target_node]
        src_indices, tgt_indices = graph.edge_index
        incoming_mask = tgt_indices == target_idx

        suppliers: Dict[str, float] = {}
        shocked_value = 0.0
        for i in torch.where(incoming_mask)[0]:
            src_idx = src_indices[i].item()
            src_node = graph.node_labels[src_idx]
            edge_value = graph.value_t[i].item()
            if edge_value <= 0:
                continue
            suppliers[src_node] = edge_value
            if src_node == shocked_node:
                shocked_value = edge_value

        if not suppliers:
            result = {"supplier_hhi": 1.0, "shocked_supplier_share": 0.0}
        else:
            total_value = sum(suppliers.values())
            shares = [v / total_value for v in suppliers.values()]
            hhi = sum(s**2 for s in shares)
            shocked_share = shocked_value / total_value if total_value > 0 else 0.0
            result = {"supplier_hhi": hhi, "shocked_supplier_share": shocked_share}

        self._cache[cache_key] = result
        return result