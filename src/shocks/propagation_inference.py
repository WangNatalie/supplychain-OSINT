#!/usr/bin/env python3
"""
propagation_inference.py

Lightweight inference helper for the per-month propagation model trained by
`src/shocks/train_propagation_model.py`.

This module predicts the target:
  - prop_yoy_dev: (actual export YoY growth) - (expected export YoY growth)

So a prediction of -0.12 means:
  - export YoY growth is predicted to be 12 percentage points below expected YoY growth.

It supports counterfactual inputs:
  - user-specified shock percentage: maps to feature `shock_yoy_change`
  - user-specified absolute shock delta: maps to feature `shock_value` (Option A: treated as effective delta)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

try:
    import joblib  # type: ignore
except Exception as e:  # pragma: no cover
    raise RuntimeError("joblib is required (usually installed with scikit-learn).") from e


def _parse_sector(node: str) -> str:
    parts = str(node).split("_", 1)
    return parts[1] if len(parts) == 2 else ""


def _parse_country(node: str) -> str:
    parts = str(node).split("_", 1)
    return parts[0] if parts else ""

def _months_bucket(months_after_shock: int) -> str:
    """
    Bucket months_after_shock the same way as training (`add_derived_columns`).
    """
    try:
        mi = int(months_after_shock)
    except Exception:
        return "NA"
    if mi <= 0:
        return "0"
    if mi == 1:
        return "1"
    if mi == 2:
        return "2"
    if 3 <= mi <= 5:
        return "3_5"
    if 6 <= mi <= 8:
        return "6_8"
    if 9 <= mi <= 12:
        return "9_12"
    return "13_plus"


@dataclass(frozen=True)
class PropagationModelPaths:
    model_joblib: Path
    schema_json: Path


class PropagationPredictor:
    """
    Loads a trained sklearn Pipeline and exposes a stable inference API.
    """

    def __init__(self, model_dir: str):
        d = Path(model_dir)
        self.paths = PropagationModelPaths(
            model_joblib=d / "model.joblib",
            schema_json=d / "feature_schema.json",
        )
        if not self.paths.model_joblib.exists():
            raise FileNotFoundError(f"Missing model file: {self.paths.model_joblib}")
        if not self.paths.schema_json.exists():
            raise FileNotFoundError(f"Missing schema file: {self.paths.schema_json}")

        self.model = joblib.load(self.paths.model_joblib)
        with open(self.paths.schema_json, "r") as f:
            self.schema: Dict[str, Any] = json.load(f)

        fs = self.schema.get("feature_schema", {})
        self.numeric_cols = list(fs.get("numeric_cols", []))
        self.categorical_cols = list(fs.get("categorical_cols", []))

        if not self.numeric_cols or not self.categorical_cols:
            raise ValueError(f"Invalid feature schema in {self.paths.schema_json}")

    def _build_row(
        self,
        *,
        shock_node: str,
        target_node: str,
        target_country: str,
        months_after_shock: int,
        observation_month: int,
        is_domestic: bool,
        shock_yoy_change: Optional[float],
        shock_value: Optional[float],
        icio_edge_value: float,
        supplier_hhi: float,
        shocked_supplier_share: float,
        target_log_gdp_per_capita: Optional[float] = None,
        target_gdp_growth: Optional[float] = None,
        target_inflation: Optional[float] = None,
        target_unemployment_rate: Optional[float] = None,
        shock_month: Optional[int] = None,
    ) -> pd.DataFrame:
        shock_yoy_change_f = float(shock_yoy_change) if shock_yoy_change is not None else None
        shock_value_f = float(shock_value) if shock_value is not None else None

        # Derived features (mirror training)
        shock_value_x_shocked_share = (
            float(shock_value_f) * float(shocked_supplier_share)
            if shock_value_f is not None
            else None
        )
        shock_yoy_change_x_shocked_share = (
            float(shock_yoy_change_f) * float(shocked_supplier_share)
            if shock_yoy_change_f is not None
            else None
        )
        shock_value_x_diversification = (
            float(shock_value_f) * (1.0 - float(supplier_hhi))
            if shock_value_f is not None
            else None
        )

        # Derived categorical features
        row: Dict[str, Any] = {
            # Explicit categorical features that were present at training time
            "target_node": str(target_node),
            "shock_node": str(shock_node),
            "target_country": target_country,
            "target_sector": _parse_sector(target_node),
            "shock_country": _parse_country(shock_node),
            "shock_sector": _parse_sector(shock_node),
            "is_domestic": str(bool(is_domestic)),
            "obs_month": f"{int(observation_month):02d}",
            "months_bucket": _months_bucket(months_after_shock),
            # Numeric
            "shock_yoy_change": shock_yoy_change_f,
            "shock_value": shock_value_f,
            "shock_value_x_shocked_share": shock_value_x_shocked_share,
            "shock_yoy_change_x_shocked_share": shock_yoy_change_x_shocked_share,
            "shock_value_x_diversification": shock_value_x_diversification,
            "icio_edge_value": icio_edge_value,
            "supplier_hhi": supplier_hhi,
            "shocked_supplier_share": shocked_supplier_share,
            "months_after_shock": months_after_shock,
            "shock_month": float(shock_month) if shock_month is not None else None,
            "target_log_gdp_per_capita": target_log_gdp_per_capita,
            "target_gdp_growth": target_gdp_growth,
            "target_inflation": target_inflation,
            "target_unemployment_rate": target_unemployment_rate,
        }

        # Ensure column order matches training selection (pipeline uses names)
        cols = self.numeric_cols + self.categorical_cols
        missing = [c for c in cols if c not in row]
        if missing:
            raise ValueError(f"Inference row is missing required features: {missing}")

        return pd.DataFrame([{c: row.get(c) for c in cols}])

    def predict_prop_yoy_dev(
        self,
        *,
        shock_node: str,
        target_node: str,
        target_country: str,
        months_after_shock: int,
        observation_month: int,
        is_domestic: bool,
        # user-specified counterfactual shock severity
        shock_yoy_change: Optional[float] = None,
        shock_value: Optional[float] = None,
        # exposure/static partner features
        icio_edge_value: float,
        supplier_hhi: float,
        shocked_supplier_share: float,
        # optional macro controls
        target_log_gdp_per_capita: Optional[float] = None,
        target_gdp_growth: Optional[float] = None,
        target_inflation: Optional[float] = None,
        target_unemployment_rate: Optional[float] = None,
        # optional: if you know shock month-of-year
        shock_month: Optional[int] = None,
    ) -> float:
        """
        Returns predicted prop_yoy_dev (percentage points, e.g. -0.12 = -12pp).

        Note: This only predicts deviation from expected YoY growth. To convert to absolute
        trade value changes, you must also supply baseline export levels + expected YoY growth.
        """
        X = self._build_row(
            shock_node=shock_node,
            target_node=target_node,
            target_country=target_country,
            months_after_shock=months_after_shock,
            observation_month=observation_month,
            is_domestic=is_domestic,
            shock_yoy_change=shock_yoy_change,
            shock_value=shock_value,
            icio_edge_value=icio_edge_value,
            supplier_hhi=supplier_hhi,
            shocked_supplier_share=shocked_supplier_share,
            target_log_gdp_per_capita=target_log_gdp_per_capita,
            target_gdp_growth=target_gdp_growth,
            target_inflation=target_inflation,
            target_unemployment_rate=target_unemployment_rate,
            shock_month=shock_month,
        )

        pred = float(self.model.predict(X)[0])
        return pred

    @staticmethod
    def convert_dev_to_absolute_delta(
        *,
        y_lag_12: float,
        expected_yoy: float,
        pred_prop_yoy_dev: float,
    ) -> float:
        """
        Convert predicted deviation-in-growth into an absolute delta (in the same units as y_lag_12).

        Inputs:
          - y_lag_12: export level at t-12 (last year, same month)
          - expected_yoy: expected YoY growth rate for t (from your expected model)
          - pred_prop_yoy_dev: predicted deviation vs expected YoY (this model output)
        """
        pred_yoy = expected_yoy + pred_prop_yoy_dev
        y_expected = y_lag_12 * (1.0 + expected_yoy)
        y_pred = y_lag_12 * (1.0 + pred_yoy)
        return y_pred - y_expected


