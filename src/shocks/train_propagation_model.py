#!/usr/bin/env python3
"""
train_propagation_model.py

Train a per-month shock propagation model from samples produced by `build_shock_dataset.py`.

Target (label): `prop_yoy_dev`
Interpretation: percentage-point deviation of export YoY growth from expected YoY growth.

Important: this script builds a leakage-safe feature set by excluding any columns derived from
target exports at time t (e.g., export_observed/export_baseline/prop_*actual/prop_*expected).
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Avoid OpenMP shared-memory issues in constrained runtimes by forcing single-threaded math
# and disabling KMP shared-memory usage.
os.environ.setdefault("KMP_USE_SHM", "0")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

try:
    import joblib  # type: ignore
except Exception as e:  # pragma: no cover
    raise RuntimeError("joblib is required (usually installed with scikit-learn).") from e


TARGET_COL = "prop_yoy_dev"
GROUP_COL = "shock_event"


@dataclass(frozen=True)
class FeatureSchema:
    """
    Explicit, leakage-safe feature schema.

    We intentionally avoid raw identifiers like `shock_event` and `target_node` because they
    are high-cardinality and can cause overfitting and poor generalization. Instead we use
    country/sector components extracted from node strings.
    """

    numeric_cols: Tuple[str, ...]
    categorical_cols: Tuple[str, ...]
    derived_cols: Tuple[str, ...]

    def to_dict(self) -> Dict[str, List[str]]:
        return {
            "numeric_cols": list(self.numeric_cols),
            "categorical_cols": list(self.categorical_cols),
            "derived_cols": list(self.derived_cols),
        }


FEATURE_SCHEMA = FeatureSchema(
    # User-controllable shock inputs
    numeric_cols=(
        "shock_yoy_change",  # user-specified shock % (YoY vs baseline)
        "shock_value",  # user-specified absolute delta (compatible with dataset)
        # exposure / network
        "icio_edge_value",
        "supplier_hhi",
        "shocked_supplier_share",
        # time controls
        "months_after_shock",
        "shock_month",
        # macro controls (may be missing)
        "target_log_gdp_per_capita",
        "target_gdp_growth",
        "target_inflation",
        "target_unemployment_rate",
    ),
    categorical_cols=(
        "target_country",
        "target_sector",
        "shock_country",
        "shock_sector",
        "is_domestic",
        "obs_month",
    ),
    derived_cols=(
        "shock_country",
        "shock_sector",
        "target_sector",
        "shock_month",
        "obs_month",
    ),
)


def _parse_sector(node: str) -> str:
    # node examples: "USA_C26", "CHN_J62_63"
    parts = str(node).split("_", 1)
    return parts[1] if len(parts) == 2 else ""


def _parse_country(node: str) -> str:
    parts = str(node).split("_", 1)
    return parts[0] if parts else ""


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["shock_country"] = df["shock_node"].map(_parse_country)
    df["shock_sector"] = df["shock_node"].map(_parse_sector)
    df["target_sector"] = df["target_node"].map(_parse_sector)

    # Extract shock month and observation month-of-year (seasonality control)
    # shock_date, observation_date are "YYYY-MM"
    df["shock_month"] = df["shock_date"].astype(str).str.split("-").str[1].astype(float)
    df["obs_month"] = df["observation_date"].astype(str).str.split("-").str[1]

    # Stabilize types
    df["is_domestic"] = df["is_domestic"].astype(str)

    return df


def leakage_safe_filter(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop obvious label-side/leakage columns (kept in file but never used as features).
    """
    # Note: we don't strictly need to drop here since we select explicit feature cols.
    # This is a guard to prevent accidental inclusion in future edits.
    banned_prefixes = (
        "prop_",  # includes prop_*actual/expected/dev/resid_std
    )
    banned_cols = {
        "export_baseline",
        "export_observed",
        "yoy_change",  # alternative target; do not use when predicting prop_yoy_dev
    }
    drop_cols = []
    for c in df.columns:
        if c in banned_cols:
            drop_cols.append(c)
        elif any(c.startswith(p) for p in banned_prefixes):
            # keep the target itself
            if c != TARGET_COL:
                drop_cols.append(c)
    return df.drop(columns=drop_cols, errors="ignore")


def build_xy_groups(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    df = add_derived_columns(df)
    df = leakage_safe_filter(df)

    required = {TARGET_COL, GROUP_COL}.union(FEATURE_SCHEMA.numeric_cols).union(FEATURE_SCHEMA.categorical_cols)
    missing = sorted([c for c in required if c not in df.columns])
    if missing:
        raise ValueError(f"Missing required columns in dataset: {missing}")

    # Keep only rows where label exists
    df = df[df[TARGET_COL].notna()].copy()

    X = df[list(FEATURE_SCHEMA.numeric_cols) + list(FEATURE_SCHEMA.categorical_cols)].copy()
    y = df[TARGET_COL].astype(float)
    groups = df[GROUP_COL].astype(str)
    return X, y, groups


def make_preprocessor(*, scale_numeric: bool) -> ColumnTransformer:
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler(with_mean=False)))
    numeric = Pipeline(steps=numeric_steps)
    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric, list(FEATURE_SCHEMA.numeric_cols)),
            ("cat", categorical, list(FEATURE_SCHEMA.categorical_cols)),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )


def make_models(random_state: int = 42) -> Dict[str, Pipeline]:
    models: Dict[str, Pipeline] = {}

    # Linear baseline: requires scaling to avoid huge-magnitude numeric features dominating.
    models["ridge"] = Pipeline(
        steps=[
            ("pre", make_preprocessor(scale_numeric=True)),
            ("model", Ridge(alpha=10.0, random_state=random_state)),
        ]
    )

    # Avoid HistGradientBoostingRegressor here because it can rely on OpenMP runtime
    # features (shared memory) that may not be available in some constrained environments.
    models["gbrt"] = Pipeline(
        steps=[
            ("pre", make_preprocessor(scale_numeric=False)),
            (
                "model",
                GradientBoostingRegressor(
                    loss="huber",
                    learning_rate=0.05,
                    max_depth=3,
                    n_estimators=500,
                    subsample=0.9,
                    random_state=random_state,
                ),
            ),
        ]
    )

    # Bagging baseline (often strong for small-ish tabular data, avoids OpenMP issues)
    # Note: RF can still be heavier; default n_estimators kept modest.
    try:
        from sklearn.ensemble import RandomForestRegressor  # local import to keep dependencies minimal

        models["rf"] = Pipeline(
            steps=[
                ("pre", make_preprocessor(scale_numeric=False)),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=400,
                        max_depth=None,
                        min_samples_leaf=10,
                        n_jobs=1,
                        random_state=random_state,
                    ),
                ),
            ]
        )
    except Exception:
        pass

    return models


def eval_regression(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mae = float(mean_absolute_error(y_true, y_pred))
    # Backwards-compatible RMSE: older sklearn versions don't support squared=False.
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(math.sqrt(mse))
    r2 = float(r2_score(y_true, y_pred))
    # Spearman can return nan if constant; coerce.
    sp = spearmanr(y_true, y_pred).correlation
    spearman = float(sp) if sp == sp else float("nan")
    return {"mae": mae, "rmse": rmse, "r2": r2, "spearman": spearman}


def slice_metrics(df_eval: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """
    Produce a small set of sliced metrics for debugging stability.
    df_eval requires: y_true, y_pred, months_after_shock, is_domestic, shocked_supplier_share
    """
    out: Dict[str, Dict[str, float]] = {}

    def _add(name: str, sub: pd.DataFrame) -> None:
        if len(sub) < 5:
            return
        out[name] = eval_regression(sub["y_true"].to_numpy(), sub["y_pred"].to_numpy())

    # months_after_shock buckets
    bins = [-999, 2, 5, 11, 999]
    labels = ["m0_2", "m3_5", "m6_11", "m12_plus"]
    df_eval = df_eval.copy()
    df_eval["bucket"] = pd.cut(df_eval["months_after_shock"], bins=bins, labels=labels)
    for b in labels:
        _add(f"months_{b}", df_eval[df_eval["bucket"] == b])

    # domestic vs foreign
    _add("domestic_true", df_eval[df_eval["is_domestic"] == "True"])
    _add("domestic_false", df_eval[df_eval["is_domestic"] == "False"])

    # high vs low exposure to shocked supplier (median split)
    med = float(df_eval["shocked_supplier_share"].median())
    _add("shocked_share_low", df_eval[df_eval["shocked_supplier_share"] <= med])
    _add("shocked_share_high", df_eval[df_eval["shocked_supplier_share"] > med])

    return out


def group_split(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    *,
    test_size: float,
    random_state: int,
) -> Tuple[np.ndarray, np.ndarray]:
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))
    return train_idx, test_idx


def filter_and_clip_labels(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    *,
    max_abs: Optional[float],
    clip_abs: Optional[float],
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    prop_yoy_dev can explode when last-year exports are tiny (YoY denominators near 0).
    We keep training stable by optionally dropping and/or clipping extreme labels.

    - max_abs: drop rows with |y| > max_abs
    - clip_abs: clamp y into [-clip_abs, +clip_abs]
    """
    X2, y2, g2 = X.copy(), y.copy(), groups.copy()

    if max_abs is not None:
        keep = y2.abs() <= float(max_abs)
        X2, y2, g2 = X2.loc[keep], y2.loc[keep], g2.loc[keep]

    if clip_abs is not None:
        c = float(clip_abs)
        y2 = y2.clip(lower=-c, upper=c)

    return X2, y2, g2


def main() -> None:
    parser = argparse.ArgumentParser(description="Train shock propagation model (target=prop_yoy_dev)")
    parser.add_argument(
        "--data",
        default=str(Path(__file__).parent / "training_data.csv"),
        help="Path to training data CSV produced by build_shock_dataset.py",
    )
    parser.add_argument(
        "--outdir",
        default=str(Path(__file__).parent / "models" / "prop_yoy_dev"),
        help="Directory to save model artifacts",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument(
        "--label-max-abs",
        type=float,
        default=5.0,
        help="Drop rows with |prop_yoy_dev| greater than this (default 5.0 = 500pp). Set 0 to disable.",
    )
    parser.add_argument(
        "--label-clip-abs",
        type=float,
        default=2.0,
        help="Clip prop_yoy_dev to [-X, X] after dropping extremes (default 2.0 = 200pp). Set 0 to disable.",
    )
    args = parser.parse_args()

    data_path = Path(args.data)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"Loading data: {data_path}")
    df = pd.read_csv(data_path)

    X, y, groups = build_xy_groups(df)
    max_abs = None if args.label_max_abs == 0 else float(args.label_max_abs)
    clip_abs = None if args.label_clip_abs == 0 else float(args.label_clip_abs)
    X, y, groups = filter_and_clip_labels(X, y, groups, max_abs=max_abs, clip_abs=clip_abs)
    print(f"Rows with label ({TARGET_COL}): {len(y):,}")
    print(f"Unique shock events: {groups.nunique():,}")
    if len(y) > 0:
        print(
            f"Label stats: mean={y.mean():+.4f} median={y.median():+.4f} "
            f"p05={y.quantile(0.05):+.4f} p95={y.quantile(0.95):+.4f}"
        )

    # Keep a copy of non-feature cols needed for sliced eval
    # Eval dataframe aligned to filtered rows (same index selection as X/y).
    df_eval_base = add_derived_columns(df).loc[X.index].copy()

    train_idx, test_idx = group_split(X, y, groups, test_size=args.test_size, random_state=args.seed)
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    print(f"Split: train={len(train_idx):,} rows, test={len(test_idx):,} rows")

    # Baseline: always 0 (no deviation beyond expected)
    y_pred_0 = np.zeros_like(y_test.to_numpy(), dtype=float)
    baseline_metrics = eval_regression(y_test.to_numpy(), y_pred_0)
    print(f"Baseline (predict 0): MAE={baseline_metrics['mae']:.4f}, RMSE={baseline_metrics['rmse']:.4f}, R2={baseline_metrics['r2']:.4f}")

    models = make_models(random_state=args.seed)
    results: Dict[str, Dict[str, object]] = {}

    best_name: Optional[str] = None
    best_mae: float = float("inf")
    best_model: Optional[Pipeline] = None

    for name, model in models.items():
        print(f"\nTraining model: {name}")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        metrics = eval_regression(y_test.to_numpy(), y_pred)

        # sliced eval needs: months_after_shock, is_domestic, shocked_supplier_share
        df_eval = df_eval_base.iloc[test_idx].copy()
        df_eval["y_true"] = y_test.to_numpy()
        df_eval["y_pred"] = y_pred
        sliced = slice_metrics(df_eval)

        results[name] = {
            "metrics": metrics,
            "sliced": sliced,
        }
        print(f"  MAE={metrics['mae']:.4f} RMSE={metrics['rmse']:.4f} R2={metrics['r2']:.4f} Spearman={metrics['spearman']:.4f}")

        if metrics["mae"] < best_mae:
            best_mae = metrics["mae"]
            best_name = name
            best_model = model

    if best_model is None or best_name is None:
        raise RuntimeError("No model trained successfully.")

    print(f"\nBest model: {best_name} (MAE={best_mae:.4f})")

    # Save artifacts
    model_path = outdir / "model.joblib"
    schema_path = outdir / "feature_schema.json"
    metrics_path = outdir / "metrics.json"

    joblib.dump(best_model, model_path)
    with open(schema_path, "w") as f:
        json.dump(
            {
                "target": TARGET_COL,
                "group_col": GROUP_COL,
                "feature_schema": FEATURE_SCHEMA.to_dict(),
                "selected_model": best_name,
            },
            f,
            indent=2,
        )
    with open(metrics_path, "w") as f:
        json.dump({"baseline_zero": baseline_metrics, "models": results}, f, indent=2)

    print(f"\nSaved model: {model_path}")
    print(f"Saved schema: {schema_path}")
    print(f"Saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()


