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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, LeaveOneGroupOut
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
        "shock_value_x_shocked_share",
        "shock_yoy_change_x_shocked_share",
        "shock_value_x_diversification",
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
        "target_node",
        "shock_country",
        "shock_sector",
        "shock_node",
        "obs_month",
        "months_bucket",
    ),
    derived_cols=(
        "shock_country",
        "shock_sector",
        "target_sector",
        "shock_month",
        "obs_month",
        "months_bucket", 
        "shock_value_x_shocked_share",
        "shock_yoy_change_x_shocked_share",
        "shock_value_x_diversification",
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

    # Bin months_after_shock into coarse buckets for stability.
    # Buckets: 0,1,2,3–5,6–8,9–12 (anything >12 falls into "13_plus").
    def _bucket(m: float) -> str:
        try:
            mi = int(m)
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

    df["months_bucket"] = df["months_after_shock"].map(_bucket)

    # Interaction features (requested)
    # - shock_value * shocked_supplier_share
    # - shock_yoy_change * shocked_supplier_share
    # - shock_value * (1 - supplier_hhi)
    df["shock_value_x_shocked_share"] = df["shock_value"].astype(float) * df["shocked_supplier_share"].astype(float)
    df["shock_yoy_change_x_shocked_share"] = df["shock_yoy_change"].astype(float) * df["shocked_supplier_share"].astype(float)
    df["shock_value_x_diversification"] = df["shock_value"].astype(float) * (1.0 - df["supplier_hhi"].astype(float))

    # Stabilize types
    df["shock_node"] = df["shock_node"].astype(str)
    df["target_node"] = df["target_node"].astype(str)

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
            # Never drop the active label column, even if it is normally banned.
            if c != TARGET_COL:
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

    from sklearn.linear_model import Ridge  # type: ignore
    models["ridge"] = Pipeline(
        steps=[
            ("pre", make_preprocessor(scale_numeric=True)),
            ("model", Ridge(alpha=10.0, random_state=random_state)),
        ]
    )
    
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


    from sklearn.ensemble import RandomForestRegressor  # type: ignore

    models["rf"] = Pipeline(
        steps=[
            ("pre", make_preprocessor(scale_numeric=False)),
            ("model", RandomForestRegressor(n_estimators=400, max_depth=None, min_samples_leaf=10, n_jobs=1, random_state=random_state)),
        ]
    )


    from catboost import CatBoostRegressor  # type: ignore

    models["catboost"] = Pipeline(
        steps=[
            ("pre", make_preprocessor(scale_numeric=False)),
            (
                "model",
                CatBoostRegressor(
                    loss_function="MAE",
                    depth=6,
                    learning_rate=0.05,
                    iterations=2000,
                    random_seed=random_state,
                    verbose=False,
                ),
            ),
        ]
    )

    from xgboost import XGBRegressor  # type: ignore

    models["xgboost"] = Pipeline(
        steps=[
            ("pre", make_preprocessor(scale_numeric=False)),
            (
                "model",
                XGBRegressor(
                    n_estimators=1500,
                    learning_rate=0.03,
                    max_depth=6,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    reg_alpha=0.0,
                    reg_lambda=1.0,
                    objective="reg:squarederror",
                    n_jobs=1,
                    random_state=random_state,
                ),
            ),
        ]
    )



    return models


def eval_regression(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mae = float(mean_absolute_error(y_true, y_pred))
    # Backwards-compatible RMSE: older sklearn versions don't support squared=False.
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(math.sqrt(mse))
    r2 = float(r2_score(y_true, y_pred))
    # Spearman can return nan (e.g., when predictions are constant); coerce.
    try:
        sp = spearmanr(y_true, y_pred).correlation
        spearman = float(sp) if sp == sp else float("nan")
    except Exception:
        spearman = float("nan")
    return {"mae": mae, "rmse": rmse, "r2": r2, "spearman": spearman}


def slice_metrics(df_eval: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """
    Produce a small set of sliced metrics for debugging stability.
    df_eval requires: y_true, y_pred, months_after_shock, shocked_supplier_share
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


def event_balanced_weights(groups: pd.Series) -> pd.Series:
    """
    Weight rows so each shock_event contributes equal total weight.
    Returns weights aligned to `groups.index`.
    """
    g = groups.astype(str)
    counts = g.value_counts()
    w = g.map(lambda x: 1.0 / float(counts[x]))
    # Normalize weights so mean weight is 1.0 (nice for logging / model stability).
    w = w / float(w.mean()) if float(w.mean()) > 0 else w
    return w.astype(float)


def _fit_with_optional_weights(model: Pipeline, X: pd.DataFrame, y: pd.Series, *, sample_weight: Optional[pd.Series]) -> None:
    """Fit model; if sample_weight provided but unsupported, fall back gracefully."""
    if sample_weight is None:
        model.fit(X, y)
        return
    try:
        model.fit(X, y, model__sample_weight=sample_weight.to_numpy())
    except TypeError:
        # Some estimators may not support sample_weight.
        model.fit(X, y)


def _predict_fold(
    model: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    *,
    sample_weight_train: Optional[pd.Series],
) -> np.ndarray:
    _fit_with_optional_weights(model, X_train, y_train, sample_weight=sample_weight_train)
    return model.predict(X_test)


def crossval_grouped(
    *,
    model: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    splitter,
    df_eval_base: pd.DataFrame,
    use_event_balanced_weighting: bool,
) -> Dict[str, object]:
    fold_metrics: List[Dict[str, float]] = []
    fold_sliced: List[Dict[str, Dict[str, float]]] = []
    fold_details: List[Dict[str, object]] = []

    for fold_i, (train_idx, test_idx) in enumerate(splitter.split(X, y, groups=groups), 1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        w_train = event_balanced_weights(groups.iloc[train_idx]) if use_event_balanced_weighting else None

        # Clone-ish: pipeline contains estimator state; easiest is to re-create via deepcopy
        import copy
        m = copy.deepcopy(model)
        y_pred = _predict_fold(m, X_train, y_train, X_test, sample_weight_train=w_train)

        metrics = eval_regression(y_test.to_numpy(), y_pred)
        fold_metrics.append(metrics)

        test_groups = sorted(set(groups.iloc[test_idx].astype(str)))
        fold_details.append(
            {
                "fold": int(fold_i),
                "n_test": int(len(test_idx)),
                "test_groups": test_groups,
                "metrics": metrics,
            }
        )

        df_eval = df_eval_base.iloc[test_idx].copy()
        df_eval["y_true"] = y_test.to_numpy()
        df_eval["y_pred"] = y_pred
        fold_sliced.append(slice_metrics(df_eval))

    # aggregate
    def _agg(key: str) -> Dict[str, float]:
        vals = np.array([m[key] for m in fold_metrics], dtype=float)
        return {"mean": float(vals.mean()), "std": float(vals.std(ddof=0))}

    summary = {k: _agg(k) for k in ["mae", "rmse", "r2", "spearman"]}
    return {"folds": fold_metrics, "summary": summary, "sliced": fold_sliced, "fold_details": fold_details}


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
        default=str(Path(__file__).parent / "training_data_clean.csv"),
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
        "--cv-folds",
        type=int,
        default=5,
        help="Number of GroupKFold splits by shock_event (default 5).",
    )
    parser.add_argument(
        "--select-by",
        choices=["groupkfold", "leave_one_event_out"],
        default="leave_one_event_out",
        help="Which evaluation scheme to use for selecting the best model (default groupkfold).",
    )
    parser.add_argument(
        "--event-balanced-weighting",
        action="store_true",
        help="Weight rows so each shock_event contributes equal total weight during training/CV.",
    )
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

    def baseline_constant_grouped(*, strategy: str, y: pd.Series, groups: pd.Series, splitter) -> dict:
        """
        Baselines that are comparable to grouped CV:
          - 'zero': predict 0
          - 'train_mean': predict mean(y_train) for each fold
          - 'train_median': predict median(y_train) for each fold
        """
        if strategy not in {"zero", "train_mean", "train_median"}:
            raise ValueError(f"Unknown baseline strategy: {strategy}")

        fold_metrics = []
        for train_idx, test_idx in splitter.split(X, y, groups=groups):
            y_train = y.iloc[train_idx].to_numpy()
            y_test = y.iloc[test_idx].to_numpy()

            if strategy == "zero":
                c = 0.0
            elif strategy == "train_mean":
                c = float(np.mean(y_train)) if y_train.size else 0.0
            else:
                c = float(np.median(y_train)) if y_train.size else 0.0

            y_pred = np.full_like(y_test, fill_value=c, dtype=float)
            fold_metrics.append(eval_regression(y_test, y_pred))

        def _agg(key: str) -> dict:
            vals = np.array([m[key] for m in fold_metrics], dtype=float)
            return {"mean": float(vals.mean()), "std": float(vals.std(ddof=0))}

        return {"folds": fold_metrics, "summary": {k: _agg(k) for k in ["mae", "rmse", "r2", "spearman"]}}

    models = make_models(random_state=args.seed)
    results: Dict[str, Dict[str, object]] = {}

    if not models:
        raise RuntimeError(
            "No models available. Install optional dependencies (catboost/xgboost/lightgbm) "
            "or keep sklearn models enabled."
        )

    # GroupKFold CV (primary)
    n_groups = int(groups.nunique())
    n_splits = int(args.cv_folds)
    if n_splits < 2:
        raise ValueError("--cv-folds must be >= 2")
    if n_splits > n_groups:
        n_splits = n_groups
        print(f"Adjusted --cv-folds to {n_splits} (only {n_groups} unique shock events).")

    print(f"\nCV: GroupKFold(n_splits={n_splits}) grouped by {GROUP_COL}")
    gkf = GroupKFold(n_splits=n_splits)

    # Leave-one-event-out stress test
    print(f"LOEO: LeaveOneGroupOut grouped by {GROUP_COL}")
    logo = LeaveOneGroupOut()

    baseline_gkf = {
        "zero": baseline_constant_grouped(strategy="zero", y=y, groups=groups, splitter=gkf),
        "train_mean": baseline_constant_grouped(strategy="train_mean", y=y, groups=groups, splitter=gkf),
        "train_median": baseline_constant_grouped(strategy="train_median", y=y, groups=groups, splitter=gkf),
    }
    baseline_loeo = {
        "zero": baseline_constant_grouped(strategy="zero", y=y, groups=groups, splitter=logo),
        "train_mean": baseline_constant_grouped(strategy="train_mean", y=y, groups=groups, splitter=logo),
        "train_median": baseline_constant_grouped(strategy="train_median", y=y, groups=groups, splitter=logo),
    }

    def _p(name: str, b: dict) -> None:
        s = b["summary"]
        print(f"Baseline({name}): MAE={s['mae']['mean']:.4f} RMSE={s['rmse']['mean']:.4f} R2={s['r2']['mean']:.4f} Spearman={s['spearman']['mean']:.4f}")

    print("\nBaselines (GroupKFold):")
    _p("0", baseline_gkf["zero"])
    _p("train_mean", baseline_gkf["train_mean"])
    _p("train_median", baseline_gkf["train_median"])
    print("\nBaselines (LOEO):")
    _p("0", baseline_loeo["zero"])
    _p("train_mean", baseline_loeo["train_mean"])
    _p("train_median", baseline_loeo["train_median"])

    best_name: Optional[str] = None
    best_score: float = float("inf")

    for name, model in models.items():
        print(f"\nEvaluating model: {name}")
        cv_res = crossval_grouped(
            model=model,
            X=X,
            y=y,
            groups=groups,
            splitter=gkf,
            df_eval_base=df_eval_base,
            use_event_balanced_weighting=bool(args.event_balanced_weighting),
        )
        loeo_res = crossval_grouped(
            model=model,
            X=X,
            y=y,
            groups=groups,
            splitter=logo,
            df_eval_base=df_eval_base,
            use_event_balanced_weighting=bool(args.event_balanced_weighting),
        )

        results[name] = {
            "groupkfold": cv_res,
            "leave_one_event_out": loeo_res,
        }

        print(f"  CV: MAE={cv_res['summary']['mae']['mean']:.4f} RMSE={cv_res['summary']['rmse']['mean']:.4f} R2={cv_res['summary']['r2']['mean']:.4f} Spearman={cv_res['summary']['spearman']['mean']:.4f}")
        print(f"  LOEO: MAE={loeo_res['summary']['mae']['mean']:.4f} RMSE={loeo_res['summary']['rmse']['mean']:.4f} R2={loeo_res['summary']['r2']['mean']:.4f} Spearman={loeo_res['summary']['spearman']['mean']:.4f}")

        mae_mean = float(cv_res["summary"]["mae"]["mean"])
        mae_std = float(cv_res["summary"]["mae"]["std"])
        print(f"  GroupKFold MAE={mae_mean:.4f}±{mae_std:.4f}")

        loeo_mae_mean = float(loeo_res["summary"]["mae"]["mean"])
        loeo_mae_std = float(loeo_res["summary"]["mae"]["std"])
        print(f"  LOEO MAE={loeo_mae_mean:.4f}±{loeo_mae_std:.4f}")

        if args.select_by == "groupkfold":
            score = mae_mean
        else:
            score = loeo_mae_mean

        if score < best_score:
            best_score = score
            best_name = name

    if best_name is None:
        raise RuntimeError("No model evaluated successfully.")

    if args.select_by == "groupkfold":
        print(f"\nBest model (by GroupKFold mean MAE): {best_name} (MAE={best_score:.4f})")
    else:
        print(f"\nBest model (by LOEO mean MAE): {best_name} (MAE={best_score:.4f})")

    # Fit best model on full data (optionally weighted) and save for inference
    best_model = models[best_name]
    w_full = event_balanced_weights(groups) if args.event_balanced_weighting else None
    _fit_with_optional_weights(best_model, X, y, sample_weight=w_full)

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
                "cv": {
                    "scheme": "GroupKFold",
                    "cv_folds": n_splits,
                    "stress_test": "LeaveOneGroupOut",
                    "group_col": GROUP_COL,
                    "event_balanced_weighting": bool(args.event_balanced_weighting),
                },
            },
            f,
            indent=2,
        )
    with open(metrics_path, "w") as f:
        json.dump({"baseline_zero": {"groupkfold": baseline_gkf, "leave_one_event_out": baseline_loeo}, "models": results}, f, indent=2)

    # LOEO fold-by-fold diagnostics for BEST selected model vs BEST baseline (by LOEO mean MAE)
    try:
        best_loeo = results.get(str(best_name), {}).get("leave_one_event_out", {})
        loeo_folds = best_loeo.get("fold_details", [])
        if loeo_folds:
            # Pick best baseline strategy under LOEO by mean MAE.
            def _baseline_mae(b: dict) -> float:
                try:
                    return float(b["summary"]["mae"]["mean"])
                except Exception:
                    return float("inf")

            best_baseline_name, _best_baseline = min(
                baseline_loeo.items(),
                key=lambda kv: _baseline_mae(kv[1]),
            )

            # Recompute fold-wise baseline aligned with LOEO folds (leakage-safe).
            logo2 = LeaveOneGroupOut()
            baseline_rows: List[Dict[str, object]] = []
            for fold_i, (train_idx, test_idx) in enumerate(logo2.split(X, y, groups=groups), 1):
                y_train = y.iloc[train_idx]
                y_test = y.iloc[test_idx].to_numpy()

                if best_baseline_name == "zero":
                    c = 0.0
                elif best_baseline_name == "train_mean":
                    c = float(y_train.mean()) if len(y_train) else 0.0
                else:
                    # train_median
                    c = float(y_train.median()) if len(y_train) else 0.0

                y_pred = np.full_like(y_test, fill_value=c, dtype=float)
                m = eval_regression(y_test, y_pred)
                g = sorted(set(groups.iloc[test_idx].astype(str)))
                baseline_rows.append({"fold": int(fold_i), "test_groups": g, "n_test": int(len(test_idx)), "metrics": m})

            fold_map = {int(d["fold"]): d for d in loeo_folds}
            base_map = {int(d["fold"]): d for d in baseline_rows}
            rows = []
            for k in sorted(fold_map.keys()):
                md = fold_map[k]
                bd = base_map.get(k)
                if bd is None:
                    continue
                event = md["test_groups"][0] if len(md["test_groups"]) == 1 else "|".join(md["test_groups"])
                rows.append(
                    {
                        "event": event,
                        "n": int(md["n_test"]),
                        "mae_model": float(md["metrics"]["mae"]),
                        "mae_base": float(bd["metrics"]["mae"]),
                        "delta_mae": float(md["metrics"]["mae"]) - float(bd["metrics"]["mae"]),
                        "rmse_model": float(md["metrics"]["rmse"]),
                        "rmse_base": float(bd["metrics"]["rmse"]),
                        "delta_rmse": float(md["metrics"]["rmse"]) - float(bd["metrics"]["rmse"]),
                    }
                )

            rows.sort(key=lambda r: r["delta_mae"], reverse=True)
            print(f"\nLOEO fold-by-fold: {best_name} vs Baseline({best_baseline_name}) (worst ΔMAE first)")
            for r in rows[:15]:
                print(
                    f"  {r['event']}: n={r['n']} | "
                    f"MAE model={r['mae_model']:.4f} vs base={r['mae_base']:.4f} (Δ={r['delta_mae']:+.4f}) | "
                    f"RMSE Δ={r['delta_rmse']:+.4f}"
                )
    except Exception as e:
        print(f"LOEO fold diagnostics skipped: {e}")

    print(f"\nSaved model: {model_path}")
    print(f"Saved schema: {schema_path}")
    print(f"Saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()


