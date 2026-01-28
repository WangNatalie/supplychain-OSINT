#!/usr/bin/env python3
"""
hyperparameter_scan.py

Hyperparameter scan for the shock propagation regression target `prop_yoy_dev`.

This script:
- loads a training CSV (default: training_data_v4.csv)
- applies the same leakage-safe feature building + stability clipping as train_propagation_model.py
- evaluates candidate pipelines using:
  - GroupKFold grouped by shock_event
  - LeaveOneGroupOut ("leave-one-event-out" stress test)
- writes a ranked CSV of results and optionally saves the best model artifact.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut, ParameterGrid
from sklearn.pipeline import Pipeline

from train_propagation_model import (
    LABEL_CLIP_ABS_DEFAULT,
    LABEL_MAX_ABS_DEFAULT,
    SHOCK_CLIP_ABS_DEFAULT,
    TARGET_COL,
    GROUP_COL,
    add_derived_columns,
    build_xy_groups,
    clip_shock_feature,
    crossval_grouped,
    event_balanced_weights,
    filter_and_clip_labels,
    make_preprocessor,
)


def _jsonable(o: object) -> str:
    try:
        return json.dumps(o, sort_keys=True)
    except Exception:
        return str(o)


def _make_model(name: str, params: Dict[str, object], *, seed: int) -> Pipeline:
    pre = make_preprocessor(scale_numeric=(name == "ridge"))
    if name == "ridge":
        est = Ridge(random_state=int(seed), **params)
    elif name == "gbrt":
        est = GradientBoostingRegressor(random_state=int(seed), **params)
    elif name == "rf":
        # Force single-threaded RF to match repo constraints.
        p2 = dict(params)
        p2.setdefault("n_jobs", 1)
        est = RandomForestRegressor(random_state=int(seed), **p2)
    elif name == "catboost":
        try:
            from catboost import CatBoostRegressor  # type: ignore
        except Exception as e:
            raise RuntimeError("catboost is not installed but was requested in the scan.") from e
        p2 = dict(params)
        # Keep consistent with repo: deterministic + quiet + single-threaded
        p2.setdefault("loss_function", "MAE")
        p2.setdefault("random_seed", int(seed))
        p2.setdefault("verbose", False)
        p2.setdefault("thread_count", 1)
        est = CatBoostRegressor(**p2)
    elif name == "xgboost":
        try:
            from xgboost import XGBRegressor  # type: ignore
        except Exception as e:
            raise RuntimeError("xgboost is not installed but was requested in the scan.") from e
        p2 = dict(params)
        # Keep consistent with repo: single-threaded, stable objective
        p2.setdefault("objective", "reg:squarederror")
        p2.setdefault("n_jobs", 1)
        p2.setdefault("random_state", int(seed))
        est = XGBRegressor(**p2)
    else:
        raise ValueError(f"Unknown model name: {name}")
    return Pipeline(steps=[("pre", pre), ("model", est)])


def main() -> None:
    ap = argparse.ArgumentParser(description="Hyperparameter scan for propagation model.")
    ap.add_argument("--data", default=str(Path(__file__).parent / "training_data_v2.csv"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cv-folds", type=int, default=5)
    ap.add_argument("--event-balanced-weighting", action="store_true")
    ap.add_argument("--select-by", choices=["groupkfold", "leave_one_event_out"], default="leave_one_event_out")
    ap.add_argument(
        "--models",
        type=str,
        default="ridge,gbrt,rf,catboost,xgboost",
        help="Comma-separated list of models to scan (choices: ridge,gbrt,rf,catboost,xgboost).",
    )
    ap.add_argument("--out-csv", default=str(Path(__file__).parent / "hyperparam_scan_results_v2.csv"))
    ap.add_argument("--save-best-outdir", default="", help="If set, save best model.joblib + schema + metrics.json to this dir.")
    args = ap.parse_args()

    data_path = Path(args.data)
    df = pd.read_csv(data_path)

    # Safety: drop negative expected values (same as train_propagation_model.py)
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
    df_eval_base = add_derived_columns(df).loc[X.index].copy()

    n_groups = int(groups.nunique())
    n_splits = int(args.cv_folds)
    if n_splits > n_groups:
        n_splits = n_groups
        print(f"Adjusted --cv-folds to {n_splits} (only {n_groups} unique shock events).")
    gkf = GroupKFold(n_splits=n_splits)
    logo = LeaveOneGroupOut()

    # Grids (kept intentionally small; expand as needed)
    grids: Dict[str, Dict[str, List[object]]] = {
        "ridge": {
            "alpha": [0.5, 1.0, 3.0, 10.0, 30.0],
        },
        "gbrt": {
            "loss": ["huber"],
            "learning_rate": [0.03, 0.05, 0.08],
            "n_estimators": [300, 600, 900],
            "max_depth": [2, 3, 4],
            "subsample": [0.8, 0.9, 1.0],
        },
        "rf": {
            "n_estimators": [200, 400, 800],
            "max_depth": [None, 12, 20],
            "min_samples_leaf": [5, 10, 20],
        },
        "catboost": {
            # Keep the grid modest; CatBoost can be expensive.
            "depth": [4, 6, 8],
            "learning_rate": [0.03, 0.05, 0.1],
            "iterations": [800, 1500],
            "l2_leaf_reg": [1.0, 3.0, 10.0],
        },
        "xgboost": {
            # Modest grid; XGBoost can be expensive.
            "n_estimators": [600, 1200],
            "learning_rate": [0.03, 0.05, 0.08],
            "max_depth": [4, 6],
            "subsample": [0.8, 0.9],
            "colsample_bytree": [0.8, 0.9],
            "reg_lambda": [1.0, 5.0],
        },
    }

    requested_models = [m.strip() for m in str(args.models).split(",") if m.strip()]
    if not requested_models:
        raise SystemExit("--models must include at least one model name.")
    unknown = sorted(set(requested_models) - set(grids.keys()))
    if unknown:
        raise SystemExit(f"Unknown model(s) in --models: {unknown}. Allowed: {sorted(grids.keys())}")
    grids = {k: v for k, v in grids.items() if k in set(requested_models)}

    # Flatten configs
    configs: List[Tuple[str, Dict[str, object]]] = []
    for model_name, grid in grids.items():
        for params in ParameterGrid(grid):
            configs.append((model_name, dict(params)))

    rows: List[Dict[str, object]] = []
    best_score = float("inf")
    best: Optional[Dict[str, object]] = None
    best_model: Optional[Pipeline] = None

    for i, (model_name, params) in enumerate(configs, 1):
        model = _make_model(model_name, params, seed=int(args.seed))
        print(f"[{i}/{len(configs)}] {model_name} params={params}")

        cv_res = crossval_grouped(
            model=copy.deepcopy(model),
            X=X,
            y=y,
            groups=groups,
            splitter=gkf,
            df_eval_base=df_eval_base,
            use_event_balanced_weighting=bool(args.event_balanced_weighting),
        )
        loeo_res = crossval_grouped(
            model=copy.deepcopy(model),
            X=X,
            y=y,
            groups=groups,
            splitter=logo,
            df_eval_base=df_eval_base,
            use_event_balanced_weighting=bool(args.event_balanced_weighting),
        )

        gkf_mae = float(cv_res["summary"]["mae"]["mean"])
        loeo_mae = float(loeo_res["summary"]["mae"]["mean"])
        score = gkf_mae if args.select_by == "groupkfold" else loeo_mae

        row = {
            "model": model_name,
            "params": _jsonable(params),
            "gkf_mae_mean": gkf_mae,
            "gkf_mae_std": float(cv_res["summary"]["mae"]["std"]),
            "loeo_mae_mean": loeo_mae,
            "loeo_mae_std": float(loeo_res["summary"]["mae"]["std"]),
            "score": float(score),
        }
        rows.append(row)

        if float(score) < float(best_score):
            best_score = float(score)
            best = {
                "model": model_name,
                "params": params,
                "cv": cv_res,
                "loeo": loeo_res,
                "score": float(score),
            }
            best_model = model

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df = pd.DataFrame(rows).sort_values(["score", "loeo_mae_mean", "gkf_mae_mean"], ascending=True)
    out_df.to_csv(out_path, index=False)
    print(f"\nSaved scan results: {out_path} (n={len(out_df)})")

    if best is None or best_model is None:
        return

    print(f"Best: {best['model']} score={best['score']:.4f} params={best['params']}")

    if args.save_best_outdir:
        try:
            import joblib  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError("joblib is required to save the model") from e

        outdir = Path(args.save_best_outdir)
        outdir.mkdir(parents=True, exist_ok=True)

        # Fit best model on full data (optionally weighted)
        w_full = event_balanced_weights(groups) if bool(args.event_balanced_weighting) else None
        m = copy.deepcopy(best_model)
        if w_full is None:
            m.fit(X, y)
        else:
            try:
                m.fit(X, y, model__sample_weight=w_full.to_numpy())
            except TypeError:
                m.fit(X, y)

        joblib.dump(m, outdir / "model.joblib")
        (outdir / "feature_schema.json").write_text(
            json.dumps(
                {
                    "target": TARGET_COL,
                    "group_col": GROUP_COL,
                    "selected_model": best["model"],
                    "selected_params": best["params"],
                    "cv_folds": int(args.cv_folds),
                    "select_by": str(args.select_by),
                },
                indent=2,
            )
        )
        (outdir / "metrics.json").write_text(
            json.dumps(
                {
                    "best": {
                        "model": best["model"],
                        "params": best["params"],
                        "score": best["score"],
                    },
                    "groupkfold": best["cv"],
                    "leave_one_event_out": best["loeo"],
                },
                indent=2,
            )
        )
        print(f"Saved best model artifacts to: {outdir}")


if __name__ == "__main__":
    main()


