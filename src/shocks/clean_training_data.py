#!/usr/bin/env python3
"""
clean_training_data.py

Clean dataset rows produced by `build_shock_dataset.py`.

Default behavior:
  - remove any rows where `shock_yoy_dev` (import YoY deviation vs expected) is positive

Optional filters can be enabled via CLI flags.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd


MIN_ROWS_PER_EVENT = 5
EVENT_COL = "shock_event"


def _bool_count(series: pd.Series) -> dict:
    s = series.dropna()
    if s.empty:
        return {"true": 0, "false": 0, "missing": int(series.isna().sum())}
    return {
        "true": int((s.astype(bool) == True).sum()),  # noqa: E712
        "false": int((s.astype(bool) == False).sum()),  # noqa: E712
        "missing": int(series.isna().sum()),
    }


def clean_training_df(
    df: pd.DataFrame,
    *,
    drop_positive_import_yoy: bool = True,
    drop_missing_import_yoy: bool = True,
    require_nonnegative_months_after_shock: bool = True,
    require_target_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Args:
        df: Raw training dataframe.
        drop_positive_import_yoy: Drop rows where shock_yoy_dev > 0.
        drop_missing_import_yoy: Drop rows where shock_yoy_dev is NaN.
        require_nonnegative_months_after_shock: Drop rows with months_after_shock < 0 (if present).
        require_target_col: If provided, drop rows where this column is NaN (useful for labels).
    """
    out = df.copy()

    if drop_positive_import_yoy:
        if "shock_yoy_dev" not in out.columns:
            raise ValueError("Missing required column `shock_yoy_dev`.")
        out = out[~(out["shock_yoy_dev"] > 0)]

    if drop_missing_import_yoy:
        if "shock_yoy_dev" not in out.columns:
            raise ValueError("Missing required column `shock_yoy_dev`.")
        out = out[out["shock_yoy_dev"].notna()]

    if require_nonnegative_months_after_shock and "months_after_shock" in out.columns:
        out = out[out["months_after_shock"].fillna(0) >= 0]

    if "shock_expected" in out.columns:
        out = out[~(out["shock_expected"] < 0)]
    if "prop_expected" in out.columns:
        out = out[~(out["prop_expected"] < 0)]

    if require_target_col is not None:
        if require_target_col not in out.columns:
            raise ValueError(f"Missing required target column `{require_target_col}`.")
        out = out[out[require_target_col].notna()]

    # Final cleanup: drop underpowered events after ALL other filters
    if EVENT_COL in out.columns:
        vc = out[EVENT_COL].value_counts()
        keep_events = set(vc[vc >= MIN_ROWS_PER_EVENT].index.astype(str))
        out = out[out[EVENT_COL].astype(str).isin(keep_events)]

    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean shock propagation training data CSV.")
    parser.add_argument(
        "--input",
        type=str,
        default=str(Path(__file__).parent / "training_data.csv"),
        help="Input CSV path (default: src/shocks/training_data.csv)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(Path(__file__).parent / "training_data_clean.csv"),
        help="Output CSV path (default: src/shocks/training_data_clean.csv)",
    )
    parser.add_argument(
        "--keep-positive-import-yoy",
        action="store_true",
        help="Keep rows where shock_yoy_dev > 0 (default: drop them).",
    )
    parser.add_argument(
        "--drop-missing-import-yoy",
        action="store_true",
        help="Also drop rows with missing shock_yoy_dev (default: keep).",
    )
    parser.add_argument(
        "--require-target-col",
        type=str,
        default=None,
        help="Drop rows where this column is missing (e.g., prop_yoy_dev).",
    )

    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    if not in_path.exists():
        raise SystemExit(f"Input file not found: {in_path}")

    df = pd.read_csv(in_path)
    n0 = len(df)

    cleaned = clean_training_df(
        df,
        drop_positive_import_yoy=not args.keep_positive_import_yoy,
        drop_missing_import_yoy=args.drop_missing_import_yoy,
        require_target_col=args.require_target_col,
    )
    n1 = len(cleaned)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(out_path, index=False)

    # Minimal console report
    if "shock_yoy_dev" in df.columns:
        pct_pos = float((df["shock_yoy_dev"] > 0).mean())
        miss = float(df["shock_yoy_dev"].isna().mean())
        print(f"Input:  {n0} rows | shock_yoy_dev>0: {pct_pos:.1%} | missing: {miss:.1%}")
    if EVENT_COL in df.columns:
        before = int(df[EVENT_COL].nunique())
        after = int(cleaned[EVENT_COL].nunique()) if len(cleaned) else 0
        # This is the *post-filter* event pruning, so it's helpful to surface explicitly.
        print(f"Event filter: dropped events with <{MIN_ROWS_PER_EVENT} rows (after all filters): {before} → {after}")
    print(f"Output: {n1} rows → {out_path}")


if __name__ == "__main__":
    main()


