#!/usr/bin/env python3
"""
leontief_propagation.py - Pure Leontief Input-Output Shock Propagation

Implements classical Leontief input-output analysis for supply chain shock propagation.
Uses ICIO technical coefficients to compute how shocks propagate through the network.

Theory:
    x = (I - A)^(-1) * f
    where:
    - x = total output vector
    - A = technical coefficients matrix (inputs per unit output)
    - f = final demand vector
    - (I - A)^(-1) = Leontief inverse matrix

For shock analysis:
    Δx = (I - A)^(-1) * Δf
    
Usage:
    python leontief_propagation.py --graph embeddings/graph_2021_labeled.pt \
        --shocked-nodes USA_C26 --magnitude 0.20 --output leontief_results.csv

Dataset-eval usage (to compare vs ML model trained in shocks/train_propagation_model.py):
    python leontief_propagation.py --graph embeddings/graph_2021_labeled.pt \
        --data shocks/training_data.csv --outdir shocks/models/leontief_prop_yoy_dev
"""

import torch
import numpy as np
import pandas as pd
import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict, Tuple, Optional, Callable, Any
from scipy import sparse
from scipy.sparse.linalg import spsolve, factorized

from sklearn.model_selection import GroupShuffleSplit

# Import ICIO parser for plain English formatting
from ICIO.ICIO_parser import (
    format_node_name,
    format_sector_name,
    format_country_name
)

try:
    # Reuse the exact target/group names + metric helpers from the training script
    from shocks.train_propagation_model import (  # type: ignore
        TARGET_COL,
        GROUP_COL,
        eval_regression,
        slice_metrics,
        add_derived_columns,
    )
except Exception:
    TARGET_COL = "prop_yoy_dev"
    GROUP_COL = "shock_event"

    def eval_regression(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        # Minimal fallback; should rarely be used if project imports work.
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        mae = float(np.mean(np.abs(y_true - y_pred))) if len(y_true) else float("nan")
        rmse = float(math.sqrt(float(np.mean((y_true - y_pred) ** 2)))) if len(y_true) else float("nan")
        return {"mae": mae, "rmse": rmse, "r2": float("nan"), "spearman": float("nan")}

    def slice_metrics(df_eval: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        return {}

    def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
        return df.copy()


def _filter_and_clip_labels_df(
    df: pd.DataFrame,
    *,
    max_abs: Optional[float],
    clip_abs: Optional[float],
) -> pd.DataFrame:
    """
    Apply the same stability logic as shocks/train_propagation_model.py, but preserve
    all original dataset columns (we need shock_node/target_node for Leontief).
    """
    df2 = df.copy()
    df2 = df2[df2[TARGET_COL].notna()].copy()
    df2[TARGET_COL] = df2[TARGET_COL].astype(float)

    if max_abs is not None:
        keep = df2[TARGET_COL].abs() <= float(max_abs)
        df2 = df2.loc[keep].copy()

    if clip_abs is not None:
        c = float(clip_abs)
        df2[TARGET_COL] = df2[TARGET_COL].clip(lower=-c, upper=c)

    return df2


def _group_train_test_split(
    df: pd.DataFrame,
    *,
    test_size: float,
    random_state: int,
) -> Tuple[np.ndarray, np.ndarray]:
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    groups = df[GROUP_COL].astype(str)
    train_idx, test_idx = next(splitter.split(df, df[TARGET_COL].astype(float), groups=groups))
    return train_idx, test_idx


class LeontiefPropagator:
    """
    Computes Leontief input-output propagation for supply chain shocks.
    """
    
    def __init__(self, graph):
        """
        Initialize Leontief propagator from graph.
        
        Args:
            graph: PyTorch Geometric graph with edge_index, edge_attr, node_labels
        """
        self.graph = graph
        self.num_nodes = graph.num_nodes
        self.node_labels = graph.node_labels
        
        if not hasattr(graph, 'node_id_to_idx'):
            raise ValueError("Graph missing node_id_to_idx mapping")
        
        # Build technical coefficients matrix A
        print("Building technical coefficients matrix A...")
        self.A = self._build_technical_coefficients()
        
        # Pre-factorize (I - A) for repeated solves (faster than computing an explicit inverse).
        print("Factorizing (I - A) for repeated solving...")
        self.I_minus_A = sparse.identity(self.num_nodes, format="csr") - self.A
        # factorized expects CSC/CSR; CSC is typically better for LU.
        self._solve: Callable[[np.ndarray], np.ndarray] = factorized(self.I_minus_A.tocsc())
        
        print(f"✓ Leontief propagator initialized ({self.num_nodes} nodes)")
    
    def _build_technical_coefficients(self) -> sparse.csr_matrix:
        """
        Build technical coefficients matrix A from ICIO data.
        
        A[i,j] = how much sector j needs from sector i per unit of output
        
        Returns:
            Sparse matrix A of size (num_nodes, num_nodes)
        """
        edge_index = self.graph.edge_index.cpu().numpy()
        edge_values = self.graph.value_t.cpu().numpy()

        src_idx, tgt_idx = edge_index

        # Total output per sector = sum of outgoing flows from that sector (vectorized).
        total_output = np.bincount(src_idx, weights=edge_values, minlength=self.num_nodes).astype(float)

        # Technical coefficients: a_ij = input_ij / output_j (j is target node)
        denom = total_output[tgt_idx]
        mask = denom > 0
        row_indices = src_idx[mask]
        col_indices = tgt_idx[mask]
        coefficients = (edge_values[mask] / denom[mask]).astype(float)

        A = sparse.coo_matrix(
            (coefficients, (row_indices, col_indices)),
            shape=(self.num_nodes, self.num_nodes),
        )

        return A.tocsr()
    
    def propagate_shock(self, shocked_nodes: Dict[str, float]) -> np.ndarray:
        """
        Propagate shocks through supply chain using Leontief multipliers.
        
        Args:
            shocked_nodes: Dict mapping node_id -> magnitude (e.g., {'USA_C26': -0.20})
        
        Returns:
            impact_vector: Array of size (num_nodes,) with impact on each node's output
        """
        # Create shock vector
        shock_vector = np.zeros(self.num_nodes)
        
        for node_id, magnitude in shocked_nodes.items():
            if node_id not in self.graph.node_id_to_idx:
                print(f"Warning: Node '{node_id}' not found in graph")
                continue
            idx = self.graph.node_id_to_idx[node_id]
            shock_vector[idx] = magnitude
        
        # Solve (I - A) x = shock_vector  => x is the total (direct+indirect) output effect
        impact_vector = self._solve(shock_vector)
        
        return impact_vector

    def predict_prop_yoy_dev(
        self,
        df: pd.DataFrame,
        *,
        shock_col: str = "shock_yoy_change",
        shock_node_col: str = "shock_node",
        target_node_col: str = "target_node",
        cache_unit_shocks: bool = True,
        missing_value: float = 0.0,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Produce a Leontief baseline prediction for the dataset target `prop_yoy_dev`.

        Heuristic mapping:
        - Treat the user-specified shock magnitude (default: shock_yoy_change) as a final-demand
          shock applied at `shock_node`.
        - Predict the resulting output deviation at `target_node` using the Leontief multipliers.

        Returns:
            (y_pred, info)
        """
        required = {shock_col, shock_node_col, target_node_col}
        missing_cols = sorted([c for c in required if c not in df.columns])
        if missing_cols:
            raise ValueError(f"Missing required columns for Leontief prediction: {missing_cols}")

        shock_nodes = df[shock_node_col].astype(str).to_numpy()
        target_nodes = df[target_node_col].astype(str).to_numpy()
        magnitudes = df[shock_col].astype(float).to_numpy()

        y_pred = np.full(shape=(len(df),), fill_value=float(missing_value), dtype=float)

        unit_cache: Dict[int, np.ndarray] = {}
        missing_shock = 0
        missing_target = 0

        for i, (s_node, t_node, mag) in enumerate(zip(shock_nodes, target_nodes, magnitudes)):
            s_idx = self.graph.node_id_to_idx.get(s_node)
            t_idx = self.graph.node_id_to_idx.get(t_node)
            if s_idx is None:
                missing_shock += 1
                continue
            if t_idx is None:
                missing_target += 1
                continue

            if cache_unit_shocks:
                x_unit = unit_cache.get(int(s_idx))
                if x_unit is None:
                    b = np.zeros(self.num_nodes, dtype=float)
                    b[int(s_idx)] = 1.0
                    x_unit = self._solve(b).astype(float)
                    unit_cache[int(s_idx)] = x_unit
                y_pred[i] = float(x_unit[int(t_idx)] * mag)
            else:
                b = np.zeros(self.num_nodes, dtype=float)
                b[int(s_idx)] = float(mag)
                x = self._solve(b)
                y_pred[i] = float(x[int(t_idx)])

        info: Dict[str, Any] = {
            "shock_col": shock_col,
            "shock_node_col": shock_node_col,
            "target_node_col": target_node_col,
            "missing_shock_nodes": int(missing_shock),
            "missing_target_nodes": int(missing_target),
            "unique_unit_shocks_solved": int(len(unit_cache)) if cache_unit_shocks else None,
        }
        return y_pred, info
    
    def compute_edge_impacts(self, node_impacts: np.ndarray) -> pd.DataFrame:
        """
        Compute edge-level impacts from node-level impacts.
        
        If node j's output drops by x%, its demand for inputs drops by x%,
        so edge i→j drops by x%.
        
        Args:
            node_impacts: Array of size (num_nodes,) with impact on each node
        
        Returns:
            DataFrame with edge-level impacts
        """
        edge_index = self.graph.edge_index.cpu().numpy()
        src_idx, tgt_idx = edge_index
        edge_values = self.graph.value_t.cpu().numpy()
        
        results = []
        
        for i in range(len(src_idx)):
            src = src_idx[i]
            tgt = tgt_idx[i]
            value = edge_values[i]
            
            # Edge impact = target node's output change
            # (if target produces less, it needs fewer inputs from source)
            target_impact = node_impacts[tgt]
            
            # Also consider source impact (if source produces less, it can supply less)
            source_impact = node_impacts[src]
            
            # Conservative: use minimum (bottleneck)
            edge_impact_pct = min(target_impact, source_impact)
            edge_impact_abs = value * edge_impact_pct
            
            results.append({
                'source': self.node_labels[src],
                'target': self.node_labels[tgt],
                'baseline_value': value,
                'impact_pct': edge_impact_pct * 100,
                'impact_abs': edge_impact_abs,
                'source_output_impact': source_impact * 100,
                'target_output_impact': target_impact * 100
            })
        
        return pd.DataFrame(results)
    
    def analyze_shock(self, shocked_nodes: Dict[str, float]) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Complete shock analysis: propagate and compute edge impacts.
        
        Args:
            shocked_nodes: Dict mapping node_id -> magnitude
        
        Returns:
            (node_impacts, edge_impacts_df)
        """
        print(f"\nPropagating shock through Leontief multipliers...")
        node_impacts = self.propagate_shock(shocked_nodes)
        
        print(f"Computing edge-level impacts...")
        edge_impacts = self.compute_edge_impacts(node_impacts)
        
        # Sort by largest drops
        edge_impacts = edge_impacts.sort_values('impact_abs', ascending=True)
        
        return node_impacts, edge_impacts
    
    def print_summary(self, shocked_nodes: Dict[str, float], 
                     node_impacts: np.ndarray, 
                     edge_impacts: pd.DataFrame):
        """Print summary of Leontief propagation results."""
        
        print("\n" + "="*80)
        print("LEONTIEF INPUT-OUTPUT PROPAGATION ANALYSIS")
        print("="*80)
        
        print(f"\nShocked Nodes:")
        for node_id, magnitude in shocked_nodes.items():
            node_name = format_node_name(node_id, include_code=True)
            print(f"  • {node_name}: {magnitude:.1%}")
        
        # Node-level summary
        print("\n" + "-"*80)
        print("NODE-LEVEL IMPACTS (Top 20 affected sectors)")
        print("-"*80)
        
        # Find top affected nodes
        node_impact_df = pd.DataFrame({
            'node': self.node_labels,
            'impact_pct': node_impacts * 100
        })
        node_impact_df = node_impact_df.sort_values('impact_pct', ascending=True)
        
        print("\n{:<50} {:>12}".format("Node", "Output Change"))
        print("-"*80)
        for _, row in node_impact_df.head(20).iterrows():
            node_name = format_node_name(row['node'])
            print("{:<50} {:>11.2f}%".format(
                node_name[:48],
                row['impact_pct']
            ))
        
        # Edge-level summary
        print("\n" + "-"*80)
        print("EDGE-LEVEL IMPACTS")
        print("-"*80)
        
        drops = edge_impacts[edge_impacts['impact_abs'] < 0]
        increases = edge_impacts[edge_impacts['impact_abs'] > 0]
        
        print(f"\nTotal edges: {len(edge_impacts):,}")
        print(f"  Edges with drops: {len(drops):,} ({100*len(drops)/len(edge_impacts):.1f}%)")
        print(f"  Edges with increases: {len(increases):,} ({100*len(increases)/len(edge_impacts):.1f}%)")
        
        if len(drops) > 0:
            print(f"\n  Max drop: {drops['impact_pct'].min():.2f}%")
            print(f"  Mean drop: {drops['impact_pct'].mean():.2f}%")
            print(f"  Total value at risk: ${drops['baseline_value'].sum():,.0f}")
            print(f"  Total impact: ${drops['impact_abs'].sum():,.0f}")
        
        # Top drops
        print("\n" + "-"*80)
        print("TOP 20 LARGEST DROPS IN TRADING FLOWS")
        print("-"*80)
        
        top_drops = drops.head(20)
        
        print("\n{:<40} {:<40} {:>15} {:>10}".format(
            "Source", "Target", "$ Change", "% Change"
        ))
        print("-"*80)
        for _, row in top_drops.iterrows():
            source_name = format_node_name(row['source'])[:38]
            target_name = format_node_name(row['target'])[:38]
            print("{:<40} {:<40} ${:>14,.0f} {:>9.2f}%".format(
                source_name,
                target_name,
                row['impact_abs'],
                row['impact_pct']
            ))
        
        print("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(
        description="Leontief input-output shock propagation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "--graph",
        required=True,
        help="Path to graph file (e.g., embeddings/graph_2021_labeled.pt)",
    )

    # Dataset-eval mode (mirrors inputs/outputs of shocks/train_propagation_model.py)
    parser.add_argument(
        "--data",
        default=None,
        help="Path to training data CSV produced by shocks/build_shock_dataset.py",
    )
    parser.add_argument(
        "--outdir",
        default=None,
        help="Directory to save Leontief baseline artifacts (metrics + predictions CSV)",
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
    parser.add_argument(
        "--shock-col",
        default="shock_yoy_change",
        help="Which dataset column to treat as the Leontief shock magnitude (default shock_yoy_change).",
    )

    # Single-shock mode (kept for backwards compatibility)
    parser.add_argument("--shocked-nodes", nargs="+", help="Node IDs to shock (e.g., USA_C26 CHN_C26)")
    parser.add_argument("--magnitude", type=float, default=0.5, help="Shock magnitude as fraction (0.5 = 50%% reduction)")
    parser.add_argument("--output", type=str, help="Path to save detailed results CSV (optional)")
    parser.add_argument("--save-matrix", action="store_true", help="Save (I-A) matrix as NPZ (for debugging/reuse)")
    
    args = parser.parse_args()
    
    # Load graph
    print(f"Loading graph from {args.graph}...")
    try:
        graph = torch.load(args.graph, map_location="cpu", weights_only=False)
    except ModuleNotFoundError as e:
        # Common when graphs were saved with torch_geometric objects but runtime lacks PyG.
        missing = str(e).strip()
        print("\nError: Failed to load graph due to a missing Python dependency.")
        print(f"  {missing}")
        print("\nThis graph was likely saved from a PyTorch Geometric (torch_geometric) object.")
        print("Fix: run in the same environment you used to create the graph (or install torch_geometric).")
        return
    
    # Initialize Leontief propagator
    propagator = LeontiefPropagator(graph)
    
    # Save matrix if requested (debugging / reuse)
    if args.save_matrix:
        matrix_path = Path(args.graph).parent / "I_minus_A.npz"
        sparse.save_npz(matrix_path, propagator.I_minus_A)
        print(f"✓ Saved (I - A) to {matrix_path}")

    # Dataset-eval mode: produce comparable outputs to shocks/train_propagation_model.py
    if args.data is not None:
        data_path = Path(args.data)
        outdir = Path(args.outdir) if args.outdir is not None else (data_path.parent / "models" / "leontief_prop_yoy_dev")
        outdir.mkdir(parents=True, exist_ok=True)

        print(f"Loading data: {data_path}")
        df_raw = pd.read_csv(data_path)

        # Keep derived columns aligned with training script (helps slicing/debug)
        df = add_derived_columns(df_raw)

        if TARGET_COL not in df.columns:
            raise ValueError(f"Dataset missing target column '{TARGET_COL}'.")
        if GROUP_COL not in df.columns:
            raise ValueError(f"Dataset missing group column '{GROUP_COL}'.")

        max_abs = None if args.label_max_abs == 0 else float(args.label_max_abs)
        clip_abs = None if args.label_clip_abs == 0 else float(args.label_clip_abs)
        df = _filter_and_clip_labels_df(df, max_abs=max_abs, clip_abs=clip_abs)

        print(f"Rows with label ({TARGET_COL}): {len(df):,}")
        print(f"Unique shock events: {df[GROUP_COL].astype(str).nunique():,}")
        if len(df) > 0:
            y = df[TARGET_COL].astype(float)
            print(
                f"Label stats: mean={y.mean():+.4f} median={y.median():+.4f} "
                f"p05={y.quantile(0.05):+.4f} p95={y.quantile(0.95):+.4f}"
            )

        train_idx, test_idx = _group_train_test_split(df, test_size=args.test_size, random_state=args.seed)
        print(f"Split: train={len(train_idx):,} rows, test={len(test_idx):,} rows")

        # Baseline: always 0 (no deviation beyond expected)
        y_test = df.iloc[test_idx][TARGET_COL].astype(float).to_numpy()
        y_pred_0 = np.zeros_like(y_test, dtype=float)
        baseline_metrics = eval_regression(y_test, y_pred_0)
        print(
            f"Baseline (predict 0): MAE={baseline_metrics['mae']:.4f}, "
            f"RMSE={baseline_metrics['rmse']:.4f}, R2={baseline_metrics.get('r2', float('nan')):.4f}"
        )

        # Leontief predictions
        df_test = df.iloc[test_idx].copy()
        y_pred, info = propagator.predict_prop_yoy_dev(df_test, shock_col=args.shock_col)
        leontief_metrics = eval_regression(y_test, y_pred)

        # Sliced evaluation (same helper as training script)
        df_eval = df_test.copy()
        df_eval["y_true"] = y_test
        df_eval["y_pred"] = y_pred
        sliced = slice_metrics(df_eval)

        print(
            f"Leontief baseline: MAE={leontief_metrics['mae']:.4f} "
            f"RMSE={leontief_metrics['rmse']:.4f} R2={leontief_metrics.get('r2', float('nan')):.4f} "
            f"Spearman={leontief_metrics.get('spearman', float('nan')):.4f}"
        )
        if info.get("missing_shock_nodes", 0) or info.get("missing_target_nodes", 0):
            print(
                f"Note: missing node IDs in graph mapping: "
                f"shock={info.get('missing_shock_nodes')} target={info.get('missing_target_nodes')}"
            )

        # Save artifacts
        metrics_path = outdir / "metrics.json"
        preds_path = outdir / "predictions.csv"

        with open(metrics_path, "w") as f:
            json.dump(
                {
                    "target": TARGET_COL,
                    "group_col": GROUP_COL,
                    "shock_col": args.shock_col,
                    "baseline_zero": baseline_metrics,
                    "leontief": {
                        "metrics": leontief_metrics,
                        "sliced": sliced,
                        "info": info,
                    },
                },
                f,
                indent=2,
            )

        # Keep a small, comparison-friendly set of columns.
        keep_cols = []
        for c in [
            GROUP_COL,
            "shock_node",
            "target_node",
            "shock_date",
            "observation_date",
            "months_after_shock",
            "is_domestic",
            "shocked_supplier_share",
            "icio_edge_value",
        ]:
            if c in df_test.columns:
                keep_cols.append(c)

        df_out = df_test[keep_cols].copy() if keep_cols else df_test.copy()
        df_out["y_true"] = y_test
        df_out["y_pred"] = y_pred
        df_out.to_csv(preds_path, index=False)

        print(f"\nSaved metrics: {metrics_path}")
        print(f"Saved predictions: {preds_path} ({len(df_out):,} rows)")
        return
    
    # Parse shocked nodes
    if not args.shocked_nodes:
        print("Error: Must specify --shocked-nodes")
        return
    
    shocked_nodes = {}
    shock_magnitude = -abs(args.magnitude)  # Convert to negative
    for node_id in args.shocked_nodes:
        shocked_nodes[node_id] = shock_magnitude
    
    # Run analysis
    node_impacts, edge_impacts = propagator.analyze_shock(shocked_nodes)
    
    # Print summary
    propagator.print_summary(shocked_nodes, node_impacts, edge_impacts)
    
    # Save results
    if args.output:
        edge_impacts.to_csv(args.output, index=False)
        print(f"\n✓ Detailed results saved to: {args.output}")
        print(f"  Contains {len(edge_impacts):,} edges")
    
    print(f"\n{'='*80}")
    print("LEONTIEF ANALYSIS COMPLETE")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()

