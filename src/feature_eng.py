from load_icio import read_icio
import torch
import pandas as pd
import numpy as np
import os
from torch_geometric.data import Data
from typing import List, Optional, Tuple
from world_data import load_indicators
import argparse

def build_node_features(edges_t: pd.DataFrame, 
                        edges_prev: Optional[pd.DataFrame] = None,
                        external_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Build node features with optional 1-year lag and external data.
    Memory-efficient: only keeps previous year, not full history.
    """
    # Basic structural features
    out_strength = edges_t.groupby("source")["value"].sum().rename("out_strength")
    in_strength  = edges_t.groupby("target")["value"].sum().rename("in_strength")
    out_degree = edges_t.groupby("source").size().rename("out_degree")
    in_degree  = edges_t.groupby("target").size().rename("in_degree")
    
    node_feats = pd.concat([in_strength, out_strength, in_degree, out_degree], axis=1)
    node_feats = node_feats.fillna(0.0)
    
    # Add 1-year lag features if available
    if edges_prev is not None:
        prev_out = edges_prev.groupby("source")["value"].sum()
        prev_in = edges_prev.groupby("target")["value"].sum()
        
        # Growth rates (memory efficient - computed on the fly)
        node_feats['out_strength_growth'] = (
            (out_strength - prev_out) / prev_out.replace(0, np.nan)
        ).fillna(0.0)
        node_feats['in_strength_growth'] = (
            (in_strength - prev_in) / prev_in.replace(0, np.nan)
        ).fillna(0.0)
        
        # Lagged values (normalized to similar scale as current)
        node_feats['out_strength_lag1'] = np.log1p(prev_out).reindex(node_feats.index).fillna(0.0)
        node_feats['in_strength_lag1'] = np.log1p(prev_in).reindex(node_feats.index).fillna(0.0)
    
    # Merge external data if provided
    if external_data is not None:
        node_feats = node_feats.join(external_data, how='left')
        for col in external_data.columns:
            if col in node_feats.columns:
                node_feats[col] = node_feats[col].fillna(node_feats[col].median())
    
    # Ensure numeric dtypes
    for col in node_feats.columns:
        node_feats[col] = pd.to_numeric(node_feats[col], errors="coerce").fillna(0.0)
    
    return node_feats

def build_edge_features(edges_t: pd.DataFrame,
                       node_feats_t: pd.DataFrame,
                       edges_prev: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Build edge features with 1-year lag only.
    """
    edges = edges_t.copy()
    
    # Basic features
    node_in_strength = node_feats_t.get("in_strength", pd.Series(dtype=float))
    node_out_strength = node_feats_t.get("out_strength", pd.Series(dtype=float))
    
    tot_in = edges["target"].map(node_in_strength).replace({0.0: np.nan})
    tot_out = edges["source"].map(node_out_strength).replace({0.0: np.nan})
    
    edges["log_value_t"] = np.log1p(edges["value"].astype(float))
    edges["share_target"] = (edges["value"].astype(float) / tot_in).fillna(0.0)
    edges["share_source"] = (edges["value"].astype(float) / tot_out).fillna(0.0)
    
    # Add 1-year lag features
    if edges_prev is not None:
        prev_edges = edges_prev[['source', 'target', 'value']].rename(
            columns={'value': 'value_prev'}
        )
        edges = edges.merge(prev_edges, on=['source', 'target'], how='left')
        edges['value_prev'] = edges['value_prev'].fillna(0.0)
        
        # Growth rate
        edges['edge_growth'] = (
            (edges['value'] - edges['value_prev']) / edges['value_prev'].replace(0, np.nan)
        ).fillna(0.0)
        
        # Lagged log value
        edges['log_value_lag1'] = np.log1p(edges['value_prev'])
        
        edges = edges.drop(columns=['value_prev'])
    
    return edges

def label_trade_drops(edges_t: pd.DataFrame, 
                     edges_t1: pd.DataFrame,
                     edges_prev: Optional[pd.DataFrame] = None,
                     drop_threshold: float = 0.1) -> pd.DataFrame:
    """
    Label trade drops with explicit tracking of edge history.
    
    Edge Categories:
    1. ESTABLISHED edges: Existed at t-1 AND exist at t (can be labeled for drops)
    2. NEW edges: Did NOT exist at t-1 but exist at t (cannot be labeled - no baseline)
    3. DISAPPEARED edges: Existed at t-1 but dropped to zero at t (informative but can't predict future)
    4. WILL_DROP edges: Exist at t and will drop >threshold at t+1 (positive label)
    5. WILL_DISAPPEAR edges: Exist at t and will be zero at t+1 (extreme positive label)
    
    Labeling Logic:
    - Only ESTABLISHED edges can receive drop labels (y_drop, y_disappear)
    - NEW edges get y_drop=0, y_disappear=0, but flagged as 'is_new_edge'=True
    - This ensures model only trains on edges with historical context
    
    Args:
        edges_t: Edges at time t (current year)
        edges_t1: Edges at time t+1 (next year, for labels)
        edges_prev: Edges at time t-1 (previous year, for history)
        drop_threshold: Threshold for drop label (default 0.1 = 10% drop)
    
    Returns:
        DataFrame with columns:
        - source, target: Edge endpoints
        - value_t, value_t1: Edge values at t and t+1
        - pct_change: Percent change from t to t+1
        - y_drop: 1 if established edge drops >threshold, 0 otherwise
        - y_disappear: 1 if established edge goes to zero, 0 otherwise
        - is_new_edge: True if edge didn't exist at t-1
        - is_established: True if edge existed at both t-1 and t
        - existed_prev: True if edge existed at t-1 (regardless of t)
    """
    # Get all edges that should be tracked at time t
    if edges_prev is not None:
        # Track union: edges from t-1 OR t
        all_edge_pairs = pd.concat([
            edges_prev[['source', 'target']],
            edges_t[['source', 'target']]
        ]).drop_duplicates()
    else:
        # First year: only track edges active at t (all are "new")
        all_edge_pairs = edges_t[['source', 'target']].copy()
    
    # Merge values from t-1, t, and t+1
    merged = all_edge_pairs.copy()
    
    # Add t-1 values (for history tracking)
    if edges_prev is not None:
        merged = merged.merge(
            edges_prev[['source', 'target', 'value']].rename(columns={'value': 'value_prev'}),
            on=['source', 'target'],
            how='left'
        )
        merged['value_prev'] = merged['value_prev'].fillna(0.0)
    else:
        merged['value_prev'] = 0.0
    
    # Add t values (current)
    merged = merged.merge(
        edges_t[['source', 'target', 'value']].rename(columns={'value': 'value_t'}),
        on=['source', 'target'],
        how='left'
    )
    merged['value_t'] = merged['value_t'].fillna(0.0)
    
    # Add t+1 values (for labels)
    merged = merged.merge(
        edges_t1[['source', 'target', 'value']].rename(columns={'value': 'value_t1'}),
        on=['source', 'target'],
        how='left'
    )
    merged['value_t1'] = merged['value_t1'].fillna(0.0)
    
    # === CRITICAL: Determine edge history status ===
    merged['existed_prev'] = merged['value_prev'] > 0  # Existed at t-1
    merged['exists_now'] = merged['value_t'] > 0       # Exists at t
    merged['is_new_edge'] = (~merged['existed_prev']) & merged['exists_now']
    merged['is_established'] = merged['existed_prev'] & merged['exists_now']
    
    # Calculate percent change (only meaningful for established edges)
    merged['pct_change'] = 0.0
    established_mask = merged['is_established']
    
    if established_mask.any():
        merged.loc[established_mask, 'pct_change'] = (
            (merged.loc[established_mask, 'value_t1'] - merged.loc[established_mask, 'value_t']) /
            merged.loc[established_mask, 'value_t']
        )
    
    # === LABELS: Only for established edges that exist at t ===
    # New edges cannot be labeled for drops (no baseline)
    # Disappeared edges (existed prev, zero at t) are excluded from active predictions
    
    active_edges = merged['exists_now']  # Only edges that exist at t can be predicted
    
    merged['y_drop'] = 0
    merged['y_disappear'] = 0
    
    # Label established edges that will drop
    drop_mask = established_mask & active_edges & (merged['pct_change'] < -drop_threshold)
    merged.loc[drop_mask, 'y_drop'] = 1
    
    # Label established edges that will completely disappear
    disappear_mask = established_mask & active_edges & (merged['value_t1'] == 0)
    merged.loc[disappear_mask, 'y_disappear'] = 1
    
    # Clean up helper columns (keep informative ones)
    merged = merged.drop(columns=['value_prev', 'exists_now', 'existed_prev'])
    
    return merged

def icio_to_edges(df: pd.DataFrame, sector_cols: list) -> pd.DataFrame:
    """Convert ICIO table to edge list."""
    df = df.loc[sector_cols, sector_cols]
    edges = df.stack().reset_index()
    edges.columns = ['source', 'target', 'value']
    edges = edges[edges["value"] > 0].dropna(subset=["value"])
    return edges

def standardize_df_columns(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """Standardize specified columns in-place for memory efficiency."""
    df = df.copy()
    for c in cols:
        if c not in df.columns:
            continue
        vals = df[c].astype(float).values
        mu = np.nanmean(vals)
        sd = np.nanstd(vals)
        if sd == 0 or np.isnan(sd):
            df[c] = 0.0
        else:
            df[c] = (vals - mu) / sd
    return df

def build_graph(
    df_t: pd.DataFrame,
    df_t1: pd.DataFrame,
    sector_cols: List[str],
    drop_threshold: float = 0.1,
    global_node_index: Optional[pd.Index] = None,
    edges_prev: Optional[pd.DataFrame] = None,
    external_data: Optional[pd.DataFrame] = None,
    year: Optional[int] = None,
    start_year: Optional[int] = None
) -> Tuple[Data, pd.DataFrame, pd.Index]:
    """
    Build graph with 1-year temporal features.
    Memory optimized for 16GB RAM.
    """
    # 1) Build edges
    edges_t = icio_to_edges(df_t, sector_cols)
    edges_t1 = icio_to_edges(df_t1, sector_cols)
    
    # 2) Node features with 1-year lag
    node_feats_t = build_node_features(edges_t, edges_prev, external_data)
    
    # 3) Edge features with 1-year lag
    edges_feat_t = build_edge_features(edges_t, node_feats_t, edges_prev)
    
    # 4) Labels
    labels = label_trade_drops(edges_t, edges_t1, edges_prev, drop_threshold)
    
    # 5) Merge
    merged = edges_feat_t.merge(
        labels[["source", "target", "value_t", "value_t1", "pct_change", "y_drop"]],
        on=["source", "target"],
        how="left"
    )
    
    # 6) Node indexing
    if global_node_index is None:
        nodes = pd.Index(sorted(set(merged["source"]) | set(merged["target"])))
    else:
        nodes = global_node_index
    node_id_map = {n: i for i, n in enumerate(nodes)}
    
    merged = merged[merged["source"].isin(nodes) & merged["target"].isin(nodes)].copy()
    
    # 7) Identify feature columns dynamically
    exclude_cols = {'source', 'target', 'value_t', 'value_t1', 'pct_change', 
                   'y_drop', 'src_id', 'tgt_id', 'value'}
    edge_feat_cols = [c for c in merged.columns if c not in exclude_cols]
    edge_feat_cols = [c for c in edge_feat_cols if merged[c].dtype in 
                     ['float64', 'float32', 'int64', 'int32']]
    
    # 8) Standardize
    merged_std = standardize_df_columns(merged, edge_feat_cols)
    
    # 9) Build edge tensors
    merged_std["src_id"] = merged_std["source"].map(node_id_map)
    merged_std["tgt_id"] = merged_std["target"].map(node_id_map)
    
    edge_index = torch.tensor(
        merged_std[["src_id", "tgt_id"]].values.T, 
        dtype=torch.long
    )
    edge_attr = torch.tensor(
        merged_std[edge_feat_cols].values, 
        dtype=torch.float32
    )
    edge_y = torch.tensor(
        merged_std["y_drop"].values, 
        dtype=torch.float32
    )
    
    # 10) Build node feature matrix
    node_feats_aligned = node_feats_t.reindex(nodes).fillna(0.0)
    node_feat_cols = list(node_feats_aligned.columns)
    node_feats_aligned_std = standardize_df_columns(node_feats_aligned, node_feat_cols)
    x = torch.tensor(
        node_feats_aligned_std[node_feat_cols].values, 
        dtype=torch.float32
    )
    
    # 11) Optional: Add year encoding (cheap feature)
    if year is not None:
        year_normalized = (year - start_year) / 20.0
        time_feat = torch.full((x.shape[0], 1), year_normalized, dtype=torch.float32)
        x = torch.cat([x, time_feat], dim=1)
    
    # 12) Assemble graph
    value_t = torch.tensor(
        merged_std["value_t"].values, 
        dtype=torch.float32
    )
    value_t1 = torch.tensor(
        merged_std["value_t1"].values, 
        dtype=torch.float32
    )
    
    graph = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=edge_y,
        value_t=value_t,
        value_t1=value_t1,
        num_nodes=len(nodes),
        node_labels=list(nodes),
        node_id_to_idx={n: i for i, n in enumerate(nodes)},
        edge_labels=list(zip(merged_std["source"], merged_std["target"]))
    )
    
    return graph, merged_std, nodes
 
def main():
    """
    Memory-efficient pipeline for 16GB RAM.
    Only keeps previous year's edges in memory.
    Usage: python feature_eng.py --start-year [start year] --end-year [end year]
    """

    ap = argparse.ArgumentParser()
    ap.add_argument("--start-year", type=int, default=1996)
    ap.add_argument("--end-year", type=int, default=2022)
    args = ap.parse_args()

    # Fixed node index from first year
    print("Reading first year ICIO table to establish node index...")
    df_first, cols = read_icio(f"ICIO/{args.start_year}_SML.csv")
    nodes_global = pd.Index(sorted(cols))
    
    edges_prev = None  # No lag for first year
    df_t = None
    cols_t = None

    # Load previous year if it exists (for lag features)
    prev_year = args.start_year - 1
    prev_file = f"ICIO/{prev_year}_SML.csv"
    if os.path.exists(prev_file):
        print(f"Loading {prev_year} for lag features...")
        df_prev, cols_prev = read_icio(prev_file)
        edges_prev = icio_to_edges(df_prev, cols_prev)
        print(f"  Lag features will be available starting from {args.start_year}")
    else:
        print(f"No {prev_year} data found - first year will have no lag features")
    
    # Process all years in single loop
    for year in range(args.start_year, args.end_year):
        # Load current year if not already loaded
        if df_t is None:
            print(f"\nReading {year} ICIO table...")
            df_t, cols_t = read_icio(f"ICIO/{year}_SML.csv")
        
        # Load next year
        print(f"Reading {year+1} ICIO table...")
        df_t1, cols_t1 = read_icio(f"ICIO/{year+1}_SML.csv")
        
        # Generate graph
        lag_status = "with lag features" if edges_prev is not None else "(no lag - first year)"
        print(f"Generating {year} graph {lag_status}...")
        external_data = load_indicators(year, nodes_global)
        graph, edges_df, _ = build_graph(
            df_t, df_t1, cols_t,
            drop_threshold=0.1,
            global_node_index=nodes_global,
            edges_prev=edges_prev,
            external_data=external_data,
            year=year,
            start_year=args.start_year
        )
        
        torch.save(graph, f"embeddings/graph_{year}_labeled.pt")
        print(f"Saved {year} graph. Nodes: {graph.num_nodes}, Edges: {graph.edge_index.shape[1]}")
        print(f"  Node features: {graph.x.shape[1]}, Edge features: {graph.edge_attr.shape[1]}")
        print(f"  Positive class: {graph.y.sum().item()}/{len(graph.y)} ({100*graph.y.mean():.2f}%)")
        
        # Update for next iteration (memory efficient)
        edges_prev = icio_to_edges(df_t, cols_t)
        df_t = df_t1
        cols_t = cols_t1
    
    print("\n✓ Pipeline complete!")
    print(f"Generated graphs for years {args.start_year}-{args.end_year-1}")
    print(f"Saved to embeddings/ directory")

if __name__ == "__main__":
    main()