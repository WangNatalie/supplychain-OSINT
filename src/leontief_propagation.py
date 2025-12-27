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
"""

import torch
import numpy as np
import pandas as pd
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
from scipy import sparse
from scipy.sparse.linalg import spsolve

# Import ICIO parser for plain English formatting
from ICIO.ICIO_parser import (
    format_node_name,
    format_sector_name,
    format_country_name
)


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
        
        # Compute Leontief inverse (I - A)^(-1)
        print("Computing Leontief inverse (I - A)^(-1)...")
        self.L = self._compute_leontief_inverse()
        
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
        
        # Compute total output per sector (sum of all outputs)
        total_output = np.zeros(self.num_nodes)
        src_idx, tgt_idx = edge_index
        
        for i in range(len(src_idx)):
            src = src_idx[i]
            value = edge_values[i]
            total_output[src] += value
        
        # Build technical coefficients: a_ij = input_ij / output_j
        # where input_ij is how much i provides to j
        row_indices = []
        col_indices = []
        coefficients = []
        
        for i in range(len(src_idx)):
            src = src_idx[i]
            tgt = tgt_idx[i]
            value = edge_values[i]
            
            # Technical coefficient: input per unit output
            if total_output[tgt] > 0:
                a_ij = value / total_output[tgt]
                row_indices.append(src)
                col_indices.append(tgt)
                coefficients.append(a_ij)
        
        # Create sparse matrix
        A = sparse.coo_matrix(
            (coefficients, (row_indices, col_indices)),
            shape=(self.num_nodes, self.num_nodes)
        )
        
        return A.tocsr()
    
    def _compute_leontief_inverse(self) -> sparse.csr_matrix:
        """
        Compute Leontief inverse L = (I - A)^(-1)
        
        This matrix captures all direct and indirect effects:
        - Direct: immediate suppliers/customers
        - Indirect: cascading effects through supply chain
        
        Returns:
            Leontief inverse matrix L
        """
        I = sparse.identity(self.num_nodes, format='csr')
        I_minus_A = I - self.A
        
        # For large matrices, we'll keep it sparse and use iterative solving
        # Converting to dense only for small networks
        if self.num_nodes < 5000:
            # Small network: compute dense inverse
            L_dense = np.linalg.inv(I_minus_A.toarray())
            return sparse.csr_matrix(L_dense)
        else:
            # Large network: keep sparse (solve on-demand per shock)
            return I_minus_A
    
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
        
        # Compute total impact: Δx = L @ Δf
        if self.num_nodes < 5000:
            # Small network: dense matrix multiplication
            impact_vector = self.L @ shock_vector
        else:
            # Large network: solve sparse system (I - A) @ x = shock_vector
            impact_vector = spsolve(self.L, shock_vector)
        
        return impact_vector
    
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
    
    parser.add_argument("--graph", required=True,
                       help="Path to graph file (e.g., embeddings/graph_2021_labeled.pt)")
    parser.add_argument("--shocked-nodes", nargs="+",
                       help="Node IDs to shock (e.g., USA_C26 CHN_C26)")
    parser.add_argument("--magnitude", type=float, default=0.5,
                       help="Shock magnitude as fraction (0.5 = 50%% reduction)")
    parser.add_argument("--output", type=str,
                       help="Path to save detailed results CSV (optional)")
    parser.add_argument("--save-matrix", action="store_true",
                       help="Save Leontief inverse matrix (for reuse)")
    
    args = parser.parse_args()
    
    # Load graph
    print(f"Loading graph from {args.graph}...")
    graph = torch.load(args.graph, map_location='cpu', weights_only=False)
    
    # Initialize Leontief propagator
    propagator = LeontiefPropagator(graph)
    
    # Save matrix if requested
    if args.save_matrix:
        matrix_path = Path(args.graph).parent / "leontief_inverse.npz"
        sparse.save_npz(matrix_path, propagator.L)
        print(f"✓ Saved Leontief inverse to {matrix_path}")
    
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

