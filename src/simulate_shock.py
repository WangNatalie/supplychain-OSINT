#!/usr/bin/env python3
"""
simulate_shock.py - Counterfactual Shock Analysis for Supply Chain Networks

Nodes can be found in row/column labels of ICIO tables.

Simulates "what-if" scenarios: What happens if Ecuador's agriculture drops 50%?

Usage Examples:
    # Simulate single node shock
    python simulate_shock.py \
        --model models/shock_propagation/best_model.pt \
        --graph embeddings/graph_2021_labeled.pt \
        --shocked-nodes ECU_A01 \
        --magnitude 0.5

    # Simulate multi-node shock (e.g., regional crisis)
    python simulate_shock.py \
        --model models/shock_propagation/best_model.pt \
        --graph embeddings/graph_2021_labeled.pt \
        --shocked-nodes ECU_A01 ECU_A02 ECU_A03 \
        --magnitude 0.3

    # Simulate edge shock (test direct edge response)
    python simulate_shock.py \
        --model models/shock_propagation/best_model.pt \
        --graph embeddings/graph_2021_labeled.pt \
        --shocked-edges USA_C26->CHN_C26 \
        --magnitude 0.2

    # Simulate combined node + edge shock
    python simulate_shock.py \
        --model models/shock_propagation/best_model.pt \
        --graph embeddings/graph_2021_labeled.pt \
        --shocked-nodes CHN_A01 \
        --shocked-edges USA_C26->CHN_C26 DEU_C26->USA_C26 \
        --magnitude 0.4
"""

import torch
import argparse
import pandas as pd
import numpy as np
from typing import List, Dict
from pathlib import Path
import sys

# Import ICIO parser functions for plain English formatting
from ICIO.ICIO_parser import (
    format_node_name,
    format_sector_name,
    format_country_name
)

class ShockSimulator:
    """Handles shock simulation and analysis"""
    
    def __init__(self, model, graph, device):
        self.model = model
        self.graph = graph
        self.device = device
        self.model.eval()
        
        if not hasattr(graph, 'node_id_to_idx'):
            raise ValueError(
                "Graph missing node_id_to_idx mapping. "
                "Please re-run feature_eng.py with updated build_graph_with_labels()."
            )
    
    def create_shock_mask(self, shocked_nodes: List[str], shock_magnitude: float) -> torch.Tensor:
        """
        Create magnitude-based shock mask for specified nodes.
        
        Args:
            shocked_nodes: List of node IDs to shock (e.g., ['USA_A01'])
            shock_magnitude: Actual shock magnitude (e.g., -0.20 for 20% reduction)
        
        Returns:
            Shock mask tensor with magnitude values (not binary)
        """
        shock_mask = torch.zeros(self.graph.num_nodes, device=self.device)
        valid_nodes = []
        
        for node_id in shocked_nodes:
            if node_id not in self.graph.node_id_to_idx:
                print(f"Warning: Node '{node_id}' not found in graph")
                print(f"  Available nodes (sample): {self.graph.node_labels[:5]}...")
                continue
            
            idx = self.graph.node_id_to_idx[node_id]
            shock_mask[idx] = shock_magnitude  # Actual magnitude, not binary 1
            valid_nodes.append(node_id)
        
        if len(valid_nodes) == 0:
            raise ValueError("No valid nodes found to shock!")
        
        # Print in plain English
        print(f"✓ Shocking {len(valid_nodes)} node(s) with {abs(shock_magnitude):.0%} reduction:")
        for node in valid_nodes:
            print(f"  • {format_node_name(node, include_code=True)}")
        
        return shock_mask
    
    def create_edge_shock_mask(self, shocked_edges: List[str], shock_magnitude: float) -> torch.Tensor:
        """
        Create magnitude-based shock mask for specified edges.
        
        Args:
            shocked_edges: List of edge specifications (e.g., ['USA_A01->CHN_A01', 'DEU_C26->USA_C26'])
            shock_magnitude: Actual shock magnitude (e.g., -0.20 for 20% reduction)
        
        Returns:
            Shock mask tensor with magnitude values (not binary)
        """
        shock_mask = torch.zeros(self.graph.edge_index.shape[1], device=self.device)
        valid_edges = []
        
        # Build edge lookup for efficient matching
        src_idx, tgt_idx = self.graph.edge_index.cpu().numpy()
        edge_to_idx = {}
        for i in range(len(src_idx)):
            src_node = self.graph.node_labels[src_idx[i]]
            tgt_node = self.graph.node_labels[tgt_idx[i]]
            edge_key = f"{src_node}->{tgt_node}"
            edge_to_idx[edge_key] = i
        
        for edge_spec in shocked_edges:
            if '->' not in edge_spec:
                print(f"Warning: Invalid edge format '{edge_spec}'. Use format 'SOURCE->TARGET'")
                continue
            
            if edge_spec not in edge_to_idx:
                print(f"Warning: Edge '{edge_spec}' not found in graph")
                # Show similar edges for debugging
                src_node = edge_spec.split('->')[0]
                similar = [k for k in list(edge_to_idx.keys())[:10] if k.startswith(src_node)]
                if similar:
                    print(f"  Similar edges: {similar[:3]}")
                continue
            
            idx = edge_to_idx[edge_spec]
            shock_mask[idx] = shock_magnitude
            valid_edges.append(edge_spec)
        
        if len(valid_edges) == 0:
            raise ValueError("No valid edges found to shock!")
        
        # Print in plain English
        print(f"✓ Shocking {len(valid_edges)} edge(s) with {abs(shock_magnitude):.0%} reduction:")
        for edge_spec in valid_edges:
            src, tgt = edge_spec.split('->')
            src_name = format_node_name(src)
            tgt_name = format_node_name(tgt)
            print(f"  • {src_name} → {tgt_name}")
        
        return shock_mask
    
    def run_simulation(self, shock_mask_nodes: torch.Tensor = None, 
                      shock_mask_edges: torch.Tensor = None) -> Dict[str, np.ndarray]:
        """
        Run baseline and shocked predictions.
        
        Args:
            shock_mask_nodes: Magnitude-based shock mask for nodes (e.g., -0.20 for shocked nodes)
            shock_mask_edges: Magnitude-based shock mask for edges (e.g., -0.20 for shocked edges)
        
        Returns:
            Dict with baseline, shocked, and propagation effect predictions
        
        Note: Shock masks already contain the actual magnitude, so model
              receives the correct signal directly (no post-processing needed)
        """
        with torch.no_grad():
            # Baseline prediction (business as usual)
            baseline_delta = self.model(
                self.graph.x,
                self.graph.edge_index,
                self.graph.edge_attr,
                shock_mask_nodes=None,
                shock_mask_edges=None
            ).cpu().numpy()
            
            # Shocked prediction (model now receives actual magnitude values)
            shocked_delta = self.model(
                self.graph.x,
                self.graph.edge_index,
                self.graph.edge_attr,
                shock_mask_nodes=shock_mask_nodes,  # Contains actual magnitude
                shock_mask_edges=shock_mask_edges   # Contains actual magnitude
            ).cpu().numpy()
        
        # No post-processing needed! Model learned magnitude relationships during training
        propagation_effect = shocked_delta - baseline_delta
        
        return {
            'baseline_delta': baseline_delta,
            'shocked_delta': shocked_delta,
            'propagation_effect': propagation_effect
        }
    
    def analyze_results(self, 
                       predictions: Dict[str, np.ndarray],
                       shocked_nodes: List[str] = None,
                       shocked_edges: List[str] = None) -> pd.DataFrame:
        """Convert predictions to interpretable DataFrame (codes only, format on display)"""
        
        if shocked_nodes is None:
            shocked_nodes = []
        if shocked_edges is None:
            shocked_edges = []
        
        # Extract edge information
        src_idx, tgt_idx = self.graph.edge_index.cpu().numpy()
        value_t = self.graph.value_t.cpu().numpy()
        
        # Reconstruct actual values from log-space predictions
        log_value_t = np.log1p(value_t)
        baseline_value_t1 = np.expm1(log_value_t + predictions['baseline_delta'])
        shocked_value_t1 = np.expm1(log_value_t + predictions['shocked_delta'])
        
        # Extract codes only (no expensive string formatting yet)
        source_codes = [self.graph.node_labels[i] for i in src_idx]
        target_codes = [self.graph.node_labels[i] for i in tgt_idx]
        
        source_countries = [node.split('_')[0] if '_' in node else node for node in source_codes]
        target_countries = [node.split('_')[0] if '_' in node else node for node in target_codes]
        source_sectors = [node.split('_')[1] if '_' in node else 'UNK' for node in source_codes]
        target_sectors = [node.split('_')[1] if '_' in node else 'UNK' for node in target_codes]
        
        # Build results DataFrame with codes only
        results = pd.DataFrame({
            # Codes (we'll format to plain English only when displaying)
            'source': source_codes,
            'target': target_codes,
            'source_country': source_countries,
            'target_country': target_countries,
            'source_sector': source_sectors,
            'target_sector': target_sectors,
            
            # Values
            'value_t': value_t,
            'baseline_value_t1': baseline_value_t1,
            'shocked_value_t1': shocked_value_t1,
            'absolute_change': shocked_value_t1 - baseline_value_t1,
            'pct_change': ((shocked_value_t1 - baseline_value_t1) / (baseline_value_t1 + 1e-8)) * 100,
            'propagation_effect_log': predictions['propagation_effect']
        })
        
        # Create edge keys for matching
        results['edge_key'] = results['source'] + '->' + results['target']
        
        # Categorize edge relationships to shock
        results['edge_type'] = 'indirect'
        
        # Mark directly shocked edges
        if shocked_edges:
            results.loc[results['edge_key'].isin(shocked_edges), 'edge_type'] = 'shocked_edge'
        
        # Mark edges connected to shocked nodes (if not already marked as shocked edge)
        if shocked_nodes:
            results.loc[
                (results['source'].isin(shocked_nodes)) & (results['edge_type'] == 'indirect'),
                'edge_type'
            ] = 'direct_outgoing'
            results.loc[
                (results['target'].isin(shocked_nodes)) & (results['edge_type'] == 'indirect'),
                'edge_type'
            ] = 'direct_incoming'
            results.loc[
                (results['source'].isin(shocked_nodes)) & 
                (results['target'].isin(shocked_nodes)) & 
                (results['edge_type'] == 'indirect'),
                'edge_type'
            ] = 'internal'
        
        # Compute impact magnitude (by absolute dollars)
        results['abs_change'] = np.abs(results['absolute_change'])
        
        return results.sort_values('abs_change', ascending=False)
    
    def print_summary(self, results: pd.DataFrame, shocked_nodes: List[str] = None, 
                     shocked_edges: List[str] = None):
        """Print comprehensive shock analysis summary"""
        
        if shocked_nodes is None:
            shocked_nodes = []
        if shocked_edges is None:
            shocked_edges = []
        
        print("\n" + "="*80)
        print("SHOCK PROPAGATION ANALYSIS")
        print("="*80)
        
        # Format shocked nodes in plain English
        if shocked_nodes:
            shocked_names = [format_node_name(node, include_code=True) for node in shocked_nodes]
            print(f"\nShocked Nodes:")
            for name in shocked_names:
                print(f"  • {name}")
        
        # Format shocked edges in plain English
        if shocked_edges:
            print(f"\nShocked Edges:")
            for edge_spec in shocked_edges:
                src, tgt = edge_spec.split('->')
                src_name = format_node_name(src)
                tgt_name = format_node_name(tgt)
                print(f"  • {src_name} → {tgt_name}")
        
        print(f"\nTotal Edges Analyzed: {len(results):,}")
        
        # Breakdown by edge type
        print("\n" + "-"*80)
        print("DIRECT EFFECTS")
        print("-"*80)
        
        # Show shocked edges first if present
        if shocked_edges:
            subset = results[results['edge_type'] == 'shocked_edge']
            if len(subset) > 0:
                print(f"\nShocked Edges (Direct):")
                print(f"  Count: {len(subset):,} edges")
                print(f"  Total value at risk: ${subset['value_t'].sum():,.0f}")
                print(f"  Mean % change: {subset['pct_change'].mean():.2f}%")
                print(f"  Total absolute change: ${subset['absolute_change'].sum():,.0f}")
                print(f"  Median % change: {subset['pct_change'].median():.2f}%")
        
        # Show node-connected edges
        for edge_type in ['direct_outgoing', 'direct_incoming', 'internal']:
            subset = results[results['edge_type'] == edge_type]
            if len(subset) == 0:
                continue
            
            print(f"\n{edge_type.replace('_', ' ').title()}:")
            print(f"  Count: {len(subset):,} edges")
            print(f"  Total value at risk: ${subset['value_t'].sum():,.0f}")
            print(f"  Mean % change: {subset['pct_change'].mean():.2f}%")
            print(f"  Total absolute change: ${subset['absolute_change'].sum():,.0f}")
            print(f"  Median % change: {subset['pct_change'].median():.2f}%")
        
        # Indirect effects
        print("\n" + "-"*80)
        print("INDIRECT EFFECTS (Propagation through network)")
        print("-"*80)
        
        indirect = results[results['edge_type'] == 'indirect']
        
        if len(indirect) > 0:
            significant_1pct = (np.abs(indirect['pct_change']) > 1).sum()
            significant_5pct = (np.abs(indirect['pct_change']) > 5).sum()
            significant_10pct = (np.abs(indirect['pct_change']) > 10).sum()
            
            print(f"\nEdges with significant propagation:")
            print(f"  >1% change:  {significant_1pct:,} ({100*significant_1pct/len(indirect):.2f}%)")
            print(f"  >5% change:  {significant_5pct:,} ({100*significant_5pct/len(indirect):.2f}%)")
            print(f"  >10% change: {significant_10pct:,} ({100*significant_10pct/len(indirect):.2f}%)")
            print(f"\n  Max indirect effect: {indirect['pct_change'].abs().max():.2f}%")
            print(f"  Mean indirect effect: {indirect['pct_change'].mean():.2f}%")
            print(f"  Std dev: {indirect['pct_change'].std():.2f}%")
        
        # Country-level aggregation
        print("\n" + "-"*80)
        print("COUNTRY-LEVEL IMPACT (Top 10 affected countries)")
        print("-"*80)
        
        country_impact = results.groupby('target_country').agg({
            'absolute_change': 'sum',
            'pct_change': 'mean',
            'value_t': 'sum'
        }).sort_values('absolute_change', key=abs, ascending=False).head(10)
        
        print("\n{:<30} {:>15} {:>12} {:>15}".format(
            "Country", "Total Impact", "Avg % Change", "Original Value"
        ))
        print("-"*80)
        for country_code, row in country_impact.iterrows():
            country_name = format_country_name(country_code)
            print("{:<30} ${:>14,.0f} {:>11.2f}% ${:>14,.0f}".format(
                country_name[:28],
                row['absolute_change'],
                row['pct_change'],
                row['value_t']
            ))
        
        # Sector-level aggregation
        print("\n" + "-"*80)
        print("SECTOR-LEVEL IMPACT (Top 10 affected sectors)")
        print("-"*80)
        
        sector_impact = results.groupby('target_sector').agg({
            'absolute_change': 'sum',
            'pct_change': 'mean',
            'value_t': 'sum'
        }).sort_values('absolute_change', key=abs, ascending=False).head(10)
        
        print("\n{:<50} {:>15} {:>12}".format(
            "Sector", "Total Impact", "Avg % Change"
        ))
        print("-"*80)
        for sector_code, row in sector_impact.iterrows():
            sector_name = format_sector_name(sector_code)
            display_name = sector_name[:48] if len(sector_name) > 48 else sector_name
            print("{:<50} ${:>14,.0f} {:>11.2f}%".format(
                display_name,
                row['absolute_change'],
                row['pct_change']
            ))
        
        # Node-level aggregation (country_sector combinations)
        print("\n" + "-"*80)
        print("NODE-LEVEL IMPACT (Top 10 affected country-sectors)")
        print("-"*80)
        
        node_impact = results.groupby('target').agg({
            'absolute_change': 'sum',
            'pct_change': 'mean',
            'value_t': 'sum'
        }).sort_values('absolute_change', key=abs, ascending=False).head(10)
        
        print("\n{:<50} {:>15} {:>12}".format(
            "Country-Sector", "Total Impact", "Avg % Change"
        ))
        print("-"*80)
        for node_code, row in node_impact.iterrows():
            node_name = format_node_name(node_code)
            display_name = node_name[:48] if len(node_name) > 48 else node_name
            print("{:<50} ${:>14,.0f} {:>11.2f}%".format(
                display_name,
                row['absolute_change'],
                row['pct_change']
            ))
        
        # Top affected edges
        print("\n" + "-"*80)
        print("TOP 15 MOST AFFECTED TRADING FLOWS")
        print("-"*80)
        
        top_edges = results.head(15)
        
        print("\n{:<40} {:<40} {:>15} {:>10}".format(
            "Source", "Target", "$ Change", "% Change"
        ))
        print("-"*80)
        for _, row in top_edges.iterrows():
            # Format codes to plain English only for display
            source_name = format_node_name(row['source'])
            target_name = format_node_name(row['target'])
            
            # Truncate long names for display
            source_display = source_name[:38] if len(source_name) > 38 else source_name
            target_display = target_name[:38] if len(target_name) > 38 else target_name
            
            print("{:<40} {:<40} ${:>14,.0f} {:>9.2f}%".format(
                source_display,
                target_display,
                row['absolute_change'],
                row['pct_change']
            ))
        
        print("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(
        description="Simulate counterfactual shocks in supply chain networks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument("--model", required=True,
                       help="Path to trained model checkpoint")
    parser.add_argument("--graph", required=True,
                       help="Path to graph file (e.g., embeddings/graph_2021_labeled.pt)")
    parser.add_argument("--shocked-nodes", nargs="+",
                       help="Node IDs to shock (e.g., ECU_AGR CHN_MFG)")
    parser.add_argument("--shocked-edges", nargs="+",
                       help="Edge specifications to shock (e.g., USA_C26->CHN_C26)")
    parser.add_argument("--magnitude", type=float, default=0.5,
                       help="Shock magnitude as fraction (0.5 = 50%% reduction)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--list-nodes", action="store_true",
                       help="List all available nodes in the graph and exit")
    parser.add_argument("--output", type=str,
                       help="Path to save detailed results CSV (optional)")
    
    args = parser.parse_args()
    
    # Load graph
    print(f"Loading graph from {args.graph}...")
    graph = torch.load(args.graph, map_location=args.device, weights_only=False)
    
    # List nodes if requested
    if args.list_nodes:
        print(f"\nAvailable nodes in graph ({len(graph.node_labels)} total):")
        print("="*80)
        
        # Group by country
        nodes_by_country = {}
        for node in graph.node_labels:
            if '_' in node:
                country_code = node.split('_')[0]
                sector_code = node.split('_')[1]
            else:
                country_code = node
                sector_code = 'N/A'
            
            if country_code not in nodes_by_country:
                nodes_by_country[country_code] = []
            nodes_by_country[country_code].append((node, sector_code))
        
        for country_code in sorted(nodes_by_country.keys()):
            nodes = nodes_by_country[country_code]
            country_name = format_country_name(country_code)
            
            print(f"\n{country_name} [{country_code}] ({len(nodes)} sectors):")
            print("-" * 80)
            
            for node, sector_code in sorted(nodes):
                if sector_code != 'N/A':
                    # Show: plain English name (and code for reference)
                    node_name = format_node_name(node)
                    print(f"  {node:<15} → {node_name}")
                else:
                    print(f"  {node:<15}")
        
        return
    
    # Load model
    print(f"Loading model from {args.model}...")
    checkpoint = torch.load(args.model, map_location=args.device, weights_only=False)
    
    # Import model class (assume it's in training.py)
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        from training import ShockPropagationGNN
    except ImportError:
        print("Error: Cannot import ShockPropagationGNN from training.py")
        print("Make sure training.py is in the same directory or in PYTHONPATH")
        return
    
    # Reconstruct model
    model_args = checkpoint['args']
    sample_graph = graph
    
    model = ShockPropagationGNN(
        node_in_dim=sample_graph.x.shape[1],
        edge_in_dim=sample_graph.edge_attr.shape[1],
        hidden_dim=model_args.get('hidden_dim', 128),
        num_layers=model_args.get('num_layers', 3),
        dropout=model_args.get('dropout', 0.3),
        use_attention=model_args.get('use_attention', False)
    ).to(args.device)
    
    state_dict = checkpoint['model_state_dict']
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('edge_mlp.7.'):
            new_key = key.replace('edge_mlp.7.', 'edge_mlp.6.')
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
    
    model.load_state_dict(new_state_dict, strict=False)
    print(f"✓ Model loaded (trained for {checkpoint['epoch']} epochs)")
    
    # Initialize simulator
    simulator = ShockSimulator(model, graph, args.device)
    
    # Validate that at least one shock type is specified
    if not args.shocked_nodes and not args.shocked_edges:
        print("Error: Must specify either --shocked-nodes or --shocked-edges (or both)")
        return
    
    # Run simulation
    print(f"\n{'='*80}")
    shock_desc = []
    if args.shocked_nodes:
        shock_desc.append(f"{len(args.shocked_nodes)} nodes")
    if args.shocked_edges:
        shock_desc.append(f"{len(args.shocked_edges)} edges")
    print(f"SIMULATING SHOCK: {args.magnitude:.1%} reduction in {' and '.join(shock_desc)}")
    print(f"{'='*80}")
    
    # Convert magnitude to negative (reduction)
    shock_magnitude = -abs(args.magnitude)
    
    # Create shock masks
    shock_mask_nodes = None
    shock_mask_edges = None
    
    if args.shocked_nodes:
        shock_mask_nodes = simulator.create_shock_mask(args.shocked_nodes, shock_magnitude)
    
    if args.shocked_edges:
        shock_mask_edges = simulator.create_edge_shock_mask(args.shocked_edges, shock_magnitude)
    
    # Run simulation
    predictions = simulator.run_simulation(shock_mask_nodes, shock_mask_edges)
    results = simulator.analyze_results(predictions, args.shocked_nodes, args.shocked_edges)
    
    # Print summary
    simulator.print_summary(results, args.shocked_nodes, args.shocked_edges)
    
    # Save detailed results to CSV if requested
    if args.output:
        # For CSV, keep codes only (processing 669K rows to plain English is slow)
        # Users can filter by codes and format specific rows if needed
        results.to_csv(args.output, index=False)
        print(f"\n✓ Detailed results saved to: {args.output}")
        print(f"  Contains {len(results):,} edges (codes saved, use ICIO_parser.py to convert to plain English)")
        print(f"  Tip: Filter CSV first, then convert top results to plain English")
    
    # Save summary statistics
    summary = {
        'shocked_nodes': args.shocked_nodes if args.shocked_nodes else [],
        'shocked_edges': args.shocked_edges if args.shocked_edges else [],
        'magnitude': args.magnitude,
        'total_edges': len(results),
        'shocked_edge_count': len(results[results['edge_type'] == 'shocked_edge']),
        'direct_outgoing': len(results[results['edge_type'] == 'direct_outgoing']),
        'direct_incoming': len(results[results['edge_type'] == 'direct_incoming']),
        'indirect': len(results[results['edge_type'] == 'indirect']),
        'total_impact': float(results['absolute_change'].sum()),
        'mean_pct_change': float(results['pct_change'].mean()),
        'median_pct_change': float(results['pct_change'].median()),
        'edges_affected_1pct': int((np.abs(results['pct_change']) > 1).sum()),
        'edges_affected_5pct': int((np.abs(results['pct_change']) > 5).sum()),
        'edges_affected_10pct': int((np.abs(results['pct_change']) > 10).sum()),
    }
        
    print(f"\n{'='*80}")
    print("SIMULATION COMPLETE")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()