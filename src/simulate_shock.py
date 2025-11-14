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
        --magnitude 0.3 \

    # Simulate cascading failure
    python simulate_shock.py \
        --model models/shock_propagation/best_model.pt \
        --graph embeddings/graph_2021_labeled.pt \
        --shocked-nodes CHN_A01 USA_A01 DEU_A01 \
        --magnitude 0.4 \
        --visualize
"""

import torch
import argparse
import pandas as pd
import numpy as np
from typing import List, Dict
from pathlib import Path
import sys

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
    
    def create_shock_mask(self, shocked_nodes: List[str]) -> torch.Tensor:
        """Create binary shock mask for specified nodes"""
        shock_mask = torch.zeros(self.graph.num_nodes, device=self.device)
        valid_nodes = []
        
        for node_id in shocked_nodes:
            if node_id not in self.graph.node_id_to_idx:
                print(f"Warning: Node '{node_id}' not found in graph")
                print(f"  Available nodes (sample): {self.graph.node_labels[:5]}...")
                continue
            
            idx = self.graph.node_id_to_idx[node_id]
            shock_mask[idx] = 1.0
            valid_nodes.append(node_id)
        
        if len(valid_nodes) == 0:
            raise ValueError("No valid nodes found to shock!")
        
        print(f"✓ Shocking {len(valid_nodes)} nodes: {valid_nodes}")
        return shock_mask
    
    def run_simulation(self, shock_mask_nodes: torch.Tensor, shock_magnitude: float = -0.20) -> Dict[str, np.ndarray]:
        """
        Run baseline and shocked predictions with magnitude scaling
        
        Args:
            shock_mask_nodes: Binary mask indicating shocked nodes
            shock_magnitude: Actual shock magnitude (e.g., -0.20 for 20% reduction)
        
        Returns:
            Dict with baseline, shocked, and scaled propagation predictions
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
            
            # Shocked prediction (model receives binary shock indicator)
            shocked_delta = self.model(
                self.graph.x,
                self.graph.edge_index,
                self.graph.edge_attr,
                shock_mask_nodes=shock_mask_nodes,
                shock_mask_edges=None
            ).cpu().numpy()
        
        # POST-PROCESSING SCALING FIX (will remove once training is fixed)
        # Model was trained on "typical" shocks (~17.5% average from training)
        # Scale the predicted effect to match the desired shock magnitude
        typical_shock_magnitude = -0.175  # Average shock magnitude from training data
        
        propagation_effect = shocked_delta - baseline_delta
        scaling_factor = shock_magnitude / typical_shock_magnitude
        scaled_propagation = propagation_effect * scaling_factor
        
        
        return {
            'baseline_delta': baseline_delta,
            'shocked_delta': baseline_delta + scaled_propagation,  # Use scaled effect
            'propagation_effect': scaled_propagation
        }
    
    def analyze_results(self, 
                       predictions: Dict[str, np.ndarray],
                       shocked_nodes: List[str]) -> pd.DataFrame:
        """Convert predictions to interpretable DataFrame"""
        
        # Extract edge information
        src_idx, tgt_idx = self.graph.edge_index.cpu().numpy()
        value_t = self.graph.value_t.cpu().numpy()
        
        # Reconstruct actual values from log-space predictions
        log_value_t = np.log1p(value_t)
        baseline_value_t1 = np.expm1(log_value_t + predictions['baseline_delta'])
        shocked_value_t1 = np.expm1(log_value_t + predictions['shocked_delta'])
        
        # Build results DataFrame
        results = pd.DataFrame({
            'source': [self.graph.node_labels[i] for i in src_idx],
            'target': [self.graph.node_labels[i] for i in tgt_idx],
            'source_country': [self.graph.node_labels[i].split('_')[0] for i in src_idx],
            'target_country': [self.graph.node_labels[i].split('_')[0] for i in tgt_idx],
            'source_sector': [self.graph.node_labels[i].split('_')[1] if '_' in self.graph.node_labels[i] else 'UNK' for i in src_idx],
            'target_sector': [self.graph.node_labels[i].split('_')[1] if '_' in self.graph.node_labels[i] else 'UNK' for i in tgt_idx],
            'value_t': value_t,
            'baseline_value_t1': baseline_value_t1,
            'shocked_value_t1': shocked_value_t1,
            'absolute_change': shocked_value_t1 - baseline_value_t1,
            'pct_change': ((shocked_value_t1 - baseline_value_t1) / (baseline_value_t1 + 1e-8)) * 100,
            'propagation_effect_log': predictions['propagation_effect']
        })
        
        # Categorize edge relationships to shock
        results['edge_type'] = 'indirect'
        results.loc[results['source'].isin(shocked_nodes), 'edge_type'] = 'direct_outgoing'
        results.loc[results['target'].isin(shocked_nodes), 'edge_type'] = 'direct_incoming'
        results.loc[
            results['source'].isin(shocked_nodes) & results['target'].isin(shocked_nodes),
            'edge_type'
        ] = 'internal'
        
        # Compute impact magnitude (by absolute dollars)
        results['abs_change'] = np.abs(results['absolute_change'])
        
        return results.sort_values('abs_change', ascending=False)
    
    def print_summary(self, results: pd.DataFrame, shocked_nodes: List[str]):
        """Print comprehensive shock analysis summary"""
        
        print("\n" + "="*80)
        print("SHOCK PROPAGATION ANALYSIS")
        print("="*80)
        
        print(f"\nShocked Nodes: {shocked_nodes}")
        print(f"Total Edges Analyzed: {len(results):,}")
        
        # Breakdown by edge type
        print("\n" + "-"*80)
        print("DIRECT EFFECTS (Edges connected to shocked nodes)")
        print("-"*80)
        
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
        
        print("\n{:<15} {:>15} {:>15} {:>15}".format(
            "Country", "Total Impact", "Avg % Change", "Original Value"
        ))
        print("-"*80)
        for country, row in country_impact.iterrows():
            print("{:<15} ${:>14,.0f} {:>14.2f}% ${:>14,.0f}".format(
                country,
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
        
        print("\n{:<15} {:>15} {:>15} {:>15}".format(
            "Sector", "Total Impact", "Avg % Change", "Original Value"
        ))
        print("-"*80)
        for sector, row in sector_impact.iterrows():
            print("{:<15} ${:>14,.0f} {:>14.2f}% ${:>14,.0f}".format(
                sector,
                row['absolute_change'],
                row['pct_change'],
                row['value_t']
            ))
        
        # Node-level aggregation (country_sector combinations)
        print("\n" + "-"*80)
        print("NODE-LEVEL IMPACT (Top 10 affected country-sectors)")
        print("-"*80)
        
        # Aggregate by target node
        node_impact = results.groupby('target').agg({
            'absolute_change': 'sum',
            'pct_change': 'mean',
            'value_t': 'sum'
        }).sort_values('absolute_change', key=abs, ascending=False).head(10)
        
        print("\n{:<20} {:>15} {:>15} {:>15}".format(
            "Country-Sector", "Total Impact", "Avg % Change", "Original Value"
        ))
        print("-"*80)
        for node, row in node_impact.iterrows():
            print("{:<20} ${:>14,.0f} {:>14.2f}% ${:>14,.0f}".format(
                node[:20],
                row['absolute_change'],
                row['pct_change'],
                row['value_t']
            ))
        
        # Top affected edges
        print("\n" + "-"*80)
        print("TOP 15 MOST AFFECTED TRADING FLOWS")
        print("-"*80)
        
        top_edges = results.head(15)[['source', 'target', 'value_t', 
                                      'baseline_value_t1', 'shocked_value_t1',
                                      'absolute_change', 'pct_change', 'edge_type']]
        
        print("\n{:<15} {:<15} {:>12} {:>12} {:>12} {:>10}".format(
            "Source", "Target", "Current", "Shocked", "Change", "% Change"
        ))
        print("-"*80)
        for _, row in top_edges.iterrows():
            print("{:<15} {:<15} ${:>11,.0f} ${:>11,.0f} ${:>11,.0f} {:>9.2f}%".format(
                row['source'][:15],
                row['target'][:15],
                row['value_t'],
                row['shocked_value_t1'],
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
    parser.add_argument("--shocked-nodes", nargs="+", required=True,
                       help="Node IDs to shock (e.g., ECU_AGR CHN_MFG)")
    parser.add_argument("--magnitude", type=float, default=0.5,
                       help="Shock magnitude as fraction (0.5 = 50%% reduction)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--list-nodes", action="store_true",
                       help="List all available nodes in the graph and exit")
    
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
                country = node.split('_')[0]
                sector = node.split('_')[1]
            else:
                country = node
                sector = 'N/A'
            
            if country not in nodes_by_country:
                nodes_by_country[country] = []
            nodes_by_country[country].append((node, sector))
        
        for country in sorted(nodes_by_country.keys()):
            nodes = nodes_by_country[country]
            print(f"\n{country} ({len(nodes)} sectors):")
            for node, sector in sorted(nodes):
                print(f"  {node}")
        
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
    
    # Run simulation
    print(f"\n{'='*80}")
    print(f"SIMULATING SHOCK: {args.magnitude:.1%} reduction in {len(args.shocked_nodes)} nodes")
    print(f"{'='*80}")
    
    shock_mask = simulator.create_shock_mask(args.shocked_nodes)
    # Convert magnitude to negative (reduction)
    shock_magnitude = -abs(args.magnitude)
    predictions = simulator.run_simulation(shock_mask, shock_magnitude=shock_magnitude)
    results = simulator.analyze_results(predictions, args.shocked_nodes)
    
    # Print summary
    simulator.print_summary(results, args.shocked_nodes)
    
    # Save summary statistics
    summary = {
        'shocked_nodes': args.shocked_nodes,
        'magnitude': args.magnitude,
        'total_edges': len(results),
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