"""
Proper evaluation for SHOCK PROPAGATION

Tests:
1. Synthetic shock injection → measure predicted downstream effects
2. Distance-based propagation decay
3. Economic context sensitivity (boom vs crisis years)
4. Sector-specific propagation patterns
"""
import torch
import numpy as np
from training import ShockPropagationGNN, create_shock_training_data, load_and_process_example
from typing import List, Dict
import random
from pathlib import Path


def inject_synthetic_shock(graph, shocked_node_indices: List[int], shock_magnitude: float = 0.5):
    """
    Create shock masks for synthetic evaluation
    
    Args:
        graph: Graph data
        shocked_node_indices: Indices of nodes to shock
        shock_magnitude: Magnitude of shock (not used in mask, but for reference)
    
    Returns:
        shock_mask_nodes, shock_mask_edges
    """
    num_nodes = graph.x.shape[0]
    num_edges = graph.edge_index.shape[1]
    
    # Node mask
    shock_mask_nodes = torch.zeros(num_nodes)
    shock_mask_nodes[shocked_node_indices] = 1.0
    
    # Edge mask: edges originating from shocked nodes
    src_idx = graph.edge_index[0]
    shock_mask_edges = torch.zeros(num_edges)
    for node_idx in shocked_node_indices:
        shock_mask_edges[src_idx == node_idx] = 1.0
    
    return shock_mask_nodes, shock_mask_edges


def compute_edge_distances_from_shock(graph, shocked_nodes: torch.Tensor):
    """
    Compute distance of each edge from shocked nodes
    
    Returns:
        distances: Tensor of shape [num_edges] with distance values
        distance_masks: Dict with masks for each distance level
    """
    src_idx, tgt_idx = graph.edge_index
    num_edges = graph.edge_index.shape[1]
    shocked_node_set = set(torch.where(shocked_nodes > 0)[0].tolist())
    
    distances = torch.full((num_edges,), 999, dtype=torch.long)
    
    # Distance 0: Edges FROM shocked nodes
    for i in range(num_edges):
        if src_idx[i].item() in shocked_node_set:
            distances[i] = 0
    
    # Distance 1: Edges from nodes that buy from shocked nodes (1-hop downstream)
    distance_1_nodes = set()
    for i in range(num_edges):
        if src_idx[i].item() in shocked_node_set:
            distance_1_nodes.add(tgt_idx[i].item())
    
    for i in range(num_edges):
        if distances[i] == 999 and src_idx[i].item() in distance_1_nodes:
            distances[i] = 1
    
    # Distance 2: Edges from nodes that buy from distance-1 nodes
    distance_2_nodes = set()
    for i in range(num_edges):
        if distances[i] == 1:
            distance_2_nodes.add(tgt_idx[i].item())
    
    for i in range(num_edges):
        if distances[i] == 999 and src_idx[i].item() in distance_2_nodes:
            distances[i] = 2
    
    # Distance 3+: Everything else
    distances[distances == 999] = 3
    
    return distances


def evaluate_single_shock(model, graph, shocked_nodes: List[int], device='cpu'):
    """
    Evaluate model's shock propagation prediction for a single shock
    
    Returns:
        Dict with propagation metrics
    """
    graph = graph.to(device)
    
    # Create shock masks
    shock_mask_nodes, shock_mask_edges = inject_synthetic_shock(graph, shocked_nodes)
    shock_mask_nodes = shock_mask_nodes.to(device)
    shock_mask_edges = shock_mask_edges.to(device)
    
    # Get baseline prediction (no shock)
    with torch.no_grad():
        baseline_pred = model(
            graph.x, graph.edge_index, graph.edge_attr,
            shock_mask_nodes=None, shock_mask_edges=None
        )
        
        # Get shocked prediction
        shocked_pred = model(
            graph.x, graph.edge_index, graph.edge_attr,
            shock_mask_nodes=shock_mask_nodes, shock_mask_edges=shock_mask_edges
        )
    
    # Compute propagation effect
    propagation_effect = shocked_pred - baseline_pred
    
    # Analyze by distance
    distances = compute_edge_distances_from_shock(graph, shock_mask_nodes)
    
    results = {}
    for dist in [0, 1, 2, 3]:
        mask = (distances == dist)
        if mask.sum() > 0:
            effects = propagation_effect[mask].cpu()
            results[f'dist_{dist}'] = {
                'n_edges': mask.sum().item(),
                'mean_effect': effects.mean().item(),
                'median_effect': effects.median().item(),
                'std_effect': effects.std().item(),
                'pct_drops': (effects < -0.01).float().mean().item() * 100,
                'pct_significant': (effects.abs() > 0.01).float().mean().item() * 100,
            }
    
    return results


def evaluate_year(model, year: int, embeddings_dir: str, n_trials: int = 20, device='cpu'):
    """
    Evaluate shock propagation for a given year with multiple random shocks
    
    Args:
        model: Trained model
        year: Year to test
        n_trials: Number of random shock scenarios to test
        device: Device to run on
    
    Returns:
        Dict with aggregated results
    """
    # Load year data
    graph_path = Path(embeddings_dir) / f"graph_{year}_labeled.pt"
    if not graph_path.exists():
        return None
    
    graph = torch.load(graph_path, map_location='cpu', weights_only=False)
    num_nodes = graph.x.shape[0]
    
    print(f"\n  Testing {year} with {n_trials} synthetic shocks...")
    
    all_results = []
    
    for trial in range(n_trials):
        # Randomly select 5-10 nodes to shock
        n_shocked = random.randint(5, 10)
        shocked_nodes = random.sample(range(num_nodes), n_shocked)
        
        # Evaluate this shock
        result = evaluate_single_shock(model, graph, shocked_nodes, device)
        all_results.append(result)
    
    # Aggregate results
    aggregated = {}
    for dist in [0, 1, 2, 3]:
        key = f'dist_{dist}'
        if any(key in r for r in all_results):
            dist_results = [r[key] for r in all_results if key in r]
            
            aggregated[key] = {
                'n_edges': np.mean([r['n_edges'] for r in dist_results]),  # Average number of edges at this distance
                'mean_effect': np.mean([r['mean_effect'] for r in dist_results]),
                'median_effect': np.median([r['median_effect'] for r in dist_results]),
                'pct_drops': np.mean([r['pct_drops'] for r in dist_results]),
                'pct_significant': np.mean([r['pct_significant'] for r in dist_results]),
            }
    
    return aggregated


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate shock propagation (not general drops)")
    parser.add_argument("--model-path", type=str, default="models/hybrid/best_model.pt",
                       help="Path to trained model")
    parser.add_argument("--embeddings-dir", type=str, default="embeddings",
                       help="Directory with graph files")
    parser.add_argument("--years", nargs="+", type=int, 
                       default=[2003, 2008, 2015, 2019, 2020, 2021],
                       help="Years to test")
    parser.add_argument("--n-trials", type=int, default=20,
                       help="Number of random shocks per year")
    parser.add_argument("--device", type=str, default="cpu",
                       help="Device to run on")
    
    args = parser.parse_args()
    
    print("="*80)
    print("SHOCK PROPAGATION EVALUATION")
    print("="*80)
    print(f"\nThis evaluates SHOCK PROPAGATION, not general drop prediction.")
    print(f"Method: Inject synthetic shocks → measure downstream effects")
    print(f"\nModel: {args.model_path}")
    print(f"Years: {args.years}")
    print(f"Trials per year: {args.n_trials}")
    
    # Load model
    print("\nLoading model...")
    checkpoint = torch.load(args.model_path, map_location='cpu', weights_only=False)
    ckpt_args = checkpoint.get('args', {})
    
    # Get dimensions
    sample_graph = torch.load(f"{args.embeddings_dir}/graph_{args.years[0]}_labeled.pt",
                              map_location='cpu', weights_only=False)
    node_dim = sample_graph.x.shape[1]
    edge_dim = sample_graph.edge_attr.shape[1]
    
    model = ShockPropagationGNN(
        node_in_dim=node_dim,
        edge_in_dim=edge_dim,
        hidden_dim=ckpt_args.get('hidden_dim', 128),
        num_layers=ckpt_args.get('num_layers', 3),
        dropout=ckpt_args.get('dropout', 0.3),
        use_attention=ckpt_args.get('use_attention', True)
    )
    
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.to(args.device)
    model.eval()
    print("✓ Model loaded")
    
    # Evaluate each year
    print("\n" + "="*80)
    print("RESULTS BY YEAR")
    print("="*80)
    
    all_year_results = {}
    
    for year in args.years:
        results = evaluate_year(model, year, args.embeddings_dir, args.n_trials, args.device)
        if results:
            all_year_results[year] = results
            
            print(f"\n{year}:")
            print(f"  {'Distance':<15} {'# Edges':<12} {'Mean Effect':<15} {'% Drops':<12} {'% Significant':<15}")
            print(f"  {'-'*75}")
            
            for dist in [0, 1, 2, 3]:
                key = f'dist_{dist}'
                if key in results:
                    r = results[key]
                    label = {0: "Direct (0-hop)", 1: "1-hop", 2: "2-hop", 3: "3+ hop"}[dist]
                    print(f"  {label:<15} {r['n_edges']:>11.0f} {r['mean_effect']:>14.4f} {r['pct_drops']:>11.1f}% {r['pct_significant']:>14.1f}%")
    
    # Summary analysis
    print("\n" + "="*80)
    print("PROPAGATION ANALYSIS")
    print("="*80)
    
    # Average across all years
    avg_by_distance = {}
    for dist in [0, 1, 2, 3]:
        key = f'dist_{dist}'
        effects = [yr[key]['mean_effect'] for yr in all_year_results.values() if key in yr]
        pct_drops = [yr[key]['pct_drops'] for yr in all_year_results.values() if key in yr]
        
        if effects:
            avg_by_distance[dist] = {
                'mean_effect': np.mean(effects),
                'pct_drops': np.mean(pct_drops)
            }
    
    print("\nAverage propagation decay:")
    for dist in [0, 1, 2, 3]:
        if dist in avg_by_distance:
            r = avg_by_distance[dist]
            label = {0: "Direct", 1: "1-hop", 2: "2-hop", 3: "3+ hop"}[dist]
            print(f"  {label:<10}: Effect={r['mean_effect']:>7.4f}, Drops={r['pct_drops']:>5.1f}%")
    
    # Check for propagation
    print("\n" + "="*80)
    print("EVALUATION")
    print("="*80)
    
    if 0 in avg_by_distance and 1 in avg_by_distance:
        direct_effect = avg_by_distance[0]['mean_effect']
        hop1_effect = avg_by_distance[1]['mean_effect']
        
        print(f"\nDirect effect: {direct_effect:.4f}")
        print(f"1-hop effect:  {hop1_effect:.4f}")
        
        if abs(direct_effect) > 0.01:
            print("✓ Model responds to direct shocks")
        else:
            print("✗ Model barely responds to direct shocks")
        
        if abs(hop1_effect) > 0.005:
            ratio = hop1_effect / direct_effect if direct_effect != 0 else 0
            print(f"✓ Propagation detected (1-hop = {ratio*100:.1f}% of direct)")
        else:
            print("⚠️  Very weak or no propagation to 1-hop neighbors")
        
        # Context sensitivity
        crisis_years = [2008, 2015, 2019]
        boom_years = [2003, 2020]
        
        crisis_effects = [all_year_results[y]['dist_1']['mean_effect'] 
                         for y in crisis_years if y in all_year_results and 'dist_1' in all_year_results[y]]
        boom_effects = [all_year_results[y]['dist_1']['mean_effect']
                       for y in boom_years if y in all_year_results and 'dist_1' in all_year_results[y]]
        
        if crisis_effects and boom_effects:
            crisis_avg = np.mean(crisis_effects)
            boom_avg = np.mean(boom_effects)
            
            print(f"\nContext sensitivity:")
            print(f"  Crisis years (2008,2015,2019): {crisis_avg:.4f} effect")
            print(f"  Boom years (2003,2020):         {boom_avg:.4f} effect")
            
            if abs(crisis_avg) > abs(boom_avg) * 1.3:
                print("  ✓ Model correctly predicts stronger propagation in crisis contexts")
            elif abs(crisis_avg) < abs(boom_avg) * 0.7:
                print("  ⚠️  Model predicts weaker propagation in crisis (unexpected)")
            else:
                print("  → Similar propagation in both contexts")
    
    print("\n" + "="*80)
    print("CONCLUSION")
    print("="*80)
    print("""
This evaluation tests SHOCK PROPAGATION using synthetic shocks.
Unlike the previous evaluation (which tested general drop prediction),
this measures whether the model can predict downstream cascade effects.

Key metrics:
- Direct effect: How much model responds to shocked nodes
- 1-hop effect: Propagation to immediate downstream partners
- Decay: Whether effect weakens with distance (expected)
- Context sensitivity: Different propagation in boom vs crisis years
""")


if __name__ == "__main__":
    main()
