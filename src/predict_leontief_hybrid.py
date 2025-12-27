#!/usr/bin/env python3
"""
predict_leontief_hybrid.py - Predict using Leontief-ML Hybrid Model

Uses trained hybrid model to predict supply chain shock propagation:
    1. Compute Leontief baseline (structural propagation)
    2. Apply ML adjustment factors (learned from historical data)
    3. Output adjusted predictions

Usage:
    python predict_leontief_hybrid.py \
        --model models/leontief_hybrid/best_model.pt \
        --graph embeddings/graph_2021_labeled.pt \
        --shocked-nodes USA_C26 --magnitude 0.20 \
        --output hybrid_predictions.csv
"""

import torch
import numpy as np
import pandas as pd
import argparse
import pickle
import json
from pathlib import Path
from typing import Dict

from leontief_propagation import LeontiefPropagator
from train_leontief_hybrid import AdjustmentMLModel
from ICIO.ICIO_parser import format_node_name


class HybridPredictor:
    """
    Combines Leontief structural propagation with ML-learned adjustments.
    """
    
    def __init__(self, model_path: str, graph_path: str):
        """
        Initialize hybrid predictor.
        
        Args:
            model_path: Path to trained ML model
            graph_path: Path to ICIO graph
        """
        print(f"Loading graph from {graph_path}...")
        graph = torch.load(graph_path, map_location='cpu', weights_only=False)
        
        print(f"Initializing Leontief propagator...")
        self.propagator = LeontiefPropagator(graph)
        
        print(f"Loading ML adjustment model from {model_path}...")
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        
        # Load model
        num_features = checkpoint['num_features']
        hidden_dim = checkpoint.get('hidden_dim', 64)
        dropout = checkpoint.get('dropout', 0.2)
        
        self.ml_model = AdjustmentMLModel(num_features, hidden_dim, dropout)
        self.ml_model.load_state_dict(checkpoint['model_state_dict'])
        self.ml_model.eval()
        
        # Load scaler
        model_dir = Path(model_path).parent
        with open(model_dir / 'scaler.pkl', 'rb') as f:
            self.scaler = pickle.load(f)
        
        # Load feature names
        with open(model_dir / 'feature_names.json', 'r') as f:
            self.feature_names = json.load(f)
        
        print(f"✓ Hybrid model loaded")
        print(f"  Features: {len(self.feature_names)}")
    
    def extract_features(self, edge_data: pd.Series, 
                        shock_magnitude: float, 
                        num_shocked_nodes: int) -> np.ndarray:
        """
        Extract features for ML model from edge data.
        """
        features = {}
        
        # Core features (must match training)
        features['leontief_prediction'] = edge_data['impact_pct'] / 100
        features['edge_value'] = edge_data['baseline_value']
        features['edge_value_log'] = np.log1p(edge_data['baseline_value'])
        features['shock_magnitude'] = shock_magnitude
        features['num_shocked_nodes'] = num_shocked_nodes
        features['source_output_impact'] = edge_data['source_output_impact'] / 100
        features['target_output_impact'] = edge_data['target_output_impact'] / 100
        
        # Optional features (set to 0 if not available)
        optional_features = [
            'geographic_distance', 'supplier_diversity',
            'substitutability', 'inventory_months'
        ]
        for feat in optional_features:
            features[feat] = 0.0
        
        # Build feature vector in correct order
        feature_vector = np.array([features.get(name, 0.0) for name in self.feature_names])
        
        return feature_vector
    
    def predict(self, shocked_nodes: Dict[str, float]) -> pd.DataFrame:
        """
        Predict shock propagation with hybrid model.
        
        Args:
            shocked_nodes: Dict mapping node_id -> magnitude
        
        Returns:
            DataFrame with Leontief baseline, ML adjustment, and final predictions
        """
        print(f"\n{'='*80}")
        print("HYBRID PREDICTION (Leontief + ML Adjustment)")
        print(f"{'='*80}\n")
        
        # Step 1: Compute Leontief baseline
        print("Step 1: Computing Leontief baseline...")
        node_impacts = self.propagator.propagate_shock(shocked_nodes)
        edge_impacts = self.propagator.compute_edge_impacts(node_impacts)
        
        # Step 2: Extract features for ML model
        print("Step 2: Extracting features for ML adjustment...")
        shock_magnitude = np.mean([abs(m) for m in shocked_nodes.values()])
        num_shocked = len(shocked_nodes)
        
        features_list = []
        for _, edge in edge_impacts.iterrows():
            features = self.extract_features(edge, shock_magnitude, num_shocked)
            features_list.append(features)
        
        X = np.array(features_list)
        X_scaled = self.scaler.transform(X)
        
        # Step 3: Predict adjustment factors
        print("Step 3: Predicting ML adjustment factors...")
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_scaled)
            adjustments = self.ml_model(X_tensor).numpy()
        
        # Step 4: Apply adjustments
        print("Step 4: Computing final adjusted predictions...")
        edge_impacts['adjustment_factor'] = adjustments
        edge_impacts['adjusted_impact_pct'] = edge_impacts['impact_pct'] * adjustments
        edge_impacts['adjusted_impact_abs'] = edge_impacts['impact_abs'] * adjustments
        
        # Sort by largest adjusted drops
        edge_impacts = edge_impacts.sort_values('adjusted_impact_abs', ascending=True)
        
        return edge_impacts
    
    def print_comparison(self, results: pd.DataFrame, shocked_nodes: Dict[str, float]):
        """Print comparison of Leontief vs Hybrid predictions."""
        
        print(f"\n{'='*80}")
        print("COMPARISON: Leontief vs Hybrid (ML-Adjusted)")
        print(f"{'='*80}")
        
        print(f"\nShocked Nodes:")
        for node_id, magnitude in shocked_nodes.items():
            node_name = format_node_name(node_id, include_code=True)
            print(f"  • {node_name}: {magnitude:.1%}")
        
        # Overall statistics
        drops = results[results['adjusted_impact_abs'] < 0]
        leontief_drops = results[results['impact_abs'] < 0]
        
        print(f"\n{'='*80}")
        print("OVERALL IMPACT")
        print(f"{'='*80}")
        
        print(f"\n{'Metric':<40} {'Leontief':<15} {'Hybrid':<15} {'Adjustment'}")
        print("-"*80)
        
        metrics = [
            ('Edges with drops', len(leontief_drops), len(drops)),
            ('Mean drop (%)', leontief_drops['impact_pct'].mean() if len(leontief_drops) > 0 else 0,
                              drops['adjusted_impact_pct'].mean() if len(drops) > 0 else 0),
            ('Max drop (%)', leontief_drops['impact_pct'].min() if len(leontief_drops) > 0 else 0,
                            drops['adjusted_impact_pct'].min() if len(drops) > 0 else 0),
            ('Total value at risk ($)', leontief_drops['baseline_value'].sum() if len(leontief_drops) > 0 else 0,
                                       drops['baseline_value'].sum() if len(drops) > 0 else 0),
        ]
        
        for metric_name, leontief_val, hybrid_val in metrics:
            if 'Edges' in metric_name or 'value' in metric_name:
                ratio = f"{hybrid_val/leontief_val:.2f}x" if leontief_val > 0 else "N/A"
                print(f"{metric_name:<40} {leontief_val:<15,.0f} {hybrid_val:<15,.0f} {ratio}")
            else:
                diff = hybrid_val - leontief_val
                print(f"{metric_name:<40} {leontief_val:<15.2f} {hybrid_val:<15.2f} {diff:+.2f}")
        
        # Top drops comparison
        print(f"\n{'='*80}")
        print("TOP 15 LARGEST DROPS (HYBRID)")
        print(f"{'='*80}")
        
        top_drops = drops.head(15)
        
        print(f"\n{'Source':<30} {'Target':<30} {'Leontief %':<12} {'Hybrid %':<12} {'Adjust'}")
        print("-"*80)
        
        for _, row in top_drops.iterrows():
            source_name = format_node_name(row['source'])[:28]
            target_name = format_node_name(row['target'])[:28]
            adj_factor = row['adjustment_factor']
            
            print(f"{source_name:<30} {target_name:<30} {row['impact_pct']:>11.2f} "
                  f"{row['adjusted_impact_pct']:>11.2f} {adj_factor:>6.2f}x")
        
        print(f"\n{'='*80}")


def main():
    parser = argparse.ArgumentParser(
        description="Predict shock propagation with Leontief-ML hybrid model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument("--model", required=True,
                       help="Path to trained hybrid model")
    parser.add_argument("--graph", required=True,
                       help="Path to ICIO graph file")
    parser.add_argument("--shocked-nodes", nargs="+", required=True,
                       help="Node IDs to shock")
    parser.add_argument("--magnitude", type=float, default=0.5,
                       help="Shock magnitude (0.5 = 50%% reduction)")
    parser.add_argument("--output", type=str,
                       help="Path to save results CSV")
    
    args = parser.parse_args()
    
    # Initialize predictor
    predictor = HybridPredictor(args.model, args.graph)
    
    # Parse shocked nodes
    shocked_nodes = {}
    shock_magnitude = -abs(args.magnitude)
    for node_id in args.shocked_nodes:
        shocked_nodes[node_id] = shock_magnitude
    
    # Run prediction
    results = predictor.predict(shocked_nodes)
    
    # Print comparison
    predictor.print_comparison(results, shocked_nodes)
    
    # Save results
    if args.output:
        results.to_csv(args.output, index=False)
        print(f"\n✓ Results saved to: {args.output}")
    
    print(f"\n{'='*80}")
    print("PREDICTION COMPLETE")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()

