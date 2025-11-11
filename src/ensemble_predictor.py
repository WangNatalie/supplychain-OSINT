#!/usr/bin/env python3
"""
ensemble_predictor.py - Confidence-weighted ensemble for shock propagation

Strategy:
1. Hybrid: Main predictor (best balanced performance)
2. Focal: Confirmation for drop predictions (expert at detecting drops)
3. Sign-corrected: Confirmation for gain predictions (expert at detecting gains)

Usage:
    from ensemble_predictor import EnsemblePredictor
    
    ensemble = EnsemblePredictor(
        hybrid_model_path="models/hybrid_model/best_model.pt",
        focal_model_path="models/focal_model/best_model.pt",
        sign_model_path="models/sign_corrected_model/best_model.pt"
    )
    
    predictions = ensemble.predict(graph, shock_masks)
"""

import torch
import numpy as np
from pathlib import Path
from training import ShockPropagationGNN, extract_edge_values
from typing import Dict, Tuple, Optional


class EnsemblePredictor:
    """
    Confidence-weighted ensemble combining three loss function models
    """
    
    def __init__(self, 
                 hybrid_model_path: str,
                 focal_model_path: str,
                 sign_model_path: str,
                 device: str = 'cpu'):
        """
        Load three models trained with different loss functions
        
        Args:
            hybrid_model_path: Path to hybrid loss model
            focal_model_path: Path to focal loss model  
            sign_model_path: Path to sign-corrected loss model
            device: Device to run on
        """
        self.device = device
        
        # Load models
        print("Loading ensemble models...")
        self.hybrid_model, self.hybrid_args = self._load_model(hybrid_model_path, "hybrid")
        self.focal_model, self.focal_args = self._load_model(focal_model_path, "focal")
        self.sign_model, self.sign_args = self._load_model(sign_model_path, "sign_corrected")
        
        print("✓ Ensemble loaded successfully")
        
        # Based on multi-year analysis:
        # Hybrid: Best overall (78.8% dir acc, 65.3% drop, 80.7% gain)
        # Focal: Best drop detection (72.9% drop acc)
        # Sign: Best gain detection (95.9% gain acc)
        
        # Confidence weights (how much to trust each model)
        self.model_confidence = {
            'hybrid': {
                'drop': 0.65,  # Decent at drops
                'gain': 0.81,  # Good at gains
                'base': 0.75   # Good overall
            },
            'focal': {
                'drop': 0.73,  # Excellent at drops
                'gain': 0.60,  # Poor at gains
                'base': 0.65   # Mediocre overall
            },
            'sign': {
                'drop': 0.60,  # Poor at drops
                'gain': 0.96,  # Excellent at gains
                'base': 0.80   # Good R² for magnitudes
            }
        }
    
    def _load_model(self, model_path: str, name: str) -> Tuple[ShockPropagationGNN, Dict]:
        """Load a single model from checkpoint"""
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        args = checkpoint.get('args', {})
        
        # Get dimensions from checkpoint or defaults
        node_dim = args.get('node_dim', None)
        edge_dim = args.get('edge_dim', None)
        
        # If not in checkpoint, we need to infer from a graph
        if node_dim is None or edge_dim is None:
            # Try to load first available graph
            for year in range(1996, 2023):
                try:
                    graph = torch.load(f"embeddings/graph_{year}_labeled.pt", 
                                     map_location='cpu', weights_only=False)
                    node_dim = graph.x.shape[1]
                    edge_dim = graph.edge_attr.shape[1]
                    break
                except:
                    continue
        
        if node_dim is None:
            raise ValueError(f"Could not determine dimensions for {name} model")
        
        model = ShockPropagationGNN(
            node_in_dim=node_dim,
            edge_in_dim=edge_dim,
            hidden_dim=args.get('hidden_dim', 64),
            num_layers=args.get('num_layers', 2),
            dropout=args.get('dropout', 0.3),
            use_attention=args.get('use_attention', True)
        )
        
        # Handle potential architecture mismatches
        state_dict = checkpoint['model_state_dict']
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('edge_mlp.7.'):
                new_key = key.replace('edge_mlp.7.', 'edge_mlp.6.')
                new_state_dict[new_key] = value
            else:
                new_state_dict[key] = value
        
        model.load_state_dict(new_state_dict, strict=False)
        model.to(self.device)
        model.eval()
        
        print(f"  ✓ Loaded {name} model")
        return model, args
    
    def predict(self, 
                graph_data: Dict,
                return_components: bool = False,
                confidence_threshold: float = 0.7) -> Dict:
        """
        Make ensemble prediction with confidence weighting
        
        Args:
            graph_data: Dict with keys: x, edge_index, edge_attr, 
                       shock_mask_nodes, shock_mask_edges
            return_components: If True, return individual model predictions
            confidence_threshold: Threshold for high-confidence predictions
        
        Returns:
            Dict with predictions and metadata
        """
        with torch.no_grad():
            # Get predictions from all models
            hybrid_pred = self.hybrid_model(
                graph_data['x'],
                graph_data['edge_index'],
                graph_data['edge_attr'],
                graph_data.get('shock_mask_nodes'),
                graph_data.get('shock_mask_edges')
            )
            
            focal_pred = self.focal_model(
                graph_data['x'],
                graph_data['edge_index'],
                graph_data['edge_attr'],
                graph_data.get('shock_mask_nodes'),
                graph_data.get('shock_mask_edges')
            )
            
            sign_pred = self.sign_model(
                graph_data['x'],
                graph_data['edge_index'],
                graph_data['edge_attr'],
                graph_data.get('shock_mask_nodes'),
                graph_data.get('shock_mask_edges')
            )
        
        # Determine sign of each prediction
        hybrid_sign = torch.sign(hybrid_pred)
        focal_sign = torch.sign(focal_pred)
        sign_sign = torch.sign(sign_pred)
        
        # Confidence-weighted ensemble
        ensemble_pred = torch.zeros_like(hybrid_pred)
        confidence_scores = torch.zeros_like(hybrid_pred)
        
        for i in range(len(hybrid_pred)):
            # Determine what hybrid predicts
            is_drop = hybrid_pred[i] < 0
            is_gain = hybrid_pred[i] > 0
            
            if is_drop:
                # Hybrid predicts drop - check if focal confirms
                focal_confirms = focal_pred[i] < 0
                sign_confirms = sign_pred[i] < 0
                
                # Weight by confidence
                weight_hybrid = self.model_confidence['hybrid']['drop']
                weight_focal = self.model_confidence['focal']['drop'] if focal_confirms else 0.3
                weight_sign = self.model_confidence['sign']['drop'] if sign_confirms else 0.2
                
                # Normalize weights
                total_weight = weight_hybrid + weight_focal + weight_sign
                weight_hybrid /= total_weight
                weight_focal /= total_weight
                weight_sign /= total_weight
                
                # Weighted prediction
                ensemble_pred[i] = (
                    weight_hybrid * hybrid_pred[i] +
                    weight_focal * focal_pred[i] +
                    weight_sign * sign_pred[i]
                )
                
                # Confidence: high if focal confirms (focal is drop expert)
                if focal_confirms:
                    confidence_scores[i] = 0.85
                elif sign_confirms:
                    confidence_scores[i] = 0.65
                else:
                    confidence_scores[i] = 0.50  # Low confidence (models disagree)
                
            elif is_gain:
                # Hybrid predicts gain - check if sign confirms
                sign_confirms = sign_pred[i] > 0
                focal_confirms = focal_pred[i] > 0
                
                # Weight by confidence
                weight_hybrid = self.model_confidence['hybrid']['gain']
                weight_sign = self.model_confidence['sign']['gain'] if sign_confirms else 0.3
                weight_focal = self.model_confidence['focal']['gain'] if focal_confirms else 0.2
                
                # Normalize
                total_weight = weight_hybrid + weight_sign + weight_focal
                weight_hybrid /= total_weight
                weight_sign /= total_weight
                weight_focal /= total_weight
                
                # Weighted prediction
                ensemble_pred[i] = (
                    weight_hybrid * hybrid_pred[i] +
                    weight_sign * sign_pred[i] +
                    weight_focal * focal_pred[i]
                )
                
                # Confidence: high if sign confirms (sign is gain expert)
                if sign_confirms:
                    confidence_scores[i] = 0.90
                elif focal_confirms:
                    confidence_scores[i] = 0.70
                else:
                    confidence_scores[i] = 0.55
            
            else:
                # Near zero - equal weighting
                ensemble_pred[i] = (
                    0.5 * hybrid_pred[i] +
                    0.25 * focal_pred[i] +
                    0.25 * sign_pred[i]
                )
                confidence_scores[i] = 0.60
        
        # Prepare output
        result = {
            'predictions': ensemble_pred,
            'confidence': confidence_scores,
            'high_confidence_mask': confidence_scores > confidence_threshold,
            'predicted_drops': (ensemble_pred < 0).sum().item(),
            'predicted_gains': (ensemble_pred > 0).sum().item(),
            'avg_confidence': confidence_scores.mean().item()
        }
        
        if return_components:
            result['hybrid_pred'] = hybrid_pred
            result['focal_pred'] = focal_pred
            result['sign_pred'] = sign_pred
            result['agreement'] = {
                'all_agree': ((hybrid_sign == focal_sign) & (focal_sign == sign_sign)).sum().item(),
                'hybrid_focal_agree': (hybrid_sign == focal_sign).sum().item(),
                'hybrid_sign_agree': (hybrid_sign == sign_sign).sum().item(),
                'focal_sign_agree': (focal_sign == sign_sign).sum().item()
            }
        
        return result
    
    def predict_shock_propagation(self,
                                  graph,
                                  shocked_nodes: torch.Tensor,
                                  shock_magnitude: float = 0.5):
        """
        Predict shock propagation with ensemble
        
        Args:
            graph: Graph data object
            shocked_nodes: Binary mask [num_nodes] indicating shocked nodes
            shock_magnitude: Magnitude of shock (0-1)
        
        Returns:
            Dict with shock propagation analysis
        """
        # Prepare data
        graph_data = {
            'x': graph.x.to(self.device),
            'edge_index': graph.edge_index.to(self.device),
            'edge_attr': graph.edge_attr.to(self.device),
            'shock_mask_nodes': shocked_nodes.to(self.device),
            'shock_mask_edges': None
        }
        
        # Get baseline (no shock)
        graph_data_baseline = graph_data.copy()
        graph_data_baseline['shock_mask_nodes'] = None
        
        baseline_result = self.predict(graph_data_baseline, return_components=True)
        shocked_result = self.predict(graph_data, return_components=True)
        
        # Compute propagation effect
        propagation_effect = shocked_result['predictions'] - baseline_result['predictions']
        
        # Analyze
        src_idx, tgt_idx = graph.edge_index
        direct_edges = torch.isin(src_idx, torch.where(shocked_nodes > 0)[0])
        
        return {
            'baseline_predictions': baseline_result['predictions'],
            'shocked_predictions': shocked_result['predictions'],
            'propagation_effect': propagation_effect,
            'direct_effect_mean': propagation_effect[direct_edges].mean().item(),
            'indirect_effect_mean': propagation_effect[~direct_edges].mean().item(),
            'total_affected': (torch.abs(propagation_effect) > 0.01).sum().item(),
            'confidence': shocked_result['confidence'],
            'avg_confidence': shocked_result['avg_confidence'],
            'model_agreement': shocked_result['agreement']
        }


def main():
    """Demo usage"""
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--hybrid-model", required=True)
    parser.add_argument("--focal-model", required=True)
    parser.add_argument("--sign-model", required=True)
    parser.add_argument("--test-year", type=int, default=2021)
    parser.add_argument("--device", default="cpu")
    
    args = parser.parse_args()
    
    # Load ensemble
    ensemble = EnsemblePredictor(
        hybrid_model_path=args.hybrid_model,
        focal_model_path=args.focal_model,
        sign_model_path=args.sign_model,
        device=args.device
    )
    
    # Load test graph
    graph = torch.load(f"embeddings/graph_{args.test_year}_labeled.pt",
                      map_location=args.device, weights_only=False)
    
    print(f"\nTesting on year {args.test_year}")
    print(f"Graph: {graph.num_nodes} nodes, {graph.edge_index.shape[1]} edges")
    
    # Prepare data
    graph_data = {
        'x': graph.x,
        'edge_index': graph.edge_index,
        'edge_attr': graph.edge_attr,
        'shock_mask_nodes': None,
        'shock_mask_edges': None
    }
    
    # Make predictions
    print("\nMaking ensemble predictions...")
    result = ensemble.predict(graph_data, return_components=True)
    
    print(f"\nResults:")
    print(f"  Predicted drops: {result['predicted_drops']:,} ({result['predicted_drops']/len(result['predictions'])*100:.1f}%)")
    print(f"  Predicted gains: {result['predicted_gains']:,} ({result['predicted_gains']/len(result['predictions'])*100:.1f}%)")
    print(f"  Average confidence: {result['avg_confidence']:.2%}")
    print(f"  High confidence predictions: {result['high_confidence_mask'].sum().item():,}")
    
    print(f"\nModel agreement:")
    for k, v in result['agreement'].items():
        print(f"  {k}: {v:,} edges ({v/len(result['predictions'])*100:.1f}%)")
    
    # Compare to ground truth if available
    if hasattr(graph, 'value_t') and hasattr(graph, 'value_t1'):
        value_t, value_t1 = extract_edge_values(graph)
        delta_log_true = torch.log1p(value_t1) - torch.log1p(value_t)
        
        # Compute accuracy
        pred_sign = torch.sign(result['predictions'])
        true_sign = torch.sign(delta_log_true)
        accuracy = (pred_sign == true_sign).float().mean()
        
        print(f"\nAccuracy vs ground truth:")
        print(f"  Direction accuracy: {accuracy:.2%}")
        
        # By confidence level
        high_conf = result['high_confidence_mask']
        high_conf_acc = (pred_sign[high_conf] == true_sign[high_conf]).float().mean()
        print(f"  High confidence accuracy: {high_conf_acc:.2%}")


if __name__ == "__main__":
    main()

