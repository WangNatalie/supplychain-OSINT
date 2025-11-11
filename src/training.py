#!/usr/bin/env python3
"""
training.py - Shock Propagation Model for Supply Chain Networks

PROBLEM: Predict how supply chain disruptions propagate through the network
- Input: Current network state + shock specification (which nodes/edges are disrupted)
- Output: Predicted changes in all edge values after shock propagates

APPROACH: Counterfactual Prediction
1. Train model to predict edge values from network features
2. At inference: Modify shocked node/edge features, predict new equilibrium
3. Compare: original predictions vs. shocked predictions = propagation effect

TRAINING DATA GENERATION:
- Each year-to-year transition is a "natural experiment"
- Edges that dropped significantly = natural shocks
- Learn how network structure + shock indicators → edge value changes
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, BatchNorm, GATConv
from torch_geometric.data import Data
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import warnings
import psutil
import os
warnings.filterwarnings('ignore')
from loss_functions import get_loss_function
from tqdm import tqdm
import time

# Epsilon for sign comparisons; overridden by --sign-eps
SIGN_EPS = 1e-4

def print_memory_usage(prefix=""):
    process = psutil.Process(os.getpid())
    mem_gb = process.memory_info().rss / 1024**3
    print(f"{prefix}Memory usage: {mem_gb:.2f} GB")

class ShockPropagationGNN(torch.nn.Module):
    """
    GNN model for predicting shock propagation in supply networks.
    
    Key Innovation: Shock-aware architecture
    - Takes current edge values + shock indicators as input
    - Predicts CHANGES in edge values (Δvalue) 
    - Can handle counterfactual "what-if" scenarios at inference
    
    Architecture:
    1. Node encoder: [base features || shock indicator || connectivity]
    2. Message passing: Propagate shock information through network
    3. Edge predictor: Predict value change for each edge
    """
    
    def __init__(self, 
                 node_in_dim: int,
                 edge_in_dim: int,
                 hidden_dim: int = 128,
                 num_layers: int = 3,
                 dropout: float = 0.3,
                 use_attention: bool = True):
        super().__init__()
        
        self.node_in_dim = node_in_dim
        self.edge_in_dim = edge_in_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_attention = use_attention
        
        # Node feature encoder
        # Input includes: base features + shock indicators (added at runtime)
        self.node_encoder = torch.nn.Sequential(
            torch.nn.Linear(node_in_dim + 2, hidden_dim),  # +2 for shock flags
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout)
        )
        
        # Graph convolution layers
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        
        for i in range(num_layers):
            if use_attention:
                self.convs.append(GATConv(hidden_dim, hidden_dim, heads=4, concat=False))
            else:
                self.convs.append(SAGEConv(hidden_dim, hidden_dim))
            self.batch_norms.append(BatchNorm(hidden_dim))
        
        # Edge value change predictor
        # Predicts Δlog(value) = log(value_t+1) - log(value_t)
        edge_mlp_in = 2 * hidden_dim + edge_in_dim + 1
        self.edge_mlp = torch.nn.Sequential(
            torch.nn.Linear(edge_mlp_in, hidden_dim),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, hidden_dim // 2),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim // 2, 1)
        )
        
        self.output_scale = torch.nn.Parameter(torch.ones(1) * 4.0)
        self.output_bias = torch.nn.Parameter(torch.zeros(1))
    
    def forward(self, x, edge_index, edge_attr, shock_mask_nodes=None, shock_mask_edges=None):
        """
        Args:
            x: Node features [num_nodes, node_in_dim]
            edge_index: Edge connectivity [2, num_edges]
            edge_attr: Edge features [num_edges, edge_in_dim]
            shock_mask_nodes: Binary [num_nodes] - 1 if node is shocked
            shock_mask_edges: Binary [num_edges] - 1 if edge is shocked
        
        Returns:
            delta_log_values: Predicted change in log(value) for each edge
        """
        num_nodes = x.shape[0]
        num_edges = edge_attr.shape[0]
        
        # Add shock indicators to node features
        if shock_mask_nodes is None:
            shock_mask_nodes = torch.zeros(num_nodes, 1, device=x.device)
        else:
            shock_mask_nodes = shock_mask_nodes.view(-1, 1).float()
        
        # Compute "downstream of shock" indicator (any neighbor is shocked)
        src_idx, tgt_idx = edge_index
        downstream_indicator = torch.zeros(num_nodes, 1, device=x.device)
        if shock_mask_nodes.sum() > 0:
            # Propagate shock signal one hop
            downstream_indicator.scatter_add_(0, tgt_idx.unsqueeze(1), shock_mask_nodes[src_idx])
            downstream_indicator = (downstream_indicator > 0).float()
        
        # Augment node features: [original || is_shocked || has_shocked_supplier]
        x_augmented = torch.cat([x, shock_mask_nodes, downstream_indicator], dim=1)
        
        x = self.node_encoder(x_augmented)
        
        for i, (conv, bn) in enumerate(zip(self.convs, self.batch_norms)):
            x = conv(x, edge_index)
            x = bn(x)
            if i < self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        src_emb = x[src_idx]
        tgt_emb = x[tgt_idx]
        
        if shock_mask_edges is not None:
            shock_mask_edges = shock_mask_edges.view(-1, 1).float()
            edge_feat = torch.cat([edge_attr, shock_mask_edges], dim=1)
        else:
            edge_feat = torch.cat([edge_attr, torch.zeros(edge_attr.size(0), 1, device=edge_attr.device)], dim=1)

        edge_input = torch.cat([src_emb, tgt_emb, edge_feat], dim=1)
        
        delta_log_value = self.edge_mlp(edge_input).squeeze(-1)
        
        delta_log_value = delta_log_value * self.output_scale + self.output_bias
        
        return delta_log_value


def load_graphs(years: List[int], embeddings_dir: str = "embeddings") -> List[Data]:
    """Load graphs with value_t and value_t1"""
    graphs = []
    for year in years:
        path = Path(embeddings_dir) / f"graph_{year}_labeled.pt"
        if not path.exists():
            print(f"Warning: {path} not found, skipping year {year}")
            continue
        graph = torch.load(path, weights_only=False)
        graph.year = year

        if not hasattr(graph, 'edge_attr'):
            print(f"Warning: Graph {year} missing edge_attr")
            continue
        
        graphs.append(graph)
    return graphs


def extract_edge_values(graph: Data) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract current and next year edge values from graph.
    Assumes these are stored during feature engineering.
    """
    # Look for value_t and value_t1 in graph
    # If not present, try to extract from edge_attr or compute from other fields
    
    if hasattr(graph, 'value_t') and hasattr(graph, 'value_t1'):
        return graph.value_t, graph.value_t1
    
    # Fallback: compute from edge features
    # This requires knowing which columns in edge_attr correspond to values
    # For now, we'll require these to be added during feature engineering
    raise ValueError(
        "Graph missing value_t and value_t1. "
        "Update feature_eng.py to include these as separate attributes."
    )


def create_shock_training_data(graph_years: List[int], embeddings_dir: str, 
                              shock_threshold: float = 0.15,
                              balanced_sampling: bool = False,
                              balance_ratio: float = 1.0) -> List[Dict]:
    """
    Create training metadata and precompute shock masks once per graph.
    
    Args:
        graph_years: Years to process
        embeddings_dir: Directory with graph files
        shock_threshold: Threshold for shock identification
        balanced_sampling: If True, create balanced gain/drop indices
        balance_ratio: Ratio of gains to drops (1.0 = equal, only used if balanced_sampling=True)
    
    Returns:
        List of metadata dicts with precomputed masks and optional balanced indices
    """
    training_data = []
    
    for year in graph_years:
        path = Path(embeddings_dir) / f"graph_{year}_labeled.pt"
        if not path.exists():
            continue
        
        # Load graph once to precompute masks
        graph = torch.load(path, map_location='cpu', weights_only=False)
        value_t, value_t1 = extract_edge_values(graph)
        pct_change = (value_t1 - value_t) / (value_t + 1e-8)
        shock_mask_edges = (pct_change < -shock_threshold).float()
        
        # Node shock mask: fraction of shocked outgoing edges > 0.2
        src_idx, _ = graph.edge_index
        num_nodes = graph.x.shape[0]
        shocked_out_degree = torch.zeros(num_nodes)
        shocked_out_degree.scatter_add_(0, src_idx, shock_mask_edges)
        out_degree = torch.zeros(num_nodes)
        out_degree.scatter_add_(0, src_idx, torch.ones_like(shock_mask_edges))
        shock_mask_nodes = (shocked_out_degree / (out_degree + 1e-8) > 0.2).float()
        
        metadata = {
            'graph_path': str(path),
            'year': year,
            'shock_threshold': shock_threshold,
            'shock_mask_edges': shock_mask_edges,
            'shock_mask_nodes': shock_mask_nodes
        }
        
        # Add balanced sampling indices if requested
        if balanced_sampling:
            delta_log = torch.log1p(value_t1) - torch.log1p(value_t)
            
            # Classify edges
            gain_mask = delta_log > SIGN_EPS
            drop_mask = delta_log < -SIGN_EPS
            neutral_mask = torch.abs(delta_log) <= SIGN_EPS
            
            gain_idx = torch.where(gain_mask)[0]
            drop_idx = torch.where(drop_mask)[0]
            neutral_idx = torch.where(neutral_mask)[0]
            
            n_gains = len(gain_idx)
            n_drops = len(drop_idx)
            
            # Sample to achieve balance_ratio
            if balance_ratio == 1.0:
                # Equal gains and drops
                n_target = min(n_gains, n_drops)
                sampled_gain_idx = gain_idx[torch.randperm(n_gains)[:n_target]]
                sampled_drop_idx = drop_idx[torch.randperm(n_drops)[:n_target]]
            else:
                # Custom ratio
                n_target = min(n_gains, int(n_drops * balance_ratio))
                sampled_gain_idx = gain_idx[torch.randperm(n_gains)[:n_target]]
                sampled_drop_idx = drop_idx[torch.randperm(n_drops)[:int(n_target/balance_ratio)]]
            
            # Add some neutral edges (10% of balanced set)
            n_neutral = int(len(sampled_gain_idx) * 0.1)
            sampled_neutral_idx = neutral_idx[torch.randperm(len(neutral_idx))[:n_neutral]] if len(neutral_idx) > 0 else torch.tensor([])
            
            # Combine and ensure long type for indexing
            balanced_edge_idx = torch.cat([sampled_gain_idx, sampled_drop_idx, sampled_neutral_idx]).long()
            
            metadata['balanced_edge_idx'] = balanced_edge_idx
            metadata['n_gains_sampled'] = len(sampled_gain_idx)
            metadata['n_drops_sampled'] = len(sampled_drop_idx)
            
            if balanced_sampling:
                print(f"  Year {year}: Original {n_gains}G/{n_drops}D → Sampled {len(sampled_gain_idx)}G/{len(sampled_drop_idx)}D")
        
        training_data.append(metadata)
    
    return training_data


def load_and_process_example(example_metadata: Dict, device, edge_sample_ratio=0.3, 
                            use_balanced=True) -> Dict:
    """
    Load graph and use precomputed shock annotations with edge sampling.
    
    Args:
        example_metadata: Metadata dict with graph path and masks
        device: Device to load tensors to
        edge_sample_ratio: Ratio of edges to sample (if not using balanced indices)
        use_balanced: If True and balanced_edge_idx exists, use balanced sampling
    
    Returns:
        Dict with graph data, masks, and targets
    """
    graph = torch.load(example_metadata['graph_path'], map_location='cpu', weights_only=False)
    
    # Extract values
    value_t, value_t1 = extract_edge_values(graph)
    
    # Use precomputed shock masks
    shock_mask_edges = example_metadata.get('shock_mask_edges')
    shock_mask_nodes = example_metadata.get('shock_mask_nodes')
    if shock_mask_edges is None or shock_mask_nodes is None:
        # Fallback to on-the-fly computation (shouldn't happen once precomputed)
        pct_change = (value_t1 - value_t) / (value_t + 1e-8)
        shock_mask_edges = (pct_change < -example_metadata['shock_threshold']).float()
        src_idx, _ = graph.edge_index
        num_nodes = graph.x.shape[0]
        shocked_out_degree = torch.zeros(num_nodes)
        shocked_out_degree.scatter_add_(0, src_idx, shock_mask_edges)
        out_degree = torch.zeros(num_nodes)
        out_degree.scatter_add_(0, src_idx, torch.ones_like(shock_mask_edges))
        shock_mask_nodes = (shocked_out_degree / (out_degree + 1e-8) > 0.2).float()
    
    # Choose sampling strategy
    if use_balanced and 'balanced_edge_idx' in example_metadata:
        # Use precomputed balanced indices
        balanced_idx = example_metadata['balanced_edge_idx']
        
        # Further subsample if needed
        if edge_sample_ratio < 1.0:
            n_sample = int(len(balanced_idx) * edge_sample_ratio)
            subsample_idx = torch.randperm(len(balanced_idx))[:n_sample]
            sampled_edge_idx = balanced_idx[subsample_idx].long()
        else:
            sampled_edge_idx = balanced_idx.long()
    else:
        # Original stratified sampling: keep all shocked edges + random sample of others
        num_edges = graph.edge_index.shape[1]
        num_sampled = int(num_edges * edge_sample_ratio)
        
        shocked_edge_idx = torch.where(shock_mask_edges > 0)[0]
        normal_edge_idx = torch.where(shock_mask_edges == 0)[0]
        
        # Keep all shocked edges
        num_normal_to_sample = max(0, num_sampled - len(shocked_edge_idx))
        sampled_normal_idx = normal_edge_idx[torch.randperm(len(normal_edge_idx))[:num_normal_to_sample]]
        
        sampled_edge_idx = torch.cat([shocked_edge_idx, sampled_normal_idx]).long()
    
    # Subsample graph
    graph.edge_index = graph.edge_index[:, sampled_edge_idx]
    graph.edge_attr = graph.edge_attr[sampled_edge_idx]
    value_t = value_t[sampled_edge_idx]
    value_t1 = value_t1[sampled_edge_idx]
    shock_mask_edges = shock_mask_edges[sampled_edge_idx]
    
    # Compute deltas (node masks already precomputed)
    log_value_t = torch.log1p(value_t)
    log_value_t1 = torch.log1p(value_t1)
    delta_log_value = log_value_t1 - log_value_t
    
    src_idx, tgt_idx = graph.edge_index
    num_nodes = graph.x.shape[0]
    
    return {
        'graph': graph.to(device),
        'shock_mask_nodes': shock_mask_nodes.to(device),
        'shock_mask_edges': shock_mask_edges.to(device),
        'target': delta_log_value.to(device),
        'value_t': value_t,
        'value_t1': value_t1,
        'year': example_metadata['year']
    }


def train_epoch(model, optimizer, training_metadata, device, loss_fn, accumulation_steps=4, edge_sample_ratio=0.3):
    """Train with gradient accumulation to reduce memory per step"""
    model.train()
    total_loss = 0
    total_samples = 0
    total_direction_correct_count = 0
    
    optimizer.zero_grad()
    
    pbar = tqdm(enumerate(training_metadata), total=len(training_metadata), 
                desc="Training", leave=False)
    
    for i, metadata in pbar:
        step_start = time.time()
            
        # Load graph only when needed (use balanced sampling if available)
        example = load_and_process_example(metadata, device, 
                                          edge_sample_ratio=edge_sample_ratio,
                                          use_balanced=True)
        load_time = time.time() - step_start

        # Forward pass
        forward_start = time.time()
        predictions = model(
            example['graph'].x, 
            example['graph'].edge_index, 
            example['graph'].edge_attr,
            example['shock_mask_nodes'],
            example['shock_mask_edges']
        )
        
        if hasattr(loss_fn, 'name') and loss_fn.name.startswith("WeightedSign"):
            loss = loss_fn(predictions, example['target'], example['value_t'])
        else:
            loss = loss_fn(predictions, example['target'])

        loss = loss / accumulation_steps
        forward_time = time.time() - forward_start

        backward_start = time.time()
        loss.backward()
        backward_time = time.time() - backward_start
        
        # Track direction accuracy with epsilon around zero (global counting)
        with torch.no_grad():
            preds_detached = predictions.detach()
            targets = example['target']
            pred_sign = torch.where(preds_detached.abs() < SIGN_EPS, torch.zeros_like(preds_detached), torch.sign(preds_detached))
            target_sign = torch.where(targets.abs() < SIGN_EPS, torch.zeros_like(targets), torch.sign(targets))
            correct_count = (pred_sign == target_sign).sum().item()
            total_direction_correct_count += correct_count
        
        total_loss += loss.item() * len(example['target']) * accumulation_steps
        total_samples += len(example['target'])
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'load': f'{load_time:.2f}s',
            'fwd': f'{forward_time:.2f}s',
            'bwd': f'{backward_time:.2f}s'
        })

        # Explicitly delete to free memory
        del example, predictions
        
        # Update weights every N steps
        if (i + 1) % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
    
    # Handle remaining gradients
    if len(training_metadata) % accumulation_steps != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()
    
    return {
        'loss': total_loss / total_samples,
        'direction_acc': (total_direction_correct_count / total_samples) if total_samples > 0 else 0.0
    }


@torch.no_grad()
def evaluate(model, training_metadata, device, edge_sample_ratio=1.0) -> Dict[str, float]:
    """Evaluate model on predicting edge value changes"""
    model.eval()
    
    all_preds = []
    all_targets = []
    all_values_t = []
    all_values_t1_true = []
    
    for metadata in training_metadata:  

        # Don't use balanced sampling for evaluation - use full unbalanced data
        example = load_and_process_example(metadata, device, 
                                          edge_sample_ratio=edge_sample_ratio,
                                          use_balanced=False)
        
        predictions = model(
            example['graph'].x,
            example['graph'].edge_index,
            example['graph'].edge_attr,
            example['shock_mask_nodes'],
            example['shock_mask_edges']
        )
        
        all_preds.append(predictions.cpu())
        all_targets.append(example['target'].cpu())
        all_values_t.append(example['value_t'])
        all_values_t1_true.append(example['value_t1'])
        
        # Free memory immediately
        del example
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    all_preds = torch.cat(all_preds).numpy()
    all_targets = torch.cat(all_targets).numpy()
    all_values_t = torch.cat(all_values_t).numpy()
    all_values_t1_true = torch.cat(all_values_t1_true).numpy()
    
    # Metrics on Δlog(value)
    mse = mean_squared_error(all_targets, all_preds)
    mae = mean_absolute_error(all_targets, all_preds)
    r2 = r2_score(all_targets, all_preds)
    
    # Metrics on actual values (reconstruct from Δlog)
    log_values_t = np.log1p(all_values_t)
    log_values_t1_pred = log_values_t + all_preds
    values_t1_pred = np.expm1(log_values_t1_pred)
    
    eps = 1e-6
    # Classic MAPE can explode when true values are ~0 (kept for reference)
    denom = np.maximum(np.abs(all_values_t1_true), eps)
    mape = float(np.mean(np.abs((values_t1_pred - all_values_t1_true) / denom)) * 100)
    # SMAPE is better behaved near zero - use this as primary percentage metric
    smape = float(np.mean(2.0 * np.abs(values_t1_pred - all_values_t1_true) /
                          (np.abs(values_t1_pred) + np.abs(all_values_t1_true) + eps)) * 100)
    
    # Directional accuracy (with epsilon around zero)
    sign_target = np.where(np.abs(all_targets) < SIGN_EPS, 0, np.sign(all_targets))
    sign_pred = np.where(np.abs(all_preds) < SIGN_EPS, 0, np.sign(all_preds))
    direction_correct_mask = (sign_target == sign_pred)
    direction_acc = float(direction_correct_mask.mean())

    # Directional accuracy split by gains (>0) and drops (<0)
    gains_mask = all_targets > SIGN_EPS
    drops_mask = all_targets < -SIGN_EPS
    gain_dir_acc = float(direction_correct_mask[gains_mask].mean()) if gains_mask.any() else 0.0
    drop_dir_acc = float(direction_correct_mask[drops_mask].mean()) if drops_mask.any() else 0.0

    # Drop detection F1 (treat "drop" as positive class)
    tp = float(np.sum((sign_pred < 0) & (sign_target < 0)))
    fp = float(np.sum((sign_pred < 0) & (sign_target >= 0)))
    fn = float(np.sum((sign_pred >= 0) & (sign_target < 0)))
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    drop_f1 = 2.0 * precision * recall / (precision + recall + 1e-8)

    # Sign distributions
    target_neg_frac = float((all_targets < -SIGN_EPS).mean())
    target_zero_frac = float((np.abs(all_targets) <= SIGN_EPS).mean())
    target_pos_frac = float((all_targets > SIGN_EPS).mean())
    pred_neg_frac = float((all_preds < -SIGN_EPS).mean())
    pred_zero_frac = float((np.abs(all_preds) <= SIGN_EPS).mean())
    pred_pos_frac = float((all_preds > SIGN_EPS).mean())

    return {
        'mse': mse,
        'rmse': np.sqrt(mse),
        'mae': mae,
        'r2': r2,
        'mape': mape,
        'smape': smape,
        # Direction metrics
        'direction_acc': direction_acc,
        'gain_dir_acc': gain_dir_acc,
        'drop_dir_acc': drop_dir_acc,
        'drop_f1': float(drop_f1),
        # Distributions
        'target_neg_frac': target_neg_frac,
        'target_zero_frac': target_zero_frac,
        'target_pos_frac': target_pos_frac,
        'pred_neg_frac': pred_neg_frac,
        'pred_zero_frac': pred_zero_frac,
        'pred_pos_frac': pred_pos_frac,
    }


def get_best_metric(metrics: Dict[str, float], metric_name: str) -> float:
    """
    Return the scalar to optimize for validation scheduling/early stopping.
    Higher-is-better metrics: r2, direction_acc
    Lower-is-better metrics: mse, rmse, mae, mape
    Falls back gracefully if the requested metric is missing.
    """
    name = (metric_name or "r2").lower()

    if name in metrics:
        value = metrics[name]
    else:
        if name == 'rmse' and 'mse' in metrics:
            value = float(np.sqrt(metrics['mse']))
        else:
            for k in ['r2', 'direction_acc', 'rmse', 'mse', 'mae', 'mape']:
                if k in metrics:
                    value = metrics[k]
                    name = k
                    break
            else:
                return 0.0

    # Determine optimization direction
    maximize = name in {'r2', 'direction_acc'}
    # For schedulers expecting a higher-is-better target, return as-is;
    # for lower-is-better, return the negative so larger is still better.
    return float(value if maximize else -value)


def simulate_shock(model, graph, shocked_nodes: List[str], 
                  shock_magnitude: float, device) -> Dict:
    """
    Simulate a counterfactual shock and predict network response.
    
    Args:
        model: Trained shock propagation model
        graph: Current network state
        shocked_nodes: List of node IDs to shock (e.g., ['ECU_AGR'])
        shock_magnitude: Fractional decrease (e.g., 0.5 = 50% drop)
        device: torch device
    
    Returns:
        Dictionary with predictions and analysis
    """
    model.eval()
    graph = graph.to(device)
    
    # TODO: Need node ID to index mapping
    # For now, assume we have a way to identify nodes
    # This would require storing node labels in the graph
    
    print(f"\n{'='*70}")
    print(f"SHOCK SIMULATION")
    print(f"{'='*70}")
    print(f"Shocked nodes: {shocked_nodes}")
    print(f"Shock magnitude: {shock_magnitude:.1%} reduction")
    
    # Create shock masks
    shock_mask_nodes = torch.zeros(graph.num_nodes, device=device)
    shock_mask_nodes[:len(shocked_nodes)] = 1.0
    
    # Baseline: predict without shock
    with torch.no_grad():
        baseline_delta = model(
            graph.x,
            graph.edge_index,
            graph.edge_attr,
            shock_mask_nodes=None,
            shock_mask_edges=None
        )
        
        # Shocked: predict with shock
        shocked_delta = model(
            graph.x,
            graph.edge_index,
            graph.edge_attr,
            shock_mask_nodes=shock_mask_nodes,
            shock_mask_edges=None
        )
    
    # Compute propagation effect
    propagation_effect = shocked_delta - baseline_delta
    
    # Analyze results
    src_idx, tgt_idx = graph.edge_index
    
    # Edges directly from shocked nodes
    direct_edges = torch.isin(src_idx, torch.where(shock_mask_nodes > 0)[0])
    
    # Indirect edges (downstream)
    indirect_edges = ~direct_edges
    
    results = {
        'baseline_delta': baseline_delta.cpu().numpy(),
        'shocked_delta': shocked_delta.cpu().numpy(),
        'propagation_effect': propagation_effect.cpu().numpy(),
        'direct_effect_mean': propagation_effect[direct_edges].mean().item(),
        'indirect_effect_mean': propagation_effect[indirect_edges].mean().item(),
        'total_affected_edges': (torch.abs(propagation_effect) > 0.01).sum().item(),
    }
    
    print(f"\nPropagation Analysis:")
    print(f"  Direct edges affected: {direct_edges.sum().item()}")
    print(f"  Avg direct effect: {results['direct_effect_mean']:.4f} (Δlog)")
    print(f"  Avg indirect effect: {results['indirect_effect_mean']:.4f} (Δlog)")
    print(f"  Total edges significantly affected: {results['total_affected_edges']}")
    
    return results

def save_checkpoint(epoch, model, optimizer, scheduler, val_metrics, args, save_dir, is_best=False):
    """Save training checkpoint with all state needed to resume"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'val_metrics': val_metrics,
        'args': vars(args),
        'best_val_r2': getattr(save_checkpoint, 'best_val_r2', -float('inf')),
        'patience_counter': getattr(save_checkpoint, 'patience_counter', 0)
    }
    
    # Save latest checkpoint
    latest_path = save_dir / "checkpoint_latest.pt"
    torch.save(checkpoint, latest_path)
    
    # Save best model separately
    if is_best:
        torch.save(checkpoint, save_dir / "best_model.pt")

# Attach state to function object for tracking across calls
save_checkpoint.best_val_r2 = -float('inf')
save_checkpoint.patience_counter = 0


def load_checkpoint(checkpoint_path, model, optimizer, scheduler, device):
    """Load checkpoint and restore training state"""
    print(f"\nLoading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    start_epoch = checkpoint['epoch'] + 1
    best_val_r2 = checkpoint.get('best_val_r2', -float('inf'))
    patience_counter = checkpoint.get('patience_counter', 0)
    
    # Restore function state
    save_checkpoint.best_val_r2 = best_val_r2
    save_checkpoint.patience_counter = patience_counter
    
    print(f"✓ Resumed from epoch {checkpoint['epoch']}")
    print(f"  Best val R²: {best_val_r2:.4f}")
    print(f"  Patience counter: {patience_counter}/{checkpoint['args'].get('patience', 15)}")
    
    return start_epoch, best_val_r2, patience_counter


def main():
    parser = argparse.ArgumentParser()
    
    # Data
    parser.add_argument("--train-start", type=int, default=1996)
    parser.add_argument("--train-end", type=int, default=2019)
    parser.add_argument("--val-start", type=int, default=2020)
    parser.add_argument("--val-end", type=int, default=2020)
    parser.add_argument("--test-start", type=int, default=2021)
    parser.add_argument("--test-end", type=int, default=2021)
    parser.add_argument("--shock-threshold", type=float, default=0.15,
                       help="Threshold for identifying natural shocks (default: 15%% drop)")
    
    # Model
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--use-attention", action="store_true",
                       help="Use GAT instead of GraphSAGE for attention mechanism")

    # Loss
    parser.add_argument("--loss", type=str, default="mse",
                   choices=['mse', 'sign_corrected', 'weighted_sign', 
                           'focal', 'hybrid'],
                   help="Loss function type")
    parser.add_argument("--loss-alpha", type=float, default=1.0,
                    help="Alpha parameter for sign-corrected losses")
    parser.add_argument("--loss-gamma", type=float, default=1.5,
                    help="Gamma parameter for focal losses")
    
    # Training
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--train-edge-sample-ratio", type=float, default=0.3,
                        help="Fraction of edges to sample per example during training (0-1)")
    parser.add_argument("--balanced-sampling", action="store_true",
                        help="Use balanced gain/drop sampling (addresses temporal imbalance)")
    parser.add_argument("--balance-ratio", type=float, default=1.0,
                        help="Ratio of gains to drops in training (1.0 = equal, only used with --balanced-sampling)")
    
    # System
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--embeddings-dir", type=str, default="embeddings")
    parser.add_argument("--save-dir", type=str, default="models")
    parser.add_argument("--best-metric", type=str, default="r2",
                        choices=['r2', 'rmse', 'mse', 'mae', 'mape', 'direction_acc'],
                        help="Validation metric to optimize (default: r2)")
    parser.add_argument("--sign-eps", type=float, default=1e-4,
                        help="Epsilon threshold to treat small deltas as zero for direction accuracy")

    # Checkpoints
    parser.add_argument("--resume", type=str, default=None,
                       help="Path to checkpoint to resume from (default: auto-detect latest)")
    parser.add_argument("--auto-resume", action="store_true",
                       help="Automatically resume from latest checkpoint if it exists")
    
    args = parser.parse_args()

    # Set global epsilon for sign comparisons
    global SIGN_EPS
    SIGN_EPS = args.sign_eps
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path("runs/shock_propagation") / datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter(log_dir)
    
    print("="*70)
    print("SHOCK PROPAGATION MODEL - Supply Chain Network")
    print("="*70)
    print(f"Device: {args.device}")
    print(f"Train: {args.train_start}-{args.train_end}")
    print(f"Val: {args.val_start}-{args.val_end}")
    print(f"Test: {args.test_start}-{args.test_end}")
    print(f"Shock threshold: {args.shock_threshold:.1%}")
    if args.balanced_sampling:
        print(f"Balanced sampling: ENABLED (ratio {args.balance_ratio}:1 gains:drops)")
    print("="*70)
    
    # Load graphs
    print("\nLoading graphs...")
    train_years = list(range(args.train_start, args.train_end + 1))
    val_years = list(range(args.val_start, args.val_end + 1))
    test_years = list(range(args.test_start, args.test_end + 1))
    
    train_graphs = load_graphs(train_years, args.embeddings_dir)
    val_graphs = load_graphs(val_years, args.embeddings_dir)
    test_graphs = load_graphs(test_years, args.embeddings_dir)
    
    print(f"Loaded {len(train_graphs)} train, {len(val_graphs)} val, {len(test_graphs)} test graphs")
    
    # Create training data with shock annotations
    if args.balanced_sampling:
        print("\nCreating balanced training data...")
    else:
        print("\nAnnotating natural shocks in historical data...")
    train_data = create_shock_training_data(
        train_years, args.embeddings_dir, args.shock_threshold,
        balanced_sampling=args.balanced_sampling,
        balance_ratio=args.balance_ratio
    )
    val_data = create_shock_training_data(
        val_years, args.embeddings_dir, args.shock_threshold,
        balanced_sampling=False  # Always unbalanced for validation
    )
    test_data = create_shock_training_data(
        test_years, args.embeddings_dir, args.shock_threshold,
        balanced_sampling=False  # Always unbalanced for test
    )
    
    print(f"Train examples: {len(train_data)}")
    print(f"Val examples: {len(val_data)}")
    print(f"Test examples: {len(test_data)}")
    
    processed_example = load_and_process_example(
        create_shock_training_data([train_years[0]], args.embeddings_dir, args.shock_threshold)[0],
        args.device,
        edge_sample_ratio=args.train_edge_sample_ratio,
        use_balanced=False  # Just getting dimensions
    )
    node_dim = processed_example['graph'].x.shape[1]
    edge_dim = processed_example['graph'].edge_attr.shape[1]
    
    print(f"\nModel dimensions:")
    print(f"  Node features: {node_dim}")
    print(f"  Edge features: {edge_dim}")
    
    model = ShockPropagationGNN(
        node_in_dim=node_dim,
        edge_in_dim=edge_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        use_attention=args.use_attention
    ).to(args.device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )

    if args.loss == 'sign_corrected':
        loss_fn = get_loss_function('sign_corrected', alpha=args.loss_alpha)
    elif args.loss == 'focal':
        loss_fn = get_loss_function('focal', gamma=args.loss_gamma)
    elif args.loss == 'hybrid':
        loss_fn = get_loss_function('hybrid', alpha=args.loss_alpha, gamma=args.loss_gamma)
    else:
        loss_fn = get_loss_function(args.loss)
    
    # Training loop
    print("\nTraining...")
    start_epoch = 1
    best_epoch = 0
    patience_counter = 0
    best_metric_value = -float('inf')
    prev_val_metrics = None

    # Check for resume flag
    if hasattr(args, 'resume') and args.resume:
        start_epoch, prev_val_metrics = load_checkpoint(
            args.resume, model, optimizer, scheduler, args.device)
    elif hasattr(args, 'auto_resume') and args.auto_resume:
        latest_checkpoint = Path(save_dir) / "checkpoint_latest.pt"
        if latest_checkpoint.exists():
            start_epoch, prev_val_metrics = load_checkpoint(
                latest_checkpoint, model, optimizer, scheduler, args.device)
    if prev_val_metrics:
        best_metric_value = get_best_metric(prev_val_metrics, args.best_metric)
    
    
    for epoch in range(start_epoch, args.epochs + 1):
        train_metrics = train_epoch(
            model, optimizer, train_data, args.device, loss_fn,
            accumulation_steps=4, edge_sample_ratio=args.train_edge_sample_ratio
        )
        val_metrics = evaluate(
            model, val_data, args.device
        )
        
        scheduler.step(get_best_metric(val_metrics, args.best_metric))
        
        writer.add_scalar('Loss/train', train_metrics['loss'], epoch)
        writer.add_scalar('Direction Accuracy/train', train_metrics['direction_acc'], epoch)
        for k, v in val_metrics.items():
            writer.add_scalar(f'Metrics/val_{k}', v, epoch)
        
        print(f"\nEpoch {epoch:3d}/{args.epochs}")
        print(f"  Train Loss: {train_metrics['loss']:.4f}")
        print(f"  Train Direction Acc: {train_metrics['direction_acc']:.2%}")
        print(f"  Val R²: {val_metrics['r2']:.4f}")
        print(f"  Val RMSE: {val_metrics['rmse']:.4f}")
        print(f"  Val Direction Acc: {val_metrics['direction_acc']:.2%}")
        
        # Check for improvement
        current_metric_value = get_best_metric(val_metrics, args.best_metric)
        is_best = current_metric_value > best_metric_value
        if is_best:
            best_metric_value = current_metric_value
            best_epoch = epoch
            patience_counter = 0
            print(f"    ✓ New best {args.best_metric}: {best_metric_value:.4f}")
        else:
            patience_counter += 1
        
        # Save checkpoint
        save_checkpoint(epoch, model, optimizer, scheduler, val_metrics, args, save_dir, is_best)
        
        # Early stopping check
        if patience_counter >= args.patience:
            print(f"\nEarly stopping at epoch {epoch}")
            print(f"Best {args.best_metric} was {best_metric_value:.4f} at epoch {best_epoch}")
            break
    
    # Final evaluation
    print("\n" + "="*70)
    print("FINAL EVALUATION")
    print("="*70)
    
    checkpoint = torch.load(save_dir / "best_model.pt", weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_metrics = evaluate(model, test_data, args.device)
    
    print(f"\nTest Set Performance:")
    print(f"  Direction Accuracy: {test_metrics['direction_acc']:.2%}")
    print(f"    - Gains: {test_metrics['gain_dir_acc']:.2%}")
    print(f"    - Drops: {test_metrics['drop_dir_acc']:.2%}")
    print(f"  Drop Detection F1: {test_metrics['drop_f1']:.3f}")
    print(f"  R²: {test_metrics['r2']:.3f}")
    print(f"  RMSE: {test_metrics['rmse']:.4f}")
    print(f"  SMAPE: {test_metrics['smape']:.1f}%")
    
    # Save results
    best_checkpoint = torch.load(save_dir / "best_model.pt", weights_only=False)
    results = {
        'test_metrics': {k: float(v) for k, v in test_metrics.items()},
        'best_val_metrics': {k: float(v) for k, v in best_checkpoint['val_metrics'].items()},
        'best_epoch': best_checkpoint['epoch'],
        'args': vars(args)
    }
    
    with open(save_dir / "results.json", 'w') as f:
        json.dump(results, f, indent=2, default=float)
    
    print(f"\n✓ Model saved to {save_dir}/")
    print(f"✓ Logs: {log_dir}/")
    
    # Demo: Simulate a counterfactual shock
    print("\n" + "="*70)
    print("DEMO: Counterfactual Shock Simulation")
    print("="*70)
    print("(Using 2021 graph as baseline)")
    
    if test_graphs:
        demo_graph = test_graphs[0]
        simulate_shock(
            model, 
            demo_graph,
            shocked_nodes=['ECU_A01'], 
            shock_magnitude=0.5,  # 50% reduction
            device=args.device
        )
    
    writer.close()


if __name__ == "__main__":
    main()