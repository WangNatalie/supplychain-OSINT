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
import pandas as pd
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
        
        # Graph convolution layers (use GAT for attention to shocked nodes)
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        
        for i in range(num_layers):
            if use_attention:
                # GAT: learns to attend to shocked/important nodes
                self.convs.append(GATConv(hidden_dim, hidden_dim, heads=4, concat=False))
            else:
                self.convs.append(SAGEConv(hidden_dim, hidden_dim))
            self.batch_norms.append(BatchNorm(hidden_dim))
        
        # Edge value change predictor
        # Predicts Δlog(value) = log(value_t+1) - log(value_t)
        edge_mlp_in = 2 * hidden_dim + edge_in_dim
        self.edge_mlp = torch.nn.Sequential(
            torch.nn.Linear(edge_mlp_in, hidden_dim),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, hidden_dim // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim // 2, 1)  # Predict Δlog(value)
        )
    
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
        
        # Encode
        x = self.node_encoder(x_augmented)
        
        # Message passing (propagate shock information)
        for i, (conv, bn) in enumerate(zip(self.convs, self.batch_norms)):
            x = conv(x, edge_index)
            x = bn(x)
            if i < self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Edge-level prediction
        src_emb = x[src_idx]
        tgt_emb = x[tgt_idx]
        
        # Optionally include edge shock indicator in edge features
        if shock_mask_edges is not None:
            shock_mask_edges = shock_mask_edges.view(-1, 1).float()
            edge_attr_augmented = torch.cat([edge_attr, shock_mask_edges], dim=1)
            # Adjust MLP input size if needed (or pad edge_attr)
            # For simplicity, we'll use original edge_attr
        
        edge_input = torch.cat([src_emb, tgt_emb, edge_attr], dim=1)
        
        # Predict change in log(value)
        delta_log_value = self.edge_mlp(edge_input).squeeze(-1)
        
        return delta_log_value


def load_graphs(years: List[int], embeddings_dir: str = "embeddings") -> List[Data]:
    """Load graphs with value_t and value_t1"""
    graphs = []
    for year in years:
        path = Path(embeddings_dir) / f"graph_{year}_labeled.pt"
        if not path.exists():
            print(f"Warning: {path} not found, skipping year {year}")
            continue
        graph = torch.load(path)
        graph.year = year
        
        # Verify required attributes
        if not hasattr(graph, 'edge_attr'):
            print(f"Warning: Graph {year} missing edge_attr")
            continue
        
        # Extract value_t and value_t1 from edge features if not already separate
        # Assuming your edge features include log_value_t and we have pct_change
        # We'll extract from the labeled data
        
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
                               shock_threshold: float = 0.15) -> List[Dict]:
    """
    Create training examples with lazy loading.
    Only stores metadata, not full graphs.
    """
    training_data = []
    
    for year in graph_years:
        path = Path(embeddings_dir) / f"graph_{year}_labeled.pt"
        if not path.exists():
            continue
            
        # Store path instead of loading graph
        training_data.append({
            'graph_path': str(path),
            'year': year,
            'shock_threshold': shock_threshold
        })
    
    return training_data


def load_and_process_example(example_metadata: Dict, device, edge_sample_ratio=0.3) -> Dict:
    """Load graph and create shock annotations with edge sampling"""
    graph = torch.load(example_metadata['graph_path'], map_location='cpu')
    
    # Extract values
    value_t, value_t1 = extract_edge_values(graph)
    
    # Sample edges to reduce memory
    num_edges = graph.edge_index.shape[1]
    num_sampled = int(num_edges * edge_sample_ratio)
    
    # Stratified sampling: keep all shocked edges + random sample of others
    pct_change = (value_t1 - value_t) / (value_t + 1e-8)
    shock_mask_edges = (pct_change < -example_metadata['shock_threshold']).float()
    
    shocked_edge_idx = torch.where(shock_mask_edges > 0)[0]
    normal_edge_idx = torch.where(shock_mask_edges == 0)[0]
    
    # Keep all shocked edges
    num_normal_to_sample = max(0, num_sampled - len(shocked_edge_idx))
    sampled_normal_idx = normal_edge_idx[torch.randperm(len(normal_edge_idx))[:num_normal_to_sample]]
    
    sampled_edge_idx = torch.cat([shocked_edge_idx, sampled_normal_idx])
    
    # Subsample graph
    graph.edge_index = graph.edge_index[:, sampled_edge_idx]
    graph.edge_attr = graph.edge_attr[sampled_edge_idx]
    value_t = value_t[sampled_edge_idx]
    value_t1 = value_t1[sampled_edge_idx]
    shock_mask_edges = shock_mask_edges[sampled_edge_idx]
    
    # Compute node masks and deltas
    log_value_t = torch.log1p(value_t)
    log_value_t1 = torch.log1p(value_t1)
    delta_log_value = log_value_t1 - log_value_t
    
    src_idx, tgt_idx = graph.edge_index
    num_nodes = graph.x.shape[0]
    
    shocked_out_degree = torch.zeros(num_nodes)
    shocked_out_degree.scatter_add_(0, src_idx, shock_mask_edges)
    
    out_degree = torch.zeros(num_nodes)
    out_degree.scatter_add_(0, src_idx, torch.ones_like(shock_mask_edges))
    shock_mask_nodes = (shocked_out_degree / (out_degree + 1e-8) > 0.2).float()
    
    return {
        'graph': graph.to(device),
        'shock_mask_nodes': shock_mask_nodes.to(device),
        'shock_mask_edges': shock_mask_edges.to(device),
        'target': delta_log_value.to(device),
        'value_t': value_t,
        'value_t1': value_t1,
        'year': example_metadata['year']
    }


def train_epoch(model, optimizer, training_metadata, device, accumulation_steps=4):
    """Train with gradient accumulation to reduce memory per step"""
    model.train()
    total_loss = 0
    total_samples = 0
    
    optimizer.zero_grad()
    
    for i, metadata in enumerate(training_metadata):
        if i % 5 == 0:
            print_memory_usage(f"   Step {i}/{len(training_metadata)} - ")
            
        # Load graph only when needed
        example = load_and_process_example(metadata, device)
        
        # Forward pass
        predictions = model(
            example['graph'].x, 
            example['graph'].edge_index, 
            example['graph'].edge_attr,
            example['shock_mask_nodes'],
            example['shock_mask_edges']
        )
        
        loss = F.mse_loss(predictions, example['target'])
        loss = loss + 0.01 * F.l1_loss(predictions, example['target'])
        
        # Scale loss by accumulation steps
        loss = loss / accumulation_steps
        loss.backward()
        
        total_loss += loss.item() * len(example['target']) * accumulation_steps
        total_samples += len(example['target'])
        
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
    
    return total_loss / total_samples


@torch.no_grad()
def evaluate(model, training_metadata, device) -> Dict[str, float]:
    """Evaluate model on predicting edge value changes"""
    model.eval()
    
    all_preds = []
    all_targets = []
    all_values_t = []
    all_values_t1_true = []
    
    for metadata in training_metadata:  # Changed from training_data
        # Load example on-demand
        example = load_and_process_example(metadata, device)
        
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
    
    mape = np.mean(np.abs((all_values_t1_true - values_t1_pred) / (all_values_t1_true + 1e-8))) * 100
    
    # Directional accuracy (did we predict the right direction of change?)
    direction_correct = (np.sign(all_targets) == np.sign(all_preds)).mean()
    
    return {
        'mse': mse,
        'rmse': np.sqrt(mse),
        'mae': mae,
        'r2': r2,
        'mape': mape,
        'direction_accuracy': direction_correct
    }


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
    # shocked_indices = [node_id_to_idx[node] for node in shocked_nodes]
    # shock_mask_nodes[shocked_indices] = 1.0
    
    # For now, shock first N nodes as demonstration
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
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--use-attention", action="store_true",
                       help="Use GAT instead of GraphSAGE for attention mechanism")
    
    # Training
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=15)
    
    # System
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--embeddings-dir", type=str, default="embeddings")
    parser.add_argument("--save-dir", type=str, default="models/shock_propagation")
    
    args = parser.parse_args()
    
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
    print("\nAnnotating natural shocks in historical data...")
    train_data = create_shock_training_data(train_years, args.embeddings_dir, args.shock_threshold)
    val_data = create_shock_training_data(val_years, args.embeddings_dir, args.shock_threshold)
    test_data = create_shock_training_data(test_years, args.embeddings_dir, args.shock_threshold)
    
    print(f"Train examples: {len(train_data)}")
    print(f"Val examples: {len(val_data)}")
    print(f"Test examples: {len(test_data)}")
    
    # Model
    sample = train_graphs[0]
    node_dim = sample.x.shape[1]
    edge_dim = sample.edge_attr.shape[1]
    
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
        optimizer, mode='max', factor=0.5, patience=5, verbose=True
    )
    
    # Training loop
    print("\nTraining...")
    best_val_r2 = -float('inf')
    best_epoch = 0
    patience_counter = 0
    
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, optimizer, train_data, args.device)
        val_metrics = evaluate(model, val_data, args.device)
        
        scheduler.step(val_metrics['r2'])
        
        writer.add_scalar('Loss/train', train_loss, epoch)
        for k, v in val_metrics.items():
            writer.add_scalar(f'Metrics/val_{k}', v, epoch)
        
        if epoch % 5 == 0 or epoch == 1:
            print(f"\nEpoch {epoch:3d}/{args.epochs}")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val R²: {val_metrics['r2']:.4f}")
            print(f"  Val RMSE: {val_metrics['rmse']:.4f}")
            print(f"  Val Direction Acc: {val_metrics['direction_accuracy']:.2%}")
        
        if val_metrics['r2'] > best_val_r2:
            best_val_r2 = val_metrics['r2']
            best_epoch = epoch
            patience_counter = 0
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': val_metrics,
                'args': vars(args)
            }, save_dir / "best_model.pt")
            print(f"  ✓ Best model saved (R²: {best_val_r2:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break
    
    # Final evaluation
    print("\n" + "="*70)
    print("FINAL EVALUATION")
    print("="*70)
    
    checkpoint = torch.load(save_dir / "best_model.pt")
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_metrics = evaluate(model, test_data, args.device)
    
    print(f"\nTest Set Performance:")
    print(f"  R²: {test_metrics['r2']:.4f}")
    print(f"  RMSE: {test_metrics['rmse']:.4f}")
    print(f"  MAE: {test_metrics['mae']:.4f}")
    print(f"  MAPE: {test_metrics['mape']:.2f}%")
    print(f"  Direction Accuracy: {test_metrics['direction_accuracy']:.2%}")
    
    # Save results
    results = {
        'test_metrics': test_metrics,
        'val_metrics': checkpoint['val_metrics'],
        'best_epoch': best_epoch,
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
            shocked_nodes=['ECU_AGR'],  # Example: Ecuador agriculture
            shock_magnitude=0.5,  # 50% reduction
            device=args.device
        )
    
    writer.close()


if __name__ == "__main__":
    main()