#!/usr/bin/env python3
"""
train_leontief_hybrid.py - Train ML Model to Adjust Leontief Predictions

Trains a machine learning model to predict adjustment factors for Leontief 
input-output predictions. The model learns "Leontief overpredicts by X%" based 
on edge features, shock characteristics, and historical observations.

Architecture:
    1. Leontief gives baseline propagation (structural)
    2. ML predicts adjustment factor (0-1 scale)
    3. Final prediction = leontief_baseline * adjustment_factor

Usage:
    # Train on historical shocks
    python train_leontief_hybrid.py \
        --shock-data shocks/historical_shocks.json \
        --icio-dir embeddings/ \
        --output-dir models/leontief_hybrid \
        --epochs 100

    # Test on new shock
    python predict_leontief_hybrid.py \
        --model models/leontief_hybrid/best_model.pt \
        --graph embeddings/graph_2021_labeled.pt \
        --shocked-nodes USA_C26 --magnitude 0.20
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import json
from pathlib import Path
from typing import List, Dict, Tuple
import argparse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle

from leontief_propagation import LeontiefPropagator


class AdjustmentMLModel(nn.Module):
    """
    Neural network to predict Leontief adjustment factors.
    
    Input features:
        - Leontief baseline prediction
        - Edge features (value, centrality, etc.)
        - Shock features (magnitude, sector, geography)
    
    Output:
        - Adjustment factor (0-1 scale, where 1 = Leontief is perfect)
    """
    
    def __init__(self, num_features: int, hidden_dim: int = 64, dropout: float = 0.2):
        super().__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()  # Output between 0 and 1
        )
    
    def forward(self, x):
        """
        Args:
            x: Feature tensor [batch_size, num_features]
        
        Returns:
            adjustment_factor: Tensor [batch_size] with values in [0, 1]
        """
        return self.mlp(x).squeeze(-1)


class ShockDataset:
    """
    Manages training data from historical shock observations.
    """
    
    def __init__(self, shock_data_path: str, icio_dir: str):
        """
        Args:
            shock_data_path: Path to JSON file with historical shocks
            icio_dir: Directory containing ICIO graph files
        """
        self.shock_data_path = shock_data_path
        self.icio_dir = Path(icio_dir)
        
        # Load shock events
        print(f"Loading shock data from {shock_data_path}...")
        with open(shock_data_path, 'r') as f:
            self.shock_events = json.load(f)
        
        print(f"✓ Loaded {len(self.shock_events)} shock events")
        
        # Cache for Leontief propagators (one per year)
        self.propagators = {}
    
    def get_propagator(self, year: int) -> LeontiefPropagator:
        """Get or create Leontief propagator for given year."""
        if year not in self.propagators:
            graph_path = self.icio_dir / f"graph_{year}_labeled.pt"
            if not graph_path.exists():
                raise FileNotFoundError(f"Graph not found: {graph_path}")
            
            print(f"Loading graph for {year}...")
            graph = torch.load(graph_path, map_location='cpu', weights_only=False)
            self.propagators[year] = LeontiefPropagator(graph)
        
        return self.propagators[year]
    
    def extract_training_instances(self) -> pd.DataFrame:
        """
        Extract training instances from historical shocks.
        
        Returns:
            DataFrame with columns:
                - leontief_prediction
                - actual_outcome
                - adjustment_factor (target)
                - edge_value
                - shock_magnitude
                - ... other features
        """
        training_instances = []
        
        for shock_event in self.shock_events:
            event_name = shock_event.get('name', 'Unknown')
            year = shock_event['year']
            shocked_nodes = shock_event['shocked_nodes']  # Dict: {node_id: magnitude}
            observed_impacts = shock_event['observed_impacts']  # List of edge observations
            
            print(f"\nProcessing: {event_name} ({year})")
            print(f"  Shocked nodes: {len(shocked_nodes)}")
            print(f"  Observed impacts: {len(observed_impacts)}")
            
            # Get Leontief propagator for this year
            propagator = self.get_propagator(year)
            
            # Compute Leontief predictions
            node_impacts = propagator.propagate_shock(shocked_nodes)
            edge_impacts = propagator.compute_edge_impacts(node_impacts)
            
            # Match observations to Leontief predictions
            for obs in observed_impacts:
                edge_key = f"{obs['source']}->{obs['target']}"
                
                # Find matching edge in Leontief results
                matching_edges = edge_impacts[
                    (edge_impacts['source'] == obs['source']) & 
                    (edge_impacts['target'] == obs['target'])
                ]
                
                if len(matching_edges) == 0:
                    print(f"  Warning: Edge {edge_key} not found in graph")
                    continue
                
                edge_data = matching_edges.iloc[0]
                
                # Compute actual outcome
                baseline = obs.get('baseline_value', edge_data['baseline_value'])
                actual = obs['actual_value']
                actual_pct_change = (actual - baseline) / baseline if baseline > 0 else 0
                
                # Leontief prediction (already computed)
                leontief_pct_change = edge_data['impact_pct'] / 100
                
                # Adjustment factor: actual / leontief
                if abs(leontief_pct_change) > 1e-6:
                    adjustment_factor = actual_pct_change / leontief_pct_change
                    # Clip to reasonable range [0, 2]
                    adjustment_factor = np.clip(adjustment_factor, 0, 2)
                else:
                    # Leontief predicted no impact
                    adjustment_factor = 1.0
                
                # Extract features
                instance = {
                    # Target
                    'adjustment_factor': adjustment_factor,
                    
                    # Predictions
                    'leontief_prediction': leontief_pct_change,
                    'actual_outcome': actual_pct_change,
                    
                    # Edge features
                    'edge_value': edge_data['baseline_value'],
                    'edge_value_log': np.log1p(edge_data['baseline_value']),
                    
                    # Shock features
                    'shock_magnitude': np.mean([abs(m) for m in shocked_nodes.values()]),
                    'num_shocked_nodes': len(shocked_nodes),
                    
                    # Node impacts
                    'source_output_impact': edge_data['source_output_impact'] / 100,
                    'target_output_impact': edge_data['target_output_impact'] / 100,
                    
                    # Metadata
                    'event_name': event_name,
                    'year': year,
                    'edge': edge_key
                }
                
                # Add custom features from observation if provided
                for key in ['geographic_distance', 'supplier_diversity', 
                           'substitutability', 'inventory_months']:
                    if key in obs:
                        instance[key] = obs[key]
                
                training_instances.append(instance)
        
        df = pd.DataFrame(training_instances)
        print(f"\n✓ Extracted {len(df)} training instances")
        
        return df
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Prepare feature matrix and target vector.
        
        Returns:
            (X, y, feature_names)
        """
        # Define feature columns (exclude target and metadata)
        exclude_cols = ['adjustment_factor', 'actual_outcome', 'event_name', 'year', 'edge']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Handle missing values
        df_features = df[feature_cols].fillna(0)
        
        X = df_features.values.astype(np.float32)
        y = df['adjustment_factor'].values.astype(np.float32)
        
        print(f"\nFeature matrix shape: {X.shape}")
        print(f"Features: {feature_cols}")
        
        return X, y, feature_cols


def train_hybrid_model(shock_data_path: str, icio_dir: str, output_dir: str,
                      hidden_dim: int = 64, dropout: float = 0.2,
                      epochs: int = 100, batch_size: int = 32,
                      learning_rate: float = 0.001, val_split: float = 0.2):
    """
    Train Leontief-ML hybrid model.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load and prepare data
    dataset = ShockDataset(shock_data_path, icio_dir)
    df = dataset.extract_training_instances()
    
    if len(df) < 10:
        raise ValueError(f"Insufficient training data: {len(df)} instances (need at least 10)")
    
    # Prepare features
    X, y, feature_names = dataset.prepare_features(df)
    
    # Save feature names
    with open(output_dir / 'feature_names.json', 'w') as f:
        json.dump(feature_names, f, indent=2)
    
    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Save scaler
    with open(output_dir / 'scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    # Train/val split
    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y, test_size=val_split, random_state=42
    )
    
    print(f"\nTrain set: {len(X_train)} instances")
    print(f"Val set: {len(X_val)} instances")
    
    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train)
    X_val_t = torch.FloatTensor(X_val)
    y_val_t = torch.FloatTensor(y_val)
    
    # Create dataloaders
    train_dataset = torch.utils.data.TensorDataset(X_train_t, y_train_t)
    val_dataset = torch.utils.data.TensorDataset(X_val_t, y_val_t)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False
    )
    
    # Initialize model
    num_features = X_train.shape[1]
    model = AdjustmentMLModel(num_features, hidden_dim, dropout)
    
    print(f"\nModel architecture:")
    print(model)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    # Training loop
    best_val_loss = float('inf')
    patience = 20
    patience_counter = 0
    
    print(f"\n{'='*80}")
    print("TRAINING")
    print(f"{'='*80}\n")
    
    for epoch in range(epochs):
        # Train
        model.train()
        train_losses = []
        
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
        
        # Validate
        model.eval()
        val_losses = []
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                y_pred = model(X_batch)
                loss = criterion(y_pred, y_batch)
                val_losses.append(loss.item())
        
        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        
        # Print progress
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch:3d}/{epochs}: Train Loss={train_loss:.6f}, Val Loss={val_loss:.6f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'feature_names': feature_names,
                'num_features': num_features,
                'hidden_dim': hidden_dim,
                'dropout': dropout
            }
            torch.save(checkpoint, output_dir / 'best_model.pt')
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch}")
            break
    
    print(f"\n{'='*80}")
    print(f"TRAINING COMPLETE")
    print(f"{'='*80}")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Model saved to: {output_dir / 'best_model.pt'}")
    
    # Evaluate on validation set
    model.load_state_dict(torch.load(output_dir / 'best_model.pt')['model_state_dict'])
    model.eval()
    
    with torch.no_grad():
        y_val_pred = model(X_val_t).numpy()
    
    # Compute metrics
    mae = np.mean(np.abs(y_val_pred - y_val))
    rmse = np.sqrt(np.mean((y_val_pred - y_val)**2))
    
    print(f"\nValidation Metrics:")
    print(f"  MAE:  {mae:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    
    # Save metrics
    metrics = {
        'val_loss': float(best_val_loss),
        'mae': float(mae),
        'rmse': float(rmse),
        'num_train': len(X_train),
        'num_val': len(X_val)
    }
    
    with open(output_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n✓ Training artifacts saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Train Leontief-ML hybrid model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument("--shock-data", required=True,
                       help="Path to historical shock data JSON")
    parser.add_argument("--icio-dir", required=True,
                       help="Directory containing ICIO graph files")
    parser.add_argument("--output-dir", default="models/leontief_hybrid",
                       help="Output directory for trained model")
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--val-split", type=float, default=0.2)
    
    args = parser.parse_args()
    
    train_hybrid_model(
        shock_data_path=args.shock_data,
        icio_dir=args.icio_dir,
        output_dir=args.output_dir,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        val_split=args.val_split
    )


if __name__ == "__main__":
    main()

