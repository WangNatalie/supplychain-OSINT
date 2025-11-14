#!/usr/bin/env python3
"""
Compare multiple loss function models across different years and economic contexts
"""

import torch
import numpy as np
from training import ShockPropagationGNN, load_and_process_example, create_shock_training_data
import pandas as pd
from pathlib import Path
import argparse
import sys

def evaluate_model_on_year(model_path, year, embeddings_dir="embeddings"):
    """Evaluate a single model on a single year"""
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    ckpt_args = checkpoint.get('args', {})
    
    # Load data for the year
    year_data = create_shock_training_data([year], embeddings_dir, 0.15)
    if not year_data:
        return None
    
    # Reconstruct model
    graph = torch.load(year_data[0]['graph_path'], map_location='cpu', weights_only=False)
    node_dim = graph.x.shape[1]
    edge_dim = graph.edge_attr.shape[1]
    
    model = ShockPropagationGNN(
        node_in_dim=node_dim,
        edge_in_dim=edge_dim,
        hidden_dim=ckpt_args.get('hidden_dim', 64),
        num_layers=ckpt_args.get('num_layers', 2),
        dropout=ckpt_args.get('dropout', 0.3),
        use_attention=ckpt_args.get('use_attention', True)
    )
    
    # Load state dict (handle potential architecture mismatches)
    state_dict = checkpoint['model_state_dict']
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('edge_mlp.7.'):
            new_key = key.replace('edge_mlp.7.', 'edge_mlp.6.')
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
    
    model.load_state_dict(new_state_dict, strict=False)
    model.eval()
    
    # Get predictions
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for metadata in year_data:
            example = load_and_process_example(metadata, 'cpu', edge_sample_ratio=1.0)
            predictions = model(
                example['graph'].x,
                example['graph'].edge_index,
                example['graph'].edge_attr,
                example['shock_mask_nodes'],
                example['shock_mask_edges']
            )
            all_preds.append(predictions.cpu().numpy())
            all_targets.append(example['target'].cpu().numpy())
    
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    
    # Compute metrics
    sign_eps = 1e-4
    
    # Direction accuracy
    pred_sign = np.where(np.abs(all_preds) < sign_eps, 0, np.sign(all_preds))
    target_sign = np.where(np.abs(all_targets) < sign_eps, 0, np.sign(all_targets))
    direction_acc = (pred_sign == target_sign).mean()
    
    # Separate by gains/drops
    gains_mask = all_targets > sign_eps
    drops_mask = all_targets < -sign_eps
    
    gain_dir_acc = (pred_sign[gains_mask] == target_sign[gains_mask]).mean() if gains_mask.any() else 0.0
    drop_dir_acc = (pred_sign[drops_mask] == target_sign[drops_mask]).mean() if drops_mask.any() else 0.0
    
    # Distribution metrics
    target_neg_pct = (all_targets < -sign_eps).mean() * 100
    target_pos_pct = (all_targets > sign_eps).mean() * 100
    pred_neg_pct = (all_preds < -sign_eps).mean() * 100
    pred_pos_pct = (all_preds > sign_eps).mean() * 100
    
    # Error types
    false_positive = ((all_targets < -sign_eps) & (all_preds > sign_eps)).sum()
    false_negative = ((all_targets > sign_eps) & (all_preds < -sign_eps)).sum()
    
    # R² and error metrics
    ss_res = np.sum((all_targets - all_preds) ** 2)
    ss_tot = np.sum((all_targets - all_targets.mean()) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    
    rmse = np.sqrt(np.mean((all_targets - all_preds) ** 2))
    mae = np.mean(np.abs(all_targets - all_preds))
    
    return {
        'year': year,
        'direction_acc': direction_acc,
        'gain_dir_acc': gain_dir_acc,
        'drop_dir_acc': drop_dir_acc,
        'r2': r2,
        'rmse': rmse,
        'mae': mae,
        'target_mean': all_targets.mean(),
        'pred_mean': all_preds.mean(),
        'target_std': all_targets.std(),
        'pred_std': all_preds.std(),
        'target_neg_pct': target_neg_pct,
        'target_pos_pct': target_pos_pct,
        'pred_neg_pct': pred_neg_pct,
        'pred_pos_pct': pred_pos_pct,
        'false_positive': false_positive,
        'false_negative': false_negative,
        'fp_fn_ratio': false_positive / (false_negative + 1e-8),
        'n_samples': len(all_targets)
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models-dir", type=str, default="models/loss_comparison_20251105_151803")
    parser.add_argument("--models-to-compare", type=str, default="focal_model,sign_corrected_model,hybrid_model")
    parser.add_argument("--years", type=str, default="2003,2008,2015,2019,2020,2021")
    args = parser.parse_args()
    # Models to compare
    models = {model: Path(args.models_dir) / f"{model}" / "best_model.pt" for model in args.models_to_compare.split(',')}
    years = [int(year) for year in args.years.split(',')]
    
    print("="*80)
    print("MODEL COMPARISON")
    print("="*80)
    print(f"\nTesting {len(models)} models across {len(years)} years")
    print(f"Models: {', '.join(models.keys())}")
    print(f"Years: {', '.join([f'{y}' for y in years])}")
    print()
    
    # Collect results
    all_results = []
    
    for model_name, model_path in models.items():
        if not model_path.exists():
            print(f"⚠️  Model not found: {model_path}")
            continue
        
        print(f"\n{'='*80}")
        print(f"Evaluating: {model_name.upper()}")
        print(f"{'='*80}")
        
        for year in years:
            print(f"  Year {year}...", end=" ", flush=True)
            
            try:
                result = evaluate_model_on_year(str(model_path), year)
                if result:
                    result['model'] = model_name
                    all_results.append(result)
                    print(f"✓ Dir acc: {result['direction_acc']:.1%}, Drop acc: {result['drop_dir_acc']:.1%}")
                else:
                    print("✗ No data")
            except Exception as e:
                print(f"✗ Error: {e}")
    
    # Create DataFrame
    df = pd.DataFrame(all_results)
    
    # Display summary tables
    print("\n" + "="*80)
    print("SUMMARY: DIRECTION ACCURACY BY YEAR")
    print("="*80)
    pivot_dir = df.pivot(index='year', columns='model', values='direction_acc')
    print(pivot_dir.to_string(float_format=lambda x: f'{x:.1%}'))
    
    print("\n" + "="*80)
    print("SUMMARY: DROP DETECTION ACCURACY BY YEAR")
    print("="*80)
    pivot_drop = df.pivot(index='year', columns='model', values='drop_dir_acc')
    print(pivot_drop.to_string(float_format=lambda x: f'{x:.1%}'))
    
    print("\n" + "="*80)
    print("SUMMARY: GAIN DETECTION ACCURACY BY YEAR")
    print("="*80)
    pivot_gain = df.pivot(index='year', columns='model', values='gain_dir_acc')
    print(pivot_gain.to_string(float_format=lambda x: f'{x:.1%}'))
    
    print("\n" + "="*80)
    print("SUMMARY: PREDICTION BIAS (% PREDICTED DROPS)")
    print("="*80)
    print("\nTrue % drops by year:")
    for year in sorted(years):
        year_data = df[df['year'] == year].iloc[0]
        print(f"  {year}: {year_data['target_neg_pct']:.1f}%")
    
    print("\nPredicted % drops by model:")
    pivot_pred = df.pivot(index='year', columns='model', values='pred_neg_pct')
    print(pivot_pred.to_string(float_format=lambda x: f'{x:.1f}%'))
    
    print("\n" + "="*80)
    print("SUMMARY: ERROR TYPE RATIO (FP:FN)")
    print("="*80)
    print("(Ratio > 1 = more false positives, < 1 = more false negatives)")
    pivot_ratio = df.pivot(index='year', columns='model', values='fp_fn_ratio')
    print(pivot_ratio.to_string(float_format=lambda x: f'{x:.2f}'))
    
    print("\n" + "="*80)
    print("SUMMARY: R² BY YEAR")
    print("="*80)
    pivot_r2 = df.pivot(index='year', columns='model', values='r2')
    print(pivot_r2.to_string(float_format=lambda x: f'{x:.3f}'))
    
    # Calculate average performance across years
    print("\n" + "="*80)
    print("AVERAGE PERFORMANCE ACROSS ALL YEARS")
    print("="*80)
    avg_metrics = df.groupby('model').agg({
        'direction_acc': 'mean',
        'gain_dir_acc': 'mean',
        'drop_dir_acc': 'mean',
        'r2': 'mean',
        'rmse': 'mean',
        'fp_fn_ratio': 'mean'
    })
    print(avg_metrics.to_string(float_format=lambda x: f'{x:.3f}'))
    
    # Best model by metric
    print("\n" + "="*80)
    print("BEST MODEL BY METRIC (averaged across years)")
    print("="*80)
    for metric in ['direction_acc', 'gain_dir_acc', 'drop_dir_acc', 'r2']:
        best = avg_metrics[metric].idxmax()
        value = avg_metrics.loc[best, metric]
        print(f"  {metric:20s}: {best:20s} ({value:.3f})")
    
    # Save results
    output_file = "model_comparison.csv"
    df.to_csv(output_file, index=False)
    print(f"\n✓ Detailed results saved to {output_file}")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    main()

