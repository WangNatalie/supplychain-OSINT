#!/usr/bin/env python3
import json
import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def load_results(comparison_dir):
    """Load all results.json files from the comparison directory"""
    results = {}
    comparison_path = Path(comparison_dir)
    
    for model_dir in comparison_path.glob("*_model"):
        loss_name = model_dir.name.replace("_model", "")
        results_file = model_dir / "results.json"
        
        if results_file.exists():
            with open(results_file) as f:
                data = json.load(f)
                results[loss_name] = data
        else:
            print(f"Warning: No results.json found for {loss_name}")
    
    return results

def create_comparison_table(results):
    """Create comparison table from results"""
    
    # Metrics to compare
    metrics = [
        ('test_metrics', 'direction_acc', 'Direction Accuracy', True),
        ('test_metrics', 'gain_dir_acc', 'Gain Dir Acc', True),
        ('test_metrics', 'drop_dir_acc', 'Drop Dir Acc', True),
        ('test_metrics', 'drop_f1', 'Drop F1', True),
        ('test_metrics', 'r2', 'R²', True),
        ('test_metrics', 'rmse', 'RMSE', False),
        ('test_metrics', 'mae', 'MAE', False),
        ('test_metrics', 'smape', 'SMAPE', False),
        ('test_metrics', 'target_neg_frac', 'True Drops %', None),
        ('test_metrics', 'pred_neg_frac', 'Pred Drops %', None),
    ]
    
    rows = []
    for loss_name, data in sorted(results.items()):
        row = {'Loss Function': loss_name}
        
        for section, metric, display_name, higher_better in metrics:
            value = data.get(section, {}).get(metric, float('nan'))
            
            # Format percentage metrics
            if 'acc' in metric.lower() or 'frac' in metric.lower():
                row[display_name] = f"{value:.2%}" if value == value else "N/A"
            elif metric in ['r2', 'drop_f1']:
                row[display_name] = f"{value:.4f}" if value == value else "N/A"
            elif metric in ['rmse', 'mae']:
                row[display_name] = f"{value:.4f}" if value == value else "N/A"
            elif metric == 'smape':
                row[display_name] = f"{value:.1f}%" if value == value else "N/A"
            else:
                row[display_name] = f"{value:.4f}" if value == value else "N/A"
        
        # Add best epoch
        row['Best Epoch'] = data.get('best_epoch', 'N/A')
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    return df

def find_best_models(results):
    """Identify best model for each metric"""
    best_models = {}
    
    key_metrics = {
        'direction_acc': ('Direction Accuracy', True),
        'drop_dir_acc': ('Drop Direction Accuracy', True),
        'drop_f1': ('Drop F1 Score', True),
        'r2': ('R² Score', True),
        'rmse': ('RMSE', False),
        'smape': ('SMAPE', False),
    }
    
    for metric, (display_name, higher_better) in key_metrics.items():
        values = {}
        for loss_name, data in results.items():
            value = data.get('test_metrics', {}).get(metric, float('nan'))
            if value == value:  # Check for NaN
                values[loss_name] = value
        
        if values:
            if higher_better:
                best_loss = max(values, key=values.get)
                best_value = values[best_loss]
            else:
                best_loss = min(values, key=values.get)
                best_value = values[best_loss]
            
            best_models[display_name] = (best_loss, best_value)
    
    return best_models

def create_comparison_plot(results, comparison_dir):
    """Create bar plot comparing key metrics across loss functions"""
    
    # Metrics to plot
    metrics = ['gain_dir_acc', 'drop_dir_acc', 'r2']
    metric_labels = ['Gain Direction Acc', 'Drop Direction Acc', 'R² Score']
    
    # Extract data
    loss_names = []
    gain_acc = []
    drop_acc = []
    r2_scores = []
    
    for loss_name, data in sorted(results.items()):
        test_metrics = data.get('test_metrics', {})
        loss_names.append(loss_name)
        gain_acc.append(test_metrics.get('gain_dir_acc', 0))
        drop_acc.append(test_metrics.get('drop_dir_acc', 0))
        r2_scores.append(test_metrics.get('r2', 0))
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(loss_names))
    width = 0.25
    
    # Create bars
    bars1 = ax.bar(x - width, gain_acc, width, label='Gain Dir Acc', alpha=0.8, color='#2ecc71')
    bars2 = ax.bar(x, drop_acc, width, label='Drop Dir Acc', alpha=0.8, color='#e74c3c')
    bars3 = ax.bar(x + width, r2_scores, width, label='R² Score', alpha=0.8, color='#3498db')
    
    # Customize plot
    ax.set_xlabel('Loss Function', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Loss Function Comparison: Key Metrics', fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(loss_names, rotation=45, ha='right')
    ax.legend(loc='lower left', fontsize=10)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0, 1.05)
    
    # Add value labels on bars
    def autolabel(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),  # 3 points vertical offset
                       textcoords="offset points",
                       ha='center', va='bottom',
                       fontsize=8, rotation=0)
    
    autolabel(bars1)
    autolabel(bars2)
    autolabel(bars3)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = Path(comparison_dir) / "loss_comparison_plot.png"
    plt.savefig(plot_path, dpi=200, bbox_inches='tight')
    print(f"✓ Plot saved to: {plot_path}")
    
    return plot_path

def main():
    if len(sys.argv) < 2:
        print("Usage: python compare_results.py <comparison_dir>")
        sys.exit(1)
    
    comparison_dir = sys.argv[1]
    
    print("\n" + "="*80)
    print("LOSS FUNCTION COMPARISON RESULTS")
    print("="*80 + "\n")
    
    # Load results
    results = load_results(comparison_dir)
    
    if not results:
        print("Error: No results found!")
        sys.exit(1)
    
    print(f"Found results for {len(results)} loss functions:\n")
    
    # Create comparison table
    df = create_comparison_table(results)
    
    # Display full table
    print("FULL COMPARISON TABLE:")
    print("-" * 80)
    print(df.to_string(index=False))
    print()
    
    # Find best models
    print("\n" + "="*80)
    print("BEST MODEL FOR EACH METRIC:")
    print("="*80 + "\n")
    
    best_models = find_best_models(results)
    for metric, (loss_name, value) in best_models.items():
        if 'acc' in metric.lower() or '%' in metric:
            print(f"  {metric:<30} {loss_name:<20} ({value:.2%})")
        elif 'F1' in metric or 'R²' in metric:
            print(f"  {metric:<30} {loss_name:<20} ({value:.4f})")
        elif 'RMSE' in metric or 'MAE' in metric:
            print(f"  {metric:<30} {loss_name:<20} ({value:.4f})")
        else:
            print(f"  {metric:<30} {loss_name:<20} ({value:.2f})")
    
    # Save to CSV
    csv_path = Path(comparison_dir) / "comparison_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n✓ Results saved to: {csv_path}")
    
    # Create comparison plot
    print("\n" + "="*80)
    print("CREATING COMPARISON PLOT")
    print("="*80 + "\n")
    create_comparison_plot(results, comparison_dir)
    
    print("\n" + "="*80)

if __name__ == "__main__":
    main()
