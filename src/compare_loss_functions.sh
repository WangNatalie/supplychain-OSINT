#!/bin/bash
# compare_loss_functions.sh - Train models with different loss functions and compare results

set -e  # Exit on error

# Configuration
EPOCHS=20
TRAIN_START=1996
TRAIN_END=2019
VAL_START=2020
VAL_END=2020
TEST_START=2021
TEST_END=2021
HIDDEN_DIM=64
NUM_LAYERS=2
DROPOUT=0.1
LR=0.002
WEIGHT_DECAY=0.0001
PATIENCE=15
TRAIN_EDGE_SAMPLE_RATIO=0.2
EMBEDDINGS_DIR="embeddings"
SIGN_EPS=0.001

# Output directory for comparison
COMPARISON_DIR="models/loss_comparison_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$COMPARISON_DIR"

echo "========================================================================"
echo "LOSS FUNCTION COMPARISON EXPERIMENT"
echo "========================================================================"
echo "Training ${EPOCHS} epochs per loss function"
echo "Results will be saved to: $COMPARISON_DIR"
echo "========================================================================"
echo ""

# Loss functions to test
declare -a LOSS_FUNCTIONS=(
    "mse"
    "huber"
    "sign_corrected"
    "focal"
    "hybrid"
)

# Train with each loss function
for LOSS in "${LOSS_FUNCTIONS[@]}"; do
    echo ""
    echo "========================================================================"
    echo "Training with loss function: $LOSS"
    echo "========================================================================"
    
    MODEL_DIR="$COMPARISON_DIR/${LOSS}_model"
    mkdir -p "$MODEL_DIR"
    
    # Set loss-specific parameters
    case $LOSS in
        "huber")
            EXTRA_ARGS="--loss-delta 0.1"
            ;;
        "sign_corrected")
            EXTRA_ARGS="--loss-alpha 1.0"
            ;;
        "focal")
            EXTRA_ARGS="--loss-gamma 1.5"
            ;;
        "hybrid")
            EXTRA_ARGS="--loss-alpha 1.0 --loss-gamma 1.5"
            ;;
        *)
            EXTRA_ARGS=""
            ;;
    esac
    
    # Train the model
    python training.py \
        --train-start $TRAIN_START \
        --train-end $TRAIN_END \
        --val-start $VAL_START \
        --val-end $VAL_END \
        --test-start $TEST_START \
        --test-end $TEST_END \
        --shock-threshold 0.15 \
        --hidden-dim $HIDDEN_DIM \
        --num-layers $NUM_LAYERS \
        --dropout $DROPOUT \
        --use-attention \
        --loss $LOSS \
        $EXTRA_ARGS \
        --epochs $EPOCHS \
        --lr $LR \
        --weight-decay $WEIGHT_DECAY \
        --patience $PATIENCE \
        --train-edge-sample-ratio $TRAIN_EDGE_SAMPLE_RATIO \
        --device cpu \
        --embeddings-dir $EMBEDDINGS_DIR \
        --save-dir "$MODEL_DIR" \
        --best-metric r2 \
        --sign-eps $SIGN_EPS \
        2>&1 | tee "$MODEL_DIR/training.log"
    
    echo "✓ Completed training with $LOSS"
    echo "  Results saved to: $MODEL_DIR/results.json"
done

echo ""
echo "========================================================================"
echo "GENERATING COMPARISON REPORT"
echo "========================================================================"

# Create Python script to parse and compare results
cat > "$COMPARISON_DIR/compare_results.py" << 'EOF'
#!/usr/bin/env python3
import json
import sys
from pathlib import Path
import pandas as pd

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
    
    print("\n" + "="*80)

if __name__ == "__main__":
    main()
EOF

chmod +x "$COMPARISON_DIR/compare_results.py"

# Run comparison
python "$COMPARISON_DIR/compare_results.py" "$COMPARISON_DIR"

echo ""
echo "========================================================================"
echo "EXPERIMENT COMPLETE"
echo "========================================================================"
echo "All results saved to: $COMPARISON_DIR"
echo "  - Individual model directories: ${COMPARISON_DIR}/*_model/"
echo "  - Comparison CSV: ${COMPARISON_DIR}/comparison_results.csv"
echo "  - Training logs: ${COMPARISON_DIR}/*_model/training.log"
echo "========================================================================"

