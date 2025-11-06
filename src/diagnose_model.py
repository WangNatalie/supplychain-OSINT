# diagnose_model.py
import torch
import numpy as np
from training import ShockPropagationGNN, load_and_process_example, create_shock_training_data
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Load model checkpoint and recover training hyperparameters
checkpoint = torch.load("models/best_model.pt", map_location='cpu', weights_only=False)
ckpt_args = checkpoint.get('args', {})

# Load validation data
val_years = [2019]
val_data = create_shock_training_data(val_years, "embeddings", 0.15)

# Reconstruct model
graph = torch.load(val_data[0]['graph_path'], map_location='cpu', weights_only=False)
node_dim = graph.x.shape[1]
edge_dim = graph.edge_attr.shape[1]

model = ShockPropagationGNN(
    node_in_dim=node_dim,
    edge_in_dim=edge_dim,
    hidden_dim=ckpt_args.get('hidden_dim', 128),
    num_layers=ckpt_args.get('num_layers', 3),
    dropout=ckpt_args.get('dropout', 0.3),
    use_attention=ckpt_args.get('use_attention', True)
)

# Map old checkpoint keys to new model architecture
# Old model had edge_mlp.7 as final layer, new has edge_mlp.6
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
    for metadata in val_data:
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

print("="*70)
print("PREDICTION DISTRIBUTION ANALYSIS")
print("="*70)

print(f"\nTarget statistics:")
print(f"  Mean: {all_targets.mean():.4f}")
print(f"  Std:  {all_targets.std():.4f}")
print(f"  Min:  {all_targets.min():.4f}")
print(f"  Max:  {all_targets.max():.4f}")
print(f"  Negative: {(all_targets < 0).sum()} ({100*(all_targets < 0).mean():.1f}%)")
print(f"  Positive: {(all_targets > 0).sum()} ({100*(all_targets > 0).mean():.1f}%)")

print(f"\nPrediction statistics:")
print(f"  Mean: {all_preds.mean():.4f}")
print(f"  Std:  {all_preds.std():.4f}")
print(f"  Min:  {all_preds.min():.4f}")
print(f"  Max:  {all_preds.max():.4f}")
print(f"  Negative: {(all_preds < 0).sum()} ({100*(all_preds < 0).mean():.1f}%)")
print(f"  Positive: {(all_preds > 0).sum()} ({100*(all_preds > 0).mean():.1f}%)")

print(f"\nDirection analysis:")
correct_dir = (np.sign(all_targets) == np.sign(all_preds)).mean()
print(f"  Correct direction: {correct_dir:.2%}")
print(f"  Predicted increases when should decrease: {((all_targets < 0) & (all_preds > 0)).sum()}")
print(f"  Predicted decreases when should increase: {((all_targets > 0) & (all_preds < 0)).sum()}")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 1. Scatter plot
ax = axes[0, 0]
ax.scatter(all_targets, all_preds, alpha=0.1, s=1)
ax.axline((0, 0), slope=1, color='red', linestyle='--', label='Perfect prediction')
ax.axhline(0, color='gray', linestyle='-', alpha=0.3)
ax.axvline(0, color='gray', linestyle='-', alpha=0.3)
ax.set_xlabel('True Δlog(value)')
ax.set_ylabel('Predicted Δlog(value)')
ax.set_title('Predictions vs Truth')
ax.legend()

# 2. Distribution comparison
ax = axes[0, 1]
ax.hist(all_targets, bins=50, alpha=0.5, label='True', density=True)
ax.hist(all_preds, bins=50, alpha=0.5, label='Predicted', density=True)
ax.axvline(0, color='red', linestyle='--')
ax.set_xlabel('Δlog(value)')
ax.set_ylabel('Density')
ax.set_title('Distribution Comparison')
ax.legend()

# 3. Residuals
ax = axes[1, 0]
residuals = all_preds - all_targets
ax.scatter(all_targets, residuals, alpha=0.1, s=1)
ax.axhline(0, color='red', linestyle='--')
ax.set_xlabel('True Δlog(value)')
ax.set_ylabel('Residual (Pred - True)')
ax.set_title('Residual Plot')

# 4. Direction confusion matrix
ax = axes[1, 1]
y_true_sign = (all_targets > 0).astype(int)
y_pred_sign = (all_preds > 0).astype(int)
cm = confusion_matrix(y_true_sign, y_pred_sign)
sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap='Blues')
ax.set_xlabel('Predicted Direction')
ax.set_ylabel('True Direction')
ax.set_title('Direction Confusion Matrix')
ax.set_xticklabels(['Decrease', 'Increase'])
ax.set_yticklabels(['Decrease', 'Increase'])

plt.tight_layout()
plt.savefig('model_diagnosis.png', dpi=200)
print(f"\n✅ Saved diagnosis plot to model_diagnosis.png")