#!/usr/bin/env python3
"""
loss_functions.py - Custom loss functions for shock propagation model

Usage:
    from loss_functions import get_loss_function
    
    loss_fn = get_loss_function('sign_corrected', alpha=1.0)
    loss = loss_fn(predictions, targets)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MSELoss(nn.Module):
    """
    Standard Mean Squared Error loss.
    
    Simple baseline - treats all errors equally.
    Good for: Well-behaved data without outliers
    """
    
    def __init__(self):
        super().__init__()
        self.name = "MSE"
    
    def forward(self, pred, target):
        """
        Args:
            pred: Predicted Δlog(value) [num_edges]
            target: True Δlog(value) [num_edges]
        """
        return F.mse_loss(pred, target)


class HuberLoss(nn.Module):
    """
    Huber loss - robust to outliers.
    
    Behaves like MSE for small errors, L1 for large errors.
    Good for: Data with occasional extreme outliers
    
    Args:
        delta: Threshold between L2 and L1 behavior (default: 0.1)
               Smaller delta = more robust to outliers
    """
    
    def __init__(self, delta=0.1):
        super().__init__()
        self.delta = delta
        self.name = f"Huber(δ={delta})"
    
    def forward(self, pred, target):
        return F.huber_loss(pred, target, delta=self.delta)


class SignCorrectedMSELoss(nn.Module):
    """
    MSE with exponential penalty for wrong direction predictions.
    
    Loss = MSE * exp(α * (1 - sign_match))
    
    Where:
        sign_match = 1 if signs match, 0 otherwise
        α = strength of direction penalty
    
    Effect:
        - Correct direction: loss = MSE * e^0 = MSE
        - Wrong direction: loss = MSE * e^α
    
    Args:
        alpha: Direction penalty strength
               0.5 = 1.6x penalty for wrong direction
               1.0 = 2.7x penalty (recommended)
               2.0 = 7.4x penalty
               10.0 = 22,026x penalty (too aggressive!)
    
    Good for: When direction matters more than magnitude
    """
    
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha
        self.name = f"SignCorrectedMSE(α={alpha})"
    
    def forward(self, pred, target):
        # Base MSE
        mse = (pred - target) ** 2
        
        # Sign correction: exp(α) if wrong, exp(0)=1 if correct
        pred_sign = torch.sign(pred)
        target_sign = torch.sign(target)
        sign_match = (pred_sign == target_sign).float()
        
        # Exponential penalty for wrong direction
        correction = torch.exp(self.alpha * (1.0 - sign_match))
        
        return (mse * correction).mean()


class WeightedSignCorrectedMSELoss(nn.Module):
    """
    Sign-corrected MSE with edge value weighting.
    
    Focuses model on economically important edges (large trade flows).
    
    Loss = MSE * sign_correction * value_weight
    
    Args:
        alpha: Direction penalty strength (default: 1.0)
        weight_power: Power for value weighting (default: 0.5)
                     0.0 = no weighting (all edges equal)
                     0.5 = sqrt weighting (recommended)
                     1.0 = linear weighting (emphasizes large edges heavily)
    
    Good for: When you care more about large, important trade flows
    """
    
    def __init__(self, alpha=1.0, weight_power=0.5):
        super().__init__()
        self.alpha = alpha
        self.weight_power = weight_power
        self.name = f"WeightedSignMSE(α={alpha}, p={weight_power})"
    
    def forward(self, pred, target, value_t=None):
        """
        Args:
            pred: Predicted Δlog(value)
            target: True Δlog(value)
            value_t: Current edge values (optional, for weighting)
        """
        # Base MSE
        mse = (pred - target) ** 2
        
        # Sign correction
        sign_match = (torch.sign(pred) == torch.sign(target)).float()
        sign_correction = torch.exp(self.alpha * (1.0 - sign_match))
        
        # Value weighting (if provided)
        if value_t is not None:
            # Convert to same device as pred
            if isinstance(value_t, torch.Tensor):
                value_t = value_t.to(pred.device)
            else:
                value_t = torch.tensor(value_t, device=pred.device)
            
            # Compute weights: larger edges get more weight
            weights = torch.pow(value_t + 1, self.weight_power)
            # Normalize so mean weight = 1
            weights = weights / weights.mean()
            # Clip to prevent extreme weights
            weights = torch.clamp(weights, 0.1, 10.0)
        else:
            weights = 1.0
        
        return (mse * sign_correction * weights).mean()


class FocalRegressionLoss(nn.Module):
    """
    Focal loss adapted for regression - focuses on hard examples.
    
    Loss = |error|^γ * |error|
    
    Where γ controls focus:
        γ = 0: Equal weighting (like MSE)
        γ = 1: Focus more on large errors (recommended)
        γ = 2: Strong focus on large errors
    
    Args:
        gamma: Focusing parameter (default: 1.0)
        direction_boost: Extra penalty for wrong direction (default: 1.5)
    
    Good for: When you want model to focus on hard-to-predict edges
    """
    
    def __init__(self, gamma=1.0, direction_boost=1.5):
        super().__init__()
        self.gamma = gamma
        self.direction_boost = direction_boost
        self.name = f"Focal(γ={gamma}, dir={direction_boost})"
    
    def forward(self, pred, target):
        # Absolute error
        abs_error = torch.abs(pred - target)
        
        # Focal weighting: harder examples (larger errors) get more weight
        focal_weight = torch.pow(abs_error, self.gamma)
        
        # Direction penalty
        wrong_sign = (torch.sign(pred) != torch.sign(target)).float()
        direction_weight = 1.0 + wrong_sign * (self.direction_boost - 1.0)
        
        # Combined loss
        loss = abs_error * focal_weight * direction_weight
        
        return loss.mean()


class HybridSignFocalLoss(nn.Module):
    """
    Combines sign correction + focal weighting for best of both worlds.
    
    Loss = MSE * focal_weight * sign_correction
    
    Args:
        alpha: Sign correction strength (default: 1.0)
        gamma: Focal parameter (default: 1.5)
    
    Good for: Balanced approach - handles direction AND magnitude well
    """
    
    def __init__(self, alpha=1.0, gamma=1.5):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.name = f"HybridSignFocal(α={alpha}, γ={gamma})"
    
    def forward(self, pred, target):
        # Squared error
        squared_error = (pred - target) ** 2
        
        # Focal weighting (focus on large errors)
        abs_error = torch.sqrt(squared_error + 1e-8)
        focal_weight = torch.pow(abs_error, self.gamma - 1)
        
        # Sign correction (penalize wrong directions)
        sign_match = (torch.sign(pred) == torch.sign(target)).float()
        sign_weight = torch.exp(self.alpha * (1.0 - sign_match))
        
        return (squared_error * focal_weight * sign_weight).mean()


class AsymmetricLoss(nn.Module):
    """
    Asymmetric loss - penalize drops more than gains.
    
    Useful for supply chain risk where disruptions matter more than growth.
    
    Loss = MSE * weight
    where weight = drop_weight if predicting/actual is drop, else 1.0
    
    Args:
        drop_weight: Extra penalty for drop predictions (default: 2.0)
                    2.0 = drops are 2x more important than gains
    
    Good for: When you care more about predicting disruptions correctly
    """
    
    def __init__(self, drop_weight=2.0):
        super().__init__()
        self.drop_weight = drop_weight
        self.name = f"Asymmetric(drop={drop_weight}x)"
    
    def forward(self, pred, target):
        # Base MSE
        mse = (pred - target) ** 2
        
        # Extra weight for drops (either predicted or actual)
        is_drop = ((pred < 0) | (target < 0)).float()
        weights = 1.0 + is_drop * (self.drop_weight - 1.0)
        
        return (mse * weights).mean()


# ============================================================================
# HELPER FUNCTION FOR EASY USE
# ============================================================================

def get_loss_function(loss_type='mse', **kwargs):
    """
    Factory function to get loss by name.
    
    Usage:
        loss_fn = get_loss_function('sign_corrected', alpha=1.0)
        loss_fn = get_loss_function('huber', delta=0.1)
        loss_fn = get_loss_function('focal', gamma=2.0)
    
    Available losses:
        'mse': Standard MSE
        'huber': Huber loss (robust to outliers)
        'sign_corrected': MSE with direction penalty
        'weighted_sign': Sign-corrected with value weighting
        'focal': Focal regression loss
        'hybrid': Sign correction + focal weighting
        'asymmetric': Extra penalty for drops
    
    Args:
        loss_type: Name of loss function
        **kwargs: Parameters for the loss function
    
    Returns:
        Loss function instance
    """
    
    loss_registry = {
        'mse': MSELoss,
        'huber': HuberLoss,
        'sign_corrected': SignCorrectedMSELoss,
        'weighted_sign': WeightedSignCorrectedMSELoss,
        'focal': FocalRegressionLoss,
        'hybrid': HybridSignFocalLoss,
        'asymmetric': AsymmetricLoss,
    }
    
    if loss_type not in loss_registry:
        available = ', '.join(loss_registry.keys())
        raise ValueError(f"Unknown loss type '{loss_type}'. Available: {available}")
    
    loss_class = loss_registry[loss_type]
    loss_fn = loss_class(**kwargs)
    
    print(f"Using loss function: {loss_fn.name}")
    return loss_fn


# ============================================================================
# COMPARISON UTILITY
# ============================================================================

def compare_losses(pred, target, value_t=None):
    """
    Compare how different losses behave on the same predictions.
    
    Usage:
        from loss_functions import compare_losses
        
        # After making predictions
        compare_losses(predictions, targets, values_t)
    """
    
    losses = {
        'MSE': MSELoss(),
        'Huber(0.1)': HuberLoss(delta=0.1),
        'SignMSE(α=0.5)': SignCorrectedMSELoss(alpha=0.5),
        'SignMSE(α=1.0)': SignCorrectedMSELoss(alpha=1.0),
        'SignMSE(α=2.0)': SignCorrectedMSELoss(alpha=2.0),
        'Focal(γ=1.0)': FocalRegressionLoss(gamma=1.0),
        'Hybrid': HybridSignFocalLoss(),
        'Asymmetric': AsymmetricLoss(),
    }
    
    print("\n" + "="*60)
    print("LOSS FUNCTION COMPARISON")
    print("="*60)
    
    # Direction accuracy
    direction_acc = (torch.sign(pred) == torch.sign(target)).float().mean()
    print(f"Direction Accuracy: {direction_acc:.2%}")
    print(f"Mean Absolute Error: {torch.abs(pred - target).mean():.4f}")
    print()
    
    print(f"{'Loss Function':<25} {'Value':<15} {'Relative'}")
    print("-"*60)
    
    base_loss = None
    for name, loss_fn in losses.items():
        try:
            if 'Weighted' in name and value_t is not None:
                loss_val = loss_fn(pred, target, value_t).item()
            else:
                loss_val = loss_fn(pred, target).item()
            
            if base_loss is None:
                base_loss = loss_val
                relative = 1.0
            else:
                relative = loss_val / base_loss
            
            print(f"{name:<25} {loss_val:<15.6f} {relative:.2f}x")
        except Exception as e:
            print(f"{name:<25} ERROR: {str(e)}")
    
    print("="*60)


if __name__ == "__main__":
    """
    Test/demo script showing how each loss behaves
    """
    
    print("Testing loss functions...\n")
    
    # Create test data
    torch.manual_seed(42)
    pred = torch.randn(1000) * 0.1
    target = torch.randn(1000) * 0.1
    value_t = torch.rand(1000) * 100 + 1  # Edge values from $1 to $101
    
    # Introduce some wrong directions
    wrong_idx = torch.randperm(1000)[:300]
    pred[wrong_idx] = -pred[wrong_idx]
    
    print(f"Test data:")
    print(f"  Samples: {len(pred)}")
    print(f"  Direction accuracy: {(torch.sign(pred) == torch.sign(target)).float().mean():.2%}")
    print(f"  Mean |error|: {torch.abs(pred - target).mean():.4f}")
    
    # Compare all losses
    compare_losses(pred, target, value_t)
