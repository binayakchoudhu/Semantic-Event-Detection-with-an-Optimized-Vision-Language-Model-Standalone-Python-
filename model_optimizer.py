"""
Model Optimization using Quantization and Pruning.

Implements INT8 dynamic quantization and structured pruning
for CLIP model optimization.
"""

import os
import torch
import torch.nn as nn
from transformers import CLIPModel
from typing import Dict, Tuple, Optional


class ModelOptimizer:
    """Optimizes CLIP models using quantization and pruning."""
    
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        self.model_name = model_name
        self.original_model = None
        self.quantized_model = None
        self.pruned_model = None
        
    def load_model(self) -> CLIPModel:
        """Load the original model."""
        print(f"Loading model: {self.model_name}")
        self.original_model = CLIPModel.from_pretrained(self.model_name)
        self.original_model.eval()
        return self.original_model
    
    def get_model_size(self, model: nn.Module) -> float:
        """Calculate model size in MB."""
        param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
        buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
        return (param_size + buffer_size) / 1024 / 1024
    
    def count_parameters(self, model: nn.Module) -> Tuple[int, int]:
        """Count total and non-zero parameters."""
        total = sum(p.numel() for p in model.parameters())
        nonzero = sum(torch.count_nonzero(p).item() for p in model.parameters())
        return total, nonzero
    
    def quantize_dynamic(self, model: Optional[CLIPModel] = None) -> nn.Module:
        """
        Apply INT8 dynamic quantization.
        
        Quantizes weights to INT8 and computes activations in INT8.
        Note: On Apple Silicon, falls back to float16 precision.
        """
        if model is None:
            if self.original_model is None:
                self.load_model()
            model = self.original_model
        
        print("Applying model optimization...")
        
        model_copy = CLIPModel.from_pretrained(self.model_name)
        model_copy.eval()
        
        # Check if quantization is supported (not on Apple Silicon)
        try:
            self.quantized_model = torch.quantization.quantize_dynamic(
                model_copy,
                {nn.Linear},
                dtype=torch.qint8
            )
            print("INT8 dynamic quantization complete.")
        except RuntimeError as e:
            if "NoQEngine" in str(e):
                print("INT8 quantization not supported on this platform.")
                print("Applying float16 precision instead...")
                model_copy = model_copy.half()  # Convert to float16
                self.quantized_model = model_copy
                print("Float16 optimization complete.")
            else:
                raise e
        
        return self.quantized_model
    
    def prune_model(
        self, 
        model: Optional[CLIPModel] = None,
        amount: float = 0.3
    ) -> CLIPModel:
        """
        Apply L1 magnitude-based pruning.
        
        Args:
            amount: Fraction of parameters to prune (0-1)
        """
        if model is None:
            if self.original_model is None:
                self.load_model()
        
        print(f"Applying pruning (amount={amount})...")
        
        self.pruned_model = CLIPModel.from_pretrained(self.model_name)
        self.pruned_model.eval()
        
        pruned_count = 0
        for name, module in self.pruned_model.named_modules():
            if isinstance(module, nn.Linear):
                # Manual L1 magnitude pruning (handles non-contiguous tensors)
                with torch.no_grad():
                    weight = module.weight.data
                    # Make contiguous and flatten
                    flat_weight = weight.contiguous().view(-1)
                    # Find threshold for pruning
                    k = int(amount * flat_weight.numel())
                    if k > 0:
                        threshold = torch.kthvalue(torch.abs(flat_weight), k).values
                        # Create mask and apply
                        mask = torch.abs(weight) > threshold
                        module.weight.data = weight * mask.float()
                        pruned_count += 1
        
        print(f"Pruning complete. Pruned {pruned_count} layers.")
        return self.pruned_model
    
    def save_optimized_model(
        self,
        model: nn.Module,
        output_path: str,
        model_type: str = "quantized"
    ) -> str:
        """Save the optimized model to disk."""
        os.makedirs(output_path, exist_ok=True)
        
        filename = f"clip_{model_type}.pt"
        filepath = os.path.join(output_path, filename)
        
        torch.save(model.state_dict(), filepath)
        print(f"Model saved to: {filepath}")
        
        return filepath
    
    def get_optimization_stats(self) -> Dict:
        """Get statistics comparing original and optimized models."""
        stats = {}
        
        if self.original_model:
            orig_size = self.get_model_size(self.original_model)
            orig_total, orig_nonzero = self.count_parameters(self.original_model)
            stats["original"] = {
                "size_mb": orig_size,
                "total_params": orig_total,
                "nonzero_params": orig_nonzero,
                "sparsity": 1 - (orig_nonzero / orig_total)
            }
        
        if self.quantized_model:
            quant_size = self.get_model_size(self.quantized_model)
            stats["quantized"] = {
                "size_mb": quant_size,
                "size_reduction": (1 - quant_size / orig_size) * 100 if self.original_model else 0
            }
        
        if self.pruned_model:
            pruned_size = self.get_model_size(self.pruned_model)
            pruned_total, pruned_nonzero = self.count_parameters(self.pruned_model)
            stats["pruned"] = {
                "size_mb": pruned_size,
                "total_params": pruned_total,
                "nonzero_params": pruned_nonzero,
                "sparsity": 1 - (pruned_nonzero / pruned_total),
                "params_removed": orig_total - pruned_nonzero if self.original_model else 0
            }
        
        return stats


def print_optimization_stats(stats: Dict):
    """Pretty print optimization statistics."""
    print("\n" + "="*60)
    print("MODEL OPTIMIZATION STATISTICS")
    print("="*60)
    
    if "original" in stats:
        orig = stats["original"]
        print(f"\nOriginal Model:")
        print(f"  Size: {orig['size_mb']:.2f} MB")
        print(f"  Parameters: {orig['total_params']:,}")
        print(f"  Sparsity: {orig['sparsity']*100:.2f}%")
    
    if "quantized" in stats:
        quant = stats["quantized"]
        print(f"\nQuantized Model (INT8):")
        print(f"  Size: {quant['size_mb']:.2f} MB")
        print(f"  Size Reduction: {quant['size_reduction']:.1f}%")
    
    if "pruned" in stats:
        pruned = stats["pruned"]
        print(f"\nPruned Model:")
        print(f"  Size: {pruned['size_mb']:.2f} MB")
        print(f"  Non-zero Parameters: {pruned['nonzero_params']:,}")
        print(f"  Sparsity: {pruned['sparsity']*100:.2f}%")
