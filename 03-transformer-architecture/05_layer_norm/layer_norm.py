"""
Step 5: Layer Normalization

Layer normalization stabilizes training by normalizing activations.
Without it, deep transformers fail to train — gradients explode or vanish.

The formula:
    LayerNorm(x) = γ × (x - μ) / √(σ² + ε) + β
    
Where:
    μ = mean of x (across the last dimension)
    σ² = variance of x (across the last dimension)
    γ = learnable scale parameter (initialized to 1)
    β = learnable shift parameter (initialized to 0)
    ε = small constant for numerical stability (typically 1e-5)

Key insight: We normalize across FEATURES, not across batch.
    - BatchNorm: normalize across batch dimension (bad for variable-length sequences)
    - LayerNorm: normalize across feature dimension (good for transformers)

Two placements:
    - Post-LN (original): x + Sublayer(LayerNorm(x))  
    - Pre-LN (modern):    LayerNorm(x + Sublayer(x))
    
Pre-LN is more stable for deep models and is used in GPT-2, LLaMA, etc.
"""

import torch
import torch.nn as nn
import math


class LayerNorm(nn.Module):
    """
    Layer Normalization — normalizes across the feature dimension.
    
    For input (batch, seq_len, d_model), normalizes each (d_model,) vector
    independently to have mean=0 and variance=1, then applies learnable
    scale (γ) and shift (β).
    
    Args:
        d_model: Feature dimension to normalize over
        eps: Small constant for numerical stability
    """
    
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        
        self.d_model = d_model
        self.eps = eps
        
        # Learnable parameters
        # γ (gamma): scale, initialized to 1
        # β (beta): shift, initialized to 0
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply layer normalization.
        
        Args:
            x: Input tensor of shape (..., d_model)
               Typically (batch, seq_len, d_model)
        
        Returns:
            Normalized tensor of same shape
        """
        # Step 1: Compute mean across last dimension
        # keepdim=True so we can broadcast in subtraction
        mean = x.mean(dim=-1, keepdim=True)
        
        # Step 2: Compute variance across last dimension
        # Using the formula: var = E[x²] - E[x]²
        # Or we can use: var = E[(x - mean)²]
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        
        # Step 3: Normalize
        # (x - mean) centers the data around 0
        # / sqrt(var + eps) scales to unit variance
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        
        # Step 4: Apply learnable scale and shift
        # This allows the model to learn the optimal distribution
        output = self.gamma * x_norm + self.beta
        
        return output


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization — a simpler, faster alternative.
    
    Used in: LLaMA, Mistral, GPT-NeoX
    
    Key difference from LayerNorm:
        - LayerNorm: normalize by mean AND variance (2 statistics)
        - RMSNorm: normalize by RMS only (1 statistic)
    
    The formula:
        RMSNorm(x) = γ × x / √(mean(x²) + ε)
    
    Why it works:
        - Empirically equivalent quality to LayerNorm
        - 10-15% faster (no mean subtraction)
        - Simpler gradient computation
    
    Args:
        d_model: Feature dimension
        eps: Small constant for numerical stability
    """
    
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        
        self.d_model = d_model
        self.eps = eps
        
        # Only scale parameter, no shift (bias)
        self.gamma = nn.Parameter(torch.ones(d_model))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply RMS normalization.
        
        Args:
            x: Input tensor of shape (..., d_model)
        
        Returns:
            Normalized tensor of same shape
        """
        # Compute RMS: root mean square
        # rms = sqrt(mean(x²))
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        
        # Normalize and scale
        output = self.gamma * (x / rms)
        
        return output


# =============================================================================
# DEMONSTRATION
# =============================================================================

def demonstrate_layer_norm():
    """
    Demonstrate LayerNorm with concrete examples showing what it does.
    """
    print("=" * 70)
    print("LAYER NORMALIZATION DEMONSTRATION")
    print("=" * 70)
    
    # Simple example with clear numbers
    print("\n1. SIMPLE EXAMPLE: What LayerNorm Actually Does")
    print("-" * 50)
    
    # Create a simple input: one token with 4 features
    x_simple = torch.tensor([[1.0, 2.0, 3.0, 4.0]])  # shape (1, 4)
    
    print(f"   Input x: {x_simple.squeeze().tolist()}")
    
    # Manual computation
    mean = x_simple.mean()
    var = x_simple.var(unbiased=False)
    std = torch.sqrt(var + 1e-5)
    
    print(f"\n   Step 1 - Compute mean:")
    print(f"      mean = (1 + 2 + 3 + 4) / 4 = {mean.item():.2f}")
    
    print(f"\n   Step 2 - Compute variance:")
    print(f"      var = ((1-2.5)² + (2-2.5)² + (3-2.5)² + (4-2.5)²) / 4")
    print(f"          = (2.25 + 0.25 + 0.25 + 2.25) / 4 = {var.item():.2f}")
    print(f"      std = √{var.item():.2f} = {std.item():.4f}")
    
    print(f"\n   Step 3 - Normalize (subtract mean, divide by std):")
    x_centered = x_simple - mean
    x_norm_manual = x_centered / std
    print(f"      (x - mean) = {x_centered.squeeze().tolist()}")
    print(f"      (x - mean) / std = {[f'{v:.4f}' for v in x_norm_manual.squeeze().tolist()]}")
    
    # Verify with our implementation
    ln = LayerNorm(4)
    ln.eval()
    # Set gamma=1, beta=0 to see pure normalization
    with torch.no_grad():
        ln.gamma.fill_(1.0)
        ln.beta.fill_(0.0)
        output = ln(x_simple)
    
    print(f"\n   Our LayerNorm output: {[f'{v:.4f}' for v in output.squeeze().tolist()]}")
    print(f"   Matches manual: {torch.allclose(output, x_norm_manual, atol=1e-4)}")
    
    # Show the effect of gamma and beta
    print(f"\n2. LEARNABLE PARAMETERS: γ (scale) and β (shift)")
    print("-" * 50)
    
    ln_learnable = LayerNorm(4)
    with torch.no_grad():
        ln_learnable.gamma.copy_(torch.tensor([2.0, 2.0, 2.0, 2.0]))  # Scale by 2
        ln_learnable.beta.copy_(torch.tensor([1.0, 1.0, 1.0, 1.0]))   # Shift by 1
        output_scaled = ln_learnable(x_simple)
    
    print(f"   With γ=2, β=1:")
    print(f"   Output = 2 × normalized + 1")
    print(f"   = {[f'{v:.4f}' for v in output_scaled.squeeze().tolist()]}")
    
    # Real transformer example
    print(f"\n3. TRANSFORMER EXAMPLE")
    print("-" * 50)
    
    batch_size, seq_len, d_model = 2, 5, 512
    x = torch.randn(batch_size, seq_len, d_model)
    
    print(f"   Input shape: {x.shape}")
    print(f"   Meaning: {batch_size} batches × {seq_len} tokens × {d_model} features")
    
    ln_transformer = LayerNorm(d_model)
    output = ln_transformer(x)
    
    print(f"   Output shape: {output.shape}")
    
    # Check that each position is normalized
    print(f"\n   Verification: Each position should have mean≈0, std≈1")
    
    # Check first batch, first token
    sample = output[0, 0, :]  # First batch, first token
    print(f"   Token [0,0]: mean={sample.mean().item():.6f}, std={sample.std().item():.4f}")
    
    sample = output[0, 1, :]  # First batch, second token
    print(f"   Token [0,1]: mean={sample.mean().item():.6f}, std={sample.std().item():.4f}")
    
    sample = output[1, 0, :]  # Second batch, first token
    print(f"   Token [1,0]: mean={sample.mean().item():.6f}, std={sample.std().item():.4f}")
    
    # Compare with BatchNorm
    print(f"\n4. LAYERNORM vs BATCHNORM: Why LayerNorm for Transformers?")
    print("-" * 50)
    print("""
   BatchNorm normalizes across BATCH dimension:
      - Computes mean/var across all samples for each feature
      - Problem: Different batches have different sequence lengths!
      - Problem: At inference, batch size = 1, statistics are unreliable
   
   LayerNorm normalizes across FEATURE dimension:
      - Computes mean/var for each token independently
      - Works with any batch size, any sequence length
      - Each token is normalized on its own
   
   Visual:
   
   BatchNorm (across ↓):          LayerNorm (across →):
   ┌─────────────────┐            ┌─────────────────┐
   │ token1 │ token2 │            │ token1 │ token2 │
   │   ↓    │   ↓    │            │  →→→   │  →→→   │
   │   ↓    │   ↓    │            │  →→→   │  →→→   │
   └─────────────────┘            └─────────────────┘
     batch                          each token alone
""")
    
    # RMSNorm comparison
    print(f"\n5. RMSNORM: The Simpler Alternative")
    print("-" * 50)
    
    rms = RMSNorm(d_model)
    output_rms = rms(x)
    
    print(f"   RMSNorm skips the mean subtraction:")
    print(f"      LayerNorm: (x - mean) / std × γ + β")
    print(f"      RMSNorm:   x / rms × γ")
    print(f"\n   Output shape: {output_rms.shape}")
    
    # Parameter count
    ln_params = sum(p.numel() for p in ln_transformer.parameters())
    rms_params = sum(p.numel() for p in rms.parameters())
    print(f"\n   Parameters:")
    print(f"      LayerNorm: {ln_params:,} (γ: {d_model}, β: {d_model})")
    print(f"      RMSNorm:   {rms_params:,} (γ: {d_model}, no β)")
    
    print("\n" + "=" * 70)
    print("KEY TAKEAWAYS")
    print("=" * 70)
    print("""
1. WHY NORMALIZE?
   - Deep networks have unstable activations (exploding/vanishing)
   - LayerNorm keeps values in a reasonable range
   - Enables training of very deep transformers (100+ layers)

2. LAYER vs BATCH NORMALIZATION
   - LayerNorm: normalize each token independently (good for sequences)
   - BatchNorm: normalize across batch (bad for variable-length text)

3. THE FORMULA
   LayerNorm(x) = γ × (x - mean) / std + β
   - Normalize to mean=0, std=1
   - Then let γ, β learn the best scale and shift

4. PRE-LN vs POST-LN
   - Post-LN (original): Attention → Add → Norm
   - Pre-LN (modern):    Norm → Attention → Add
   - Pre-LN is more stable, used in GPT-2, LLaMA, etc.

5. RMSNORM (modern alternative)
   - Skips mean subtraction, only uses root-mean-square
   - 10-15% faster, same quality
   - Used in LLaMA, Mistral
""")


def test_layer_norm():
    """Test LayerNorm and RMSNorm implementations."""
    print("\n" + "=" * 70)
    print("TESTING LAYER NORMALIZATION")
    print("=" * 70)
    
    # Test 1: Output shape
    print("\n✓ Test 1: Output shapes")
    d_model = 512
    batch_size, seq_len = 2, 10
    
    ln = LayerNorm(d_model)
    x = torch.randn(batch_size, seq_len, d_model)
    output = ln(x)
    
    assert output.shape == x.shape, f"Expected {x.shape}, got {output.shape}"
    print(f"   LayerNorm: {x.shape} → {output.shape} ✓")
    
    # Test 2: RMSNorm shape
    rms = RMSNorm(d_model)
    output_rms = rms(x)
    
    assert output_rms.shape == x.shape
    print(f"   RMSNorm: {x.shape} → {output_rms.shape} ✓")
    
    # Test 3: Normalized output has mean≈0, std≈1
    print("\n✓ Test 2: LayerNorm produces mean≈0, std≈1")
    
    # With gamma=1, beta=0
    ln_test = LayerNorm(d_model)
    with torch.no_grad():
        ln_test.gamma.fill_(1.0)
        ln_test.beta.fill_(0.0)
    
    output_test = ln_test(x)
    
    # Check each position
    for b in range(batch_size):
        for s in range(min(3, seq_len)):  # Check first 3 positions
            sample = output_test[b, s, :]
            mean = sample.mean().item()
            std = sample.std().item()
            assert abs(mean) < 1e-5, f"Mean should be ≈0, got {mean}"
            assert abs(std - 1.0) < 0.1, f"Std should be ≈1, got {std}"
    
    print(f"   Mean ≈ 0, Std ≈ 1 for all positions ✓")
    
    # Test 4: Compare with PyTorch's LayerNorm
    print("\n✓ Test 3: Matches PyTorch's nn.LayerNorm")
    
    pytorch_ln = nn.LayerNorm(d_model)
    our_ln = LayerNorm(d_model)
    
    # Copy parameters
    with torch.no_grad():
        our_ln.gamma.copy_(pytorch_ln.weight)
        our_ln.beta.copy_(pytorch_ln.bias)
    
    output_ours = our_ln(x)
    output_pytorch = pytorch_ln(x)
    
    assert torch.allclose(output_ours, output_pytorch, atol=1e-5), \
        "Our LayerNorm should match PyTorch's"
    print(f"   Outputs match PyTorch implementation ✓")
    
    # Test 5: Gradients flow
    print("\n✓ Test 4: Gradients flow correctly")
    
    ln_grad = LayerNorm(d_model)
    x_grad = torch.randn(batch_size, seq_len, d_model, requires_grad=True)
    
    output = ln_grad(x_grad)
    loss = output.sum()
    loss.backward()
    
    assert x_grad.grad is not None, "No gradient for input"
    assert ln_grad.gamma.grad is not None, "No gradient for gamma"
    assert ln_grad.beta.grad is not None, "No gradient for beta"
    print(f"   Gradients computed for input, γ, and β ✓")
    
    # Test 6: Works with different shapes
    print("\n✓ Test 5: Handles various input shapes")
    
    for shape in [(1, 1, d_model), (4, 20, d_model), (1, 100, d_model)]:
        test_x = torch.randn(*shape)
        test_out = ln(test_x)
        assert test_out.shape == shape
    print(f"   Various shapes all work ✓")
    
    print("\n" + "=" * 70)
    print("ALL TESTS PASSED ✓")
    print("=" * 70)


if __name__ == "__main__":
    test_layer_norm()
    demonstrate_layer_norm()
