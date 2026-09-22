"""
Step 4: Position-wise Feed-Forward Network (FFN)

The FFN is the "other half" of each transformer block. While attention
lets tokens communicate with each other, the FFN processes each token
INDEPENDENTLY through a small neural network.

The formula (original transformer):
    FFN(x) = ReLU(x × W₁ + b₁) × W₂ + b₂

Key pattern: EXPAND then CONTRACT
    Input:  d_model (512)
    Hidden: d_ff (2048) — 4× expansion
    Output: d_model (512)

Why expand? More dimensions = more capacity to learn complex functions.
The expansion gives the model "room to think" before compressing back.

Position-wise means:
    - Same weights applied to EVERY position
    - But each position processed INDEPENDENTLY (no mixing between tokens)
    - Token interactions happen in attention, not here
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network from "Attention Is All You Need".
    
    Architecture:
        Linear(d_model → d_ff) → ReLU → Dropout → Linear(d_ff → d_model) → Dropout
    
    Args:
        d_model: Input and output dimension (e.g., 512)
        d_ff: Hidden layer dimension (e.g., 2048, typically 4 × d_model)
        dropout: Dropout probability
    """
    
    def __init__(self, d_model: int, d_ff: int = None, dropout: float = 0.1):
        super().__init__()
        
        # Default: 4× expansion (as in original paper)
        if d_ff is None:
            d_ff = 4 * d_model
        
        self.d_model = d_model
        self.d_ff = d_ff
        
        # Two linear transformations with ReLU in between
        self.linear1 = nn.Linear(d_model, d_ff)    # Expand
        self.linear2 = nn.Linear(d_ff, d_model)    # Contract
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply position-wise feed-forward network.
        
        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
            
        Returns:
            Output tensor of shape (batch, seq_len, d_model)
        
        The same transformation is applied to each position independently.
        """
        # Step 1: Expand to d_ff dimensions
        # (batch, seq, d_model) → (batch, seq, d_ff)
        hidden = self.linear1(x)
        
        # Step 2: Non-linearity (ReLU in original, modern models use others)
        hidden = F.relu(hidden)
        
        # Step 3: Dropout for regularization
        hidden = self.dropout(hidden)
        
        # Step 4: Contract back to d_model
        # (batch, seq, d_ff) → (batch, seq, d_model)
        output = self.linear2(hidden)
        
        # Step 5: Final dropout
        output = self.dropout(output)
        
        return output


class SwiGLUFeedForward(nn.Module):
    """
    SwiGLU Feed-Forward Network — the modern replacement for ReLU FFN.
    
    Used in: LLaMA, PaLM, Mistral, and most state-of-the-art models.
    
    Key insight: Instead of ReLU, use a GATED activation:
        SwiGLU(x) = Swish(x × W_gate) ⊙ (x × W_up)
        
    Where:
        - Swish(x) = x × sigmoid(x)  — smooth, non-monotonic activation
        - ⊙ = element-wise multiplication (the "gate")
        - W_gate and W_up are separate projections
    
    Architecture:
        x → W_gate → Swish ─┐
                            ├─→ element-wise multiply → W_down → output
        x → W_up ───────────┘
    
    Why better than ReLU?
        1. Gating allows selective information flow
        2. Swish has better gradient properties than ReLU
        3. Empirically: ~1-2% better on language tasks
    
    Note: d_ff is often 8/3 × d_model (not 4×) to keep similar param count,
    since we now have 3 matrices instead of 2.
    """
    
    def __init__(self, d_model: int, d_ff: int = None, dropout: float = 0.1):
        super().__init__()
        
        # Default: 8/3 × d_model (keeps similar param count as 4× with 2 matrices)
        # Many implementations round to nice numbers
        if d_ff is None:
            d_ff = int(8 / 3 * d_model)
            # Round to multiple of 64 for efficiency
            d_ff = ((d_ff + 63) // 64) * 64
        
        self.d_model = d_model
        self.d_ff = d_ff
        
        # Three linear layers (vs two in standard FFN)
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)  # For Swish gate
        self.w_up = nn.Linear(d_model, d_ff, bias=False)    # For the value
        self.w_down = nn.Linear(d_ff, d_model, bias=False)  # Project back
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply SwiGLU feed-forward network.
        
        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
            
        Returns:
            Output tensor of shape (batch, seq_len, d_model)
        """
        # Gate path: x → W_gate → Swish
        gate = F.silu(self.w_gate(x))  # silu = swish = x * sigmoid(x)
        
        # Value path: x → W_up
        up = self.w_up(x)
        
        # Gated combination: element-wise multiply
        hidden = gate * up
        
        # Project back to d_model
        output = self.w_down(hidden)
        output = self.dropout(output)
        
        return output


# =============================================================================
# DEMONSTRATION
# =============================================================================

def demonstrate_feed_forward():
    """
    Demonstrate FFN with shape tracking and comparison between variants.
    """
    print("=" * 70)
    print("FEED-FORWARD NETWORK DEMONSTRATION")
    print("=" * 70)
    
    # Configuration
    d_model = 512
    d_ff = 2048  # 4× expansion
    batch_size = 2
    seq_len = 10
    
    print(f"\n1. CONFIGURATION")
    print(f"   d_model (input/output): {d_model}")
    print(f"   d_ff (hidden):          {d_ff}")
    print(f"   Expansion ratio:        {d_ff / d_model}×")
    
    # Create modules
    ffn_relu = FeedForward(d_model, d_ff, dropout=0.0)
    ffn_swiglu = SwiGLUFeedForward(d_model, dropout=0.0)
    
    # Create input
    torch.manual_seed(42)
    x = torch.randn(batch_size, seq_len, d_model)
    
    print(f"\n2. INPUT")
    print(f"   Shape: {x.shape}")
    print(f"   Meaning: {batch_size} batches × {seq_len} tokens × {d_model} dims")
    
    # Forward pass
    print(f"\n3. STANDARD FFN (ReLU)")
    print(f"   Architecture: Linear → ReLU → Linear")
    
    ffn_relu.eval()  # Disable dropout
    with torch.no_grad():
        output_relu = ffn_relu(x)
    
    print(f"   Input:  {x.shape}")
    print(f"   Hidden: (batch, seq, {d_ff})  [intermediate, not returned]")
    print(f"   Output: {output_relu.shape}")
    
    # SwiGLU
    print(f"\n4. SWIGLU FFN (Modern)")
    print(f"   Architecture: Swish(x·W_gate) ⊙ (x·W_up) → W_down")
    print(f"   d_ff (auto): {ffn_swiglu.d_ff}")
    
    with torch.no_grad():
        output_swiglu = ffn_swiglu(x)
    
    print(f"   Output: {output_swiglu.shape}")
    
    # Parameter comparison
    print(f"\n5. PARAMETER COUNT COMPARISON")
    
    relu_params = sum(p.numel() for p in ffn_relu.parameters())
    swiglu_params = sum(p.numel() for p in ffn_swiglu.parameters())
    
    print(f"\n   Standard FFN (ReLU):")
    print(f"      W1: {d_model} × {d_ff} = {d_model * d_ff:,}")
    print(f"      b1: {d_ff:,}")
    print(f"      W2: {d_ff} × {d_model} = {d_ff * d_model:,}")
    print(f"      b2: {d_model:,}")
    print(f"      Total: {relu_params:,}")
    
    print(f"\n   SwiGLU FFN:")
    print(f"      W_gate: {d_model} × {ffn_swiglu.d_ff} = {d_model * ffn_swiglu.d_ff:,}")
    print(f"      W_up:   {d_model} × {ffn_swiglu.d_ff} = {d_model * ffn_swiglu.d_ff:,}")
    print(f"      W_down: {ffn_swiglu.d_ff} × {d_model} = {ffn_swiglu.d_ff * d_model:,}")
    print(f"      (no biases)")
    print(f"      Total: {swiglu_params:,}")
    
    print(f"\n   Ratio: {swiglu_params / relu_params:.2f}× (SwiGLU vs ReLU)")
    
    # Position-wise demonstration
    print(f"\n6. POSITION-WISE: Each Token Processed Independently")
    print(f"   Same weights, different inputs → different outputs")
    
    # Demonstrate that processing tokens individually gives same result
    ffn_check = FeedForward(d_model, d_ff, dropout=0.0)
    ffn_check.eval()
    
    with torch.no_grad():
        full_out = ffn_check(x)
        single_out = ffn_check(x[:, 0:1, :])
    
    matches = torch.allclose(full_out[:, 0:1, :], single_out, atol=1e-6)
    print(f"\n   Process token 0 alone vs in sequence:")
    print(f"   Single token result matches: {matches} ✓")
    
    # Shape walkthrough
    print(f"\n7. SHAPE WALKTHROUGH (Standard FFN)")
    print(f"""
    Input x:     ({batch_size}, {seq_len}, {d_model})
                      ↓
    Linear1:     ({batch_size}, {seq_len}, {d_ff})     [expand]
                      ↓
    ReLU:        ({batch_size}, {seq_len}, {d_ff})     [non-linearity]
                      ↓
    Linear2:     ({batch_size}, {seq_len}, {d_model})  [contract]
                      ↓
    Output:      ({batch_size}, {seq_len}, {d_model})
    """)
    
    print("=" * 70)
    print("KEY TAKEAWAYS")
    print("=" * 70)
    print("""
1. EXPAND-CONTRACT PATTERN
   - Input: d_model (512)
   - Hidden: d_ff (2048) — 4× more room to learn
   - Output: d_model (512)
   
2. POSITION-WISE = INDEPENDENT
   - Same weights for all positions
   - No token-to-token interaction (that's attention's job)
   - Can be computed in parallel across positions
   
3. WHY FFN MATTERS
   - Attention: tokens communicate, gather information
   - FFN: process gathered information, add non-linearity
   - Together: attention + FFN = one transformer block

4. SWIGLU IS BETTER (modern choice)
   - Gating allows selective information flow
   - Swish activation has better gradients than ReLU
   - Used in LLaMA, Mistral, PaLM, etc.

5. MOST PARAMETERS ARE HERE
   - In a transformer, ~2/3 of parameters are in FFN layers
   - Attention has fewer params but more compute (quadratic)
""")


def test_feed_forward():
    """Test FFN implementations for correctness."""
    print("\n" + "=" * 70)
    print("TESTING FEED-FORWARD NETWORKS")
    print("=" * 70)
    
    # Test 1: Output shape
    print("\n✓ Test 1: Output shapes")
    d_model, d_ff = 512, 2048
    batch_size, seq_len = 2, 10
    
    ffn = FeedForward(d_model, d_ff, dropout=0.0)
    ffn.eval()
    x = torch.randn(batch_size, seq_len, d_model)
    
    with torch.no_grad():
        output = ffn(x)
    
    assert output.shape == x.shape, f"Expected {x.shape}, got {output.shape}"
    print(f"   FeedForward: {x.shape} → {output.shape} ✓")
    
    # Test 2: SwiGLU shape
    print("\n✓ Test 2: SwiGLU output shape")
    swiglu = SwiGLUFeedForward(d_model, dropout=0.0)
    swiglu.eval()
    
    with torch.no_grad():
        output_swiglu = swiglu(x)
    
    assert output_swiglu.shape == x.shape, f"Expected {x.shape}, got {output_swiglu.shape}"
    print(f"   SwiGLU: {x.shape} → {output_swiglu.shape} ✓")
    
    # Test 3: Position-wise independence
    print("\n✓ Test 3: Position-wise independence")
    ffn_test = FeedForward(d_model, d_ff, dropout=0.0)
    ffn_test.eval()
    
    with torch.no_grad():
        full_output = ffn_test(x)
        
        # Process each position separately
        for pos in range(seq_len):
            single_pos = x[:, pos:pos+1, :]
            single_output = ffn_test(single_pos)
            
            assert torch.allclose(full_output[:, pos:pos+1, :], single_output, atol=1e-6), \
                f"Position {pos} mismatch"
    
    print(f"   Each position processed identically ✓")
    
    # Test 4: Different sequence lengths
    print("\n✓ Test 4: Handles different sequence lengths")
    for seq_l in [1, 5, 20, 100]:
        test_input = torch.randn(1, seq_l, d_model)
        with torch.no_grad():
            test_output = ffn(test_input)
        assert test_output.shape == (1, seq_l, d_model)
    print(f"   Sequence lengths [1, 5, 20, 100] all work ✓")
    
    # Test 5: Default d_ff
    print("\n✓ Test 5: Default d_ff = 4 × d_model")
    ffn_default = FeedForward(d_model=256)
    assert ffn_default.d_ff == 1024, f"Expected 1024, got {ffn_default.d_ff}"
    print(f"   d_model=256 → d_ff={ffn_default.d_ff} ✓")
    
    # Test 6: Gradient flow
    print("\n✓ Test 6: Gradients flow correctly")
    ffn_grad = FeedForward(d_model, d_ff, dropout=0.0)
    x_grad = torch.randn(batch_size, seq_len, d_model, requires_grad=True)
    
    output = ffn_grad(x_grad)
    loss = output.sum()
    loss.backward()
    
    assert x_grad.grad is not None, "No gradient computed"
    assert not torch.all(x_grad.grad == 0), "Gradient is all zeros"
    print(f"   Gradients computed and non-zero ✓")
    
    print("\n" + "=" * 70)
    print("ALL TESTS PASSED ✓")
    print("=" * 70)


if __name__ == "__main__":
    test_feed_forward()
    demonstrate_feed_forward()
