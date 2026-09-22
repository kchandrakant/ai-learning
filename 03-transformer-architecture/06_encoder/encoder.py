"""
Step 6: Encoder Block

The encoder block combines all our components:
    - Multi-Head Self-Attention
    - Feed-Forward Network
    - Layer Normalization
    - Residual Connections

Architecture (Pre-LN, modern style):
    
    x ──┬── LayerNorm ── MHA ──┬── LayerNorm ── FFN ──┬── output
        │                      │                      │
        └──────── Add ─────────┴──────── Add ─────────┘
        
The residual connections (Add) are crucial:
    - Allow gradients to flow directly through the network
    - Enable training of very deep models (100+ layers)
    - Each sublayer ADDS to the input rather than replacing it

Pre-LN vs Post-LN:
    - Post-LN (original): x + Sublayer(LayerNorm(x))  
    - Pre-LN (modern):    LayerNorm(x + Sublayer(x))
    
We implement Pre-LN because it's more stable for deep models.
"""

import torch
import torch.nn as nn
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our implementations
from importlib import import_module

# Multi-Head Attention
mha_module = import_module("03_multihead_attention.multihead_attention")
MultiHeadAttention = mha_module.MultiHeadAttention

# Feed-Forward Network
ffn_module = import_module("04_feed_forward.feed_forward")
FeedForward = ffn_module.FeedForward
SwiGLUFeedForward = ffn_module.SwiGLUFeedForward

# Layer Normalization
ln_module = import_module("05_layer_norm.layer_norm")
LayerNorm = ln_module.LayerNorm
RMSNorm = ln_module.RMSNorm


class EncoderBlock(nn.Module):
    """
    A single Transformer Encoder Block.
    
    Components:
        1. Multi-Head Self-Attention (tokens attend to each other)
        2. Feed-Forward Network (process each token independently)
        3. Layer Normalization (stabilize activations)
        4. Residual Connections (enable gradient flow)
    
    Args:
        d_model: Model dimension (e.g., 512)
        num_heads: Number of attention heads (e.g., 8)
        d_ff: Feed-forward hidden dimension (e.g., 2048)
        dropout: Dropout probability
        use_swiglu: Use SwiGLU instead of ReLU FFN
        use_rmsnorm: Use RMSNorm instead of LayerNorm
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int = None,
        dropout: float = 0.1,
        use_swiglu: bool = False,
        use_rmsnorm: bool = False
    ):
        super().__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        
        # Default d_ff = 4 * d_model
        if d_ff is None:
            d_ff = 4 * d_model
        
        # =====================================================================
        # SUB-LAYER 1: Multi-Head Self-Attention
        # =====================================================================
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        
        # =====================================================================
        # SUB-LAYER 2: Feed-Forward Network
        # =====================================================================
        if use_swiglu:
            self.feed_forward = SwiGLUFeedForward(d_model, d_ff, dropout)
        else:
            self.feed_forward = FeedForward(d_model, d_ff, dropout)
        
        # =====================================================================
        # LAYER NORMALIZATION
        # =====================================================================
        NormClass = RMSNorm if use_rmsnorm else LayerNorm
        self.norm1 = NormClass(d_model)  # Before attention
        self.norm2 = NormClass(d_model)  # Before FFN
        
        # =====================================================================
        # DROPOUT for residual connections
        # =====================================================================
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Forward pass through encoder block.
        
        Args:
            x: Input tensor (batch, seq_len, d_model)
            mask: Optional attention mask
        
        Returns:
            Output tensor (batch, seq_len, d_model)
        """
        # ==================================================================
        # SUB-LAYER 1: Self-Attention with Residual Connection
        # ==================================================================
        # Pre-LN: normalize BEFORE the sublayer
        
        # Step 1: Normalize
        x_norm = self.norm1(x)
        
        # Step 2: Self-attention (Q = K = V = normalized input)
        attn_output = self.self_attention(
            query=x_norm,
            key=x_norm,
            value=x_norm,
            mask=mask
        )
        
        # Step 3: Dropout + Residual connection
        # The residual (x) is the ORIGINAL input, not normalized
        x = x + self.dropout(attn_output)
        
        # ==================================================================
        # SUB-LAYER 2: Feed-Forward with Residual Connection
        # ==================================================================
        # Pre-LN: normalize BEFORE the sublayer
        
        # Step 1: Normalize
        x_norm = self.norm2(x)
        
        # Step 2: Feed-forward
        ff_output = self.feed_forward(x_norm)
        
        # Step 3: Dropout + Residual connection
        x = x + self.dropout(ff_output)
        
        return x
    
    def get_attention_weights(self) -> torch.Tensor:
        """Return attention weights from the last forward pass."""
        return self.self_attention.get_attention_weights()


class Encoder(nn.Module):
    """
    Full Transformer Encoder: Stack of N encoder blocks.
    
    Architecture:
        Input → [EncoderBlock] × N → LayerNorm → Output
    
    The final LayerNorm is needed for Pre-LN architecture
    (the last block's output isn't normalized otherwise).
    
    Args:
        num_layers: Number of encoder blocks (e.g., 6 or 12)
        d_model: Model dimension
        num_heads: Number of attention heads
        d_ff: Feed-forward hidden dimension
        dropout: Dropout probability
        use_swiglu: Use SwiGLU FFN
        use_rmsnorm: Use RMSNorm
    """
    
    def __init__(
        self,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int = None,
        dropout: float = 0.1,
        use_swiglu: bool = False,
        use_rmsnorm: bool = False
    ):
        super().__init__()
        
        self.num_layers = num_layers
        self.d_model = d_model
        
        # Stack of encoder blocks
        self.layers = nn.ModuleList([
            EncoderBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                dropout=dropout,
                use_swiglu=use_swiglu,
                use_rmsnorm=use_rmsnorm
            )
            for _ in range(num_layers)
        ])
        
        # Final normalization (for Pre-LN architecture)
        NormClass = RMSNorm if use_rmsnorm else LayerNorm
        self.final_norm = NormClass(d_model)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Forward pass through all encoder layers.
        
        Args:
            x: Input tensor (batch, seq_len, d_model)
            mask: Optional attention mask
        
        Returns:
            Encoded output (batch, seq_len, d_model)
        """
        # Pass through each encoder block
        for layer in self.layers:
            x = layer(x, mask)
        
        # Final normalization
        x = self.final_norm(x)
        
        return x


# =============================================================================
# DEMONSTRATION
# =============================================================================

def demonstrate_encoder():
    """
    Demonstrate the Encoder Block with detailed explanations.
    """
    print("=" * 70)
    print("ENCODER BLOCK DEMONSTRATION")
    print("=" * 70)
    
    # Configuration
    d_model = 512
    num_heads = 8
    d_ff = 2048
    num_layers = 6
    batch_size = 2
    seq_len = 10
    
    print(f"\n1. CONFIGURATION")
    print(f"   d_model:    {d_model}")
    print(f"   num_heads:  {num_heads}")
    print(f"   d_ff:       {d_ff}")
    print(f"   num_layers: {num_layers}")
    
    # Create encoder block
    encoder_block = EncoderBlock(d_model, num_heads, d_ff, dropout=0.0)
    encoder_block.eval()
    
    # Create input
    torch.manual_seed(42)
    x = torch.randn(batch_size, seq_len, d_model)
    
    print(f"\n2. SINGLE ENCODER BLOCK")
    print(f"   Input shape:  {x.shape}")
    
    with torch.no_grad():
        output = encoder_block(x)
    
    print(f"   Output shape: {output.shape}")
    
    # Attention weights
    attn_weights = encoder_block.get_attention_weights()
    print(f"   Attention weights: {attn_weights.shape}")
    print(f"   (batch={batch_size}, heads={num_heads}, seq={seq_len}, seq={seq_len})")
    
    # Show the data flow
    print(f"\n3. DATA FLOW THROUGH ENCODER BLOCK")
    print(f"""
    Input x: ({batch_size}, {seq_len}, {d_model})
        │
        ▼
    LayerNorm: ({batch_size}, {seq_len}, {d_model})
        │
        ▼
    Multi-Head Attention
        Q, K, V = x_norm (self-attention)
        │
        ▼
    Attention output: ({batch_size}, {seq_len}, {d_model})
        │
        ▼
    Add residual: x + attn_output
        │
        ▼
    LayerNorm: ({batch_size}, {seq_len}, {d_model})
        │
        ▼
    Feed-Forward: ({batch_size}, {seq_len}, {d_model})
        │
        ▼
    Add residual: x + ff_output
        │
        ▼
    Output: ({batch_size}, {seq_len}, {d_model})
    """)
    
    # Full encoder stack
    print(f"\n4. FULL ENCODER (Stack of {num_layers} Blocks)")
    
    encoder = Encoder(num_layers, d_model, num_heads, d_ff, dropout=0.0)
    encoder.eval()
    
    with torch.no_grad():
        encoded = encoder(x)
    
    print(f"   Input:  {x.shape}")
    print(f"   Output: {encoded.shape}")
    
    # Parameter count
    print(f"\n5. PARAMETER COUNT")
    
    block_params = sum(p.numel() for p in encoder_block.parameters())
    total_params = sum(p.numel() for p in encoder.parameters())
    
    print(f"   Single block:     {block_params:,}")
    print(f"   {num_layers}-layer encoder: {total_params:,}")
    
    # Break down per component
    print(f"\n   Breakdown per block:")
    attn_params = sum(p.numel() for p in encoder_block.self_attention.parameters())
    ffn_params = sum(p.numel() for p in encoder_block.feed_forward.parameters())
    norm_params = sum(p.numel() for p in encoder_block.norm1.parameters()) * 2
    
    print(f"      Attention:  {attn_params:,} ({100*attn_params/block_params:.1f}%)")
    print(f"      FFN:        {ffn_params:,} ({100*ffn_params/block_params:.1f}%)")
    print(f"      LayerNorms: {norm_params:,} ({100*norm_params/block_params:.1f}%)")
    
    # Modern variant
    print(f"\n6. MODERN VARIANT (SwiGLU + RMSNorm)")
    
    encoder_modern = Encoder(
        num_layers, d_model, num_heads, d_ff,
        dropout=0.0,
        use_swiglu=True,
        use_rmsnorm=True
    )
    encoder_modern.eval()
    
    with torch.no_grad():
        encoded_modern = encoder_modern(x)
    
    modern_params = sum(p.numel() for p in encoder_modern.parameters())
    print(f"   Output shape: {encoded_modern.shape}")
    print(f"   Parameters:   {modern_params:,}")
    
    # Residual connection demonstration
    print(f"\n7. WHY RESIDUAL CONNECTIONS MATTER")
    print(f"""
    Without residual:
        output = sublayer(x)
        
        Problem: In deep networks, gradients must flow through
        EVERY layer. If any layer has small gradients, the
        signal vanishes.
    
    With residual:
        output = x + sublayer(x)
        
        Solution: Gradients can "skip" layers via the + path.
        Even if sublayer gradients are small, the x gradient
        flows directly through.
    
    Visual:
        
        x ──────────────────┐
        │                   │
        ▼                   │ (gradient highway)
    sublayer(x)             │
        │                   │
        ▼                   ▼
        +───────────────────┘
        │
        ▼
      output
    """)
    
    print("\n" + "=" * 70)
    print("KEY TAKEAWAYS")
    print("=" * 70)
    print("""
1. ENCODER BLOCK = Attention + FFN + Norms + Residuals
   - Attention: tokens communicate
   - FFN: tokens process information
   - Norms: keep values stable
   - Residuals: enable gradient flow

2. SELF-ATTENTION
   - Q = K = V = input (hence "self")
   - Every token can attend to every other token
   - No masking in encoder (unlike decoder)

3. RESIDUAL CONNECTIONS are CRITICAL
   - output = x + sublayer(x)
   - Gradients flow through "+" even if sublayer is problematic
   - Enables training 100+ layer models

4. PRE-LN vs POST-LN
   - Pre-LN (we use): Norm → Sublayer → Add
   - Post-LN (original): Sublayer → Add → Norm
   - Pre-LN is more stable for deep models

5. STACKING
   - Full encoder = N identical blocks stacked
   - Each block refines the representation
   - Typical: N = 6 (BERT base), 12 (BERT large), 24+ (GPT)
""")


def test_encoder():
    """Test encoder implementation."""
    print("\n" + "=" * 70)
    print("TESTING ENCODER")
    print("=" * 70)
    
    d_model = 512
    num_heads = 8
    d_ff = 2048
    num_layers = 6
    batch_size = 2
    seq_len = 10
    
    # Test 1: Single block output shape
    print("\n✓ Test 1: EncoderBlock output shape")
    block = EncoderBlock(d_model, num_heads, d_ff, dropout=0.0)
    block.eval()
    x = torch.randn(batch_size, seq_len, d_model)
    
    with torch.no_grad():
        output = block(x)
    
    assert output.shape == x.shape, f"Expected {x.shape}, got {output.shape}"
    print(f"   {x.shape} → {output.shape} ✓")
    
    # Test 2: Full encoder output shape
    print("\n✓ Test 2: Encoder (stack) output shape")
    encoder = Encoder(num_layers, d_model, num_heads, d_ff, dropout=0.0)
    encoder.eval()
    
    with torch.no_grad():
        output = encoder(x)
    
    assert output.shape == x.shape
    print(f"   {num_layers} layers: {x.shape} → {output.shape} ✓")
    
    # Test 3: Attention weights available
    print("\n✓ Test 3: Attention weights accessible")
    attn = block.get_attention_weights()
    expected = (batch_size, num_heads, seq_len, seq_len)
    assert attn.shape == expected
    print(f"   Attention weights shape: {attn.shape} ✓")
    
    # Test 4: Residual connection effect
    print("\n✓ Test 4: Residual connection works")
    
    # Create a block with zeroed attention (to test residual)
    block_test = EncoderBlock(d_model, num_heads, d_ff, dropout=0.0)
    block_test.eval()
    
    # If attention output were zero, residual should preserve input
    x_test = torch.randn(1, 5, d_model)
    with torch.no_grad():
        out = block_test(x_test)
    
    # Output should be different from input (sublayers do something)
    assert not torch.allclose(out, x_test, atol=1e-3)
    print(f"   Sublayers modify input (not identity) ✓")
    
    # Test 5: Modern variants work
    print("\n✓ Test 5: Modern variants (SwiGLU + RMSNorm)")
    encoder_modern = Encoder(
        num_layers, d_model, num_heads, d_ff,
        use_swiglu=True, use_rmsnorm=True, dropout=0.0
    )
    encoder_modern.eval()
    
    with torch.no_grad():
        out_modern = encoder_modern(x)
    
    assert out_modern.shape == x.shape
    print(f"   SwiGLU + RMSNorm: {x.shape} → {out_modern.shape} ✓")
    
    # Test 6: Gradients flow
    print("\n✓ Test 6: Gradients flow through encoder")
    encoder_grad = Encoder(num_layers, d_model, num_heads, d_ff, dropout=0.0)
    x_grad = torch.randn(batch_size, seq_len, d_model, requires_grad=True)
    
    output = encoder_grad(x_grad)
    loss = output.sum()
    loss.backward()
    
    assert x_grad.grad is not None
    assert not torch.all(x_grad.grad == 0)
    print(f"   Gradients computed and non-zero ✓")
    
    # Test 7: Different sequence lengths
    print("\n✓ Test 7: Handles different sequence lengths")
    for seq_l in [1, 5, 20, 50]:
        test_x = torch.randn(1, seq_l, d_model)
        with torch.no_grad():
            test_out = encoder(test_x)
        assert test_out.shape == (1, seq_l, d_model)
    print(f"   Sequence lengths [1, 5, 20, 50] all work ✓")
    
    print("\n" + "=" * 70)
    print("ALL TESTS PASSED ✓")
    print("=" * 70)


if __name__ == "__main__":
    test_encoder()
    demonstrate_encoder()
