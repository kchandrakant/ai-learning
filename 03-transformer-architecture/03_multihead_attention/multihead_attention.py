"""
Step 3: Multi-Head Attention

The key insight: A single attention head can only focus on one "type" of 
relationship at a time. Multiple heads allow the model to jointly attend 
to information from different representation subspaces.

The formula:
    MultiHead(Q, K, V) = Concat(head_1, ..., head_h) × W_O
    
    where head_i = Attention(Q × W_Q^i, K × W_K^i, V × W_V^i)

Key dimensions:
    d_model = 512  (total model dimension)
    num_heads = 8  (number of parallel attention heads)
    d_k = d_v = d_model / num_heads = 64  (dimension per head)

The magic: We DON'T create h separate attention modules!
Instead, we use a single large projection matrix and reshape.
This is computationally equivalent but much more efficient.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our scaled dot-product attention from Step 2
# Note: Python module names can't start with digits, so we import differently
from importlib import import_module
attention_module = import_module("02_attention.scaled_dot_product_attention")
scaled_dot_product_attention = attention_module.scaled_dot_product_attention
create_causal_mask = attention_module.create_causal_mask


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention mechanism from "Attention Is All You Need".
    
    Instead of performing a single attention function with d_model dimensions,
    we project Q, K, V into h different subspaces, perform attention in parallel,
    and concatenate the results.
    
    Args:
        d_model: Total dimension of the model (e.g., 512)
        num_heads: Number of parallel attention heads (e.g., 8)
        dropout: Dropout probability for attention weights
    """
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        
        # Validate that d_model is divisible by num_heads
        assert d_model % num_heads == 0, \
            f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # Dimension per head
        
        # =====================================================================
        # PROJECTION MATRICES
        # =====================================================================
        # Instead of h separate (d_model, d_k) matrices, we use one (d_model, d_model)
        # matrix and then split the output into h heads.
        #
        # W_Q: Projects input to queries for ALL heads at once
        # W_K: Projects input to keys for ALL heads at once  
        # W_V: Projects input to values for ALL heads at once
        # W_O: Projects concatenated head outputs back to d_model
        
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        
        # Store attention weights for visualization
        self.attention_weights = None
    
    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Split the last dimension into (num_heads, d_k) and transpose.
        
        Input:  (batch, seq_len, d_model)
        Output: (batch, num_heads, seq_len, d_k)
        
        This rearranges the tensor so each head can be processed in parallel.
        """
        batch_size, seq_len, d_model = x.shape
        
        # Reshape: (batch, seq, d_model) → (batch, seq, num_heads, d_k)
        x = x.view(batch_size, seq_len, self.num_heads, self.d_k)
        
        # Transpose: (batch, seq, num_heads, d_k) → (batch, num_heads, seq, d_k)
        # Now each head is a separate "batch" dimension
        return x.transpose(1, 2)
    
    def combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reverse of split_heads: combine heads back into single tensor.
        
        Input:  (batch, num_heads, seq_len, d_k)
        Output: (batch, seq_len, d_model)
        """
        batch_size, num_heads, seq_len, d_k = x.shape
        
        # Transpose: (batch, num_heads, seq, d_k) → (batch, seq, num_heads, d_k)
        x = x.transpose(1, 2)
        
        # Reshape: (batch, seq, num_heads, d_k) → (batch, seq, d_model)
        # contiguous() is needed because transpose doesn't change memory layout
        return x.contiguous().view(batch_size, seq_len, self.d_model)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Compute multi-head attention.
        
        Args:
            query: (batch, seq_len_q, d_model) - what we're looking for
            key:   (batch, seq_len_k, d_model) - what each position contains
            value: (batch, seq_len_v, d_model) - information to retrieve
            mask:  Optional mask for attention scores
            
        Returns:
            output: (batch, seq_len_q, d_model)
            
        Note: For self-attention, query = key = value (same input)
              For cross-attention (decoder), query comes from decoder,
              key/value come from encoder
        """
        batch_size = query.size(0)
        
        # =====================================================================
        # Step 1: Linear projections for all heads at once
        # =====================================================================
        # Input: (batch, seq, d_model)
        # After projection: still (batch, seq, d_model)
        # But now contains the concatenated projections for all h heads
        
        Q = self.W_Q(query)  # (batch, seq_q, d_model)
        K = self.W_K(key)    # (batch, seq_k, d_model)
        V = self.W_V(value)  # (batch, seq_v, d_model)
        
        # =====================================================================
        # Step 2: Split into multiple heads
        # =====================================================================
        # Reshape from (batch, seq, d_model) to (batch, num_heads, seq, d_k)
        # Each head now has its own d_k-dimensional view of Q, K, V
        
        Q = self.split_heads(Q)  # (batch, num_heads, seq_q, d_k)
        K = self.split_heads(K)  # (batch, num_heads, seq_k, d_k)
        V = self.split_heads(V)  # (batch, num_heads, seq_v, d_k)
        
        # =====================================================================
        # Step 3: Scaled dot-product attention for all heads in parallel
        # =====================================================================
        # The attention function treats num_heads as part of the batch dimension
        # So we're computing h attention operations in parallel!
        
        # Adjust mask shape if needed: (batch, 1, 1, seq_k) or (batch, 1, seq_q, seq_k)
        if mask is not None:
            # Add head dimension if not present
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)  # (batch, 1, seq_q, seq_k)
        
        attn_output, attention_weights = scaled_dot_product_attention(
            Q, K, V, mask=mask, dropout=self.dropout
        )
        # attn_output: (batch, num_heads, seq_q, d_k)
        # attention_weights: (batch, num_heads, seq_q, seq_k)
        
        # Store for visualization
        self.attention_weights = attention_weights
        
        # =====================================================================
        # Step 4: Concatenate heads
        # =====================================================================
        # Combine all heads back: (batch, num_heads, seq, d_k) → (batch, seq, d_model)
        
        concat_output = self.combine_heads(attn_output)  # (batch, seq_q, d_model)
        
        # =====================================================================
        # Step 5: Final linear projection
        # =====================================================================
        # This allows the model to learn how to best combine the head outputs
        
        output = self.W_O(concat_output)  # (batch, seq_q, d_model)
        
        return output
    
    def get_attention_weights(self) -> torch.Tensor:
        """Return the attention weights from the last forward pass."""
        return self.attention_weights


# =============================================================================
# DEMONSTRATION AND VISUALIZATION
# =============================================================================

def demonstrate_multihead_attention():
    """
    Demonstrate Multi-Head Attention with detailed shape tracking
    and visualization of different heads.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    print("=" * 70)
    print("MULTI-HEAD ATTENTION DEMONSTRATION")
    print("=" * 70)
    
    # Configuration
    d_model = 64       # Small for demonstration (typically 512)
    num_heads = 4      # Number of attention heads (typically 8)
    seq_len = 6
    batch_size = 1
    
    sentence = ["The", "cat", "sat", "on", "the", "mat"]
    
    print(f"\n1. CONFIGURATION")
    print(f"   d_model (total dimension): {d_model}")
    print(f"   num_heads: {num_heads}")
    print(f"   d_k (dimension per head): {d_model // num_heads}")
    print(f"   Sentence: {sentence}")
    
    # Create the multi-head attention module
    mha = MultiHeadAttention(d_model, num_heads, dropout=0.0)
    mha.eval()  # Disable dropout for visualization
    
    # Create input (simulating embedded tokens)
    torch.manual_seed(42)
    x = torch.randn(batch_size, seq_len, d_model)
    
    print(f"\n2. INPUT")
    print(f"   x shape: {x.shape}")
    print(f"   Interpretation: {batch_size} batch × {seq_len} tokens × {d_model} dimensions")
    
    # For self-attention: query = key = value
    print(f"\n3. SELF-ATTENTION (query = key = value = x)")
    
    # Forward pass
    with torch.no_grad():
        output = mha(query=x, key=x, value=x)
    
    print(f"   Output shape: {output.shape}")
    
    # Get attention weights
    attn_weights = mha.get_attention_weights()
    print(f"   Attention weights shape: {attn_weights.shape}")
    print(f"   Interpretation: {batch_size} batch × {num_heads} heads × {seq_len} queries × {seq_len} keys")
    
    # =========================================================================
    # Visualize attention patterns for each head
    # =========================================================================
    print(f"\n4. ATTENTION PATTERNS BY HEAD")
    print(f"   Each head learns different relationships!\n")
    
    fig, axes = plt.subplots(1, num_heads, figsize=(4 * num_heads, 4))
    
    for head_idx in range(num_heads):
        ax = axes[head_idx] if num_heads > 1 else axes
        
        # Get this head's attention weights
        head_weights = attn_weights[0, head_idx].detach().numpy()
        
        # Plot heatmap
        im = ax.imshow(head_weights, cmap='Blues', vmin=0, vmax=1)
        
        # Labels
        ax.set_xticks(range(seq_len))
        ax.set_yticks(range(seq_len))
        ax.set_xticklabels(sentence, rotation=45, ha='right', fontsize=9)
        ax.set_yticklabels(sentence, fontsize=9)
        
        ax.set_xlabel('Key (attending to)', fontsize=10)
        if head_idx == 0:
            ax.set_ylabel('Query (from)', fontsize=10)
        ax.set_title(f'Head {head_idx + 1}', fontsize=12, fontweight='bold')
        
        # Add colorbar
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    plt.suptitle('Multi-Head Attention: Each Head Learns Different Patterns', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('multihead_attention_visualization.png', dpi=150, bbox_inches='tight')
    print(f"   Saved: multihead_attention_visualization.png")
    
    # =========================================================================
    # Demonstrate with causal mask (decoder-style)
    # =========================================================================
    print(f"\n5. WITH CAUSAL MASK (decoder self-attention)")
    
    causal_mask = create_causal_mask(seq_len)
    
    with torch.no_grad():
        output_masked = mha(query=x, key=x, value=x, mask=causal_mask)
    
    attn_weights_masked = mha.get_attention_weights()
    
    # Visualize masked attention
    fig, axes = plt.subplots(1, num_heads, figsize=(4 * num_heads, 4))
    
    for head_idx in range(num_heads):
        ax = axes[head_idx] if num_heads > 1 else axes
        head_weights = attn_weights_masked[0, head_idx].detach().numpy()
        
        im = ax.imshow(head_weights, cmap='Blues', vmin=0, vmax=1)
        ax.set_xticks(range(seq_len))
        ax.set_yticks(range(seq_len))
        ax.set_xticklabels(sentence, rotation=45, ha='right', fontsize=9)
        ax.set_yticklabels(sentence, fontsize=9)
        
        ax.set_xlabel('Key (attending to)', fontsize=10)
        if head_idx == 0:
            ax.set_ylabel('Query (from)', fontsize=10)
        ax.set_title(f'Head {head_idx + 1}', fontsize=12, fontweight='bold')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    plt.suptitle('Multi-Head Attention with Causal Mask (No Future Peeking)', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('multihead_causal_visualization.png', dpi=150, bbox_inches='tight')
    print(f"   Saved: multihead_causal_visualization.png")
    
    # =========================================================================
    # Shape walkthrough
    # =========================================================================
    print(f"\n6. SHAPE WALKTHROUGH")
    print(f"""
    Input x:           ({batch_size}, {seq_len}, {d_model})
                       ↓
    Project Q, K, V:   ({batch_size}, {seq_len}, {d_model})  [same shape, different learned weights]
                       ↓
    Split heads:       ({batch_size}, {num_heads}, {seq_len}, {d_model // num_heads})
                       ↓
    Attention:         ({batch_size}, {num_heads}, {seq_len}, {d_model // num_heads})  [output]
                       ({batch_size}, {num_heads}, {seq_len}, {seq_len})  [attention weights]
                       ↓
    Combine heads:     ({batch_size}, {seq_len}, {d_model})
                       ↓
    Output projection: ({batch_size}, {seq_len}, {d_model})
    """)
    
    # =========================================================================
    # Key takeaways
    # =========================================================================
    print("=" * 70)
    print("KEY TAKEAWAYS")
    print("=" * 70)
    print("""
1. MULTIPLE HEADS = MULTIPLE RELATIONSHIP TYPES
   - Each head has its own W_Q, W_K, W_V projections (embedded in the large matrices)
   - Different heads learn to focus on different patterns

2. EFFICIENT IMPLEMENTATION via reshape
   - One large matrix multiply, then reshape into heads
   - Mathematically equivalent to h separate attention operations
   - But MUCH faster due to parallelism

3. PARAMETER COUNT is roughly the same as single-head
   - h heads × (d_model/h) dimensions ≈ 1 head × d_model dimensions
   - We're splitting capacity, not adding it

4. W_O learns to COMBINE head outputs
   - After concatenation, W_O learns optimal combination
   - This is a key learnable parameter

5. CAUSAL MASK works identically
   - Applied to all heads simultaneously
   - Prevents future token attention in decoder

6. RESEARCH SHOWS heads SPECIALIZE:
   - Positional heads (attend to nearby tokens)
   - Syntactic heads (subject-verb dependencies)  
   - Rare word heads (attend to infrequent tokens)
   - Separator heads (attend to punctuation/special tokens)
""")


def test_multihead_attention():
    """Test Multi-Head Attention implementation correctness."""
    print("\n" + "=" * 70)
    print("TESTING MULTI-HEAD ATTENTION")
    print("=" * 70)
    
    # Test 1: Basic shape test
    print("\n✓ Test 1: Output shapes")
    d_model, num_heads = 512, 8
    batch_size, seq_len = 2, 10
    
    mha = MultiHeadAttention(d_model, num_heads, dropout=0.0)  # No dropout for testing
    mha.eval()
    x = torch.randn(batch_size, seq_len, d_model)
    
    with torch.no_grad():
        output = mha(x, x, x)
    assert output.shape == (batch_size, seq_len, d_model), \
        f"Expected {(batch_size, seq_len, d_model)}, got {output.shape}"
    print(f"   Input: {x.shape} → Output: {output.shape} ✓")
    
    # Test 2: Attention weights shape
    print("\n✓ Test 2: Attention weights shape")
    attn_weights = mha.get_attention_weights()
    expected_shape = (batch_size, num_heads, seq_len, seq_len)
    assert attn_weights.shape == expected_shape, \
        f"Expected {expected_shape}, got {attn_weights.shape}"
    print(f"   Attention weights: {attn_weights.shape} ✓")
    
    # Test 3: Attention weights sum to 1
    print("\n✓ Test 3: Attention weights sum to 1 (per query)")
    row_sums = attn_weights.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5), \
        "Attention weights should sum to 1"
    print(f"   Row sums ≈ 1.0 ✓")
    
    # Test 4: Cross-attention (different sequence lengths)
    print("\n✓ Test 4: Cross-attention (encoder-decoder)")
    seq_len_q, seq_len_kv = 8, 12
    query = torch.randn(batch_size, seq_len_q, d_model)
    key = torch.randn(batch_size, seq_len_kv, d_model)
    value = torch.randn(batch_size, seq_len_kv, d_model)
    
    with torch.no_grad():
        output = mha(query, key, value)
    assert output.shape == (batch_size, seq_len_q, d_model), \
        f"Cross-attention output shape mismatch"
    print(f"   Query: {query.shape}, Key/Value: {key.shape} → Output: {output.shape} ✓")
    
    # Test 5: Causal mask
    print("\n✓ Test 5: Causal mask zeros future attention")
    mha_test = MultiHeadAttention(d_model, num_heads, dropout=0.0)
    mha_test.eval()
    
    causal_mask = create_causal_mask(seq_len)
    with torch.no_grad():
        _ = mha_test(x, x, x, mask=causal_mask)
    
    attn_masked = mha_test.get_attention_weights()
    # Check upper triangle is zero (future positions)
    for i in range(seq_len):
        for j in range(i + 1, seq_len):
            assert torch.allclose(attn_masked[:, :, i, j], torch.zeros(batch_size, num_heads), atol=1e-5), \
                f"Position ({i}, {j}) should have zero attention"
    print(f"   Future positions correctly masked ✓")
    
    # Test 6: d_model must be divisible by num_heads
    print("\n✓ Test 6: Dimension validation")
    try:
        bad_mha = MultiHeadAttention(d_model=512, num_heads=7)
        print("   ERROR: Should have raised assertion!")
    except AssertionError:
        print("   Correctly rejects d_model=512, num_heads=7 ✓")
    
    print("\n" + "=" * 70)
    print("ALL TESTS PASSED ✓")
    print("=" * 70)


if __name__ == "__main__":
    # Run tests first
    test_multihead_attention()
    
    # Then run demonstration with visualization
    demonstrate_multihead_attention()
