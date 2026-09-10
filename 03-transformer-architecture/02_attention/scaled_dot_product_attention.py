"""
Step 2: Scaled Dot-Product Attention

This is THE core innovation of the transformer. It allows each token to 
"look at" every other token and decide what's relevant.

The formula:
    Attention(Q, K, V) = softmax(QK^T / √d_k) × V

Where:
    Q (Query):  "What am I looking for?" - shape (seq_len, d_k)
    K (Key):    "What do I contain?" - shape (seq_len, d_k)
    V (Value):  "What information do I provide?" - shape (seq_len, d_v)
    d_k:        Dimension of keys (used for scaling)

The process:
    1. Compute similarity scores: Q @ K^T → (seq_len, seq_len)
    2. Scale by √d_k to prevent softmax saturation
    3. Apply softmax to get attention weights (rows sum to 1)
    4. Weighted sum of values: weights @ V → (seq_len, d_v)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor = None,
    dropout: nn.Dropout = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute scaled dot-product attention.
    
    Args:
        query: (batch, seq_len, d_k) - what we're looking for
        key:   (batch, seq_len, d_k) - what each position contains
        value: (batch, seq_len, d_v) - information to retrieve
        mask:  (batch, 1, seq_len) or (batch, seq_len, seq_len) - positions to ignore
        dropout: Optional dropout layer for attention weights
        
    Returns:
        output: (batch, seq_len, d_v) - attended values
        attention_weights: (batch, seq_len, seq_len) - attention pattern
    """
    d_k = query.size(-1)
    
    # Step 1: Compute attention scores
    # Q @ K^T: (batch, seq_len, d_k) @ (batch, d_k, seq_len) → (batch, seq_len, seq_len)
    scores = torch.matmul(query, key.transpose(-2, -1))
    
    # Step 2: Scale by √d_k
    # Without scaling, large d_k → large dot products → softmax becomes too peaked
    scores = scores / math.sqrt(d_k)
    
    # Step 3: Apply mask (if provided)
    # Mask sets certain positions to -inf so softmax gives them ~0 weight
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    
    # Step 4: Softmax to get attention weights
    # Each row sums to 1 (each query position distributes attention across all keys)
    attention_weights = F.softmax(scores, dim=-1)
    
    # Optional dropout on attention weights
    if dropout is not None:
        attention_weights = dropout(attention_weights)
    
    # Step 5: Weighted sum of values
    # (batch, seq_len, seq_len) @ (batch, seq_len, d_v) → (batch, seq_len, d_v)
    output = torch.matmul(attention_weights, value)
    
    return output, attention_weights


def create_causal_mask(seq_len: int) -> torch.Tensor:
    """
    Create a causal (look-ahead) mask for autoregressive decoding.
    
    Position i can only attend to positions 0, 1, ..., i (not future positions).
    
    Returns:
        mask: (1, seq_len, seq_len) - lower triangular matrix of 1s
    """
    mask = torch.tril(torch.ones(seq_len, seq_len))
    return mask.unsqueeze(0)  # Add batch dimension


def create_padding_mask(seq: torch.Tensor, pad_idx: int = 0) -> torch.Tensor:
    """
    Create a padding mask to ignore padding tokens.
    
    Args:
        seq: (batch, seq_len) - token indices
        pad_idx: Index of the padding token
        
    Returns:
        mask: (batch, 1, seq_len) - 1 for real tokens, 0 for padding
    """
    mask = (seq != pad_idx).unsqueeze(1)
    return mask


# =============================================================================
# Demonstration
# =============================================================================

if __name__ == "__main__":
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    print("=" * 60)
    print("SCALED DOT-PRODUCT ATTENTION DEMONSTRATION")
    print("=" * 60)
    
    # Example: "The cat sat on the mat"
    sentence = ["The", "cat", "sat", "on", "the", "mat"]
    seq_len = len(sentence)
    d_k = 8  # Dimension of queries/keys
    d_v = 8  # Dimension of values
    batch_size = 1
    
    print(f"\n1. SETUP")
    print(f"   Sentence: {sentence}")
    print(f"   Sequence length: {seq_len}")
    print(f"   d_k (query/key dim): {d_k}")
    print(f"   d_v (value dim): {d_v}")
    
    # Create random Q, K, V (in real transformer, these come from linear projections)
    torch.manual_seed(42)
    Q = torch.randn(batch_size, seq_len, d_k)
    K = torch.randn(batch_size, seq_len, d_k)
    V = torch.randn(batch_size, seq_len, d_v)
    
    print(f"\n2. INPUT SHAPES")
    print(f"   Q shape: {Q.shape} - one query vector per position")
    print(f"   K shape: {K.shape} - one key vector per position")
    print(f"   V shape: {V.shape} - one value vector per position")
    
    # Compute attention without mask
    output, attention_weights = scaled_dot_product_attention(Q, K, V)
    
    print(f"\n3. OUTPUT SHAPES")
    print(f"   Attention weights: {attention_weights.shape} - who attends to whom")
    print(f"   Output: {output.shape} - attended representations")
    
    # Show attention weights
    print(f"\n4. ATTENTION WEIGHTS (each row sums to 1)")
    print(f"   Shows how much each position attends to every other position:\n")
    weights = attention_weights[0].detach().numpy()
    
    # Print header
    print("          ", end="")
    for word in sentence:
        print(f"{word:>6}", end=" ")
    print()
    
    # Print weights
    for i, word in enumerate(sentence):
        print(f"   {word:>5} [", end="")
        for j in range(seq_len):
            print(f"{weights[i,j]:.3f}", end=" ")
        print(f"] sum={weights[i].sum():.3f}")
    
    # Demonstrate causal mask (for decoder)
    print(f"\n5. CAUSAL MASK (for autoregressive decoding)")
    causal_mask = create_causal_mask(seq_len)
    print(f"   Each position can only see itself and previous positions:\n")
    print("          ", end="")
    for word in sentence:
        print(f"{word:>5}", end=" ")
    print()
    
    mask_np = causal_mask[0].numpy()
    for i, word in enumerate(sentence):
        print(f"   {word:>5} [", end="")
        for j in range(seq_len):
            print(f"  {int(mask_np[i,j])}", end="  ")
        print("]")
    
    # Compute attention with causal mask
    output_masked, attention_weights_masked = scaled_dot_product_attention(
        Q, K, V, mask=causal_mask
    )
    
    print(f"\n6. ATTENTION WITH CAUSAL MASK")
    print(f"   Future positions get zero attention:\n")
    weights_masked = attention_weights_masked[0].detach().numpy()
    
    print("          ", end="")
    for word in sentence:
        print(f"{word:>6}", end=" ")
    print()
    
    for i, word in enumerate(sentence):
        print(f"   {word:>5} [", end="")
        for j in range(seq_len):
            print(f"{weights_masked[i,j]:.3f}", end=" ")
        print(f"] sum={weights_masked[i].sum():.3f}")
    
    # Create visualization
    print(f"\n7. CREATING VISUALIZATION...")
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot 1: Attention weights without mask
    ax1 = axes[0]
    im1 = ax1.imshow(weights, cmap='Blues', vmin=0, vmax=1)
    ax1.set_xticks(range(seq_len))
    ax1.set_yticks(range(seq_len))
    ax1.set_xticklabels(sentence, rotation=45, ha='right')
    ax1.set_yticklabels(sentence)
    ax1.set_xlabel('Key (attending to)')
    ax1.set_ylabel('Query (from)')
    ax1.set_title('Attention Weights\n(No Mask)')
    plt.colorbar(im1, ax=ax1)
    
    # Plot 2: Causal mask
    ax2 = axes[1]
    im2 = ax2.imshow(mask_np, cmap='Greys', vmin=0, vmax=1)
    ax2.set_xticks(range(seq_len))
    ax2.set_yticks(range(seq_len))
    ax2.set_xticklabels(sentence, rotation=45, ha='right')
    ax2.set_yticklabels(sentence)
    ax2.set_xlabel('Key position')
    ax2.set_ylabel('Query position')
    ax2.set_title('Causal Mask\n(1=can attend, 0=blocked)')
    plt.colorbar(im2, ax=ax2)
    
    # Plot 3: Attention weights with causal mask
    ax3 = axes[2]
    im3 = ax3.imshow(weights_masked, cmap='Blues', vmin=0, vmax=1)
    ax3.set_xticks(range(seq_len))
    ax3.set_yticks(range(seq_len))
    ax3.set_xticklabels(sentence, rotation=45, ha='right')
    ax3.set_yticklabels(sentence)
    ax3.set_xlabel('Key (attending to)')
    ax3.set_ylabel('Query (from)')
    ax3.set_title('Attention Weights\n(With Causal Mask)')
    plt.colorbar(im3, ax=ax3)
    
    plt.tight_layout()
    plt.savefig('attention_visualization.png', dpi=150)
    print(f"   Saved: attention_visualization.png")
    
    # Key takeaways
    print("\n" + "=" * 60)
    print("KEY TAKEAWAYS")
    print("=" * 60)
    print("""
1. Attention computes PAIRWISE RELATIONSHIPS between all positions
   - Q @ K^T gives a (seq_len × seq_len) matrix of scores

2. SCALING by √d_k prevents softmax from becoming too peaked
   - Large dot products → softmax outputs nearly one-hot → bad gradients

3. SOFTMAX converts scores to weights (each row sums to 1)
   - Each position distributes its "attention budget" across all positions

4. OUTPUT is a weighted sum of values
   - Each position's output is a mix of all value vectors

5. MASKING controls what positions can attend to
   - Padding mask: ignore padding tokens
   - Causal mask: prevent looking at future (for autoregressive generation)
""")
