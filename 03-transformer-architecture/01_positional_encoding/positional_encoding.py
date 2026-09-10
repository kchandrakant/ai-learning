"""
Step 1: Positional Encoding

The transformer processes all tokens in parallel, so it has no inherent sense
of word order. We need to inject position information into the embeddings.

The original paper uses sinusoidal positional encoding:
    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

Key insight: Different dimensions oscillate at different frequencies.
- Early dimensions (small i): High frequency, captures fine position differences
- Later dimensions (large i): Low frequency, captures coarse position patterns

This creates a unique "fingerprint" for each position that the model can learn to use.
"""

import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """
    Sinusoidal Positional Encoding from "Attention Is All You Need" (2017).
    
    This is ADDED to the token embeddings, not concatenated.
    The result: each token embedding now "knows" its position in the sequence.
    """
    
    def __init__(self, d_model: int, max_seq_len: int = 5000, dropout: float = 0.1):
        """
        Args:
            d_model: Dimension of the model (embedding size). Must be even.
            max_seq_len: Maximum sequence length to pre-compute encodings for.
            dropout: Dropout probability applied after adding positional encoding.
        """
        super().__init__()
        
        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)
        
        # Create a matrix of shape (max_seq_len, d_model) to store all encodings
        pe = torch.zeros(max_seq_len, d_model)
        
        # Position indices: [0, 1, 2, ..., max_seq_len-1]
        # Shape: (max_seq_len, 1) - column vector for broadcasting
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        
        # Dimension indices for the divisor term
        # We only need d_model/2 values because each creates both sin and cos
        # div_term = 10000^(2i/d_model) = exp(2i * -log(10000)/d_model)
        # Using exp/log is more numerically stable than power
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        # Apply sin to even indices (0, 2, 4, ...)
        pe[:, 0::2] = torch.sin(position * div_term)
        
        # Apply cos to odd indices (1, 3, 5, ...)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add batch dimension: (max_seq_len, d_model) -> (1, max_seq_len, d_model)
        pe = pe.unsqueeze(0)
        
        # Register as buffer (not a parameter - won't be updated during training)
        # Buffers are saved with the model and moved to GPU with .to(device)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input embeddings.
        
        Args:
            x: Input embeddings of shape (batch_size, seq_len, d_model)
            
        Returns:
            Embeddings with positional information added, same shape as input
        """
        seq_len = x.size(1)
        
        # Add positional encoding (broadcasting handles batch dimension)
        # pe[:, :seq_len] selects only the positions we need
        x = x + self.pe[:, :seq_len, :]
        
        return self.dropout(x)


# =============================================================================
# Let's test and visualize!
# =============================================================================

if __name__ == "__main__":
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    print("=" * 60)
    print("POSITIONAL ENCODING DEMONSTRATION")
    print("=" * 60)
    
    # Create a positional encoding layer
    d_model = 64  # Small for visualization
    max_len = 100
    pe_layer = PositionalEncoding(d_model=d_model, max_seq_len=max_len, dropout=0.0)
    
    # Get the raw positional encodings (without dropout, without input)
    pe_values = pe_layer.pe.squeeze(0).numpy()  # Shape: (max_len, d_model)
    
    print(f"\n1. ENCODING SHAPE")
    print(f"   Positional encoding matrix: {pe_values.shape}")
    print(f"   (max_seq_len={max_len}, d_model={d_model})")
    
    # Show encoding for first few positions
    print(f"\n2. SAMPLE ENCODINGS (first 8 dimensions)")
    print(f"   Position 0: {pe_values[0, :8].round(3)}")
    print(f"   Position 1: {pe_values[1, :8].round(3)}")
    print(f"   Position 2: {pe_values[2, :8].round(3)}")
    
    # Demonstrate how it's used with actual embeddings
    print(f"\n3. USAGE WITH EMBEDDINGS")
    batch_size = 2
    seq_len = 10
    
    # Simulate token embeddings (normally from nn.Embedding)
    fake_embeddings = torch.randn(batch_size, seq_len, d_model)
    print(f"   Input embedding shape: {fake_embeddings.shape}")
    
    # Add positional encoding
    output = pe_layer(fake_embeddings)
    print(f"   Output shape (unchanged): {output.shape}")
    
    # Verify the encoding is position-dependent
    print(f"\n4. POSITION UNIQUENESS CHECK")
    # Compare encodings at different positions
    pos_0 = pe_values[0]
    pos_1 = pe_values[1]
    pos_50 = pe_values[50]
    
    dist_0_1 = torch.norm(torch.tensor(pos_0 - pos_1)).item()
    dist_0_50 = torch.norm(torch.tensor(pos_0 - pos_50)).item()
    print(f"   Distance between pos 0 and pos 1: {dist_0_1:.4f}")
    print(f"   Distance between pos 0 and pos 50: {dist_0_50:.4f}")
    print(f"   (Nearby positions are more similar than distant ones)")
    
    # Create visualization
    print(f"\n5. CREATING VISUALIZATION...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Heatmap of positional encodings
    ax1 = axes[0, 0]
    im1 = ax1.imshow(pe_values[:50, :], aspect='auto', cmap='RdBu', vmin=-1, vmax=1)
    ax1.set_xlabel('Dimension')
    ax1.set_ylabel('Position')
    ax1.set_title('Positional Encoding Heatmap\n(First 50 positions)')
    plt.colorbar(im1, ax=ax1)
    
    # Plot 2: Individual dimension curves
    ax2 = axes[0, 1]
    positions = range(100)
    ax2.plot(positions, pe_values[:, 0], label='dim 0 (sin, high freq)', alpha=0.8)
    ax2.plot(positions, pe_values[:, 1], label='dim 1 (cos, high freq)', alpha=0.8)
    ax2.plot(positions, pe_values[:, 20], label='dim 20 (sin, med freq)', alpha=0.8)
    ax2.plot(positions, pe_values[:, 62], label='dim 62 (sin, low freq)', alpha=0.8)
    ax2.set_xlabel('Position')
    ax2.set_ylabel('Encoding Value')
    ax2.set_title('Different Dimensions = Different Frequencies')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Encoding vectors for specific positions
    ax3 = axes[1, 0]
    for pos in [0, 5, 10, 25, 50]:
        ax3.plot(pe_values[pos, :], label=f'Position {pos}', alpha=0.7)
    ax3.set_xlabel('Dimension')
    ax3.set_ylabel('Encoding Value')
    ax3.set_title('Encoding Vectors for Different Positions')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Similarity matrix (dot product between positions)
    ax4 = axes[1, 1]
    similarity = pe_values[:30] @ pe_values[:30].T
    im4 = ax4.imshow(similarity, cmap='viridis')
    ax4.set_xlabel('Position')
    ax4.set_ylabel('Position')
    ax4.set_title('Position Similarity Matrix\n(Dot product of encodings)')
    plt.colorbar(im4, ax=ax4)
    
    plt.tight_layout()
    plt.savefig('01_positional_encoding/positional_encoding_visualization.png', dpi=150)
    print(f"   Saved: 01_positional_encoding/positional_encoding_visualization.png")
    
    print("\n" + "=" * 60)
    print("KEY TAKEAWAYS:")
    print("=" * 60)
    print("""
1. Each position gets a UNIQUE encoding vector
2. Low dimensions = high frequency (fine detail)
3. High dimensions = low frequency (coarse patterns)  
4. Nearby positions have similar encodings (smooth)
5. The encoding is ADDED to embeddings, not concatenated
6. It's pre-computed once and reused (efficient!)
""")
