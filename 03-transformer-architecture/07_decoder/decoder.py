"""
Step 7: Decoder Block

The decoder has THREE sub-layers (vs encoder's TWO):
    1. Masked Self-Attention — attend to past tokens only
    2. Cross-Attention — attend to encoder output
    3. Feed-Forward Network — process each position

Architecture:
    
    decoder_input
         │
         ▼
    ┌─────────────────┐
    │ Masked Self-Attn│ ← Can only see past tokens (causal mask)
    │ Q=K=V=decoder   │
    └────────┬────────┘
             │ + residual
             ▼
    ┌─────────────────┐
    │  Cross-Attention│ ← Attends to encoder output
    │ Q=decoder       │
    │ K=V=encoder_out │
    └────────┬────────┘
             │ + residual
             ▼
    ┌─────────────────┐
    │  Feed-Forward   │
    └────────┬────────┘
             │ + residual
             ▼
        output

The causal mask is the key difference from encoder:
    - Encoder: each token sees ALL tokens
    - Decoder: each token sees only PAST tokens (including itself)

This enables autoregressive generation:
    Token 1 → predicts Token 2
    Token 1,2 → predicts Token 3
    Token 1,2,3 → predicts Token 4
    ...
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
create_causal_mask = import_module("02_attention.scaled_dot_product_attention").create_causal_mask

# Feed-Forward Network
ffn_module = import_module("04_feed_forward.feed_forward")
FeedForward = ffn_module.FeedForward
SwiGLUFeedForward = ffn_module.SwiGLUFeedForward

# Layer Normalization
ln_module = import_module("05_layer_norm.layer_norm")
LayerNorm = ln_module.LayerNorm
RMSNorm = ln_module.RMSNorm


class DecoderBlock(nn.Module):
    """
    A single Transformer Decoder Block.
    
    Three sub-layers:
        1. Masked Self-Attention (causal — no future peeking)
        2. Cross-Attention (attend to encoder output)
        3. Feed-Forward Network
    
    Each sub-layer has:
        - Layer normalization (Pre-LN)
        - Residual connection
        - Dropout
    
    Args:
        d_model: Model dimension
        num_heads: Number of attention heads
        d_ff: Feed-forward hidden dimension
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
        
        if d_ff is None:
            d_ff = 4 * d_model
        
        # =====================================================================
        # SUB-LAYER 1: Masked Self-Attention
        # =====================================================================
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        
        # =====================================================================
        # SUB-LAYER 2: Cross-Attention (encoder-decoder attention)
        # =====================================================================
        self.cross_attention = MultiHeadAttention(d_model, num_heads, dropout)
        
        # =====================================================================
        # SUB-LAYER 3: Feed-Forward Network
        # =====================================================================
        if use_swiglu:
            self.feed_forward = SwiGLUFeedForward(d_model, d_ff, dropout)
        else:
            self.feed_forward = FeedForward(d_model, d_ff, dropout)
        
        # =====================================================================
        # LAYER NORMALIZATION (one per sub-layer)
        # =====================================================================
        NormClass = RMSNorm if use_rmsnorm else LayerNorm
        self.norm1 = NormClass(d_model)  # Before self-attention
        self.norm2 = NormClass(d_model)  # Before cross-attention
        self.norm3 = NormClass(d_model)  # Before FFN
        
        # =====================================================================
        # DROPOUT
        # =====================================================================
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        self_attn_mask: torch.Tensor = None,
        cross_attn_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Forward pass through decoder block.
        
        Args:
            x: Decoder input (batch, tgt_seq_len, d_model)
            encoder_output: Encoder output (batch, src_seq_len, d_model)
            self_attn_mask: Causal mask for self-attention
            cross_attn_mask: Optional mask for cross-attention
        
        Returns:
            Output tensor (batch, tgt_seq_len, d_model)
        """
        # ==================================================================
        # SUB-LAYER 1: Masked Self-Attention
        # ==================================================================
        # Decoder can only attend to past positions (causal)
        
        x_norm = self.norm1(x)
        self_attn_output = self.self_attention(
            query=x_norm,
            key=x_norm,
            value=x_norm,
            mask=self_attn_mask  # Causal mask!
        )
        x = x + self.dropout(self_attn_output)
        
        # ==================================================================
        # SUB-LAYER 2: Cross-Attention
        # ==================================================================
        # Query from decoder, Key/Value from encoder
        # This is how decoder "looks at" the source sequence
        
        x_norm = self.norm2(x)
        cross_attn_output = self.cross_attention(
            query=x_norm,           # Decoder asks questions
            key=encoder_output,     # Encoder provides keys
            value=encoder_output,   # Encoder provides values
            mask=cross_attn_mask
        )
        x = x + self.dropout(cross_attn_output)
        
        # ==================================================================
        # SUB-LAYER 3: Feed-Forward
        # ==================================================================
        x_norm = self.norm3(x)
        ff_output = self.feed_forward(x_norm)
        x = x + self.dropout(ff_output)
        
        return x
    
    def get_self_attention_weights(self) -> torch.Tensor:
        """Return self-attention weights from the last forward pass."""
        return self.self_attention.get_attention_weights()
    
    def get_cross_attention_weights(self) -> torch.Tensor:
        """Return cross-attention weights from the last forward pass."""
        return self.cross_attention.get_attention_weights()


class Decoder(nn.Module):
    """
    Full Transformer Decoder: Stack of N decoder blocks.
    
    Args:
        num_layers: Number of decoder blocks
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
        
        # Stack of decoder blocks
        self.layers = nn.ModuleList([
            DecoderBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                dropout=dropout,
                use_swiglu=use_swiglu,
                use_rmsnorm=use_rmsnorm
            )
            for _ in range(num_layers)
        ])
        
        # Final normalization
        NormClass = RMSNorm if use_rmsnorm else LayerNorm
        self.final_norm = NormClass(d_model)
    
    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        self_attn_mask: torch.Tensor = None,
        cross_attn_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Forward pass through all decoder layers.
        
        Args:
            x: Decoder input (batch, tgt_seq_len, d_model)
            encoder_output: Encoder output (batch, src_seq_len, d_model)
            self_attn_mask: Causal mask for self-attention
            cross_attn_mask: Optional mask for cross-attention
        
        Returns:
            Decoded output (batch, tgt_seq_len, d_model)
        """
        for layer in self.layers:
            x = layer(x, encoder_output, self_attn_mask, cross_attn_mask)
        
        x = self.final_norm(x)
        
        return x


class DecoderOnlyBlock(nn.Module):
    """
    Decoder-Only Block (GPT-style).
    
    This is the architecture used by GPT, LLaMA, and most modern LLMs.
    No encoder, no cross-attention — just masked self-attention + FFN.
    
    Architecture:
        x → Masked Self-Attention → FFN → output
        
    The entire model IS the decoder (no separate encoder).
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
        
        if d_ff is None:
            d_ff = 4 * d_model
        
        # Only two sub-layers (no cross-attention)
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        
        if use_swiglu:
            self.feed_forward = SwiGLUFeedForward(d_model, d_ff, dropout)
        else:
            self.feed_forward = FeedForward(d_model, d_ff, dropout)
        
        NormClass = RMSNorm if use_rmsnorm else LayerNorm
        self.norm1 = NormClass(d_model)
        self.norm2 = NormClass(d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Forward pass (decoder-only, no encoder needed).
        
        Args:
            x: Input (batch, seq_len, d_model)
            mask: Causal mask
        
        Returns:
            Output (batch, seq_len, d_model)
        """
        # Masked self-attention
        x_norm = self.norm1(x)
        attn_output = self.self_attention(x_norm, x_norm, x_norm, mask)
        x = x + self.dropout(attn_output)
        
        # Feed-forward
        x_norm = self.norm2(x)
        ff_output = self.feed_forward(x_norm)
        x = x + self.dropout(ff_output)
        
        return x


# =============================================================================
# DEMONSTRATION
# =============================================================================

def demonstrate_decoder():
    """
    Demonstrate the Decoder Block with focus on masking and cross-attention.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    print("=" * 70)
    print("DECODER BLOCK DEMONSTRATION")
    print("=" * 70)
    
    # Configuration
    d_model = 512
    num_heads = 8
    d_ff = 2048
    batch_size = 1
    src_seq_len = 6  # Encoder sequence (source)
    tgt_seq_len = 4  # Decoder sequence (target, being generated)
    
    print(f"\n1. CONFIGURATION")
    print(f"   d_model:     {d_model}")
    print(f"   num_heads:   {num_heads}")
    print(f"   src_seq_len: {src_seq_len} (encoder input)")
    print(f"   tgt_seq_len: {tgt_seq_len} (decoder input)")
    
    # Create decoder block
    decoder_block = DecoderBlock(d_model, num_heads, d_ff, dropout=0.0)
    decoder_block.eval()
    
    # Simulate encoder output
    torch.manual_seed(42)
    encoder_output = torch.randn(batch_size, src_seq_len, d_model)
    
    # Decoder input (tokens generated so far)
    decoder_input = torch.randn(batch_size, tgt_seq_len, d_model)
    
    # Create causal mask for self-attention
    causal_mask = create_causal_mask(tgt_seq_len)
    
    print(f"\n2. INPUTS")
    print(f"   Encoder output: {encoder_output.shape}")
    print(f"   Decoder input:  {decoder_input.shape}")
    print(f"   Causal mask:    {causal_mask.shape}")
    
    # Show causal mask
    print(f"\n3. CAUSAL MASK (self-attention)")
    print(f"   Prevents attending to future positions:")
    print(f"   (1 = can attend, 0 = blocked)")
    
    # causal_mask shape is (1, tgt_seq_len, tgt_seq_len)
    # Values: 1 = can attend, 0 = blocked (will become -inf)
    mask_visual = causal_mask[0].numpy()
    for i in range(tgt_seq_len):
        row = ['✓' if mask_visual[i, j] == 1 else '✗' for j in range(tgt_seq_len)]
        print(f"   Position {i}: [{' '.join(row)}]")
    
    print(f"""
   Reading: Position 0 can see [0]
            Position 1 can see [0, 1]
            Position 2 can see [0, 1, 2]
            Position 3 can see [0, 1, 2, 3]
    """)
    
    # Forward pass
    print(f"\n4. FORWARD PASS")
    
    with torch.no_grad():
        output = decoder_block(
            x=decoder_input,
            encoder_output=encoder_output,
            self_attn_mask=causal_mask
        )
    
    print(f"   Output shape: {output.shape}")
    
    # Attention weights
    self_attn = decoder_block.get_self_attention_weights()
    cross_attn = decoder_block.get_cross_attention_weights()
    
    print(f"\n5. ATTENTION PATTERNS")
    print(f"   Self-attention:  {self_attn.shape}")
    print(f"     (batch, heads, tgt_seq, tgt_seq)")
    print(f"   Cross-attention: {cross_attn.shape}")
    print(f"     (batch, heads, tgt_seq, src_seq)")
    
    # Visualize attention patterns
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Self-attention (head 0)
    ax = axes[0]
    self_attn_h0 = self_attn[0, 0].detach().numpy()
    im = ax.imshow(self_attn_h0, cmap='Blues', vmin=0, vmax=1)
    ax.set_xlabel('Key (attending to)')
    ax.set_ylabel('Query (from)')
    ax.set_title('Self-Attention (Causal)\nHead 1', fontweight='bold')
    ax.set_xticks(range(tgt_seq_len))
    ax.set_yticks(range(tgt_seq_len))
    ax.set_xticklabels([f'tgt_{i}' for i in range(tgt_seq_len)])
    ax.set_yticklabels([f'tgt_{i}' for i in range(tgt_seq_len)])
    plt.colorbar(im, ax=ax)
    
    # Cross-attention (head 0)
    ax = axes[1]
    cross_attn_h0 = cross_attn[0, 0].detach().numpy()
    im = ax.imshow(cross_attn_h0, cmap='Greens', vmin=0, vmax=1)
    ax.set_xlabel('Key (encoder positions)')
    ax.set_ylabel('Query (decoder positions)')
    ax.set_title('Cross-Attention\nHead 1', fontweight='bold')
    ax.set_xticks(range(src_seq_len))
    ax.set_yticks(range(tgt_seq_len))
    ax.set_xticklabels([f'src_{i}' for i in range(src_seq_len)])
    ax.set_yticklabels([f'tgt_{i}' for i in range(tgt_seq_len)])
    plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    plt.savefig('decoder_attention.png', dpi=150, bbox_inches='tight')
    print(f"\n   Saved: decoder_attention.png")
    
    # Compare architectures
    print(f"\n6. ARCHITECTURE COMPARISON")
    print(f"""
    ENCODER (BERT-style)         DECODER (with encoder)       DECODER-ONLY (GPT-style)
    ─────────────────────        ────────────────────────     ────────────────────────
    Self-Attention               Masked Self-Attention        Masked Self-Attention
    (bidirectional)              (causal)                     (causal)
         │                            │                            │
         ▼                            ▼                            ▼
        FFN                      Cross-Attention                  FFN
                                 (to encoder)                
                                      │                       
                                      ▼                       
                                     FFN                      
    
    Use case:                    Use case:                    Use case:
    - Understanding              - Translation                - Text generation
    - Classification             - Summarization              - Code completion
    - NER, QA                    - Any seq2seq                - Chat/dialogue
    """)
    
    # Decoder-only example
    print(f"\n7. DECODER-ONLY (GPT-style)")
    
    decoder_only = DecoderOnlyBlock(d_model, num_heads, d_ff, dropout=0.0)
    decoder_only.eval()
    
    gpt_input = torch.randn(batch_size, 10, d_model)
    gpt_mask = create_causal_mask(10)
    
    with torch.no_grad():
        gpt_output = decoder_only(gpt_input, gpt_mask)
    
    print(f"   Input:  {gpt_input.shape}")
    print(f"   Output: {gpt_output.shape}")
    print(f"   No encoder needed! Just causal self-attention + FFN.")
    
    # Parameter comparison
    print(f"\n8. PARAMETER COUNT COMPARISON")
    
    decoder_block_params = sum(p.numel() for p in decoder_block.parameters())
    decoder_only_params = sum(p.numel() for p in decoder_only.parameters())
    
    print(f"   Full decoder block (with cross-attn): {decoder_block_params:,}")
    print(f"   Decoder-only block (GPT-style):       {decoder_only_params:,}")
    print(f"   Difference: {decoder_block_params - decoder_only_params:,} (the cross-attention)")
    
    print("\n" + "=" * 70)
    print("KEY TAKEAWAYS")
    print("=" * 70)
    print("""
1. DECODER HAS THREE SUB-LAYERS (vs encoder's two)
   - Masked self-attention (causal)
   - Cross-attention (to encoder)
   - Feed-forward

2. CAUSAL MASK prevents future peeking
   - Position i can only see positions 0, 1, ..., i
   - Enables autoregressive generation
   - Upper triangle of attention matrix is -inf

3. CROSS-ATTENTION connects decoder to encoder
   - Query: from decoder (what am I looking for?)
   - Key, Value: from encoder (what does the source say?)
   - Lets decoder "look at" the source sequence

4. DECODER-ONLY (GPT-style) is simpler
   - No encoder, no cross-attention
   - Just masked self-attention + FFN
   - Used by GPT, LLaMA, Claude, etc.

5. USE CASES
   - Encoder-only (BERT): classification, understanding
   - Encoder-decoder: translation, summarization
   - Decoder-only (GPT): generation, chat, code
""")


def test_decoder():
    """Test decoder implementation."""
    print("\n" + "=" * 70)
    print("TESTING DECODER")
    print("=" * 70)
    
    d_model = 512
    num_heads = 8
    d_ff = 2048
    num_layers = 6
    batch_size = 2
    src_seq_len = 10
    tgt_seq_len = 8
    
    # Test 1: DecoderBlock output shape
    print("\n✓ Test 1: DecoderBlock output shape")
    block = DecoderBlock(d_model, num_heads, d_ff, dropout=0.0)
    block.eval()
    
    encoder_output = torch.randn(batch_size, src_seq_len, d_model)
    decoder_input = torch.randn(batch_size, tgt_seq_len, d_model)
    causal_mask = create_causal_mask(tgt_seq_len)
    
    with torch.no_grad():
        output = block(decoder_input, encoder_output, causal_mask)
    
    assert output.shape == decoder_input.shape
    print(f"   Decoder input: {decoder_input.shape}")
    print(f"   Encoder output: {encoder_output.shape}")
    print(f"   Output: {output.shape} ✓")
    
    # Test 2: Full decoder stack
    print("\n✓ Test 2: Decoder (stack) output shape")
    decoder = Decoder(num_layers, d_model, num_heads, d_ff, dropout=0.0)
    decoder.eval()
    
    with torch.no_grad():
        output = decoder(decoder_input, encoder_output, causal_mask)
    
    assert output.shape == decoder_input.shape
    print(f"   {num_layers} layers: {decoder_input.shape} → {output.shape} ✓")
    
    # Test 3: Attention weights shapes
    print("\n✓ Test 3: Attention weights shapes")
    
    self_attn = block.get_self_attention_weights()
    cross_attn = block.get_cross_attention_weights()
    
    assert self_attn.shape == (batch_size, num_heads, tgt_seq_len, tgt_seq_len)
    assert cross_attn.shape == (batch_size, num_heads, tgt_seq_len, src_seq_len)
    print(f"   Self-attention: {self_attn.shape} ✓")
    print(f"   Cross-attention: {cross_attn.shape} ✓")
    
    # Test 4: Causal mask works (no future attention)
    print("\n✓ Test 4: Causal mask prevents future attention")
    
    for i in range(tgt_seq_len):
        for j in range(i + 1, tgt_seq_len):
            # Future positions should have ~0 attention
            attn_val = self_attn[:, :, i, j].abs().max().item()
            assert attn_val < 1e-5, f"Position ({i},{j}) should be masked"
    print(f"   No attention to future positions ✓")
    
    # Test 5: Decoder-only block
    print("\n✓ Test 5: DecoderOnlyBlock (GPT-style)")
    gpt_block = DecoderOnlyBlock(d_model, num_heads, d_ff, dropout=0.0)
    gpt_block.eval()
    
    gpt_input = torch.randn(batch_size, tgt_seq_len, d_model)
    gpt_mask = create_causal_mask(tgt_seq_len)
    
    with torch.no_grad():
        gpt_output = gpt_block(gpt_input, gpt_mask)
    
    assert gpt_output.shape == gpt_input.shape
    print(f"   GPT-style: {gpt_input.shape} → {gpt_output.shape} ✓")
    
    # Test 6: Gradients flow
    print("\n✓ Test 6: Gradients flow through decoder")
    decoder_grad = Decoder(num_layers, d_model, num_heads, d_ff, dropout=0.0)
    enc_out = torch.randn(batch_size, src_seq_len, d_model, requires_grad=True)
    dec_in = torch.randn(batch_size, tgt_seq_len, d_model, requires_grad=True)
    
    output = decoder_grad(dec_in, enc_out, causal_mask)
    loss = output.sum()
    loss.backward()
    
    assert dec_in.grad is not None
    assert enc_out.grad is not None  # Gradient flows to encoder too!
    print(f"   Gradients flow to decoder input ✓")
    print(f"   Gradients flow to encoder output ✓")
    
    # Test 7: Modern variants
    print("\n✓ Test 7: Modern variants (SwiGLU + RMSNorm)")
    decoder_modern = Decoder(
        num_layers, d_model, num_heads, d_ff,
        use_swiglu=True, use_rmsnorm=True, dropout=0.0
    )
    decoder_modern.eval()
    
    with torch.no_grad():
        out_modern = decoder_modern(decoder_input, encoder_output, causal_mask)
    
    assert out_modern.shape == decoder_input.shape
    print(f"   SwiGLU + RMSNorm works ✓")
    
    print("\n" + "=" * 70)
    print("ALL TESTS PASSED ✓")
    print("=" * 70)


if __name__ == "__main__":
    test_decoder()
    demonstrate_decoder()
