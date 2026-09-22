"""
Step 8: Complete Transformer

This module brings everything together into complete, working transformers:

1. ENCODER-DECODER TRANSFORMER (Original paper)
   - For sequence-to-sequence tasks (translation, summarization)
   - Encoder processes source, decoder generates target
   
2. DECODER-ONLY TRANSFORMER (GPT-style)
   - For language modeling and generation
   - No encoder, just masked self-attention
   - This is what GPT, LLaMA, Claude, Mistral use

Architecture components:
    - Token embeddings (vocab → d_model)
    - Positional encoding (add position information)
    - Encoder/Decoder stacks
    - Output projection (d_model → vocab)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our implementations
from importlib import import_module

# Positional Encoding
pe_module = import_module("01_positional_encoding.positional_encoding")
PositionalEncoding = pe_module.PositionalEncoding

# Encoder
encoder_module = import_module("06_encoder.encoder")
Encoder = encoder_module.Encoder

# Decoder
decoder_module = import_module("07_decoder.decoder")
Decoder = decoder_module.Decoder
DecoderOnlyBlock = decoder_module.DecoderOnlyBlock

# For masks
attention_module = import_module("02_attention.scaled_dot_product_attention")
create_causal_mask = attention_module.create_causal_mask

# Layer norms
ln_module = import_module("05_layer_norm.layer_norm")
LayerNorm = ln_module.LayerNorm
RMSNorm = ln_module.RMSNorm


class Transformer(nn.Module):
    """
    Full Encoder-Decoder Transformer (original "Attention Is All You Need").
    
    Use cases:
        - Machine translation (English → French)
        - Summarization (article → summary)
        - Any sequence-to-sequence task
    
    Architecture:
        Source → Embedding + PE → Encoder → encoded representation
        Target → Embedding + PE → Decoder (with cross-attention) → Output → Logits
    
    Args:
        src_vocab_size: Source vocabulary size
        tgt_vocab_size: Target vocabulary size
        d_model: Model dimension (embedding size)
        num_heads: Number of attention heads
        num_encoder_layers: Number of encoder blocks
        num_decoder_layers: Number of decoder blocks
        d_ff: Feed-forward hidden dimension
        max_seq_len: Maximum sequence length
        dropout: Dropout probability
        use_swiglu: Use SwiGLU instead of ReLU
        use_rmsnorm: Use RMSNorm instead of LayerNorm
    """
    
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 512,
        num_heads: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        d_ff: int = 2048,
        max_seq_len: int = 5000,
        dropout: float = 0.1,
        use_swiglu: bool = False,
        use_rmsnorm: bool = False
    ):
        super().__init__()
        
        self.d_model = d_model
        
        # =====================================================================
        # EMBEDDINGS
        # =====================================================================
        # Convert token IDs to vectors
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        
        # Positional encoding (shared or separate)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_len, dropout)
        
        # =====================================================================
        # ENCODER
        # =====================================================================
        self.encoder = Encoder(
            num_layers=num_encoder_layers,
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            dropout=dropout,
            use_swiglu=use_swiglu,
            use_rmsnorm=use_rmsnorm
        )
        
        # =====================================================================
        # DECODER
        # =====================================================================
        self.decoder = Decoder(
            num_layers=num_decoder_layers,
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            dropout=dropout,
            use_swiglu=use_swiglu,
            use_rmsnorm=use_rmsnorm
        )
        
        # =====================================================================
        # OUTPUT PROJECTION
        # =====================================================================
        # Project d_model → vocab_size for next token prediction
        self.output_projection = nn.Linear(d_model, tgt_vocab_size)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with small values."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def encode(
        self,
        src: torch.Tensor,
        src_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Encode source sequence.
        
        Args:
            src: Source token IDs (batch, src_seq_len)
            src_mask: Optional mask for padding
        
        Returns:
            Encoded representation (batch, src_seq_len, d_model)
        """
        # Embed and scale
        x = self.src_embedding(src) * math.sqrt(self.d_model)
        
        # Add positional encoding
        x = self.positional_encoding(x)
        
        # Pass through encoder
        encoded = self.encoder(x, src_mask)
        
        return encoded
    
    def decode(
        self,
        tgt: torch.Tensor,
        encoder_output: torch.Tensor,
        tgt_mask: torch.Tensor = None,
        cross_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Decode target sequence given encoder output.
        
        Args:
            tgt: Target token IDs (batch, tgt_seq_len)
            encoder_output: Encoder output (batch, src_seq_len, d_model)
            tgt_mask: Causal mask for target self-attention
            cross_mask: Optional mask for cross-attention
        
        Returns:
            Decoded representation (batch, tgt_seq_len, d_model)
        """
        # Embed and scale
        x = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        
        # Add positional encoding
        x = self.positional_encoding(x)
        
        # Pass through decoder
        decoded = self.decoder(x, encoder_output, tgt_mask, cross_mask)
        
        return decoded
    
    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: torch.Tensor = None,
        tgt_mask: torch.Tensor = None,
        cross_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Full forward pass: encode source, decode target, project to logits.
        
        Args:
            src: Source token IDs (batch, src_seq_len)
            tgt: Target token IDs (batch, tgt_seq_len)
            src_mask: Optional mask for source padding
            tgt_mask: Causal mask for target (auto-created if None)
            cross_mask: Optional mask for cross-attention
        
        Returns:
            Logits over target vocabulary (batch, tgt_seq_len, tgt_vocab_size)
        """
        # Create causal mask if not provided
        if tgt_mask is None:
            tgt_seq_len = tgt.size(1)
            tgt_mask = create_causal_mask(tgt_seq_len).to(tgt.device)
        
        # Encode source
        encoder_output = self.encode(src, src_mask)
        
        # Decode target
        decoder_output = self.decode(tgt, encoder_output, tgt_mask, cross_mask)
        
        # Project to vocabulary
        logits = self.output_projection(decoder_output)
        
        return logits


class GPTTransformer(nn.Module):
    """
    Decoder-Only Transformer (GPT-style).
    
    This is the architecture used by:
        - GPT-2, GPT-3, GPT-4
        - LLaMA, LLaMA 2
        - Mistral
        - Claude
        - Most modern LLMs
    
    Use cases:
        - Text generation
        - Code completion
        - Chat/dialogue
        - Any autoregressive task
    
    Architecture:
        Input → Embedding + PE → Decoder × N → Norm → Output Projection → Logits
    
    Args:
        vocab_size: Vocabulary size
        d_model: Model dimension
        num_heads: Number of attention heads
        num_layers: Number of decoder blocks
        d_ff: Feed-forward hidden dimension
        max_seq_len: Maximum sequence length
        dropout: Dropout probability
        use_swiglu: Use SwiGLU (LLaMA-style)
        use_rmsnorm: Use RMSNorm (LLaMA-style)
        tie_weights: Tie input and output embeddings
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 768,
        num_heads: int = 12,
        num_layers: int = 12,
        d_ff: int = None,
        max_seq_len: int = 2048,
        dropout: float = 0.1,
        use_swiglu: bool = False,
        use_rmsnorm: bool = False,
        tie_weights: bool = True
    ):
        super().__init__()
        
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        
        if d_ff is None:
            d_ff = 4 * d_model
        
        # =====================================================================
        # EMBEDDINGS
        # =====================================================================
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_len, dropout)
        
        # =====================================================================
        # DECODER BLOCKS (no cross-attention)
        # =====================================================================
        self.layers = nn.ModuleList([
            DecoderOnlyBlock(
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
        
        # =====================================================================
        # OUTPUT PROJECTION
        # =====================================================================
        self.output_projection = nn.Linear(d_model, vocab_size, bias=False)
        
        # Tie weights (embedding and output projection share weights)
        if tie_weights:
            self.output_projection.weight = self.token_embedding.weight
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Forward pass for language modeling.
        
        Args:
            input_ids: Token IDs (batch, seq_len)
            mask: Causal mask (auto-created if None)
        
        Returns:
            Logits over vocabulary (batch, seq_len, vocab_size)
        """
        seq_len = input_ids.size(1)
        
        # Create causal mask if not provided
        if mask is None:
            mask = create_causal_mask(seq_len).to(input_ids.device)
        
        # Embed tokens
        x = self.token_embedding(input_ids) * math.sqrt(self.d_model)
        
        # Add positional encoding
        x = self.positional_encoding(x)
        
        # Pass through decoder layers
        for layer in self.layers:
            x = layer(x, mask)
        
        # Final normalization
        x = self.final_norm(x)
        
        # Project to vocabulary
        logits = self.output_projection(x)
        
        return logits
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: int = None
    ) -> torch.Tensor:
        """
        Generate tokens autoregressively.
        
        Args:
            input_ids: Starting tokens (batch, seq_len)
            max_new_tokens: How many new tokens to generate
            temperature: Sampling temperature (1.0 = normal, <1 = more focused)
            top_k: Only sample from top k tokens
        
        Returns:
            Generated token IDs (batch, seq_len + max_new_tokens)
        """
        self.eval()
        
        for _ in range(max_new_tokens):
            # Truncate if too long
            if input_ids.size(1) > self.max_seq_len:
                input_ids = input_ids[:, -self.max_seq_len:]
            
            # Get predictions
            logits = self.forward(input_ids)
            
            # Take logits for last position
            logits = logits[:, -1, :] / temperature
            
            # Optional top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            
            # Sample from distribution
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to sequence
            input_ids = torch.cat([input_ids, next_token], dim=1)
        
        return input_ids


# =============================================================================
# DEMONSTRATION
# =============================================================================

def demonstrate_transformer():
    """
    Demonstrate both transformer architectures.
    """
    print("=" * 70)
    print("COMPLETE TRANSFORMER DEMONSTRATION")
    print("=" * 70)
    
    # =========================================================================
    # Part 1: Encoder-Decoder Transformer
    # =========================================================================
    print("\n" + "=" * 70)
    print("PART 1: ENCODER-DECODER TRANSFORMER (Translation)")
    print("=" * 70)
    
    # Configuration
    src_vocab_size = 10000  # English vocabulary
    tgt_vocab_size = 12000  # French vocabulary
    d_model = 512
    num_heads = 8
    num_layers = 6
    
    print(f"\n1. CONFIGURATION")
    print(f"   src_vocab_size: {src_vocab_size}")
    print(f"   tgt_vocab_size: {tgt_vocab_size}")
    print(f"   d_model:        {d_model}")
    print(f"   num_heads:      {num_heads}")
    print(f"   num_layers:     {num_layers} (encoder) + {num_layers} (decoder)")
    
    # Create model
    transformer = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_encoder_layers=num_layers,
        num_decoder_layers=num_layers,
        dropout=0.0
    )
    transformer.eval()
    
    # Simulated translation: "The cat sat" → "Le chat assis"
    batch_size = 2
    src_seq_len = 5
    tgt_seq_len = 4
    
    # Random token IDs (in real use, these come from tokenizer)
    torch.manual_seed(42)
    src = torch.randint(0, src_vocab_size, (batch_size, src_seq_len))
    tgt = torch.randint(0, tgt_vocab_size, (batch_size, tgt_seq_len))
    
    print(f"\n2. INPUT SHAPES")
    print(f"   Source (English): {src.shape}")
    print(f"   Target (French):  {tgt.shape}")
    
    # Forward pass
    with torch.no_grad():
        logits = transformer(src, tgt)
    
    print(f"\n3. OUTPUT")
    print(f"   Logits shape: {logits.shape}")
    print(f"   Interpretation: {batch_size} batches × {tgt_seq_len} positions × {tgt_vocab_size} vocab")
    
    # Parameter count
    total_params = sum(p.numel() for p in transformer.parameters())
    print(f"\n4. PARAMETERS: {total_params:,}")
    
    # =========================================================================
    # Part 2: Decoder-Only Transformer (GPT-style)
    # =========================================================================
    print("\n" + "=" * 70)
    print("PART 2: DECODER-ONLY TRANSFORMER (GPT-style)")
    print("=" * 70)
    
    vocab_size = 50000
    d_model_gpt = 768
    num_heads_gpt = 12
    num_layers_gpt = 12
    
    print(f"\n1. CONFIGURATION (GPT-2 small scale)")
    print(f"   vocab_size:  {vocab_size}")
    print(f"   d_model:     {d_model_gpt}")
    print(f"   num_heads:   {num_heads_gpt}")
    print(f"   num_layers:  {num_layers_gpt}")
    
    gpt = GPTTransformer(
        vocab_size=vocab_size,
        d_model=d_model_gpt,
        num_heads=num_heads_gpt,
        num_layers=num_layers_gpt,
        dropout=0.0,
        use_swiglu=False,
        use_rmsnorm=False,
        tie_weights=True
    )
    gpt.eval()
    
    # Input: "The cat sat on the"
    seq_len = 10
    input_ids = torch.randint(0, vocab_size, (1, seq_len))
    
    print(f"\n2. INPUT")
    print(f"   Shape: {input_ids.shape}")
    
    # Forward pass
    with torch.no_grad():
        logits = gpt(input_ids)
    
    print(f"\n3. OUTPUT")
    print(f"   Logits shape: {logits.shape}")
    
    # Generation example
    print(f"\n4. GENERATION")
    
    # Start with a short prompt
    prompt = torch.randint(0, vocab_size, (1, 5))
    print(f"   Prompt shape: {prompt.shape}")
    
    generated = gpt.generate(prompt, max_new_tokens=10, temperature=1.0, top_k=50)
    print(f"   Generated shape: {generated.shape}")
    print(f"   (Added {generated.shape[1] - prompt.shape[1]} new tokens)")
    
    # Parameter count
    gpt_params = sum(p.numel() for p in gpt.parameters())
    print(f"\n5. PARAMETERS: {gpt_params:,}")
    
    # =========================================================================
    # Part 3: Modern Variant (LLaMA-style)
    # =========================================================================
    print("\n" + "=" * 70)
    print("PART 3: MODERN VARIANT (LLaMA-style)")
    print("=" * 70)
    
    llama = GPTTransformer(
        vocab_size=vocab_size,
        d_model=d_model_gpt,
        num_heads=num_heads_gpt,
        num_layers=num_layers_gpt,
        dropout=0.0,
        use_swiglu=True,   # LLaMA uses SwiGLU
        use_rmsnorm=True,  # LLaMA uses RMSNorm
        tie_weights=True
    )
    llama.eval()
    
    with torch.no_grad():
        logits_llama = llama(input_ids)
    
    llama_params = sum(p.numel() for p in llama.parameters())
    
    print(f"\n   Uses SwiGLU + RMSNorm (like LLaMA)")
    print(f"   Output shape: {logits_llama.shape}")
    print(f"   Parameters: {llama_params:,}")
    print(f"   (More params due to SwiGLU's 3 matrices vs 2)")
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("ARCHITECTURE COMPARISON")
    print("=" * 70)
    print(f"""
    ┌─────────────────────────────────────────────────────────────────┐
    │ Architecture          │ Components        │ Use Case            │
    ├───────────────────────┼───────────────────┼─────────────────────┤
    │ Encoder-Decoder       │ Encoder + Decoder │ Translation         │
    │ (Original Transformer)│ + Cross-Attention │ Summarization       │
    │                       │                   │ Seq2Seq tasks       │
    ├───────────────────────┼───────────────────┼─────────────────────┤
    │ Decoder-Only (GPT)    │ Decoder only      │ Text generation     │
    │                       │ Masked self-attn  │ Code completion     │
    │                       │ No encoder        │ Chat, dialogue      │
    ├───────────────────────┼───────────────────┼─────────────────────┤
    │ Modern (LLaMA)        │ Decoder-only      │ Same as GPT         │
    │                       │ + SwiGLU          │ Better quality      │
    │                       │ + RMSNorm         │ Faster training     │
    │                       │ + RoPE (optional) │                     │
    └───────────────────────┴───────────────────┴─────────────────────┘
    
    Parameter counts:
      Encoder-Decoder (6+6 layers, d=512):   {total_params:,}
      GPT-style (12 layers, d=768):          {gpt_params:,}
      LLaMA-style (12 layers, d=768):        {llama_params:,}
    """)
    
    print("\n" + "=" * 70)
    print("CONGRATULATIONS! YOU'VE BUILT A TRANSFORMER FROM SCRATCH!")
    print("=" * 70)
    print("""
    Components implemented:
    ✓ Positional Encoding (sinusoidal)
    ✓ Scaled Dot-Product Attention
    ✓ Multi-Head Attention
    ✓ Feed-Forward Network (ReLU + SwiGLU)
    ✓ Layer Normalization (LayerNorm + RMSNorm)
    ✓ Encoder Block + Full Encoder
    ✓ Decoder Block + Full Decoder
    ✓ Complete Transformer (Encoder-Decoder + Decoder-Only)
    
    What's next?
    - Try training on a small dataset
    - Explore the 'beyond/' folder for RoPE, GQA, KV-cache
    - Move to Course 04: LLM Internals
    """)


def test_transformer():
    """Test transformer implementations."""
    print("\n" + "=" * 70)
    print("TESTING TRANSFORMERS")
    print("=" * 70)
    
    # Test 1: Encoder-Decoder Transformer
    print("\n✓ Test 1: Encoder-Decoder Transformer")
    
    transformer = Transformer(
        src_vocab_size=1000,
        tgt_vocab_size=1000,
        d_model=256,
        num_heads=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        dropout=0.0
    )
    transformer.eval()
    
    src = torch.randint(0, 1000, (2, 10))
    tgt = torch.randint(0, 1000, (2, 8))
    
    with torch.no_grad():
        logits = transformer(src, tgt)
    
    assert logits.shape == (2, 8, 1000)
    print(f"   src: {src.shape}, tgt: {tgt.shape} → logits: {logits.shape} ✓")
    
    # Test 2: GPT Transformer
    print("\n✓ Test 2: GPT Transformer (Decoder-Only)")
    
    gpt = GPTTransformer(
        vocab_size=1000,
        d_model=256,
        num_heads=4,
        num_layers=4,
        dropout=0.0
    )
    gpt.eval()
    
    input_ids = torch.randint(0, 1000, (2, 15))
    
    with torch.no_grad():
        logits = gpt(input_ids)
    
    assert logits.shape == (2, 15, 1000)
    print(f"   input: {input_ids.shape} → logits: {logits.shape} ✓")
    
    # Test 3: Generation
    print("\n✓ Test 3: Autoregressive generation")
    
    prompt = torch.randint(0, 1000, (1, 5))
    generated = gpt.generate(prompt, max_new_tokens=10, temperature=1.0)
    
    assert generated.shape == (1, 15)  # 5 prompt + 10 generated
    print(f"   prompt: {prompt.shape} → generated: {generated.shape} ✓")
    
    # Test 4: Gradients
    print("\n✓ Test 4: Gradients flow")
    
    gpt_train = GPTTransformer(vocab_size=1000, d_model=128, num_heads=2, num_layers=2)
    input_ids = torch.randint(0, 1000, (2, 10))
    
    logits = gpt_train(input_ids)
    loss = logits.sum()
    loss.backward()
    
    # Check that embedding has gradient
    assert gpt_train.token_embedding.weight.grad is not None
    print(f"   Gradients computed ✓")
    
    # Test 5: Modern variant
    print("\n✓ Test 5: LLaMA-style (SwiGLU + RMSNorm)")
    
    llama = GPTTransformer(
        vocab_size=1000,
        d_model=256,
        num_heads=4,
        num_layers=4,
        use_swiglu=True,
        use_rmsnorm=True,
        dropout=0.0
    )
    llama.eval()
    
    with torch.no_grad():
        logits = llama(input_ids)
    
    assert logits.shape == (2, 10, 1000)
    print(f"   SwiGLU + RMSNorm works ✓")
    
    # Test 6: Tied weights
    print("\n✓ Test 6: Weight tying (embedding = output projection)")
    
    gpt_tied = GPTTransformer(vocab_size=1000, d_model=128, num_heads=2, num_layers=2, tie_weights=True)
    
    # Check that weights are the same object
    assert gpt_tied.token_embedding.weight is gpt_tied.output_projection.weight
    print(f"   Weights are tied ✓")
    
    print("\n" + "=" * 70)
    print("ALL TESTS PASSED ✓")
    print("=" * 70)


if __name__ == "__main__":
    test_transformer()
    demonstrate_transformer()
