"""
Demo: Train a Tiny Transformer

This demo trains a small GPT-style transformer on character-level language modeling.
The task: given some characters, predict the next character.

Example:
    Input:  "The cat sa"
    Output: "t" (predict next character)

We'll use a tiny model and a small dataset so it trains in seconds on CPU.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time


# =============================================================================
# TINY TRANSFORMER (self-contained for demo)
# =============================================================================

class TinyAttention(nn.Module):
    """Simplified multi-head attention."""
    
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_qkv = nn.Linear(d_model, 3 * d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        B, T, C = x.shape
        
        # Project to Q, K, V
        qkv = self.W_qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        
        # Reshape for multi-head
        q = q.view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        
        # Attention scores
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        
        return self.W_o(out)


class TinyBlock(nn.Module):
    """Transformer block with pre-norm."""
    
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = TinyAttention(d_model, num_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout)
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        x = x + self.dropout(self.attn(self.ln1(x), mask))
        x = x + self.ffn(self.ln2(x))
        return x


class TinyGPT(nn.Module):
    """
    A tiny GPT for character-level language modeling.
    """
    
    def __init__(self, vocab_size, d_model=64, num_heads=4, num_layers=4, 
                 max_seq_len=128, dropout=0.1):
        super().__init__()
        
        self.max_seq_len = max_seq_len
        
        # Embeddings
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TinyBlock(d_model, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        # Output
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Weight tying
        self.head.weight = self.token_emb.weight
        
        # Create causal mask
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(max_seq_len, max_seq_len)).view(1, 1, max_seq_len, max_seq_len)
        )
    
    def forward(self, idx):
        B, T = idx.shape
        
        # Embeddings
        tok_emb = self.token_emb(idx)
        pos_emb = self.pos_emb(torch.arange(T, device=idx.device))
        x = self.dropout(tok_emb + pos_emb)
        
        # Causal mask for this sequence length
        mask = self.mask[:, :, :T, :T]
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x, mask)
        
        # Output
        x = self.ln_f(x)
        logits = self.head(x)
        
        return logits
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0):
        """Generate new tokens autoregressively."""
        for _ in range(max_new_tokens):
            # Crop to max_seq_len
            idx_cond = idx if idx.size(1) <= self.max_seq_len else idx[:, -self.max_seq_len:]
            
            # Get predictions
            logits = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            
            # Sample
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # Append
            idx = torch.cat([idx, idx_next], dim=1)
        
        return idx


# =============================================================================
# DATASET
# =============================================================================

class CharDataset:
    """Character-level dataset."""
    
    def __init__(self, text, seq_len):
        self.seq_len = seq_len
        
        # Build vocabulary
        chars = sorted(list(set(text)))
        self.char_to_idx = {ch: i for i, ch in enumerate(chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(chars)}
        self.vocab_size = len(chars)
        
        # Encode text
        self.data = torch.tensor([self.char_to_idx[ch] for ch in text], dtype=torch.long)
    
    def __len__(self):
        return len(self.data) - self.seq_len
    
    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_len]
        y = self.data[idx + 1:idx + self.seq_len + 1]
        return x, y
    
    def decode(self, indices):
        """Convert indices back to string."""
        return ''.join([self.idx_to_char[i.item()] for i in indices])
    
    def encode(self, text):
        """Convert string to indices."""
        return torch.tensor([self.char_to_idx[ch] for ch in text], dtype=torch.long)


# =============================================================================
# TRAINING
# =============================================================================

def train():
    print("=" * 70)
    print("TRAINING A TINY TRANSFORMER")
    print("=" * 70)
    
    # =========================================================================
    # Data
    # =========================================================================
    # Simple repeating text pattern for quick learning
    text = """
    The quick brown fox jumps over the lazy dog.
    A quick brown fox jumps over a lazy dog.
    The lazy dog sleeps while the fox jumps.
    Quick foxes jump over lazy dogs all day.
    The dog is lazy but the fox is quick.
    """ * 50  # Repeat to have enough data
    
    seq_len = 32
    dataset = CharDataset(text, seq_len)
    
    print(f"\n1. DATA")
    print(f"   Text length:  {len(text)} characters")
    print(f"   Vocab size:   {dataset.vocab_size} characters")
    print(f"   Sequence len: {seq_len}")
    print(f"   Vocabulary:   {''.join(sorted(dataset.char_to_idx.keys()))}")
    
    # =========================================================================
    # Model
    # =========================================================================
    model = TinyGPT(
        vocab_size=dataset.vocab_size,
        d_model=64,
        num_heads=4,
        num_layers=4,
        max_seq_len=seq_len,
        dropout=0.1
    )
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"\n2. MODEL")
    print(f"   d_model:    64")
    print(f"   num_heads:  4")
    print(f"   num_layers: 4")
    print(f"   Parameters: {num_params:,}")
    
    # =========================================================================
    # Training setup
    # =========================================================================
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    batch_size = 32
    num_epochs = 500
    
    print(f"\n3. TRAINING")
    print(f"   Device:     {device}")
    print(f"   Batch size: {batch_size}")
    print(f"   Epochs:     {num_epochs}")
    print(f"\n   Training...")
    
    # Create batches
    def get_batch():
        indices = torch.randint(len(dataset), (batch_size,))
        x = torch.stack([dataset[i][0] for i in indices]).to(device)
        y = torch.stack([dataset[i][1] for i in indices]).to(device)
        return x, y
    
    # Training loop
    start_time = time.time()
    losses = []
    
    model.train()
    for epoch in range(num_epochs):
        x, y = get_batch()
        
        # Forward
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, dataset.vocab_size), y.view(-1))
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        # Log progress
        if (epoch + 1) % 100 == 0:
            print(f"   Epoch {epoch+1:3d}: loss = {loss.item():.4f}")
    
    elapsed = time.time() - start_time
    print(f"\n   Training completed in {elapsed:.1f} seconds")
    print(f"   Final loss: {losses[-1]:.4f}")
    print(f"   Loss reduction: {losses[0]:.4f} → {losses[-1]:.4f} ({100*(losses[0]-losses[-1])/losses[0]:.1f}% decrease)")
    
    # =========================================================================
    # Generation
    # =========================================================================
    print(f"\n4. GENERATION")
    print(f"   Let's see what the model learned!\n")
    
    model.eval()
    
    prompts = ["The ", "A quick ", "The lazy ", "fox "]
    
    for prompt in prompts:
        # Encode prompt
        prompt_ids = dataset.encode(prompt).unsqueeze(0).to(device)
        
        # Generate
        generated_ids = model.generate(prompt_ids, max_new_tokens=40, temperature=0.8)
        
        # Decode
        generated_text = dataset.decode(generated_ids[0])
        
        print(f"   Prompt: \"{prompt}\"")
        print(f"   Output: \"{generated_text}\"")
        print()
    
    # =========================================================================
    # Loss curve
    # =========================================================================
    print(f"\n5. LOSS CURVE (ASCII)")
    print(f"   " + "-" * 52)
    
    # Simple ASCII plot
    max_loss = max(losses)
    min_loss = min(losses)
    height = 8
    width = 50
    
    for row in range(height, -1, -1):
        threshold = min_loss + (max_loss - min_loss) * row / height
        line = "   │"
        for i in range(0, len(losses), len(losses) // width + 1):
            if losses[i] >= threshold:
                line += "█"
            else:
                line += " "
        if row == height:
            line += f" {max_loss:.2f}"
        elif row == 0:
            line += f" {min_loss:.2f}"
        print(line)
    
    print(f"   └" + "─" * 50 + "→ epochs")
    print(f"    0{' ' * 22}250{' ' * 21}500")
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"""
    What happened:
    1. We created a tiny transformer ({num_params:,} parameters)
    2. Trained it on simple text for {num_epochs} epochs
    3. Loss decreased from {losses[0]:.2f} to {losses[-1]:.2f}
    4. The model learned to generate coherent text!
    
    Key observations:
    - Even this tiny model learns patterns in the data
    - It picks up word structure and common phrases
    - More data + bigger model = better results
    
    This is exactly how GPT/LLaMA work, just at a much larger scale:
    - GPT-2:  1.5 billion parameters
    - GPT-3:  175 billion parameters
    - LLaMA:  7-70 billion parameters
    
    Same architecture, same training loop, just MUCH bigger!
    """)


if __name__ == "__main__":
    train()
