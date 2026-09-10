# Deep Learning to Transformers: A Step-by-Step Journey

This course bridges the gap between classical ML and the Transformer architecture. Starting from the perceptron, we build through neural networks, CNNs, RNNs, and sequence-to-sequence models to arrive at the attention mechanism that revolutionized NLP.

---

## 🎯 Prerequisites

- ML Foundations course (or equivalent)
- Python, NumPy basics
- Calculus (derivatives, chain rule)
- Linear algebra (matrices, vectors)

---

## 📚 Part 1: Neural Network Fundamentals

From single neurons to deep networks.

### Step 1: The Perceptron
**The origin of neural networks**

**What we'll cover:**
- Biological inspiration
- The perceptron algorithm
- Linear decision boundaries
- XOR problem and its limitations
- Why we need more layers

**Key equation:**
```
output = 1 if (w·x + b) > 0 else 0
```

---

### Step 2: Multi-Layer Perceptrons (MLPs)
**Stacking layers for power**

**What we'll cover:**
- Hidden layers
- Activation functions (sigmoid, tanh, ReLU)
- Universal approximation theorem
- Forward pass computation
- Fully connected architecture

**Key insight:**
```
Hidden layers can learn ANY function (given enough neurons)
```

---

### Step 3: Backpropagation
**How neural networks learn**

**What we'll cover:**
- The chain rule in action
- Forward and backward passes
- Computing gradients layer by layer
- Computational graphs
- Implementing backprop from scratch

**The algorithm:**
```
1. Forward pass: compute outputs
2. Compute loss
3. Backward pass: compute gradients (chain rule)
4. Update weights: w = w - lr × gradient
```

---

### Step 4: Training Deep Networks
**Making deep learning work**

**What we'll cover:**
- Weight initialization (Xavier, He)
- Batch normalization
- Dropout regularization
- Learning rate schedules
- Gradient clipping
- Vanishing/exploding gradients

**Key techniques:**
```
BatchNorm → stable training
Dropout → regularization
Residual connections → very deep networks
```

---

### Step 5: PyTorch Fundamentals
**The framework for deep learning**

**What we'll cover:**
- Tensors and operations
- Autograd: automatic differentiation
- nn.Module and building models
- DataLoaders and batching
- Training loops
- GPU acceleration

**Pattern:**
```python
model = MyModel()
optimizer = torch.optim.Adam(model.parameters())
for batch in dataloader:
    loss = criterion(model(batch), targets)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

---

## 📚 Part 2: Convolutional Neural Networks

Learning from spatial structure.

### Step 6: Convolutions
**The operation that powers computer vision**

**What we'll cover:**
- Convolution operation
- Filters/kernels
- Stride and padding
- Feature maps
- Parameter sharing
- Translation equivariance

**Key insight:**
```
Convolution = sliding window dot product
Learns local patterns that work anywhere in the image
```

---

### Step 7: CNN Architectures
**Building blocks of vision models**

**What we'll cover:**
- Pooling layers
- Classic architectures (LeNet, AlexNet, VGG)
- Residual networks (ResNet)
- 1x1 convolutions
- CNN for sequence data (1D convolutions)

**Architecture pattern:**
```
Conv → ReLU → Pool → Conv → ReLU → Pool → Flatten → FC → Output
```

---

## 📚 Part 3: Sequence Models

Learning from sequential data.

### Step 8: Recurrent Neural Networks (RNNs)
**Processing sequences**

**What we'll cover:**
- Sequential data challenges
- RNN architecture
- Hidden state as memory
- Backpropagation through time (BPTT)
- Vanishing gradient problem

**Key equation:**
```
h_t = tanh(W_hh × h_{t-1} + W_xh × x_t + b)
```

---

### Step 9: LSTM and GRU
**Solving the vanishing gradient**

**What we'll cover:**
- Long-term dependencies problem
- LSTM architecture (forget, input, output gates)
- GRU as simplified LSTM
- When to use which
- Bidirectional RNNs

**LSTM gates:**
```
Forget gate: what to discard from memory
Input gate: what new info to store
Output gate: what to output from memory
```

---

### Step 10: Sequence-to-Sequence Models
**The encoder-decoder paradigm**

**What we'll cover:**
- Encoder-decoder architecture
- Context vector bottleneck
- Teacher forcing
- Beam search decoding
- Applications: translation, summarization

**Architecture:**
```
Input → [Encoder RNN] → Context Vector → [Decoder RNN] → Output
```

---

## 📚 Part 4: The Attention Revolution

From fixed context to dynamic attention.

### Step 11: Attention Mechanism
**The breakthrough idea**

**What we'll cover:**
- The bottleneck problem
- Bahdanau attention (additive)
- Luong attention (multiplicative)
- Attention weights visualization
- Attention as soft lookup

**Key equations:**
```
score(h_t, h_s) = how relevant is source h_s to target h_t
attention_weights = softmax(scores)
context = weighted sum of encoder states
```

---

### Step 12: Self-Attention
**Attending to yourself**

**What we'll cover:**
- From cross-attention to self-attention
- Query, Key, Value formulation
- Scaled dot-product attention
- Why scaling matters
- Comparing positions within a sequence

**The equation:**
```
Attention(Q, K, V) = softmax(QK^T / √d_k) × V
```

---

### Step 13: Multi-Head Attention
**Attending to different aspects**

**What we'll cover:**
- Multiple attention heads
- Different representation subspaces
- Concatenation and projection
- What different heads learn
- Computational considerations

**Architecture:**
```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) × W_O
where head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
```

---

### Step 14: Positional Encoding
**Where am I in the sequence?**

**What we'll cover:**
- Why position matters
- Sinusoidal encodings
- Learned positional embeddings
- Relative position encodings
- RoPE and ALiBi (modern approaches)

**Sinusoidal formula:**
```
PE(pos, 2i) = sin(pos / 10000^(2i/d))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
```

---

### Step 15: The Transformer Architecture
**Attention Is All You Need**

**What we'll cover:**
- Full transformer architecture
- Encoder stack
- Decoder stack with masked attention
- Layer normalization and residuals
- Feed-forward networks
- Why transformers dominate

**This leads directly to the transformer-architecture course!**

---

## 🗂️ Project Structure

```
deep-learning-to-transformers/
├── LEARNING_PATH.md
├── requirements.txt
├── verify_setup.py
│
├── 01_perceptron/
├── 02_mlp/
├── 03_backpropagation/
├── 04_training_deep_networks/
├── 05_pytorch_fundamentals/
├── 06_convolutions/
├── 07_cnn_architectures/
├── 08_rnn/
├── 09_lstm_gru/
├── 10_seq2seq/
├── 11_attention/
├── 12_self_attention/
├── 13_multihead_attention/
├── 14_positional_encoding/
├── 15_transformer_architecture/
│
├── demo/
└── evolutions/
```

---

## 🚀 Let's Begin!

Start with **Step 1: The Perceptron** — where it all began.

After completing this course, continue to **transformer-architecture** for deep implementation details.

---

## 📖 References

### Papers
- "Learning representations by back-propagating errors" — Rumelhart et al., 1986
- "Long Short-Term Memory" — Hochreiter & Schmidhuber, 1997
- "Neural Machine Translation by Jointly Learning to Align and Translate" — Bahdanau et al., 2014
- "Attention Is All You Need" — Vaswani et al., 2017

### Resources
- 3Blue1Brown Neural Network series
- Andrej Karpathy's Neural Networks: Zero to Hero
- "Deep Learning" — Goodfellow, Bengio, Courville
