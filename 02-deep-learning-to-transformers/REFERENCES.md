# Deep Learning to Transformers: Key References

A curated collection of seminal papers tracing the evolution from perceptrons to transformers.

---

## 📚 How to Use This Document

- **📖 Essential**: Foundational papers everyone should read
- **🔧 Hands-on**: Papers with implementations in this course
- **📝 Reference**: Historical context and deeper exploration

---

## The Foundations (1980s-1990s)

### Learning Representations by Back-propagating Errors
**Rumelhart, Hinton, Williams, Nature 1986**

The paper that made neural networks trainable. Introduced backpropagation for multi-layer networks.

- **Key innovations**: Backpropagation algorithm, gradient descent for neural networks
- **Impact**: Foundation for all deep learning
- **Link**: [Nature](https://www.nature.com/articles/323533a0)
- **Status**: 📖 Essential

---

### Gradient-Based Learning Applied to Document Recognition
**LeCun et al., 1998**

Introduced Convolutional Neural Networks (LeNet) for image recognition.

- **Key innovations**: Convolutional layers, pooling, end-to-end learning
- **Impact**: Foundation for computer vision
- **Link**: [IEEE](http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf)
- **Status**: 🔧 Hands-on

---

## Deep Learning Renaissance (2012-2015)

### ImageNet Classification with Deep Convolutional Neural Networks
**Krizhevsky, Sutskever, Hinton, NeurIPS 2012**

AlexNet - the breakthrough that started the deep learning revolution.

- **Key innovations**: ReLU activation, dropout, GPU training, data augmentation
- **Impact**: Proved deep learning works at scale
- **Link**: [NeurIPS](https://papers.nips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html)
- **Status**: 📖 Essential

---

### Deep Residual Learning for Image Recognition
**He et al., CVPR 2016**

Introduced ResNets with skip connections, enabling training of very deep networks.

- **Key innovations**: Residual connections, batch normalization at scale
- **Impact**: Enabled 100+ layer networks, used everywhere
- **Link**: [arXiv:1512.03385](https://arxiv.org/abs/1512.03385)
- **Status**: 🔧 Hands-on

---

### Batch Normalization: Accelerating Deep Network Training
**Ioffe & Szegedy, ICML 2015**

Technique that dramatically speeds up training and improves stability.

- **Key innovations**: Normalizing activations, learnable scale/shift
- **Impact**: Standard component in modern networks
- **Link**: [arXiv:1502.03167](https://arxiv.org/abs/1502.03167)
- **Status**: 🔧 Hands-on

---

### Dropout: A Simple Way to Prevent Neural Networks from Overfitting
**Srivastava et al., JMLR 2014**

Regularization technique that randomly drops units during training.

- **Key innovations**: Stochastic regularization, ensemble interpretation
- **Impact**: Standard regularization technique
- **Link**: [JMLR](https://jmlr.org/papers/v15/srivastava14a.html)
- **Status**: 🔧 Hands-on

---

## Sequence Models (2014-2017)

### Sequence to Sequence Learning with Neural Networks
**Sutskever, Vinyals, Le, NeurIPS 2014**

Encoder-decoder architecture for machine translation.

- **Key innovations**: Seq2seq framework, reversing input sequences
- **Impact**: Foundation for neural machine translation
- **Link**: [arXiv:1409.3215](https://arxiv.org/abs/1409.3215)
- **Status**: 🔧 Hands-on

---

### Long Short-Term Memory
**Hochreiter & Schmidhuber, Neural Computation 1997**

Solved the vanishing gradient problem in RNNs.

- **Key innovations**: Gated memory cells, forget gates
- **Impact**: Enabled long-range sequence modeling
- **Link**: [MIT Press](https://www.mitpressjournals.org/doi/abs/10.1162/neco.1997.9.8.1735)
- **Status**: 🔧 Hands-on

---

### Learning Phrase Representations using RNN Encoder-Decoder
**Cho et al., EMNLP 2014**

Introduced Gated Recurrent Units (GRU), a simpler alternative to LSTM.

- **Key innovations**: GRU architecture, phrase-level representations
- **Impact**: Efficient alternative to LSTM
- **Link**: [arXiv:1406.1078](https://arxiv.org/abs/1406.1078)
- **Status**: 🔧 Hands-on

---

## The Attention Revolution (2015-2017)

### Neural Machine Translation by Jointly Learning to Align and Translate
**Bahdanau, Cho, Bengio, ICLR 2015**

Introduced attention mechanism for sequence-to-sequence models.

- **Key innovations**: Attention mechanism, alignment model
- **Impact**: Foundation for transformers
- **Link**: [arXiv:1409.0473](https://arxiv.org/abs/1409.0473)
- **Status**: 🔧 Hands-on

---

### Attention Is All You Need
**Vaswani et al., NeurIPS 2017**

The paper that started the transformer era.

- **Key innovations**: Self-attention, multi-head attention, positional encoding
- **Impact**: Foundation for BERT, GPT, and all modern LLMs
- **Link**: [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
- **Status**: 🔧 Hands-on

---

## Deep Learning in Neural Networks: An Overview
**Jürgen Schmidhuber, 2015**

Comprehensive historical survey of deep learning.

- **Key value**: Historical context, credit assignment, complete bibliography
- **Link**: [arXiv:1404.7828](https://arxiv.org/abs/1404.7828)
- **Status**: 📝 Reference

---

## Online Resources

| Resource | Description | Link |
|----------|-------------|------|
| The Illustrated Transformer | Visual explanation of attention | [jalammar.github.io](https://jalammar.github.io/illustrated-transformer/) |
| Andrej Karpathy - GPT from Scratch | Building GPT step-by-step | [YouTube](https://www.youtube.com/watch?v=kCc8FmEb1nY) |
| 3Blue1Brown - Neural Networks | Visual intuition | [YouTube](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi) |
| The Annotated Transformer | Line-by-line code walkthrough | [Harvard NLP](https://nlp.seas.harvard.edu/annotated-transformer/) |
| CS231n CNN for Visual Recognition | Stanford CNN course | [cs231n.stanford.edu](https://cs231n.stanford.edu/) |

---

## Implementation Resources

| Resource | Description | Link |
|----------|-------------|------|
| PyTorch Tutorials | Official deep learning tutorials | [pytorch.org](https://pytorch.org/tutorials/) |
| d2l.ai | Dive into Deep Learning book | [d2l.ai](https://d2l.ai/) |
| nanoGPT | Minimal GPT by Karpathy | [GitHub](https://github.com/karpathy/nanoGPT) |

---

*Last updated: September 2026*
