# Deep Learning to Transformers Demos

Interactive demonstrations of deep learning concepts.

## Available Demos

### 1. Perceptron Learning (`perceptron_demo.py`)
Watch the perceptron learn decision boundaries.

```bash
python demo/perceptron_demo.py
```

### 2. Backprop Visualization (`backprop_demo.py`)
Step through forward and backward passes.

```bash
python demo/backprop_demo.py
```

### 3. Activation Functions (`activation_demo.py`)
Compare sigmoid, tanh, ReLU, and their gradients.

```bash
python demo/activation_demo.py
```

### 4. CNN Feature Maps (`cnn_demo.py`)
Visualize what convolutional filters detect.

```bash
python demo/cnn_demo.py
```

### 5. RNN Sequence Processing (`rnn_demo.py`)
See hidden states evolve through a sequence.

```bash
python demo/rnn_demo.py
```

### 6. LSTM Gates (`lstm_demo.py`)
Visualize forget, input, and output gates.

```bash
python demo/lstm_demo.py
```

### 7. Attention Visualization (`attention_demo.py`)
Interactive attention weight heatmaps.

```bash
python demo/attention_demo.py
```

### 8. Self-Attention (`self_attention_demo.py`)
See how tokens attend to each other.

```bash
python demo/self_attention_demo.py
```

### 9. Positional Encoding (`positional_encoding_demo.py`)
Visualize sinusoidal position patterns.

```bash
python demo/positional_encoding_demo.py
```

### 10. Mini Transformer (`mini_transformer_demo.py`)
Train a small transformer on a toy task.

```bash
python demo/mini_transformer_demo.py
```

## Running Demos

```bash
pip install -r requirements.txt
python demo/<demo_name>.py
```

## Recommended Order

1. Start with `perceptron_demo.py` and `backprop_demo.py`
2. Move to `activation_demo.py` and `cnn_demo.py`
3. Sequence models: `rnn_demo.py` → `lstm_demo.py`
4. Attention: `attention_demo.py` → `self_attention_demo.py`
5. Finish with `mini_transformer_demo.py`

---

These demos complement the course modules with hands-on visualization.
