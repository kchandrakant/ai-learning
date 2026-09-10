# Module 8: Audio & Speech

Processing sound with neural networks.

## Overview

Audio is another modality. Models can transcribe, generate, and understand speech and music.

## Key Topics

### Audio Representations
```
Waveform: Raw amplitude over time (1D signal)
- High sample rate: 16kHz-44kHz
- Long sequences!

Spectrogram: Frequency × Time (2D, like image)
- STFT converts waveform to spectrogram
- Visualize energy at each frequency over time

Mel Spectrogram: Human-perception weighted
- Mel scale matches human hearing
- Most common input for models
```

### Speech Recognition (ASR)
```
Whisper (OpenAI):
- Encoder-decoder transformer
- Trained on 680K hours of audio
- Multilingual, robust

Architecture: Audio → Mel spectrogram → Encoder → Decoder → Text
```

### Text-to-Speech (TTS)
```
Convert text to natural speech:
- Tacotron: Attention-based spectrogram generation
- WaveNet: Autoregressive waveform generation
- Modern: Diffusion-based, voice cloning

Pipeline: Text → Acoustic model → Vocoder → Waveform
```

### Audio-Language Models
```
Combine audio understanding with LLMs:
- Audio encoder → projection → LLM
- Similar pattern to vision-language

Applications: Audio captioning, audio QA
```

### Music Generation
```
MusicLM, Suno:
- Text-to-music generation
- Long-range structure is challenging
- Style, genre, instrument control
```

## Exercises

1. Transcribe audio with Whisper
2. Visualize spectrograms
3. Generate speech from text

## Key Insight

Audio follows similar patterns to vision: encode to embeddings, process with transformers.
