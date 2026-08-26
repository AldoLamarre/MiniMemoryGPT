# MiniMemoryGPT

A lightweight Hugging Face PyTorch project for training a tiny causal language model on TinyStories data.

The project explores persistent long-term sequence memory by combining Transformer blocks with an LSTM memory layer in a Transformer–LSTM–Transformer architecture.

## Architecture

```text
Input
  ↓
Transformer
  ↓
LSTM Memory
  ↓
Transformer
  ↓
Language Model Head
```

## Motivation 

Built as a hands-on exploration of Transformer and LSTM sequence modeling, while practicing AI-assisted development with GitHub Copilot.

## Results

| Model | BPTT Horizon | Best Eval Loss | Epoch |
|---|---:|---:|---:|
| Sequential Memory | — | **2.2670** | 8 |
| Persistent Memory | 32 | 2.4211 | 8 |
| Persistent Memory | 256 | **2.3626** | 8 |
| Persistent Memory | 1024 | 2.3802 | 8 |