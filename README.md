# MiniMemoryGPT

A lightweight Hugging Face PyTorch project for training a tiny causal language model on TinyStories data.

## Project structure
- `minimemorygpt/config.py` – model config
- `minimemorygpt/model.py` – custom mini transformer with attention + LSTM blocks
- `minimemorygpt/data.py` – dataset loading and tokenization
- `train.py` – training entrypoint
- `evaluate.py` – generation / evaluation entrypoint

## Install

```bash
python -m pip install torch transformers datasets accelerate
```

## Train on a laptop RTX 4060 (8GB VRAM)

These defaults are designed to fit a 4060-class GPU without exploding memory:

```bash
python train.py \
  --output_dir runs/tinystory-mini \
  --batch_size 1 \
  --gradient_accumulation_steps 4 \
  --block_size 128 \
  --n_embd 128 \
  --n_layer 2 \
  --n_head 4 \
  --intermediate_size 256 \
  --max_train_samples 8000 \
  --max_eval_samples 500 \
  --epochs 2
```

If your card supports BF16, the script will use it automatically. You can also force it:

```bash
python train.py --bf16 ...
```

## Persistent-memory LSTM experiment

This project also includes a separate custom training loop for a story-aware recurrent-memory model. It keeps LSTM state across chunks within a story and resets it between stories, with optional BPTT truncation.

```bash
python train_persistent.py \
  --output_dir runs/persistent_memory \
  --batch_size 1 \
  --epochs 1 \
  --block_size 128 \
  --embedding_dim 128 \
  --hidden_dim 128 \
  --num_layers 2 \
  --bptt_length 32
```

## Persistent miniGPT experiment

This variant keeps the recurrent state in the middle, between two transformer stages.

```bash
python train_persistent_minigpt.py \
  --output_dir runs/persistent_minigpt \
  --batch_size 1 \
  --epochs 1 \
  --block_size 128 \
  --embedding_dim 128 \
  --hidden_dim 128 \
  --num_layers 2 \
  --n_head 4 \
  --bptt_length 32
```

## Generate

```bash
python evaluate.py --model_dir runs/tinystory-mini --prompt "Once upon a time" --max_new_tokens 80
```
