# Concrete ML demo (experimental)

This separate PowerSoftmax attention path explores encrypted inference without
changing MiniMemoryGPT's Transformer–LSTM model. The tiny 8-token model completed
the end-to-end demo, but its generated text is poor. The full-size encrypted run
was stopped because key generation took too long on the laptop.

## Run the smoke demo

Install the optional dependencies in an isolated Python 3.12 environment:

```bash
python3.12 -m venv .venv-concrete
source .venv-concrete/bin/activate
python -m pip install -r requirements-concrete.txt
```

Train, compile, and run the small demo:

```bash
python train_concrete_attention.py \
  --output_dir runs/power-smoke \
  --block_size 8 --n_embd 8 --n_head 2 \
  --intermediate_size 16 --n_layer 1 \
  --max_train_samples 8 --max_eval_samples 4 \
  --epochs 1 --range_finetune_epochs 1

python compile_concrete_attention.py \
  --model_dir runs/power-smoke \
  --output_dir runs/power-smoke-fhe \
  --calibration_samples 2 --max_eval_samples 4 --n_bits 3

python run_concrete_attention.py \
  --model_dir runs/power-smoke \
  --fhe_dir runs/power-smoke-fhe \
  --prompt "Once upon a time" --max_new_tokens 1
```

First inference generates client keys and may take a while. Later runs reuse
cached keys. This demo verifies the pipeline, not model quality or practical
large-model performance.

## Plaintext model comparison

The 128-token, 128-hidden-size PowerSoftmax checkpoint was compared against the
saved MiniMemoryGPT baseline on 500 TinyStories validation examples (62,672
non-padding targets):

| Model path | Validation loss ↓ | Perplexity ↓ | Next-token accuracy ↑ |
| --- | ---: | ---: | ---: |
| MiniMemoryGPT baseline | 3.8151 | 45.38 | 35.25% |
| PowerSoftmax, training operations | **3.2271** | **25.21** | **39.88%** |
| PowerSoftmax, polynomial operations (plaintext) | 3.2919 | 26.89 | 38.93% |

These are plaintext validation results; they do not measure Concrete ML
quantization or encrypted inference accuracy.
