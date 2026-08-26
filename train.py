import argparse
import os
from datetime import datetime

import torch
from transformers import Trainer, TrainingArguments

from minimemorygpt.config import MiniTinyStoryGPTConfig
from minimemorygpt.data import get_tokenizer, prepare_tinystories_datasets
from minimemorygpt.model import MiniTinyStoryGPTModel


def build_run_dir(base_dir: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_dir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def main():
    parser = argparse.ArgumentParser(description="Train a compact TinyStories GPT on a laptop GPU.")
    parser.add_argument("--output_dir", type=str, default="runs")
    parser.add_argument("--model_name", type=str, default="gpt2")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--max_train_samples", type=int, default=8000)
    parser.add_argument("--max_eval_samples", type=int, default=500)
    parser.add_argument("--block_size", type=int, default=128)
    parser.add_argument("--n_embd", type=int, default=128)
    parser.add_argument("--n_layer", type=int, default=2)
    parser.add_argument("--n_head", type=int, default=4)
    parser.add_argument("--intermediate_size", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action="store_true", help="Use FP16 on CUDA if supported.")
    parser.add_argument("--bf16", action="store_true", help="Use BF16 on CUDA if supported.")
    parser.add_argument(
        "--memory_mode",
        type=str,
        choices=["hybrid", "attention", "lstm", "none"],
        default="hybrid",
        help="Model memory mode: hybrid=attention+LSTM, attention=attention only, lstm=LSTM only, none=feedforward only.",
    )
    parser.add_argument(
        "--bptt_length",
        type=int,
        default=-1,
        help="Chunk recurrent state updates for truncated backpropagation. Use -1 to disable BPTT truncation.",
    )
    parser.add_argument(
        "--detach_recurrent_state",
        action="store_true",
        default=True,
        help="Detach recurrent state after each BPTT chunk to avoid unbounded graph growth.",
    )
    args = parser.parse_args()

    run_dir = build_run_dir(args.output_dir)
    print(f"Using run directory: {run_dir}")

    torch.manual_seed(args.seed)

    tokenizer = get_tokenizer(args.model_name)
    train_ds, eval_ds = prepare_tinystories_datasets(
        tokenizer,
        block_size=args.block_size,
        max_train_samples=args.max_train_samples,
        max_eval_samples=args.max_eval_samples,
    )

    config = MiniTinyStoryGPTConfig(
        vocab_size=len(tokenizer),
        n_positions=args.block_size,
        n_embd=args.n_embd,
        n_layer=args.n_layer,
        n_head=args.n_head,
        intermediate_size=args.intermediate_size,
        dropout=args.dropout,
        pad_token_id=tokenizer.pad_token_id,
        memory_mode=args.memory_mode,
        bptt_length=None if args.bptt_length <= 0 else args.bptt_length,
        detach_recurrent_state=args.detach_recurrent_state,
    )

    model = MiniTinyStoryGPTModel(config)
    if (
        hasattr(model, "gradient_checkpointing_enable")
        and getattr(model, "supports_gradient_checkpointing", False)
    ):
        model.gradient_checkpointing_enable()

    if args.bf16:
        use_bf16 = True
    elif args.fp16:
        use_bf16 = False
    elif torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        use_bf16 = True
    else:
        use_bf16 = False

    use_fp16 = args.fp16 and torch.cuda.is_available() and not use_bf16

    training_args = TrainingArguments(
        output_dir=run_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=max(1, args.batch_size),
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        logging_steps=25,
        eval_strategy="steps",
        eval_steps=250,
        save_steps=500,
        save_total_limit=2,
        bf16=use_bf16,
        fp16=use_fp16,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
    )
    trainer.train()
    trainer.save_model(run_dir)
    tokenizer.save_pretrained(run_dir)
    print(f"Best model saved in: {run_dir}")


if __name__ == "__main__":
    main()
