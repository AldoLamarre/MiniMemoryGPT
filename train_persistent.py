import argparse
import os

from datasets import load_dataset

from minimemorygpt.data import get_tokenizer
from persistent_memory.trainer import PersistentMemoryConfig, PersistentMemoryTrainer


def main():
    parser = argparse.ArgumentParser(description="Train a persistent-memory LSTM baseline using story chunks and BPTT.")
    parser.add_argument("--output_dir", type=str, default="runs/persistent_memory")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--save_steps", type=int, default=0)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--max_train_samples", type=int, default=1000)
    parser.add_argument("--max_eval_samples", type=int, default=100)
    parser.add_argument("--block_size", type=int, default=512)
    parser.add_argument("--embedding_dim", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--state_dropout_prob", type=float, default=0.0)
    parser.add_argument("--bptt_length", type=lambda value: None if str(value).lower() in {"null", "none", "nil"} else int(value), default=32)
    parser.add_argument("--detach_state", dest="detach_state", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bf16", action="store_true", help="Use BF16 mixed precision.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    tokenizer = get_tokenizer("gpt2")
    dataset = load_dataset("roneneldan/TinyStories")

    train_subset = dataset["train"].select(range(min(args.max_train_samples, len(dataset["train"]))))
    eval_subset = dataset["validation"].select(range(min(args.max_eval_samples, len(dataset["validation"]))))

    config = PersistentMemoryConfig(
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        epochs=args.epochs,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        learning_rate=args.learning_rate,
        vocab_size=len(tokenizer),
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        state_dropout_prob=args.state_dropout_prob,
        block_size=args.block_size,
        bptt_length=args.bptt_length,
        detach_state=args.detach_state,
        max_train_samples=args.max_train_samples,
        max_eval_samples=args.max_eval_samples,
        seed=args.seed,
        bf16=args.bf16,
    )

    trainer = PersistentMemoryTrainer(config=config, tokenizer=tokenizer)
    trainer.fit(train_subset, eval_subset)


if __name__ == "__main__":
    main()
