import argparse
import os

from train import main as train_main
from evaluate import main as evaluate_main


def main():
    parser = argparse.ArgumentParser(description="MiniMemoryGPT CLI entrypoint")
    parser.add_argument(
        "--mode",
        choices=["train", "evaluate"],
        default="train",
        help="Select whether to train the model or generate text.",
    )
    parser.add_argument("--output_dir", type=str, default="runs")
    parser.add_argument("--model_dir", type=str, default="runs")
    parser.add_argument("--prompt", type=str, default="Once upon a time")
    parser.add_argument("--max_new_tokens", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.8)
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
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument(
        "--memory_mode",
        type=str,
        choices=["hybrid", "attention", "lstm", "none"],
        default="hybrid",
    )
    parser.add_argument("--bptt_length", type=int, default=-1)
    parser.add_argument("--detach_recurrent_state", action="store_true", default=True)
    args = parser.parse_args()

    if args.mode == "train":
        import sys

        sys.argv = [
            "train.py",
            "--output_dir", args.output_dir,
            "--batch_size", str(args.batch_size),
            "--gradient_accumulation_steps", str(args.gradient_accumulation_steps),
            "--epochs", str(args.epochs),
            "--max_train_samples", str(args.max_train_samples),
            "--max_eval_samples", str(args.max_eval_samples),
            "--block_size", str(args.block_size),
            "--n_embd", str(args.n_embd),
            "--n_layer", str(args.n_layer),
            "--n_head", str(args.n_head),
            "--intermediate_size", str(args.intermediate_size),
            "--learning_rate", str(args.learning_rate),
            "--weight_decay", str(args.weight_decay),
            "--seed", str(args.seed),
            "--memory_mode", args.memory_mode,
            "--bptt_length", str(args.bptt_length),
        ]
        if args.detach_recurrent_state:
            sys.argv.append("--detach_recurrent_state")
        if args.fp16:
            sys.argv.append("--fp16")
        if args.bf16:
            sys.argv.append("--bf16")
        train_main()
    else:
        import sys

        if args.model_dir == "runs":
            candidates = sorted(
                [os.path.join(args.model_dir, d) for d in os.listdir(args.model_dir) if d.startswith("run_")],
                reverse=True,
            )
            if candidates:
                args.model_dir = candidates[0]

        sys.argv = [
            "evaluate.py",
            "--model_dir", args.model_dir,
            "--prompt", args.prompt,
            "--max_new_tokens", str(args.max_new_tokens),
            "--temperature", str(args.temperature),
        ]
        evaluate_main()


if __name__ == "__main__":
    main()
