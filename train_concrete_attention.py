import argparse
import os

import torch
from transformers import Trainer, TrainingArguments

from minimemorygpt.concrete_attention import ConcreteAttentionGPT
from minimemorygpt.config import MiniTinyStoryGPTConfig
from minimemorygpt.data import get_tokenizer, prepare_tinystories_datasets


class RangeAwareTrainer(Trainer):
    def __init__(self, *args, range_loss_weight: float = 0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.range_loss_weight = range_loss_weight

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(**inputs)
        loss = outputs.loss
        if model.training and self.range_loss_weight and outputs.range_loss is not None:
            loss = loss + self.range_loss_weight * outputs.range_loss
        return (loss, outputs) if return_outputs else loss


def main():
    parser = argparse.ArgumentParser(
        description="Train the PowerSoftmax attention-only Concrete ML model."
    )
    parser.add_argument("--output_dir", type=str, default="runs/concrete_attention")
    parser.add_argument("--model_name", type=str, default="gpt2")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--epochs", type=float, default=1)
    parser.add_argument("--max_train_samples", type=int, default=1000)
    parser.add_argument("--max_eval_samples", type=int, default=100)
    parser.add_argument("--block_size", type=int, default=32)
    parser.add_argument("--n_embd", type=int, default=32)
    parser.add_argument("--n_layer", type=int, default=1)
    parser.add_argument("--n_head", type=int, default=2)
    parser.add_argument("--intermediate_size", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--power_softmax_power", type=int, default=4)
    parser.add_argument("--power_softmax_epsilon", type=float, default=1e-3)
    parser.add_argument(
        "--attention_score_range",
        "--attention_score_clip",
        dest="attention_score_range",
        type=float,
        default=2.0,
        help="Target absolute attention-score range used during range-loss fine-tuning.",
    )
    parser.add_argument("--range_finetune_epochs", type=float, default=1.0)
    parser.add_argument("--range_loss_weight", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    args = parser.parse_args()

    if args.n_embd % args.n_head != 0:
        parser.error("--n_embd must be divisible by --n_head")
    if args.power_softmax_power < 2 or args.power_softmax_power % 2:
        parser.error("--power_softmax_power must be a positive even integer >= 2")
    if args.range_finetune_epochs < 0 or args.range_loss_weight < 0:
        parser.error("Range fine-tuning epochs and weight must be non-negative")

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
        memory_mode="attention",
        power_softmax_power=args.power_softmax_power,
        power_softmax_epsilon=args.power_softmax_epsilon,
        attention_score_range=args.attention_score_range,
    )
    model = ConcreteAttentionGPT(config)

    if args.bf16:
        use_bf16 = True
    elif args.fp16:
        use_bf16 = False
    else:
        use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    use_fp16 = args.fp16 and torch.cuda.is_available() and not use_bf16

    os.makedirs(args.output_dir, exist_ok=True)
    def make_training_args(output_dir, epochs):
        return TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            num_train_epochs=epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            logging_steps=25,
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            bf16=use_bf16,
            fp16=use_fp16,
            report_to=[],
            seed=args.seed,
        )

    trainer = RangeAwareTrainer(
        model=model,
        args=make_training_args(os.path.join(args.output_dir, "pretrain"), args.epochs),
        train_dataset=train_ds,
        eval_dataset=eval_ds,
    )
    trainer.train()

    if args.range_finetune_epochs:
        trainer = RangeAwareTrainer(
            model=model,
            args=make_training_args(
                os.path.join(args.output_dir, "range_finetune"),
                args.range_finetune_epochs,
            ),
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            range_loss_weight=args.range_loss_weight,
        )
        trainer.train()

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"PowerSoftmax attention model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
