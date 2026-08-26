import argparse

import torch
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer

from persistent_memory.minigpt import PersistentMiniGPTModel


def _make_tokenizer(model_dir: str):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _pad_story_batch(tokenizer, texts, device):
    tokenized = [
        torch.tensor(tokenizer.encode(text, truncation=False), dtype=torch.long, device=device)
        for text in texts
    ]
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    lengths = torch.tensor([ids.numel() for ids in tokenized], device=device, dtype=torch.long)
    batch = pad_sequence(tokenized, batch_first=True, padding_value=pad_token_id)
    return batch, lengths


def _build_shifted_targets(batch_ids, lengths, tokenizer):
    labels = torch.full_like(batch_ids, tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0)
    if batch_ids.size(1) > 1:
        labels[:, :-1] = batch_ids[:, 1:]
    positions = torch.arange(batch_ids.size(1), device=batch_ids.device).unsqueeze(0)
    loss_mask = (positions + 1) < lengths.unsqueeze(1)
    return labels, loss_mask


def forward_story_batch(model, tokenizer, stories, device, bptt_length=None, detach_state=True):
    if len(stories) == 0:
        return None, None

    batch_ids, lengths = _pad_story_batch(tokenizer, stories, device)
    batch_size = batch_ids.size(0)
    state = model.reset_state(batch_size, device)

    max_seq_len = batch_ids.size(1)
    block_size = getattr(model.config, "block_size", 128)
    if bptt_length is not None and bptt_length > 0:
        chunk_size = min(bptt_length, block_size)
    else:
        chunk_size = block_size
    labels, loss_mask = _build_shifted_targets(batch_ids, lengths, tokenizer)

    total_loss = 0.0
    total_tokens = 0
    for start in range(0, max_seq_len, chunk_size):
        end = min(start + chunk_size, max_seq_len)
        chunk_input = batch_ids[:, start:end]
        chunk_labels = labels[:, start:end]
        chunk_loss_mask = loss_mask[:, start:end]
        valid_tokens = int(chunk_loss_mask.sum().item())

        if valid_tokens == 0:
            continue

        active_mask = lengths > start
        attention_mask = chunk_input.ne(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0)

        outputs = model(
            input_ids=chunk_input,
            state=state,
            labels=chunk_labels,
            active_mask=active_mask,
            loss_mask=chunk_loss_mask,
            attention_mask=attention_mask,
        )

        if outputs["loss"] is not None:
            total_loss += float(outputs["loss"].item()) * valid_tokens
            total_tokens += valid_tokens

        state = outputs["state"]
        if detach_state:
            state = (state[0].detach(), state[1].detach())

    if total_tokens == 0:
        return None, state
    return total_loss / total_tokens, state


def evaluate_persistent_model(model_dir: str, split: str = "validation", max_samples: int = 50, batch_size: int = 8, bptt_length: int | None = 32):
    tokenizer = _make_tokenizer(model_dir)
    model = PersistentMiniGPTModel.from_pretrained(model_dir)
    model.eval()
    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    dataset = load_dataset("roneneldan/TinyStories")
    subset = dataset[split].select(range(min(max_samples, len(dataset[split]))))

    total_loss = 0.0
    count = 0
    with torch.no_grad():
        for i in range(0, len(subset), batch_size):
            batch_texts = subset["text"][i : i + batch_size]
            if isinstance(batch_texts, str):
                batch_texts = [batch_texts]
            loss, _ = forward_story_batch(
                model=model,
                tokenizer=tokenizer,
                stories=batch_texts,
                device=model.device,
                bptt_length=bptt_length,
                detach_state=True,
            )
            if loss is not None:
                total_loss += float(loss)
                count += 1

    avg_loss = total_loss / max(1, count)
    return avg_loss


def main():
    parser = argparse.ArgumentParser(description="Evaluate a persistent-memory MiniGPT model using the same stateful recurrent logic as training.")
    parser.add_argument("--model_dir", type=str, required=True, help="Checkpoint directory to load, e.g. runs/state_dropout_010/run_...")
    parser.add_argument("--split", type=str, default="validation", choices=["train", "validation"], help="Dataset split to evaluate")
    parser.add_argument("--max_samples", type=int, default=50, help="Maximum number of stories to evaluate")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--bptt_length", type=int, default=32)
    args = parser.parse_args()

    loss = evaluate_persistent_model(
        model_dir=args.model_dir,
        split=args.split,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
        bptt_length=args.bptt_length,
    )
    print(f"persistent_eval_loss={loss:.6f}")


if __name__ == "__main__":
    main()
