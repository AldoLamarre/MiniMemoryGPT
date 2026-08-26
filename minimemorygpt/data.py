from datasets import load_dataset


def get_tokenizer(tokenizer_name: str = "gpt2"):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def prepare_tinystories_datasets(
    tokenizer,
    block_size: int = 256,
    max_train_samples: int | None = None,
    max_eval_samples: int | None = None,
):
    dataset = load_dataset("roneneldan/TinyStories")

    def tokenize(batch):
        outputs = tokenizer(
            batch["text"],
            truncation=True,
            max_length=block_size,
            padding="max_length",
            return_attention_mask=True,
        )
        outputs["labels"] = outputs["input_ids"].copy()
        return outputs

    train_split = dataset["train"]
    valid_split = dataset["validation"]

    if max_train_samples is not None:
        train_split = train_split.select(range(min(max_train_samples, len(train_split))))
    if max_eval_samples is not None:
        valid_split = valid_split.select(range(min(max_eval_samples, len(valid_split))))

    train_ds = train_split.map(tokenize, batched=True, remove_columns=["text"])
    valid_ds = valid_split.map(tokenize, batched=True, remove_columns=["text"])
    return train_ds, valid_ds
