import argparse
import hashlib
import os
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

from minimemorygpt.concrete_attention import ConcreteAttentionGPT


def default_key_dir(fhe_dir: str) -> str:
    circuit_path = str(Path(fhe_dir).expanduser().resolve())
    circuit_id = hashlib.sha256(circuit_path.encode("utf-8")).hexdigest()[:16]
    return os.path.join(
        os.path.expanduser("~"),
        ".cache",
        "minimemorygpt",
        "fhe_keys",
        circuit_id,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Generate text with Concrete ML encrypted attention inference."
    )
    parser.add_argument("--model_dir", required=True, help="Trained attention checkpoint.")
    parser.add_argument("--fhe_dir", required=True, help="Concrete ML client/server artifacts.")
    parser.add_argument("--prompt", default="Once upon a time")
    parser.add_argument("--max_new_tokens", type=int, default=1)
    parser.add_argument("--key_dir", help="Directory for client private/evaluation keys.")
    args = parser.parse_args()

    if args.max_new_tokens < 1:
        parser.error("--max_new_tokens must be positive")
    if not os.path.isdir(args.model_dir):
        parser.error(f"Model checkpoint directory does not exist: {args.model_dir}")
    if not os.path.isdir(args.fhe_dir):
        parser.error(f"Concrete ML artifact directory does not exist: {args.fhe_dir}")

    try:
        from concrete.ml.deployment import FHEModelClient, FHEModelServer
    except ImportError as error:
        raise SystemExit(
            "Concrete ML is not installed in this Python environment. Activate the "
            "Concrete environment and install requirements-concrete.txt."
        ) from error

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = ConcreteAttentionGPT.from_pretrained(args.model_dir).eval()
    if tokenizer.pad_token_id is None:
        raise ValueError("The tokenizer must define a pad token for fixed-size contexts.")

    token_ids = tokenizer(
        args.prompt,
        return_tensors="pt",
        truncation=True,
        max_length=model.config.n_positions,
    )["input_ids"]
    if token_ids.size(1) == 0:
        raise ValueError("The prompt must contain at least one token.")
    if token_ids.size(1) > model.config.n_positions:
        token_ids = token_ids[:, -model.config.n_positions :]

    key_dir = args.key_dir or default_key_dir(args.fhe_dir)
    client = FHEModelClient(path_dir=args.fhe_dir, key_dir=key_dir)
    client.generate_private_and_evaluation_keys()
    evaluation_keys = client.get_serialized_evaluation_keys()

    server = FHEModelServer(path_dir=args.fhe_dir)
    server.load()

    generated_ids = token_ids[0].tolist()
    for _ in range(args.max_new_tokens):
        context_ids = generated_ids[-model.config.n_positions :]
        prompt_length = len(context_ids)
        padded_ids = torch.full(
            (1, model.config.n_positions),
            tokenizer.pad_token_id,
            dtype=torch.long,
        )
        padded_ids[0, :prompt_length] = torch.tensor(context_ids, dtype=torch.long)

        with torch.inference_mode():
            embedded_context = (
                model.get_attention_input(padded_ids)
                .cpu()
                .numpy()
                .astype(np.float32)
            )

        encrypted_input = client.quantize_encrypt_serialize(embedded_context)
        encrypted_output = server.run(encrypted_input, evaluation_keys)
        hidden_states = client.deserialize_decrypt_dequantize(encrypted_output)
        hidden_states = torch.as_tensor(hidden_states, dtype=model.lm_head.weight.dtype)
        if hidden_states.shape != (
            1,
            model.config.n_positions,
            model.config.n_embd,
        ):
            raise RuntimeError(
                "Unexpected decrypted hidden-state shape: "
                f"{tuple(hidden_states.shape)}"
            )

        with torch.inference_mode():
            next_token_logits = model.lm_head(
                hidden_states[0, prompt_length - 1]
            )
        generated_ids.append(int(next_token_logits.argmax(dim=-1)))

    print(tokenizer.decode(generated_ids, skip_special_tokens=True))


if __name__ == "__main__":
    main()
