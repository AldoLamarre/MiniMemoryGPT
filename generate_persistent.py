import argparse
import torch
import torch.nn.functional as F

from minimemorygpt.data import get_tokenizer
from persistent_memory.minigpt import PersistentMiniGPTModel


def generate_story(
    model_dir: str = "runs/persistent_50m_BPTT_256/run_20260826_192033",
    prompt: str = "Once upon a time",
    max_new_tokens: int = 100,
    temperature: float = 0.8,
    device: str | None = None,
) -> str:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device_obj = torch.device(device)

    tokenizer = get_tokenizer(model_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = PersistentMiniGPTModel.from_pretrained(model_dir).to(device_obj).eval()

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device_obj)
    state = model.reset_state(1, device_obj)

    tokens = input_ids[0].tolist()
    with torch.no_grad():
        out = model(input_ids=input_ids, state=state)
        state = out["state"]
        logits = out["logits"][:, -1, :] / max(temperature, 1e-5)
        next_token = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
        tokens.append(next_token.item())

        for _ in range(max_new_tokens):
            if next_token.item() == tokenizer.eos_token_id:
                break
            out = model(input_ids=next_token, state=state)
            state = out["state"]
            logits = out["logits"][:, -1, :] / max(temperature, 1e-5)
            next_token = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
            tokens.append(next_token.item())

    return tokenizer.decode(tokens, skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser(description="Generate a story using a trained persistent-memory MiniGPT model.")
    parser.add_argument(
        "--model_dir",
        type=str,
        default="runs/persistent_50m_BPTT_256/run_20260826_192033",
        help="Path to the persistent model checkpoint directory",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Once upon a time, there was a playful puppy named Max.",
        help="Initial prompt for the story",
    )
    parser.add_argument("--max_new_tokens", type=int, default=100, help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    args = parser.parse_args()

    story = generate_story(
        model_dir=args.model_dir,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )
    print("\n--- Generated Story ---\n")
    print(story)


if __name__ == "__main__":
    main()
