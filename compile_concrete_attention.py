import argparse
import os

import numpy as np
import torch
from transformers import AutoTokenizer

from minimemorygpt.concrete_attention import ConcreteAttentionGPT
from minimemorygpt.data import prepare_tinystories_datasets


def main():
    parser = argparse.ArgumentParser(
        description="Quantize and compile a trained PowerSoftmax model for Concrete ML."
    )
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--calibration_samples", type=int, default=32)
    parser.add_argument("--max_eval_samples", type=int, default=128)
    parser.add_argument("--n_bits", type=int, default=6)
    args = parser.parse_args()
    if args.calibration_samples < 1:
        parser.error("--calibration_samples must be positive")
    if not 2 <= args.n_bits <= 8:
        parser.error("--n_bits must be between 2 and 8")

    try:
        from concrete.ml.deployment import FHEModelDev
        from concrete.ml.torch.compile import compile_torch_model
    except ImportError as error:
        raise SystemExit(
            "Concrete ML is not installed. Install the optional dependencies with "
            "`pip install -r requirements-concrete.txt`."
        ) from error

    model = ConcreteAttentionGPT.from_pretrained(args.model_dir).eval()
    model.set_polynomial_inference()
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    _, eval_ds = prepare_tinystories_datasets(
        tokenizer,
        block_size=model.config.n_positions,
        max_train_samples=1,
        max_eval_samples=args.max_eval_samples,
    )

    calibration_inputs = []
    with torch.inference_mode():
        for sample in eval_ds:
            input_ids = torch.tensor(sample["input_ids"], dtype=torch.long).unsqueeze(0)
            embedded_context = model.get_attention_input(input_ids)
            calibration_inputs.append(embedded_context.squeeze(0).cpu().numpy())
            if len(calibration_inputs) == args.calibration_samples:
                break
    if len(calibration_inputs) != args.calibration_samples:
        raise ValueError(
            f"Requested {args.calibration_samples} calibration samples, "
            f"but only found {len(calibration_inputs)}."
        )

    inputset = np.stack(calibration_inputs)
    encrypted_model = model.encrypted_attention_encoder().eval()
    quantized_module = compile_torch_model(
        encrypted_model,
        inputset,
        n_bits=args.n_bits,
        rounding_threshold_bits={"n_bits": 6, "method": "approximate"},
    )
    os.makedirs(args.output_dir, exist_ok=True)
    FHEModelDev(path_dir=args.output_dir, model=quantized_module).save()
    print(
        f"Quantized Concrete ML client/server artifacts saved to {args.output_dir}. "
        "Generate private/evaluation keys on the client."
    )


if __name__ == "__main__":
    main()
