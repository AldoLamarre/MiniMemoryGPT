import os
from dataclasses import dataclass
from datetime import datetime

import torch
from transformers import GenerationConfig, TrainerState
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from .hf_config import PersistentMemoryHFConfig
from .model import PersistentMemoryLSTMModel
from .minigpt import PersistentMiniGPTModel


@dataclass
class PersistentMemoryConfig:
    output_dir: str = "runs/persistent_memory"
    architecture: str = "lstm"
    batch_size: int = 2
    gradient_accumulation_steps: int = 1
    epochs: int = 2
    save_steps: int = 0
    save_total_limit: int = 2
    learning_rate: float = 3e-4
    vocab_size: int = 50257
    embedding_dim: int = 128
    hidden_dim: int = 128
    num_layers: int = 2
    dropout: float = 0.1
    state_dropout_prob: float = 0.0
    block_size: int = 512
    bptt_length: int | None = 32
    detach_state: bool = True
    n_head: int = 4
    pre_layers: int = 1
    post_layers: int = 1
    intermediate_size: int | None = None
    max_train_samples: int = 2000
    max_eval_samples: int = 200
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    bf16: bool = False


class PersistentMemoryTrainer:
    def __init__(self, config: PersistentMemoryConfig, tokenizer):
        self.config = config
        self.tokenizer = tokenizer
        self.device = torch.device(config.device)
        hf_config = PersistentMemoryHFConfig(
            architecture=config.architecture,
            vocab_size=config.vocab_size,
            embedding_dim=config.embedding_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            dropout=config.dropout,
            state_dropout_prob=config.state_dropout_prob,
            block_size=config.block_size,
            n_head=config.n_head,
            pre_layers=config.pre_layers,
            post_layers=config.post_layers,
            intermediate_size=config.intermediate_size,
            pad_token_id=getattr(tokenizer, "pad_token_id", None),
            bos_token_id=getattr(tokenizer, "bos_token_id", None),
            eos_token_id=getattr(tokenizer, "eos_token_id", None),
        )
        if config.architecture == "lstm":
            self.model = PersistentMemoryLSTMModel(hf_config).to(self.device)
        elif config.architecture in {"minigpt", "transformer_lstm_transformer"}:
            self.model = PersistentMiniGPTModel(hf_config).to(self.device)
        else:
            raise ValueError(f"Unknown architecture: {config.architecture}")
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.learning_rate)
        self.log_history = []
        self.generation_config = GenerationConfig(
            pad_token_id=getattr(tokenizer, "pad_token_id", None),
            bos_token_id=getattr(tokenizer, "bos_token_id", None),
            eos_token_id=getattr(tokenizer, "eos_token_id", None),
        )
        self.model.generation_config = self.generation_config

    def _pad_story_batch(self, texts):
        tokenized = [torch.tensor(self.tokenizer.encode(text, truncation=False), dtype=torch.long, device=self.device) for text in texts]
        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        if not tokenized:
            empty = torch.empty((0, 0), dtype=torch.long, device=self.device)
            return empty, torch.empty((0,), dtype=torch.long, device=self.device)
        lengths = torch.tensor([ids.numel() for ids in tokenized], device=self.device, dtype=torch.long)
        batch = pad_sequence(tokenized, batch_first=True, padding_value=pad_token_id)
        return batch, lengths

    def _build_shifted_targets(self, batch_ids, lengths):
        labels = torch.full_like(batch_ids, self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0)
        if batch_ids.size(1) > 1:
            labels[:, :-1] = batch_ids[:, 1:]
        positions = torch.arange(batch_ids.size(1), device=batch_ids.device).unsqueeze(0)
        loss_mask = positions + 1 < lengths.unsqueeze(1)
        return labels, loss_mask

    def _forward_batch(self, stories, state=None):
        if len(stories) == 0:
            return None, state, 0

        batch_ids, lengths = self._pad_story_batch(stories)
        batch_size = batch_ids.size(0)
        if state is None:
            state = self.model.reset_state(batch_size, self.device)

        max_seq_len = batch_ids.size(1)
        if self.config.bptt_length is not None and self.config.bptt_length > 0:
            chunk_size = min(self.config.bptt_length, self.config.block_size)
        else:
            chunk_size = self.config.block_size
        labels, loss_mask = self._build_shifted_targets(batch_ids, lengths)

        total_loss = torch.tensor(0.0, device=self.device)
        total_tokens = 0
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=(self.config.bf16 and self.device.type == "cuda")):
            for start in range(0, max_seq_len, chunk_size):
                end = min(start + chunk_size, max_seq_len)
                chunk_input = batch_ids[:, start:end]
                active_mask = lengths > start
                chunk_attention_mask = chunk_input.ne(self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0)

                chunk_labels = labels[:, start:end]
                chunk_mask = loss_mask[:, start:end]
                valid_tokens = int(chunk_mask.sum().item())
                if valid_tokens == 0:
                    outputs = self.model(
                        input_ids=chunk_input,
                        state=state,
                        labels=None,
                        active_mask=active_mask,
                        loss_mask=None,
                        attention_mask=chunk_attention_mask,
                    )
                else:
                    outputs = self.model(
                        input_ids=chunk_input,
                        state=state,
                        labels=chunk_labels,
                        active_mask=active_mask,
                        loss_mask=chunk_mask,
                        attention_mask=chunk_attention_mask,
                    )
                    if outputs["loss"] is not None:
                        total_loss = total_loss + outputs["loss"] * valid_tokens
                        total_tokens += valid_tokens
                state = outputs["state"]
                if self.config.detach_state:
                    state = (state[0].detach(), state[1].detach())

        if total_tokens == 0:
            return None, state, 0
        return total_loss / total_tokens, state, total_tokens

    def fit(self, train_dataset, eval_dataset):
        self.model.train()
        run_dir = self._make_run_dir(self.config.output_dir)
        os.makedirs(run_dir, exist_ok=True)

        best_loss = float("inf")
        best_checkpoint_path = None
        dataloader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)
        grad_accum_steps = max(1, self.config.gradient_accumulation_steps)
        save_steps = max(0, self.config.save_steps)
        global_step = 0

        for epoch in range(self.config.epochs):
            self.model.train()
            epoch_loss = 0.0
            epoch_tokens = 0
            steps = 0
            self.optimizer.zero_grad()
            for batch_idx, batch in enumerate(dataloader):
                if isinstance(batch, dict):
                    texts = batch["text"] if isinstance(batch["text"], list) else [batch["text"]]
                else:
                    texts = [item["text"] if isinstance(item, dict) else item for item in batch]
                story_loss, _, story_tokens = self._forward_batch(texts)
                if story_loss is None:
                    continue
                (story_loss / grad_accum_steps).backward()
                epoch_loss += story_loss.item() * max(1, story_tokens)
                epoch_tokens += max(1, story_tokens)
                steps += 1
                global_step += 1
                should_step = ((batch_idx + 1) % grad_accum_steps == 0) or ((batch_idx + 1) == len(dataloader))
                if should_step:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                if save_steps > 0 and global_step % save_steps == 0:
                    self._save_checkpoint(run_dir, f"checkpoint-step-{global_step:08d}", epoch, global_step)

            eval_loss = self._evaluate(eval_dataset)
            if epoch_tokens == 0:
                epoch_tokens = max(1, steps)
            self.log_history.append(
                {
                    "epoch": float(epoch + 1),
                    "loss": epoch_loss / epoch_tokens,
                    "eval_loss": eval_loss,
                }
            )
            print(f"epoch={epoch + 1} train_loss={epoch_loss / epoch_tokens:.4f} eval_loss={eval_loss:.4f}")
            self._save_checkpoint(
                run_dir,
                f"checkpoint-epoch-{epoch + 1:03d}",
                epoch + 1,
                global_step,
                best_metric=best_loss if best_loss < float("inf") else None,
                best_model_checkpoint=best_checkpoint_path,
            )
            if eval_loss < best_loss:
                best_loss = eval_loss
                best_checkpoint_path = os.path.join(run_dir, f"checkpoint-epoch-{epoch + 1:03d}")
                self.model.save_pretrained(run_dir, safe_serialization=True)
                self.tokenizer.save_pretrained(run_dir)
                self._save_trainer_state(
                    run_dir,
                    epoch=epoch + 1,
                    global_step=global_step,
                    best_metric=best_loss,
                    best_model_checkpoint=best_checkpoint_path,
                )
            else:
                self._save_trainer_state(
                    run_dir,
                    epoch=epoch + 1,
                    global_step=global_step,
                    best_metric=best_loss if best_loss < float("inf") else None,
                    best_model_checkpoint=best_checkpoint_path,
                )
            self._trim_checkpoints(run_dir, self.config.save_total_limit)

        return run_dir

    def _evaluate(self, dataset_split):
        self.model.eval()
        total_loss = 0.0
        total_tokens = 0
        with torch.no_grad():
            for item in dataset_split:
                loss, _, token_count = self._forward_batch([item["text"]])
                if loss is not None:
                    total_loss += loss.item() * max(1, token_count)
                    total_tokens += max(1, token_count)

        self.model.train()
        if total_tokens == 0:
            return float("nan")
        return total_loss / total_tokens

    def _make_run_dir(self, base_dir: str) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(base_dir, f"run_{timestamp}")
        os.makedirs(run_dir, exist_ok=True)
        return run_dir

    def _save_checkpoint(
        self,
        run_dir: str,
        checkpoint_name: str,
        epoch: int,
        global_step: int,
        best_metric: float | None = None,
        best_model_checkpoint: str | None = None,
    ):
        checkpoint_dir = os.path.join(run_dir, checkpoint_name)
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.model.save_pretrained(checkpoint_dir, safe_serialization=True)
        self.generation_config.save_pretrained(checkpoint_dir)
        torch.save(self.optimizer.state_dict(), os.path.join(checkpoint_dir, "optimizer.pt"))
        torch.save(self._capture_rng_state(), os.path.join(checkpoint_dir, "rng_state.pth"))
        torch.save(
            {
                "output_dir": self.config.output_dir,
                "gradient_accumulation_steps": self.config.gradient_accumulation_steps,
                "epochs": self.config.epochs,
                "save_steps": self.config.save_steps,
                "save_total_limit": self.config.save_total_limit,
                "learning_rate": self.config.learning_rate,
                "batch_size": self.config.batch_size,
                "block_size": self.config.block_size,
                "bptt_length": self.config.bptt_length,
                "detach_state": self.config.detach_state,
                "state_dropout_prob": self.config.state_dropout_prob,
                "architecture": self.config.architecture,
                "bf16": self.config.bf16,
            },
            os.path.join(checkpoint_dir, "training_args.bin"),
        )
        self._save_trainer_state(
            checkpoint_dir,
            epoch=epoch,
            global_step=global_step,
            best_metric=best_metric,
            best_model_checkpoint=best_model_checkpoint,
        )

    def _save_trainer_state(
        self,
        output_dir: str,
        epoch: int,
        global_step: int,
        best_metric: float | None,
        best_model_checkpoint: str | None,
    ):
        state = TrainerState()
        state.best_global_step = global_step if best_metric is not None else None
        state.best_metric = best_metric
        state.best_model_checkpoint = best_model_checkpoint
        state.epoch = float(epoch)
        state.global_step = global_step
        state.log_history = list(self.log_history)
        state.save_to_json(os.path.join(output_dir, "trainer_state.json"))

    def _trim_checkpoints(self, run_dir: str, save_total_limit: int):
        if save_total_limit <= 0:
            return
        checkpoint_dirs = [
            os.path.join(run_dir, name)
            for name in os.listdir(run_dir)
            if name.startswith("checkpoint-") and os.path.isdir(os.path.join(run_dir, name))
        ]
        if len(checkpoint_dirs) <= save_total_limit:
            return
        checkpoint_dirs.sort(key=os.path.getmtime)
        for checkpoint_dir in checkpoint_dirs[:-save_total_limit]:
            for root, dirs, files in os.walk(checkpoint_dir, topdown=False):
                for file_name in files:
                    os.remove(os.path.join(root, file_name))
                for dir_name in dirs:
                    os.rmdir(os.path.join(root, dir_name))
            os.rmdir(checkpoint_dir)

    def _capture_rng_state(self):
        rng_state = {
            "python": None,
            "torch": torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            rng_state["cuda"] = torch.cuda.get_rng_state_all()
        return rng_state
