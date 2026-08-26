import torch
from torch import nn
from transformers import GenerationMixin, PreTrainedModel

from .recurrent import StatefulLSTMCore
from .hf_config import PersistentMemoryHFConfig


class PersistentMemoryLSTMModel(PreTrainedModel, GenerationMixin):
    config_class = PersistentMemoryHFConfig
    base_model_prefix = "model"

    def __init__(self, config: PersistentMemoryHFConfig):
        super().__init__(config)
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.position_embedding = nn.Embedding(int(config.block_size), config.embedding_dim)
        self.core = StatefulLSTMCore(
            config.embedding_dim,
            config.hidden_dim,
            config.num_layers,
            config.dropout,
            state_dropout_prob=config.state_dropout_prob,
        )
        self.norm = nn.LayerNorm(config.hidden_dim)
        self.output = nn.Linear(config.hidden_dim, config.vocab_size)
        self.post_init()

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def reset_state(self, batch_size: int, device):
        return self.core.reset_state(batch_size, device)

    @staticmethod
    def _apply_story_mask(state, active_mask):
        return StatefulLSTMCore.apply_story_mask(state, active_mask)

    def forward(self, input_ids, state=None, labels=None, active_mask=None, loss_mask=None, attention_mask=None, **kwargs):
        if input_ids.dim() != 2:
            raise ValueError(f"Expected input_ids shape [batch, seq], got {input_ids.shape}")

        batch_size, seq_len = input_ids.shape
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)

        hidden, state = self.core(
            x,
            recurrent_state=state,
            active_mask=active_mask,
            attention_mask=attention_mask,
            detach_state=False,
        )
        logits = self.output(self.norm(hidden))

        loss = None
        if labels is not None:
            loss_labels = labels.clone()
            if loss_mask is not None:
                loss_labels = loss_labels.masked_fill(~loss_mask, -100)
            elif attention_mask is not None:
                loss_labels = loss_labels.masked_fill(~attention_mask, -100)
            elif active_mask is not None:
                loss_labels = loss_labels.masked_fill(~active_mask.unsqueeze(1), -100)
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(logits.reshape(-1, logits.size(-1)), loss_labels.reshape(-1))

        return {"logits": logits, "loss": loss, "state": state}
