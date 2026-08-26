import torch
from torch import nn
from transformers import GenerationMixin, PreTrainedModel

from .hf_config import PersistentMemoryHFConfig
from .recurrent import StatefulLSTMCore


class CausalTransformerBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, intermediate_size: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(hidden_size)
        self.ff = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(intermediate_size, hidden_size),
            nn.Dropout(dropout),
        )

    def forward(self, hidden_states, attention_mask=None):
        seq_len = hidden_states.size(1)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=hidden_states.device, dtype=torch.bool),
            diagonal=1,
        )
        key_padding_mask = None if attention_mask is None else ~attention_mask.bool()

        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        attn_out, _ = self.attn(
            hidden_states,
            hidden_states,
            hidden_states,
            attn_mask=causal_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        hidden_states = residual + attn_out

        residual = hidden_states
        hidden_states = self.norm2(hidden_states)
        hidden_states = residual + self.ff(hidden_states)
        return hidden_states


class PersistentMiniGPTModel(PreTrainedModel, GenerationMixin):
    config_class = PersistentMemoryHFConfig
    base_model_prefix = "model"

    def __init__(self, config: PersistentMemoryHFConfig):
        super().__init__(config)
        self.config = config

        intermediate_size = config.intermediate_size or config.embedding_dim * 4

        self.token_embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.position_embedding = nn.Embedding(int(config.block_size), config.embedding_dim)
        self.pre_blocks = nn.ModuleList(
            [CausalTransformerBlock(config.embedding_dim, config.n_head, intermediate_size, config.dropout) for _ in range(config.pre_layers)]
        )
        self.to_lstm = nn.Linear(config.embedding_dim, config.hidden_dim) if config.embedding_dim != config.hidden_dim else nn.Identity()
        self.lstm_core = StatefulLSTMCore(
            config.hidden_dim,
            config.hidden_dim,
            config.num_layers,
            config.dropout,
            state_dropout_prob=config.state_dropout_prob,
        )
        self.from_lstm = nn.Linear(config.hidden_dim, config.embedding_dim) if config.embedding_dim != config.hidden_dim else nn.Identity()
        self.post_blocks = nn.ModuleList(
            [CausalTransformerBlock(config.embedding_dim, config.n_head, intermediate_size, config.dropout) for _ in range(config.post_layers)]
        )
        self.norm = nn.LayerNorm(config.embedding_dim)
        self.output = nn.Linear(config.embedding_dim, config.vocab_size)
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
        return self.lstm_core.reset_state(batch_size, device)

    @staticmethod
    def _apply_story_mask(state, active_mask):
        return StatefulLSTMCore.apply_story_mask(state, active_mask)

    def forward(self, input_ids, state=None, labels=None, active_mask=None, loss_mask=None, attention_mask=None):
        if input_ids.dim() != 2:
            raise ValueError(f"Expected input_ids shape [batch, seq], got {input_ids.shape}")

        batch_size, seq_len = input_ids.shape
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        hidden_states = self.token_embedding(input_ids) + self.position_embedding(positions)

        for block in self.pre_blocks:
            hidden_states = block(hidden_states, attention_mask=attention_mask)

        hidden_states = self.to_lstm(hidden_states)
        hidden_states, state = self.lstm_core(
            hidden_states,
            recurrent_state=state,
            active_mask=active_mask,
            attention_mask=attention_mask,
            detach_state=False,
        )
        hidden_states = self.from_lstm(hidden_states)

        for block in self.post_blocks:
            hidden_states = block(hidden_states, attention_mask=attention_mask)

        logits = self.output(self.norm(hidden_states))

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
