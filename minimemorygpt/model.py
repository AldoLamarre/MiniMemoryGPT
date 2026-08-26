import torch
from torch import nn
from transformers import GenerationMixin, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

from .config import MiniTinyStoryGPTConfig


class HybridAttentionLSTMBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        intermediate_size: int,
        dropout: float,
        use_lstm: bool = True,
        use_attention: bool = True,
    ):
        super().__init__()
        self.use_lstm = use_lstm
        self.use_attention = use_attention

        if self.use_attention:
            self.norm1 = nn.LayerNorm(hidden_size)
            self.attn1 = nn.MultiheadAttention(
                embed_dim=hidden_size,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True,
            )

        if self.use_lstm:
            self.norm2 = nn.LayerNorm(hidden_size)
            self.lstm = nn.LSTM(
                input_size=hidden_size,
                hidden_size=hidden_size,
                num_layers=2,
                batch_first=True,
                dropout=0.0,
            )

        if self.use_attention:
            self.norm3 = nn.LayerNorm(hidden_size)
            self.attn2 = nn.MultiheadAttention(
                embed_dim=hidden_size,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True,
            )

        self.norm4 = nn.LayerNorm(hidden_size)
        self.ff = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(intermediate_size, hidden_size),
            nn.Dropout(dropout),
        )

    def _run_lstm(self, hidden_states, attention_mask=None, recurrent_state=None, bptt_length=None, detach_state=True):
        if not self.use_lstm:
            return hidden_states, recurrent_state

        if bptt_length is None or hidden_states.size(1) <= bptt_length:
            lstm_out, state = self.lstm(hidden_states, recurrent_state)
            if detach_state:
                state = (state[0].detach(), state[1].detach())
            return lstm_out, state

        outputs = []
        state = recurrent_state
        for start in range(0, hidden_states.size(1), bptt_length):
            end = min(start + bptt_length, hidden_states.size(1))
            chunk = hidden_states[:, start:end]
            chunk_mask = None if attention_mask is None else attention_mask[:, start:end]
            chunk_lstm, state = self.lstm(chunk, state)
            if detach_state:
                state = (state[0].detach(), state[1].detach())
            outputs.append(chunk_lstm)
        return torch.cat(outputs, dim=1), state

    def forward(self, hidden_states, attention_mask=None, recurrent_state=None, bptt_length=None, detach_state=True):
        seq_len = hidden_states.size(1)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=hidden_states.device, dtype=torch.bool),
            diagonal=1,
        )

        if self.use_attention:
            residual = hidden_states
            hidden_states = self.norm1(hidden_states)
            attn_out, _ = self.attn1(
                hidden_states,
                hidden_states,
                hidden_states,
                attn_mask=causal_mask,
                key_padding_mask=None if attention_mask is None else (attention_mask == 0),
                need_weights=False,
            )
            hidden_states = residual + attn_out

        if self.use_lstm:
            residual = hidden_states
            hidden_states = self.norm2(hidden_states)
            lstm_out, recurrent_state = self._run_lstm(
                hidden_states,
                attention_mask=attention_mask,
                recurrent_state=recurrent_state,
                bptt_length=bptt_length,
                detach_state=detach_state,
            )
            hidden_states = residual + lstm_out

        if self.use_attention:
            residual = hidden_states
            hidden_states = self.norm3(hidden_states)
            attn_out, _ = self.attn2(
                hidden_states,
                hidden_states,
                hidden_states,
                attn_mask=causal_mask,
                key_padding_mask=None if attention_mask is None else (attention_mask == 0),
                need_weights=False,
            )
            hidden_states = residual + attn_out

        residual = hidden_states
        hidden_states = self.norm4(hidden_states)
        hidden_states = self.ff(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states, recurrent_state


class MiniTinyStoryGPTModel(PreTrainedModel, GenerationMixin):
    config_class = MiniTinyStoryGPTConfig
    base_model_prefix = "model"

    def __init__(self, config: MiniTinyStoryGPTConfig):
        super().__init__(config)
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)
        self.h = nn.ModuleList(
            [
                HybridAttentionLSTMBlock(
                    hidden_size=config.n_embd,
                    num_heads=config.n_head,
                    intermediate_size=config.intermediate_size,
                    dropout=config.dropout,
                    use_lstm=config.use_lstm,
                    use_attention=config.use_attention,
                )
                for _ in range(config.n_layer)
            ]
        )
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
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

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        position_ids=None,
        inputs_embeds=None,
        recurrent_state=None,
        **kwargs,
    ):
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("Either input_ids or inputs_embeds must be provided.")
            seq_len = input_ids.shape[1]
            position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
            hidden_states = self.wte(input_ids) + self.wpe(position_ids)
        else:
            hidden_states = inputs_embeds

        if attention_mask is not None:
            attention_mask = attention_mask.to(hidden_states.device)

        block_state = recurrent_state
        for block in self.h:
            hidden_states, block_state = block(
                hidden_states,
                attention_mask=attention_mask,
                recurrent_state=block_state,
                bptt_length=self.config.bptt_length,
                detach_state=self.config.detach_recurrent_state,
            )

        hidden_states = self.ln_f(hidden_states)
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None,
            cross_attentions=None,
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, **kwargs):
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

    def get_input_embeddings(self):
        return self.wte

    def set_input_embeddings(self, value):
        self.wte = value

    def resize_token_embeddings(self, new_num_tokens=None):
        old = self.wte
        new_embeddings = nn.Embedding(new_num_tokens, old.embedding_dim)
        new_embeddings.to(old.weight.device)
        nn.init.normal_(new_embeddings.weight, mean=0.0, std=0.02)
        self.wte = new_embeddings
        return new_embeddings

