from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

from .config import MiniTinyStoryGPTConfig


def _range_penalty(values, limit: float):
    return F.relu(values.abs() - limit).square().mean()


def _goldschmidt_reciprocal(values, upper_bound: float, steps: int):
    reciprocal = values * 0.0 + (1.0 / upper_bound)
    residual = values * reciprocal
    for _ in range(steps):
        correction = 2.0 - residual
        reciprocal = reciprocal * correction
        residual = residual * correction
    return reciprocal


def _goldschmidt_inverse_sqrt(values, upper_bound: float, steps: int):
    inverse_sqrt = values * 0.0 + (1.0 / (upper_bound ** 0.5))
    residual = values * inverse_sqrt * inverse_sqrt
    for _ in range(steps):
        correction = 1.5 - 0.5 * residual
        inverse_sqrt = inverse_sqrt * correction
        residual = residual * correction * correction
    return inverse_sqrt


class ConcreteLayerNorm(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        epsilon: float,
        input_range: float,
        inverse_sqrt_steps: int,
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.hidden_size = hidden_size
        self.epsilon = epsilon
        self.input_range = input_range
        self.inverse_sqrt_steps = inverse_sqrt_steps
        self.variance_upper_bound = input_range**2 + epsilon

    def forward(self, hidden_states, range_losses=None, polynomial_inference=False):
        mean = hidden_states.sum(dim=-1, keepdim=True) / self.hidden_size
        centered = hidden_states - mean
        variance = (centered * centered).sum(dim=-1, keepdim=True) / self.hidden_size
        if range_losses is not None:
            range_losses.append(_range_penalty(hidden_states, self.input_range))
        variance_with_epsilon = variance + self.epsilon
        if polynomial_inference:
            inverse_std = _goldschmidt_inverse_sqrt(
                variance_with_epsilon,
                self.variance_upper_bound,
                self.inverse_sqrt_steps,
            )
        else:
            inverse_std = torch.rsqrt(variance_with_epsilon)
        return centered * inverse_std * self.weight + self.bias


class ConcreteGELU(nn.Module):
    input_range = 2.5

    def __init__(self, input_range: float, polynomial_inference: bool = False):
        super().__init__()
        self.input_range = input_range
        self.polynomial_inference = polynomial_inference

    def forward(self, hidden_states, range_losses=None, polynomial_inference=None):
        if range_losses is not None:
            range_losses.append(_range_penalty(hidden_states, self.input_range))
        use_polynomial = (
            self.polynomial_inference
            if polynomial_inference is None
            else polynomial_inference
        )
        if not use_polynomial:
            return F.gelu(hidden_states)

        # Degree-11 odd polynomial fit for sigmoid on [-4.25, 4.25].
        value = 1.702 * hidden_states
        squared = value * value
        sigmoid_delta = value * (
            0.2496353654616534
            + squared
            * (
                -0.020144167887172652
                + squared
                * (
                    0.0016898111691979157
                    + squared
                    * (
                        -0.00010343287358345755
                        + squared
                        * (
                            3.7283924620209375e-6
                            - squared * 5.7292322724565884e-8
                        )
                    )
                )
            )
        )
        return hidden_states * (0.5 + sigmoid_delta)


class PowerSoftmax(nn.Module):
    def __init__(
        self,
        power: int,
        epsilon: float,
        score_range: float,
        sequence_length: int,
        reciprocal_steps: int,
        polynomial_inference: bool = False,
    ):
        super().__init__()
        self.power = power
        self.epsilon = epsilon
        self.score_range = score_range
        self.sequence_length = sequence_length
        self.reciprocal_steps = reciprocal_steps
        self.polynomial_inference = polynomial_inference

    def forward(self, scores, causal_mask, range_losses=None, polynomial_inference=None):
        use_polynomial = (
            self.polynomial_inference
            if polynomial_inference is None
            else polynomial_inference
        )
        if range_losses is not None:
            range_losses.append(_range_penalty(scores, self.score_range))
        if use_polynomial:
            scaled_scores = scores * (self.sequence_length ** (-1.0 / self.power))
        else:
            causal_scores = scores * causal_mask
            scale = causal_scores.abs().amax(dim=-1, keepdim=True) + 1e-6
            scaled_scores = causal_scores / scale

        scaled_scores = scaled_scores * causal_mask
        powered_scores = scaled_scores
        for _ in range(self.power - 1):
            powered_scores = powered_scores * scaled_scores

        powered_scores = powered_scores * causal_mask
        if use_polynomial:
            mean_power = powered_scores.sum(dim=-1, keepdim=True) / self.sequence_length
            denominator = mean_power + (
                self.epsilon / (self.sequence_length * self.sequence_length)
            )
            reciprocal = _goldschmidt_reciprocal(
                denominator,
                upper_bound=1.0
                + self.epsilon / (self.sequence_length * self.sequence_length),
                steps=self.reciprocal_steps,
            )
            return powered_scores * reciprocal / self.sequence_length

        denominator = powered_scores.sum(dim=-1, keepdim=True) + self.epsilon
        return powered_scores / denominator


class ConcreteAttentionHead(nn.Module):
    def __init__(self, config: MiniTinyStoryGPTConfig, head_dim: int):
        super().__init__()
        self.query = nn.Linear(config.n_embd, head_dim)
        self.key = nn.Linear(config.n_embd, head_dim)
        self.value = nn.Linear(config.n_embd, head_dim)
        self.head_dim = head_dim
        self.power_softmax = PowerSoftmax(
            power=config.power_softmax_power,
            epsilon=config.power_softmax_epsilon,
            score_range=config.attention_score_range,
            sequence_length=config.n_positions,
            reciprocal_steps=config.concrete_polynomial_reciprocal_steps,
            polynomial_inference=config.polynomial_inference,
        )
        self.attention_dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        hidden_states,
        causal_mask,
        range_losses=None,
        polynomial_inference=False,
    ):
        query = self.query(hidden_states)
        key = self.key(hidden_states)
        value = self.value(hidden_states)
        scores = torch.matmul(query, key.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attention_weights = self.power_softmax(
            scores,
            causal_mask,
            range_losses=range_losses,
            polynomial_inference=polynomial_inference,
        )
        return torch.matmul(self.attention_dropout(attention_weights), value)


class ConcreteAttentionBlock(nn.Module):
    def __init__(self, config: MiniTinyStoryGPTConfig):
        super().__init__()
        head_dim = config.n_embd // config.n_head
        self.heads = nn.ModuleList(
            [ConcreteAttentionHead(config, head_dim) for _ in range(config.n_head)]
        )
        self.attention_output = nn.Linear(config.n_embd, config.n_embd)
        self.attention_norm = ConcreteLayerNorm(
            config.n_embd,
            config.concrete_layer_norm_epsilon,
            config.concrete_layer_norm_input_range,
            config.concrete_inverse_sqrt_steps,
        )
        self.feed_forward_norm = ConcreteLayerNorm(
            config.n_embd,
            config.concrete_layer_norm_epsilon,
            config.concrete_layer_norm_input_range,
            config.concrete_inverse_sqrt_steps,
        )
        self.feed_forward = nn.Sequential(
            nn.Linear(config.n_embd, config.intermediate_size),
            nn.Dropout(config.dropout),
            nn.Linear(config.intermediate_size, config.n_embd),
            nn.Dropout(config.dropout),
        )
        self.gelu = ConcreteGELU(config.concrete_gelu_input_range)
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(config.n_positions, config.n_positions)),
        )

    def forward(self, hidden_states, range_losses=None, polynomial_inference=False):
        sequence_length = hidden_states.size(-2)
        causal_mask = self.causal_mask[:sequence_length, :sequence_length]
        normalized_states = self.attention_norm(
            hidden_states, range_losses, polynomial_inference
        )
        context = torch.cat(
            [
                head(
                    normalized_states,
                    causal_mask,
                    range_losses,
                    polynomial_inference,
                )
                for head in self.heads
            ],
            dim=-1,
        )
        hidden_states = hidden_states + self.attention_output(context)
        normalized_states = self.feed_forward_norm(
            hidden_states, range_losses, polynomial_inference
        )
        feed_forward_hidden = self.feed_forward[0](normalized_states)
        feed_forward_hidden = self.gelu(
            feed_forward_hidden, range_losses, polynomial_inference
        )
        feed_forward_hidden = self.feed_forward[1](feed_forward_hidden)
        feed_forward_hidden = self.feed_forward[2](feed_forward_hidden)
        feed_forward_hidden = self.feed_forward[3](feed_forward_hidden)
        return hidden_states + feed_forward_hidden


class ConcreteAttentionCore(nn.Module):
    def __init__(self, config: MiniTinyStoryGPTConfig):
        super().__init__()
        self.final_norm = ConcreteLayerNorm(
            config.n_embd,
            config.concrete_layer_norm_epsilon,
            config.concrete_layer_norm_input_range,
            config.concrete_inverse_sqrt_steps,
        )
        self.blocks = nn.ModuleList(
            [ConcreteAttentionBlock(config) for _ in range(config.n_layer)]
        )

    def forward(self, hidden_states, range_losses=None, polynomial_inference=False):
        for block in self.blocks:
            hidden_states = block(hidden_states, range_losses, polynomial_inference)
        hidden_states = self.final_norm(
            hidden_states, range_losses, polynomial_inference
        )
        if range_losses:
            return hidden_states, torch.stack(range_losses).sum()
        return hidden_states


class EncryptedAttentionEncoder(nn.Module):
    def __init__(self, attention_core: ConcreteAttentionCore):
        super().__init__()
        self.attention_core = attention_core

    def forward(self, input_embeddings):
        return self.attention_core(input_embeddings, polynomial_inference=True)


@dataclass
class ConcreteCausalLMOutput(CausalLMOutputWithCrossAttentions):
    range_loss: torch.Tensor | None = None


class ConcreteAttentionGPT(PreTrainedModel):
    config_class = MiniTinyStoryGPTConfig
    base_model_prefix = "model"

    def __init__(self, config: MiniTinyStoryGPTConfig):
        if config.memory_mode != "attention":
            raise ValueError("ConcreteAttentionGPT requires memory_mode='attention'")
        super().__init__(config)
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)
        self.attention_core = ConcreteAttentionCore(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.polynomial_inference = config.polynomial_inference
        self.post_init()

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_attention_input(self, input_ids):
        sequence_length = input_ids.size(1)
        if sequence_length > self.config.n_positions:
            raise ValueError(
                f"Input sequence length {sequence_length} exceeds "
                f"configured positions {self.config.n_positions}"
            )
        position_ids = torch.arange(sequence_length, device=input_ids.device).unsqueeze(0)
        return self.wte(input_ids) + self.wpe(position_ids)

    def encrypted_attention_encoder(self):
        return EncryptedAttentionEncoder(self.attention_core)

    def set_polynomial_inference(self, enabled: bool = True):
        self.polynomial_inference = enabled
        self.config.polynomial_inference = enabled
        for module in self.modules():
            if isinstance(module, PowerSoftmax):
                module.polynomial_inference = enabled

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        hidden_states = self.get_attention_input(input_ids)
        range_losses = []
        result = self.attention_core(
            hidden_states,
            range_losses,
            self.polynomial_inference,
        )
        if range_losses:
            hidden_states, range_loss = result
        else:
            hidden_states = result
            range_loss = None
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            if attention_mask is not None:
                shift_labels = shift_labels.masked_fill(attention_mask[..., 1:] == 0, -100)
            loss = nn.CrossEntropyLoss(ignore_index=-100)(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )

        return ConcreteCausalLMOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None,
            cross_attentions=None,
            range_loss=range_loss,
        )
