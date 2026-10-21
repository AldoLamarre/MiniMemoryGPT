from transformers import PretrainedConfig


class MiniTinyStoryGPTConfig(PretrainedConfig):
    model_type = "mini_tinystory_gpt"

    def __init__(
        self,
        vocab_size=50257,
        n_positions=256,
        n_embd=256,
        n_layer=2,
        n_head=4,
        intermediate_size=512,
        dropout=0.1,
        pad_token_id=50256,
        memory_mode="hybrid",
        bptt_length=None,
        detach_recurrent_state=True,
        power_softmax_power=4,
        power_softmax_epsilon=1e-3,
        attention_score_clip=2.0,
        attention_score_range=None,
        concrete_layer_norm_epsilon=1e-5,
        concrete_layer_norm_input_range=4.0,
        concrete_gelu_input_range=2.5,
        concrete_polynomial_reciprocal_steps=32,
        concrete_inverse_sqrt_steps=24,
        polynomial_inference=False,
        **kwargs,
    ):
        if memory_mode not in {"hybrid", "attention", "lstm", "none"}:
            raise ValueError("memory_mode must be one of: hybrid, attention, lstm, none")
        if bptt_length is not None and bptt_length <= 0:
            raise ValueError("bptt_length must be None or a positive integer")
        if not isinstance(power_softmax_power, int) or power_softmax_power < 2 or power_softmax_power % 2:
            raise ValueError("power_softmax_power must be a positive even integer")
        if power_softmax_epsilon <= 0:
            raise ValueError("power_softmax_epsilon must be positive")
        if attention_score_clip <= 0:
            raise ValueError("attention_score_clip must be positive")
        if attention_score_range is not None and attention_score_range <= 0:
            raise ValueError("attention_score_range must be positive")
        if concrete_layer_norm_epsilon <= 0:
            raise ValueError("concrete_layer_norm_epsilon must be positive")
        if concrete_layer_norm_input_range <= 0 or concrete_gelu_input_range <= 0:
            raise ValueError("Concrete polynomial input ranges must be positive")
        if concrete_polynomial_reciprocal_steps <= 0 or concrete_inverse_sqrt_steps <= 0:
            raise ValueError("Concrete polynomial iteration counts must be positive")

        self.vocab_size = vocab_size
        self.n_positions = n_positions
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.intermediate_size = intermediate_size
        self.dropout = dropout
        self.memory_mode = memory_mode
        self.bptt_length = bptt_length
        self.detach_recurrent_state = detach_recurrent_state
        self.power_softmax_power = power_softmax_power
        self.power_softmax_epsilon = power_softmax_epsilon
        self.attention_score_clip = attention_score_clip
        self.attention_score_range = (
            attention_score_clip if attention_score_range is None else attention_score_range
        )
        self.concrete_layer_norm_epsilon = concrete_layer_norm_epsilon
        self.concrete_layer_norm_input_range = concrete_layer_norm_input_range
        self.concrete_gelu_input_range = concrete_gelu_input_range
        self.concrete_polynomial_reciprocal_steps = concrete_polynomial_reciprocal_steps
        self.concrete_inverse_sqrt_steps = concrete_inverse_sqrt_steps
        self.polynomial_inference = polynomial_inference
        self.use_lstm = memory_mode in {"hybrid", "lstm"}
        self.use_attention = memory_mode in {"hybrid", "attention"}
        self.hidden_size = n_embd
        self.num_hidden_layers = n_layer
        self.num_attention_heads = n_head
        self.num_key_value_heads = n_head
        super().__init__(pad_token_id=pad_token_id, **kwargs)
