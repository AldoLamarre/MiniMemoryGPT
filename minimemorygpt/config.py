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
        **kwargs,
    ):
        if memory_mode not in {"hybrid", "attention", "lstm", "none"}:
            raise ValueError("memory_mode must be one of: hybrid, attention, lstm, none")
        if bptt_length is not None and bptt_length <= 0:
            raise ValueError("bptt_length must be None or a positive integer")

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
        self.use_lstm = memory_mode in {"hybrid", "lstm"}
        self.use_attention = memory_mode in {"hybrid", "attention"}
        self.hidden_size = n_embd
        self.num_hidden_layers = n_layer
        self.num_attention_heads = n_head
        self.num_key_value_heads = n_head
        super().__init__(pad_token_id=pad_token_id, **kwargs)
