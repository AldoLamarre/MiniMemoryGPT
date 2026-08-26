from transformers import PretrainedConfig


class PersistentMemoryHFConfig(PretrainedConfig):
    model_type = "persistent_memory_mini_gpt"

    def __init__(
        self,
        architecture: str = "lstm",
        vocab_size: int = 50257,
        embedding_dim: int = 128,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        state_dropout_prob: float = 0.0,
        block_size: int = 512,
        n_head: int = 4,
        pre_layers: int = 1,
        post_layers: int = 1,
        intermediate_size: int | None = None,
        pad_token_id: int | None = None,
        bos_token_id: int | None = None,
        eos_token_id: int | None = None,
        **kwargs,
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )
        self.architecture = architecture
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.state_dropout_prob = state_dropout_prob
        self.block_size = block_size
        self.n_head = n_head
        self.pre_layers = pre_layers
        self.post_layers = post_layers
        self.intermediate_size = intermediate_size
