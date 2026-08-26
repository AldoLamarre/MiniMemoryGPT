import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class StatefulLSTMCore(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, dropout: float, state_dropout_prob: float = 0.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.state_dropout_prob = state_dropout_prob
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )

    def reset_state(self, batch_size: int, device):
        zeros = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        return zeros.clone(), zeros.clone()

    @staticmethod
    def apply_story_mask(state, active_mask):
        if state is None or active_mask is None:
            return state
        h, c = state
        finished = ~active_mask
        if finished.any():
            h = h.clone()
            c = c.clone()
            h[:, finished, :] = 0
            c[:, finished, :] = 0
        return h, c

    def apply_state_dropout(self, state):
        if state is None or not self.training or self.state_dropout_prob <= 0.0:
            return state
        h, c = state
        keep_prob = 1.0 - self.state_dropout_prob
        if keep_prob <= 0.0:
            zeros_h = torch.zeros_like(h)
            zeros_c = torch.zeros_like(c)
            return zeros_h, zeros_c
        mask = torch.empty_like(h).bernoulli_(keep_prob).div_(keep_prob)
        return h * mask, c * mask

    def forward(self, hidden_states, recurrent_state=None, active_mask=None, attention_mask=None, detach_state=True):
        batch_size = hidden_states.size(0)
        if recurrent_state is None:
            recurrent_state = self.reset_state(batch_size, hidden_states.device)
        recurrent_state = self.apply_story_mask(recurrent_state, active_mask)

        if attention_mask is None:
            outputs, recurrent_state = self.lstm(hidden_states, recurrent_state)
        else:
            lengths = attention_mask.to(dtype=torch.long).sum(dim=1)
            outputs = hidden_states.new_zeros((batch_size, hidden_states.size(1), self.hidden_dim))

            active_rows = lengths > 0
            if active_rows.any():
                active_idx = active_rows.nonzero(as_tuple=False).squeeze(1)
                active_inputs = hidden_states.index_select(0, active_idx)
                active_lengths = lengths.index_select(0, active_idx)
                active_state = (
                    recurrent_state[0].index_select(1, active_idx),
                    recurrent_state[1].index_select(1, active_idx),
                )
                packed_inputs = pack_padded_sequence(
                    active_inputs,
                    active_lengths.to("cpu"),
                    batch_first=True,
                    enforce_sorted=False,
                )
                packed_outputs, active_state = self.lstm(packed_inputs, active_state)
                active_outputs, _ = pad_packed_sequence(
                    packed_outputs,
                    batch_first=True,
                    total_length=hidden_states.size(1),
                )
                if outputs.dtype != active_outputs.dtype:
                    outputs = outputs.to(dtype=active_outputs.dtype)
                outputs.index_copy_(0, active_idx, active_outputs)

                h = recurrent_state[0].to(dtype=active_state[0].dtype).clone()
                c = recurrent_state[1].to(dtype=active_state[1].dtype).clone()
                h[:, active_idx, :] = active_state[0]
                c[:, active_idx, :] = active_state[1]
                recurrent_state = (h, c)

        recurrent_state = self.apply_story_mask(recurrent_state, active_mask)
        if detach_state and recurrent_state is not None:
            recurrent_state = (recurrent_state[0].detach(), recurrent_state[1].detach())
        recurrent_state = self.apply_state_dropout(recurrent_state)
        return outputs, recurrent_state
