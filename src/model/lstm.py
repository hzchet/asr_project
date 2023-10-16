import torch
from torch.nn import Sequential, LSTM, LayerNorm, Linear
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from src.base import BaseModel


class LSTMBaseline(BaseModel):
    def __init__(self, input_size: int, hidden_size: int, n_class: int, num_layers: int = 1, bidirectional=False, **batch):
        super().__init__(input_size, hidden_size, **batch)
        if bidirectional:
            D = 2
        else:
            D = 1
        self.lstm1 = LSTM(input_size, hidden_size, num_layers=num_layers, bidirectional=bidirectional, batch_first=True)
        self.layer_norm = LayerNorm(hidden_size * D)
        self.lstm2 = LSTM(hidden_size * D, hidden_size, num_layers=num_layers, bidirectional=bidirectional, batch_first=True)
        self.linear = Linear(in_features=hidden_size * D, out_features=n_class)
        
    def forward(self, spectrogram, spectrogram_length, **batch):
        packed_input = pack_padded_sequence(
            spectrogram.permute(0, 2, 1), 
            spectrogram_length, 
            batch_first=True,
            enforce_sorted=False
        )
        output, hidden = self.lstm1(packed_input)
        output, lengths = pad_packed_sequence(output, batch_first=True)
        
        output = self.layer_norm(output)
        packed_input = pack_padded_sequence(
            output, 
            lengths, 
            batch_first=True,
            enforce_sorted=False
        )
        
        output, hidden = self.lstm2(packed_input, hidden)
        output, lengths = pad_packed_sequence(output, batch_first=True)
        
        logits = self.linear(output)
        return {"logits": logits}

    def transform_input_lengths(self, input_lengths):
        return input_lengths  # we don't reduce time dimension here
