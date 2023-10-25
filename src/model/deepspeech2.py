import math
from typing import List, Type, Tuple

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class NormalizedRNN(nn.Module):
    def __init__(
        self,
        input_size: int,
        rnn_type: Type = nn.GRU,
        hidden_size: int = 800,
        bidirectional: bool = True,
        rnn_norm: str = None
    ):
        super().__init__()
        assert rnn_norm in ('batchnorm', 'layernorm', None)
        
        D = 2 if bidirectional else 1
        self.rnn = rnn_type(input_size, hidden_size, bidirectional=bidirectional, batch_first=True)
        if rnn_norm == 'batchnorm':
            self.rnn_norm = nn.BatchNorm1d(D * hidden_size)
        elif rnn_norm == 'layernorm':
            self.rnn_norm = nn.LayerNorm(D * hidden_size)
        else:
            self.rnn_norm = nn.Identity()
    
    def forward(self, x, lengths, h=None):
        packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        x, h = self.rnn(packed, h)
        
        padded, lengths = pad_packed_sequence(x, batch_first=True)
        
        if isinstance(self.rnn_norm, nn.BatchNorm1d):
            padded = padded.transpose(1, 2)
            
        x = self.rnn_norm(padded)
        
        if isinstance(self.rnn_norm, nn.BatchNorm1d):
            x = x.transpose(1, 2)
        
        return x, lengths, h


class DeepSpeech2(nn.Module):
    def __init__(
        self, 
        channels: List[int], 
        kernel_sizes: List[Tuple[int, int]], 
        strides: List[Tuple[int, int]],
        paddings: List[Tuple[int, int]],
        n_feats: int = 128,
        n_class: int = 28,
        rnn_type: Type = nn.GRU,
        hidden_size: int = 800,
        bidirectional: bool = True,
        num_rnn_blocks: int = 5,
        rnn_norm: str = None,
        **batch
    ):
        super().__init__()
        self.channels = channels
        self.kernel_sizes = kernel_sizes
        self.strides = strides
        self.paddings = paddings
        
        self.conv_blocks, rnn_input_size = self._init_conv_blocks(n_feats, channels, kernel_sizes, 
                                                                  strides, paddings)
        self.rnn_blocks, fc_in_features = self._init_rnn_blocks(rnn_input_size, rnn_type, hidden_size, 
                                                bidirectional, rnn_norm, num_rnn_blocks)
        self.fc = nn.Linear(fc_in_features, n_class)
        
    def _init_conv_blocks(
        self,
        n_feats: int,
        channels: List[int], 
        kernel_sizes: List[Tuple[int, int]], 
        strides: List[Tuple[int, int]],
        paddings: List[Tuple[int, int]],
    ):
        assert len(channels) == len(kernel_sizes) == len(strides) == len(paddings)
        self.num_conv_layers = len(channels)
        
        channels = [1] + channels
        conv_blocks = []
        C, H = 1, n_feats
        for i in range(self.num_conv_layers):
            conv_blocks.append(nn.Sequential(
                nn.Conv2d(in_channels=channels[i], out_channels=channels[i + 1], 
                          kernel_size=kernel_sizes[i], stride=strides[i], padding=paddings[i]),
                nn.BatchNorm2d(channels[i + 1]),
                nn.Hardtanh(0, 20, inplace=True)
            ))
            
            C = channels[i + 1]
            H = int(math.floor((H + 2 * paddings[i][0] - (kernel_sizes[i][0] - 1) - 1) / strides[i][0] + 1))
        
        return nn.Sequential(*conv_blocks), C * H
    
    def _init_rnn_blocks(
        self,
        input_size: int,
        rnn_type: Type = nn.GRU,
        hidden_size: int = 800,
        bidirectional: bool = True,
        rnn_norm: str = None,
        num_rnn_blocks: int = 5
    ):
        assert rnn_norm in ('batchnorm', 'layernorm', None)
        D = 2 if bidirectional else 1
        rnn_blocks = []
        for i in range(num_rnn_blocks):
            rnn_blocks.append(NormalizedRNN(input_size, rnn_type, hidden_size, bidirectional, rnn_norm))
            input_size = D * hidden_size

        return nn.ModuleList(rnn_blocks), D * hidden_size
        
    def forward(self, spectrogram, spectrogram_length, **batch):
        x = spectrogram.unsqueeze(1)
        x = self.conv_blocks(x)
        B, C, F, T = x.shape

        x = x.permute(0, 3, 1, 2).reshape(B, T, C * F)
        lengths, h = self.transform_input_lengths(spectrogram_length), None
        for rnn_block in self.rnn_blocks:
            x, lengths, h = rnn_block(x, lengths, h)
        
        logits = self.fc(x)
        
        return {"logits": logits}

    def transform_input_lengths(self, input_lengths):
        for i in range(self.num_conv_layers):
            input_lengths = torch.floor((input_lengths + 2 * self.paddings[i][1] - (self.kernel_sizes[i][1] - 1) - 1) / self.strides[i][1] + 1).to(int)

        return input_lengths
