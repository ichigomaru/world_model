# RNN, Diffusion Policy

import torch
import torch.nn as nn


class RNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(RNN, self).__init__()
        self.rnn = nn.RNN(input_dim, hidden_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim, output_dim)
        self.hidden_dim = hidden_dim

    def initialize(self, batch_size, device):
        self.hidden = torch.zeros(1, batch_size, self.hidden_dim, device=device)

    def forward(self, input):
        out, hidden = self.rnn(input, self.hidden)
        output = self.linear(out)
        self.hidden = hidden
        return output
