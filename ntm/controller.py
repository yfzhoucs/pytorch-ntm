"""LSTM Controller."""
import torch
from torch import nn
from torch.nn import Parameter
import numpy as np


if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

class LSTMController(nn.Module):
    """An NTM controller based on LSTM."""
    def __init__(self, num_inputs, num_outputs, num_layers):
        super(LSTMController, self).__init__()

        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.num_layers = num_layers

        print(num_inputs, num_outputs)

        self.lstm = nn.LSTM(input_size=num_inputs // 3 * 4,
                            hidden_size=num_outputs,
                            num_layers=num_layers)

        # The hidden state is a learned parameter
        self.lstm_h_bias = Parameter(torch.randn(self.num_layers, 1, self.num_outputs) * 0.05)
        self.lstm_c_bias = Parameter(torch.randn(self.num_layers, 1, self.num_outputs) * 0.05)

        self.reset_parameters()

        self.x_attn = nn.MultiheadAttention(embed_dim=num_inputs // 3, num_heads=2)
        self.x_attn_2 = nn.MultiheadAttention(embed_dim=num_inputs // 3, num_heads=2)
        self.ctrl_attn = nn.MultiheadAttention(embed_dim=num_inputs // 3, num_heads=1)

    def create_new_state(self, batch_size):
        # Dimension: (num_layers * num_directions, batch, hidden_size)
        lstm_h = self.lstm_h_bias.clone().repeat(1, batch_size, 1).to(device)
        lstm_c = self.lstm_c_bias.clone().repeat(1, batch_size, 1).to(device)
        return lstm_h, lstm_c

    def reset_parameters(self):
        for p in self.lstm.parameters():
            if p.dim() == 1:
                nn.init.constant_(p, 0)
            else:
                stdev = 5 / (np.sqrt(self.num_inputs +  self.num_outputs))
                nn.init.uniform_(p, -stdev, stdev)

    def size(self):
        return self.num_inputs, self.num_outputs

    def forward(self, prev_reads, 
            k_x, v_x, 
            k_ctrl, v_ctrl, 
            prev_state):
        prev_reads = prev_reads.unsqueeze(0)
        # print('qkv', prev_reads.shape, k_x.shape, v_x.shape)
        x_attn_outp, _ = self.x_attn(prev_reads, k_x, v_x)
        x_attn_outp_2, _ = self.x_attn_2(prev_reads, k_x, v_x)
        ctrl_attn_outp, _ = self.ctrl_attn(prev_reads, k_ctrl, v_ctrl)
        # print('qxava', prev_reads.shape, x_attn_outp.shape, ctrl_attn_outp.shape)
        x = torch.cat((prev_reads, x_attn_outp, x_attn_outp_2, ctrl_attn_outp), dim=2)
        # print(x.shape)
        outp, state = self.lstm(x, prev_state)
        # print(outp.shape)
        # input()
        return outp.squeeze(0), state
