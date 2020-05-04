import torch.nn as nn

class NTMSensor(nn.Module):
    def __init__(self, outp_dim, mode='BCD'):
        super(NTMSensor, self).__init__()
        if mode == 'BCD':
            num_input_features = 5
        num_hidden_features = outp_dim // 2
        self.lstm = nn.LSTM(num_input_features, num_hidden_features,
                                num_layers=3,
                                bidirectional=True)

        # self.k_proj = nn.Linear(num_hidden_features * 2, num_hidden_features * 2)
        # self.v_proj = nn.Linear(num_hidden_features * 2, num_hidden_features * 2)
    
    def forward(self, x):
        # print(x.shape)
        x, _ = self.lstm(x)
        return x, x