import torch
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class LSTMNET(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=2, num_layers=2):
        super(LSTMNET, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers)
        self.reg = nn.Linear(30*hidden_size, output_size)
        
    def forward(self, x):
        r_out, (h_n, h_c) = self.rnn(x, None)   # None represents zero initial hidden state
        r_out = r_out.reshape(r_out.shape[0], r_out.shape[1]*r_out.shape[2])
        r_out = torch.sin(r_out)
        out = self.reg(r_out)
        return out
