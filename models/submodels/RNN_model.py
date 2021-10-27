import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.rnn = nn.RNN(input_size, hidden_dim, n_layers, batch_first=True)   

        self.fc = nn.Linear(hidden_dim, output_size)

    def forward(self, features):
        batch_size = features.batch_sizes.max()
        # batch_size = features.shape[0]
        hidden0 = self.init_hidden(batch_size)

        out, hidden = self.rnn(features, hidden0)

        out, _ = pad_packed_sequence(out, batch_first=True)

        out = self.fc(out)

        return out, hidden
    
    def init_hidden(self, batch_size):
        return torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(device)