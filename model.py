import torch
from torch import nn



class Encoder(nn.Module):
    """ Sequence to sequence bidirectional LTSM encoder network """    
    def __init__(self, vocab_size, embedding_size=300, hidden_size=256, 
                 num_layers=2, drop_p=0.5):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers=num_layers, 
                            dropout=drop_p, bidirectional=True)
        
    def forward(self, input, hidden):
        embedded = self.embedding(input)
        output, hidden = self.lstm(embedded, hidden)
        return output, hidden
    
    def init_hidden(self, device='cpu'):
        """ Create two tensors with shape (num_layers * num_directions, batch, hidden_size)
            for the hidden state and cell state
        """
        h_0, c_0 = torch.zeros(2, 2*self.num_layers, 1, self.hidden_size, device=device)
        
        return h_0, c_0



class Decoder(nn.Module):
    """ Sequence to sequence bidirectional LSTM decoder network with attention 
        Attention implementation from:
        http://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
    """
    
    def __init__(self, vocab_size, embedding_size=300, hidden_size=256, 
                       num_layers=2, drop_p=0.1, max_length=50):
        
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.max_length = max_length

        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.attn = nn.Linear(self.hidden_size + embedding_size, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2 + embedding_size, self.hidden_size)
        self.dropout = nn.Dropout(drop_p)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers=num_layers, 
                            dropout=drop_p, bidirectional=True)
        
        self.out = nn.Linear(2 * hidden_size, vocab_size)
        self.softmax = nn.LogSoftmax(dim=1)
        
    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input)
        embedded = self.dropout(embedded)
        
        # Learns the attention vector (a probability distribution) here for weighting
        # encoder outputs based on the decoder input and encoder hidden vector
        attn_weights = F.softmax(self.attn(torch.cat((embedded[0], hidden[0][0]), 1)), dim=1)
        
        # Applies the attention vector (again, a probability distribution) to the encoder
        # outputs which weight the encoder_outputs
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))
        
        # Now the decoder input is combined with the weighted encoder_outputs and
        # passed through a linear transformation as input to the LSTM layer
        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)
        output = F.relu(output)
        
        output, hidden = self.lstm(output, hidden)
        output = self.out(output).view(1, -1)
        output = self.softmax(output)
    
        return output, hidden, attn_weights
        
    def init_hidden(self, device='cpu'):
        """ Create two tensors with shape (num_layers * num_directions, batch, hidden_size)
            for the hidden state and cell state
        """
        h_0, c_0 = torch.zeros(2, 2*self.num_layers, 1, self.hidden_size, device=device)
        return h_0, c_0