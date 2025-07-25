import torch
import torch.nn as nn

class MLPAdapter(nn.Module):
    """
    Simple MLP adapter that maps CLIP features to mBART decoder prefix.
    """
    def __init__(self, **kwargs):
        super(MLPAdapter, self).__init__()
        self.hidden = nn.Sequential(
            nn.Linear(kwargs['clip_dim'], kwargs['prefix_length'] * kwargs['mbart_dim']),
            nn.PReLU(),
        )
        self.prefix_length, self.mbart_dim = kwargs['prefix_length'], kwargs['mbart_dim']
    
    def forward(self, x):
        x = x.float()
        return self.hidden(x).view(-1, self.prefix_length, self.mbart_dim)
    
    def reset(self):
        nn.init.xavier_uniform_(self.hidden[0].weight)

class HiddenMLPAdapter(nn.Module):
    """
    Two-layer MLP adapter with hidden layer.
    """
    def __init__(self, **kwargs):
        super(HiddenMLPAdapter, self).__init__()
        self.hidden = nn.Sequential(
            nn.Linear(kwargs['clip_dim'], kwargs['mbart_dim']),
            nn.ReLU(),
            nn.Linear(kwargs['mbart_dim'], kwargs['prefix_length'] * kwargs['mbart_dim']),
            nn.PReLU(),
        )
        self.prefix_length, self.mbart_dim = kwargs['prefix_length'], kwargs['mbart_dim']
    
    def forward(self, x):
        x = x.float()
        return self.hidden(x).view(-1, self.prefix_length, self.mbart_dim)
    
    def reset(self):
        nn.init.xavier_uniform_(self.hidden[0].weight)
        nn.init.xavier_uniform_(self.hidden[2].weight)

class TransformerAdapter(nn.Module):
    """
    Transformer-based adapter with self-attention.
    """
    def __init__(self, **kwargs):
        super(TransformerAdapter, self).__init__()
        self.prefix_length, self.mbart_dim = kwargs['prefix_length'], kwargs['mbart_dim']
        self.projector = nn.Linear(kwargs['clip_dim'], self.prefix_length * kwargs['mbart_dim'])
        encoder_layer = nn.TransformerEncoderLayer(
            d_model = kwargs['mbart_dim'],
            nhead = 2,
            dim_feedforward = kwargs['mbart_dim']//3,
            batch_first = True
        )
        self.transformer = nn.TransformerEncoder(
                encoder_layer = encoder_layer, 
                num_layers = kwargs['num_encoder_layers']
            )
    
    def forward(self, x):
        x = self.projector(x).view(-1, self.prefix_length, self.mbart_dim)
        return self.transformer(x) 