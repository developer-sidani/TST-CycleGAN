import torch
import torch.nn as nn


class MLPAdapter(nn.Module):
    """
    Simple MLP adapter to transform CLIP embeddings to mBART embeddings.
    Similar to the adapter used in CLIPTrans.
    """

    def __init__(self, clip_dim, mbart_dim, prefix_length):
        super(MLPAdapter, self).__init__()
        self.hidden = nn.Sequential(
            nn.Linear(clip_dim, prefix_length * mbart_dim),
            nn.PReLU(),
        )
        self.prefix_length = prefix_length
        self.mbart_dim = mbart_dim

    def forward(self, x):
        x = x.float()
        return self.hidden(x).view(-1, self.prefix_length, self.mbart_dim)

    def reset(self):
        nn.init.xavier_uniform_(self.hidden[0].weight)


class HiddenMLPAdapter(nn.Module):
    """
    MLP adapter with a hidden layer for more complex transformations.
    """

    def __init__(self, clip_dim, mbart_dim, prefix_length):
        super(HiddenMLPAdapter, self).__init__()
        self.hidden = nn.Sequential(
            nn.Linear(clip_dim, mbart_dim),
            nn.ReLU(),
            nn.Linear(mbart_dim, prefix_length * mbart_dim),
            nn.PReLU(),
        )
        self.prefix_length = prefix_length
        self.mbart_dim = mbart_dim

    def forward(self, x):
        x = x.float()
        return self.hidden(x).view(-1, self.prefix_length, self.mbart_dim)

    def reset(self):
        nn.init.xavier_uniform_(self.hidden[0].weight)
        nn.init.xavier_uniform_(self.hidden[2].weight)


class TransformerAdapter(nn.Module):
    """
    Adapter using transformer encoder layers for more context-aware adaptation.
    """

    def __init__(self, clip_dim, mbart_dim, prefix_length, num_encoder_layers=1):
        super(TransformerAdapter, self).__init__()
        self.prefix_length = prefix_length
        self.mbart_dim = mbart_dim
        self.projector = nn.Linear(clip_dim, prefix_length * mbart_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=mbart_dim,
            nhead=4,
            dim_feedforward=mbart_dim // 2,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_encoder_layers
        )

    def forward(self, x):
        x = x.float()
        x = self.projector(x).view(-1, self.prefix_length, self.mbart_dim)
        return self.transformer(x)