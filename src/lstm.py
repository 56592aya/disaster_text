# lstm.py
import torch
import torch.nn as nn
import numpy as np
import config

class ModelLSTM(nn.Module):
    """docstring for ModelLSTM."""
    def __init__(self, embedding_np):
        """initilizes the model object

        Args:
            embedding_np (np.ndarray): is matrix of embeddings(rows=words, columns embedding representation)
        """
        super(ModelLSTM, self).__init__()
        
        # the rest is configuration of the layer units
        num_words, embed_dim = embedding_np.shape

        # Embedding Layer
        self.embedding = nn.Embedding(num_embeddings=num_words, embedding_dim=embed_dim)
        # Let the weights be non-trainable
        self.embedding.weight = nn.Parameter(torch.from_numpy(embedding_np) requires_grad=False)

        # LSTM layer
        # batch_first – If True, then the input and output tensors are provided as (batch, seq, feature). Default: False
        self.lstm = nn.LSTM(input_size=embed_dim, hidden_size=config.HIDDEN_SIZE, bidirectional=True, batch_first=True)

        # Output Linear Layer with one outut, |in| = |H(for avg pool)| + |H(for max pool)|
        self.linear = nn.Linear(2*config.HIDDEN_SIZE, 1)

    def forward(self, x):
        # Apply layers and create a forward pass
        
        # apply emedding
        x = self.embedding(x)

        # apply lstm
        # the underline is (hn, cn)
        x, _  = self.lstm(x)
        
        # apply poolings
        avg_pool = torch.mean(x, 1)
        max_pool, _ = torch.max(x, 1)
        # concat
        out = torch.cat((avg_pool, max_pool), 1)
        # apply linear
        out = self.linear(out)

        return out


