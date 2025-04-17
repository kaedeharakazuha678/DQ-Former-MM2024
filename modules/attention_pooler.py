import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from einops import repeat
from modules.transformer import TransformerEncoder

class AttentionPooler(nn.Module):
    def __init__(self, num_queries,embedding_dim, num_heads=8):
        super(AttentionPooler, self).__init__()
        self.query = Parameter(torch.randn(num_queries, embedding_dim)).cuda()
        self.transformer_layer = TransformerEncoder(embed_dim=embedding_dim, num_heads=num_heads, layers=1)

    def forward(self, key, value):
        # input: (batch_size, seq_len, embedding_dim)
        queries = repeat(self.query, 'n e -> b n e', b=key.shape[0])
        queries = queries.permute(1, 0, 2)
        key = key.permute(1, 0, 2)
        value = value.permute(1, 0, 2)
        attn_output = self.transformer_layer(queries, key, value)

        return attn_output


