import torch
from torch import einsum, nn
from modules.transformer import TransformerEncoder

def FeedForward(dim, mult=4):
    "scale up the hidden dimension of the input"
    inner_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim, bias=False),
        nn.GELU(),
        nn.Linear(inner_dim, dim, bias=False),
    )

def bottleneckadapter(dim,mult=4):
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, dim//mult, bias=False),
        nn.ReLU(),
        nn.Linear(dim//mult, dim, bias=False),
    )


class GatedCrossAttentionBlock(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        layers = 1,
        attn_dropout = 0.1,
        relu_dropout = 0.1,
        res_dropout = 0.1,
        embed_dropout = 0,
        attn_mask = False,
        ff_mult = 4,
    ):
        super().__init__()
        self.attn = TransformerEncoder(embed_dim=embed_dim, 
                                       num_heads=num_heads,
                                       layers=layers, 
                                       attn_dropout=attn_dropout, 
                                       relu_dropout=relu_dropout,
                                       res_dropout=res_dropout,
                                       embed_dropout=embed_dropout,
                                       attn_mask=attn_mask  
                                       )
        self.attn_gate = nn.Parameter(torch.tensor([0.0]))

        self.ff = FeedForward(embed_dim, ff_mult)
        self.ff_gate = nn.Parameter(torch.tensor([0.0]))

    def forward(self,query, key, value):
        x = (self.attn(query, key, value)* self.attn_gate.tanh() + query)
        x = self.ff(x) * self.ff_gate.tanh() + x

        return x