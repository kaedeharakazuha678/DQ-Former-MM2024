from torch import einsum, nn
from torch.nn.modules.module import Module
from modules.transformer import TransformerEncoder


class PredictLayer(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0.5, residual=True):
        super().__init__()
        self.dropout = dropout
        self.ffn = self.FeedForward(input_dim, mult=1)
        self.pred_head = nn.Linear(input_dim, output_dim)
        self.residual = residual
        
    def FeedForward(self, dim, mult=4):
        "scale up the hidden dimension of the input"
        inner_dim = int(dim * mult)
        return nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, inner_dim, bias=False),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(inner_dim, dim, bias=False),
        )
    
    def adapter(self, dim, inner_dim):
        return nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, inner_dim, bias=False),
            nn.ReLU(),
            nn.Linear(inner_dim, dim, bias=False),
        )
             
    def add_module(self, name: str, module: Module) -> None:
        return super().add_module(name, module)
        
    def forward(self, x):
        if self.residual:
            x = x + self.ffn(x)
        else:
            x = self.ffn(x)
        x = self.pred_head(x)
        return x
