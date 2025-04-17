import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, Tuple, Optional, Callable
from torch import Tensor


class MyTransformerEncoderLayer(nn.TransformerEncoderLayer):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
                 device=None, dtype=None):
        super(MyTransformerEncoderLayer, self).__init__(d_model, nhead, dim_feedforward, dropout,
                                                            activation, layer_norm_eps, batch_first, 
                                                            norm_first, device, dtype)
                                                            
    def forward(
            self,
            src: Tensor,
            src_mask: Optional[Tensor] = None,
            src_key_padding_mask: Optional[Tensor] = None,
            is_causal: bool = False,
            need_weights: bool = False,
            ) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            is_causal: If specified, applies a causal mask as src_mask.
              Default: ``False``.
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        src_key_padding_mask = F._canonical_mask(
            mask=src_key_padding_mask,
            mask_name="src_key_padding_mask",
            other_type=F._none_or_dtype(src_mask),
            other_name="src_mask",
            target_type=src.dtype
        )

        src_mask = F._canonical_mask(
            mask=src_mask,
            mask_name="src_mask",
            other_type=None,
            other_name="",
            target_type=src.dtype,
            check_other=False,
        )
     
        x = src
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask, is_causal=is_causal)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask, is_causal=is_causal))
            x = self.norm2(x + self._ff_block(x))

        if need_weights:
            attn_weights = self.self_attn(src,src,src,attn_mask=src_mask,key_padding_mask=src_key_padding_mask,is_causal=is_causal,need_weights=True)[1]
            attn_weights = attn_weights.squeeze() # [B, N, N]          
            return x, attn_weights       
        else:
            return x
    
    def _sa_block(self, x: Tensor,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor], is_causal: bool = False) -> Tensor:
        x = self.self_attn(x, x, x, attn_mask=attn_mask,key_padding_mask=key_padding_mask,need_weights=False, is_causal=is_causal)[0]
        return self.dropout1(x)
