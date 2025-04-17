import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.my_transformer import MyTransformerEncoderLayer
from .model import MyModel
from utils.helper import PredictLayer

class Semantic_encoder(nn.Module):
    def __init__(self, n_heads, n_layers, d_model, dropout):
        super(Semantic_encoder, self).__init__()
        self.semantic_encoder = nn.ModuleList([
            MyTransformerEncoderLayer(d_model=d_model, nhead=n_heads, dropout=dropout, batch_first=True)
            for _ in range(n_layers)])
    
    def forward(self, x, need_weights=False):
        """
            Forward pass of the semantic encoder.

            Args:
                x (torch.Tensor): Input tensor.
                need_weights (bool): Flag indicating whether to return attention weights.

            Returns:
                torch.Tensor or (torch.Tensor, list): Encoded representation of input tensor.
                    If need_weights is True, also returns a list of attention weights.
        """
        if need_weights:
            attention_weights = []
            for i in range(len(self.semantic_encoder)):
                x, attn_weights = self.semantic_encoder[i].forward(x, need_weights=True)
                attention_weights.append(attn_weights.detach().cpu().numpy())
            return x, attention_weights
        else:
            for i in range(len(self.semantic_encoder)):
                x = self.semantic_encoder[i].forward(x)
            return x

class SingleStreamModel(MyModel):
    def __init__(self, args):
        super(SingleStreamModel, self).__init__(args)
        self.d_model = args.intermediate_dim
        self.n_head = args.num_heads
        self.n_layer = args.num_layers
        self.dropout = args.dropout
        self.enable_modality_embedding = args.enable_modality_embedding
        self.modalities = args.modalities  # ['t', 'a', 'v', 'ta', 'tv', 'av', 'tav']

        if 't' in self.modalities:
            self.text_encoder = self.get_text_encoder(adapter_mode=False, frozen_mode=False)
            self.text_dim = self.text_encoder.config.hidden_size
            self.proj_text = nn.Conv1d(self.text_dim, self.d_model, kernel_size=1)

        if 'a' in self.modalities:
            self.audio_encoder = self.get_audio_encoder(adapter_mode=False, frozen_mode=False)
            self.audio_dim = self.audio_encoder.config.hidden_size
            self.proj_audio = nn.Conv1d(self.audio_dim, self.d_model, kernel_size=4, stride=2, padding=2)

        if 'v' in self.modalities:
            self.vision_encoder = self.get_vision_encoder(adapter_mode=False, frozen_mode=False)
            self.vision_dim = self.vision_encoder.config.hidden_size
            self.proj_vision = nn.Conv1d(self.vision_dim, self.d_model, kernel_size=4, stride=2, padding=2)

        self.modality_embedding_func = nn.Embedding(4, self.d_model, padding_idx=0)
        self.shared_semantic_encoder = Semantic_encoder(self.n_head, self.n_layer, self.d_model, self.dropout)
        self.pred_head = PredictLayer(self.d_model, self.output_dim, dropout=self.dropout, residual=True)

    def modality_embedding(self, seq_len: int = None, modality_type: str = None):
        modality_type_mapping = {'text': 1, 'audio': 2, 'vision': 3}
        assert modality_type in modality_type_mapping, "modality_type should be text, audio, vision"
        modality_id = modality_type_mapping[modality_type]
        token_ids = torch.full((seq_len,), modality_id, dtype=torch.long).to('cuda')
        return self.modality_embedding_func(token_ids)

    def forward(self, audio_value=None, audio_attention_mask=None, input_ids=None, 
                attention_mask=None, queries=None, text=None, labels=None, pixel_values=None):
        return_dict = {}
        embeddings = []

        if 't' in self.modalities and input_ids is not None:
            t_features = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
            t_embed = t_features.last_hidden_state
            t_embed = self.proj_text(t_embed.permute(0, 2, 1)).permute(0, 2, 1)
            if self.enable_modality_embedding:
                t_embed += self.modality_embedding_func(torch.full((t_embed.size(1),), 1, device=t_embed.device))
            embeddings.append(t_embed)

        if 'a' in self.modalities and audio_value is not None:
            a_features = self.audio_encoder(audio_value, output_hidden_states=True)
            a_embed = a_features.last_hidden_state
            a_embed = self.proj_audio(a_embed.permute(0, 2, 1)).permute(0, 2, 1)
            if self.enable_modality_embedding:
                a_embed += self.modality_embedding_func(torch.full((a_embed.size(1),), 2, device=a_embed.device))
            embeddings.append(a_embed)

        if 'v' in self.modalities and pixel_values is not None:
            v_features = self.vision_encoder(pixel_values, output_hidden_states=True)
            v_embed = v_features.last_hidden_state
            v_embed = self.proj_vision(v_embed.permute(0, 2, 1)).permute(0, 2, 1)
            if self.enable_modality_embedding:
                v_embed += self.modality_embedding_func(torch.full((v_embed.size(1),),3, device=v_embed.device))
            embeddings.append(v_embed)

        if not embeddings:
            raise ValueError("At least one modality should be provided.")

        multi_embed = torch.cat(embeddings, dim=1)
        multi_embed = self.shared_semantic_encoder(multi_embed, need_weights=False)
        fusion_out = torch.mean(multi_embed, dim=1)
        logits = self.pred_head(fusion_out)

        loss = self.calculate_task_loss(logits, labels) if labels is not None else None

        return {"loss": loss, "logits": logits, "return_dict": return_dict}