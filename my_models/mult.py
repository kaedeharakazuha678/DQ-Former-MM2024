import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.transformer import TransformerEncoder
from modules.adapter import BottleneckAdapter as Adapter
from .model import MyModel
from utils.helper import PredictLayer


class MultModel(MyModel):
    def __init__(self, args):
        super(MultModel, self).__init__(args)
        self.modalities = args.modalities # should be one of ['t', 'a', 'v', 'ta', 'tv', 'av', 'tav']
        self.num_heads = args.num_heads
        self.dropout = args.dropout
        self.attn_dropout = args.attn_dropout
        self.relu_dropout = args.relu_dropout
        self.res_dropout = args.res_dropout
        self.layers = args.fusion_layers
        self.hidden_dim = args.intermediate_dim
        self.conv1d_kernel_size = 1

        self.text_encoder = self.get_text_encoder() if 't' in self.modalities else None
        self.audio_encoder = self.get_audio_encoder() if 'a' in self.modalities else None
        self.vision_encoder = self.get_vision_encoder() if 'v' in self.modalities else None

        self.proj = nn.ModuleDict()
        if 't' in self.modalities:
            self.text_dim = self.text_encoder.config.hidden_size
            self.proj['t'] = nn.Conv1d(self.text_dim, self.hidden_dim, self.conv1d_kernel_size, bias=False)
        if 'a' in self.modalities:
            self.audio_dim = self.audio_encoder.config.hidden_size
            self.proj['a'] = nn.Conv1d(self.audio_dim, self.hidden_dim, self.conv1d_kernel_size, bias=False)
        if 'v' in self.modalities:
            self.vision_dim = self.vision_encoder.config.hidden_size
            self.proj['v'] = nn.Conv1d(self.vision_dim, self.hidden_dim, self.conv1d_kernel_size, bias=False)

        self.cross_attn = nn.ModuleDict()
        for src in self.modalities:
            for tgt in self.modalities:
                if src != tgt:
                    self_type = f"{src[0]}{tgt[0]}"  
                    self.cross_attn[f"{src}_with_{tgt}"] = self.get_network(self_type)

        self.self_attn = nn.ModuleDict()
        for modality in self.modalities:
            self_type = f"{modality[0]}_mem"
            self.self_attn[modality] = self.get_network(self_type)

        combined_dim = self.hidden_dim * len(self.modalities)
        output_dim = self.hidden_dim // 2
        self.proj1 = nn.Linear(combined_dim, combined_dim)
        self.proj2 = nn.Linear(combined_dim, combined_dim)
        self.out_layer = nn.Linear(combined_dim, output_dim)
        self.fc_out = nn.Linear(output_dim, self.output_dim)

    def get_network(self, self_type, layers=-1):
        embed_dim = self.hidden_dim
        attn_dropout = self.attn_dropout

        if layers == -1:
            layers = self.layers

        return TransformerEncoder(
            embed_dim=embed_dim,
            num_heads=self.num_heads,
            layers=layers,
            attn_dropout=attn_dropout,
            relu_dropout=self.relu_dropout,
            res_dropout=self.res_dropout,
            embed_dropout=self.dropout,
            attn_mask=True
        )

    def forward(self, audio_value=None, audio_attention_mask=None, input_ids=None, 
                attention_mask=None, queries=None, text=None, labels=None, pixel_values=None):
        embeddings = {}
        if 't' in self.modalities:
            t_output = self.text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
            embeddings['t'] = t_output.last_hidden_state

        if 'a' in self.modalities:
            a_output = self.audio_encoder(
                audio_value,
                output_hidden_states=True
            )
            embeddings['a'] = a_output.last_hidden_state

        if 'v' in self.modalities:
            v_output = self.vision_encoder(
               pixel_values,
                output_hidden_states=True
            )
            embeddings['v'] = v_output.last_hidden_state

        hidden_sizes = [emb.size(-1) for emb in embeddings.values()]
        assert all(h == hidden_sizes[0] for h in hidden_sizes), "模态隐藏维度不一致"

        emos_out = self.forward_features(embeddings)
        labels = labels

        loss = self.calculate_task_loss(emos_out, labels) if labels is not None else None
        logits = emos_out

        return {
            "loss": loss,
            "logits": logits,
            "return_dict": {}
        }

    def forward_features(self, embeddings):
        proj_embeddings = {}
        for modality, emb in embeddings.items():
            x = emb.transpose(1, 2)  # (batch, hidden, seq_len)
            proj_x = self.proj[modality](x).permute(2, 0, 1)  # (seq_len, batch, hidden)
            proj_embeddings[modality] = proj_x

        modality_features = {}
        for modality in self.modalities:
            cross_attn_outputs = []
            for other_modality in self.modalities:
                if other_modality != modality:
                    cross_attn_layer = self.cross_attn[f"{modality}_with_{other_modality}"]
                    cross_output = cross_attn_layer(
                        proj_embeddings[modality],
                        proj_embeddings[other_modality],
                        proj_embeddings[other_modality]
                    )
                    cross_attn_outputs.append(cross_output)
            
            combined = sum(cross_attn_outputs) if cross_attn_outputs else proj_embeddings[modality]
            
            self_attn_layer = self.self_attn[modality]
            self_output = self_attn_layer(combined)
            if isinstance(self_output, tuple):
                self_output = self_output[0]
            modality_features[modality] = self_output[-1]

        last_hs = torch.cat([modality_features[m] for m in self.modalities], dim=1)

        last_hs_proj = self.proj2(F.dropout(F.relu(self.proj1(last_hs), inplace=True), p=self.dropout))
        last_hs_proj += last_hs
        features = self.out_layer(last_hs_proj)
        emos_out = self.fc_out(features)
        return emos_out