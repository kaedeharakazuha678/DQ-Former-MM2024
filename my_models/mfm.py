import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.loss import CrossEntropyLoss
from modules.transformer import TransformerEncoder
from modules.adapter import BottleneckAdapter as Adapter
from .model import MyModel

#TODO

class MFM(MyModel):
    def __init__(self, args):
        super(MFM, self).__init__(args)
        self.num_heads = args.num_heads
        self.dropout = args.dropout
        self.attn_dropout = args.attn_dropout
        self.relu_dropout = args.relu_dropout
        self.res_dropout = args.res_dropout
        self.train_mode = args.do_train

        self.text_encoder = self.get_text_encoder(self.adapter_mode, self.frozen_mode)
        self.text_dim = self.text_encoder.config.hidden_size
        self.audio_encoder = self.get_audio_encoder(self.adapter_mode, self.frozen_mode)
        self.audio_dim = self.audio_encoder.config.hidden_size

        self.attn_mask = True
        self.layers = args.fusion_layers 
        self.hidden_dim = args.intermediate_dim # 128
        self.conv1d_kernel_size = 1

        # adapted for two modalities
        # combined_dim = 2 * (self.hidden_dim + self.hidden_dim )
        combined_dim = self.hidden_dim + self.hidden_dim
        output_dim = self.hidden_dim // 2
        
        # 1. Temporal convolutional layers
        self.proj_t = nn.Conv1d(self.text_dim,  self.hidden_dim, kernel_size=self.conv1d_kernel_size, padding=0, bias=False)
        self.proj_a = nn.Conv1d(self.audio_dim, self.hidden_dim, kernel_size=self.conv1d_kernel_size, padding=0, bias=False)
        
        # 2. Crossmodal Attentions
        self.trans_t_with_a = self.get_network(self_type='la')
        self.trans_a_with_t = self.get_network(self_type='al')
        
        # 3. Self Attentions (Could be replaced by LSTMs, GRUs, etc.)
        self.trans_t_mem = self.get_network(self_type='l_mem', layers=3)
        self.trans_a_mem = self.get_network(self_type='a_mem', layers=3)

        # Projection layers
        self.proj1 = nn.Linear(combined_dim, combined_dim)
        self.proj2 = nn.Linear(combined_dim, combined_dim)
        self.out_layer = nn.Linear(combined_dim, output_dim)
        
        # cls layers
        self.fc_out = nn.Linear(output_dim, self.output_dim)
        
    def get_network(self, self_type='l', layers=-1):
        if self_type in ['l', 'al', 'vl']:
            embed_dim, attn_dropout = self.hidden_dim, self.attn_dropout
        elif self_type in ['a', 'la', 'va']:
            embed_dim, attn_dropout = self.hidden_dim, self.attn_dropout
        elif self_type in ['v', 'lv', 'av']:
            embed_dim, attn_dropout = self.hidden_dim, self.attn_dropout
        elif self_type == 'l_mem':
            # embed_dim, attn_dropout = 2*self.hidden_dim, self.attn_dropout
            embed_dim, attn_dropout = self.hidden_dim, self.attn_dropout
        elif self_type == 'a_mem':
            # embed_dim, attn_dropout = 2*self.hidden_dim, self.attn_dropout
            embed_dim, attn_dropout = self.hidden_dim, self.attn_dropout
        elif self_type == 'v_mem':
            # embed_dim, attn_dropout = 2*self.hidden_dim, self.attn_dropout
            embed_dim, attn_dropout = self.hidden_dim, self.attn_dropout
        else:
            raise ValueError("Unknown network type")
        
        if layers == -1:
            layers = self.layers
        
        return TransformerEncoder(embed_dim=embed_dim,
                                num_heads=self.num_heads,
                                layers=layers,
                                attn_dropout=attn_dropout,
                                relu_dropout=self.dropout,
                                res_dropout=self.dropout,
                                embed_dropout=self.dropout,
                                attn_mask=self.attn_mask)
        

    def forward(self, audio, input_ids, attention_mask, queries, text=None, labels=None):
        return_dict = {}
        ce_loss = CrossEntropyLoss()

        t_output = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        a_output = self.audio_encoder(audio, output_hidden_states=True)
        
        a_embedding = a_output.last_hidden_state # shape: (batch_size, seq_len, hidden_size_a)
        t_embedding = t_output.last_hidden_state # shape: (batch_size, seq_len, hidden_size_t)

        t_hidden_size = t_embedding.size(-1)
        a_hidden_size = a_embedding.size(-1)
        assert t_hidden_size == a_hidden_size, "text_hidden_size != audio_hidden_size"

        emos_out = self.forward_features(t_embedding, a_embedding) # shape: (batch_size, seq_len, output_size)

        # return features, emos_out, vals_out, interloss
        if labels is not None:
            loss = ce_loss.get_loss(emos_out, labels)
            logits = emos_out

        return {"loss": loss, 
                "logits": logits, 
                "return_dict": return_dict}

    def forward_features(self, t_embedding, a_embedding):
        x_t = t_embedding.transpose(1, 2)
        x_a = a_embedding.transpose(1, 2)

        proj_x_t = self.proj_t(x_t).permute(2, 0, 1)  # shape: (seq_len, batch_size, hidden_size)
        proj_x_a = self.proj_a(x_a).permute(2, 0, 1)
        
        h_t_with_as = self.trans_t_with_a(proj_x_t, proj_x_a, proj_x_a) # shape: (seq_len, batch_size, hidden_size)
        h_ls = self.trans_t_mem(h_t_with_as)  # shape: (seq_len, batch_size, hidden_size)
        if type(h_ls) == tuple:
            h_ls = h_ls[0]
        last_h_l = h_ls[-1] # shape: (batch_size, hidden_size) get the last hidden state
        
        h_a_with_ts = self.trans_a_with_t(proj_x_a, proj_x_t, proj_x_t)
        h_as = self.trans_a_mem(h_a_with_ts)
        if type(h_as) == tuple:
            h_as = h_as[0]
        last_h_a = h_as[-1]
        
        last_hs = torch.cat([last_h_l, last_h_a], dim=1) # shape: (batch_size, hidden_size)
        # FFN layer and residual connection
        last_hs_proj = self.proj2(F.dropout(F.relu(self.proj1(last_hs), inplace=True), p=self.dropout, training=self.training))
        last_hs_proj += last_hs
        
        # linear layer
        features = self.out_layer(last_hs_proj)
        # classification layer
        emos_out = self.fc_out(features)

        return emos_out
