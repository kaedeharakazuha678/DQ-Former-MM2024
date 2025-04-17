import torch
from torch import nn
import torch.nn.functional as F
from modules.transformer import TransformerEncoder, MRUEncoderBlock, Linear
from utils.loss import LabelSmoothingCELoss, CrossEntropyLoss
from modules.adapter import BottleneckAdapter as Adapter
from .model import MyModel

#TODO

class PMR(MyModel):
    def __init__(self,args):
        super(PMR, self).__init__(args)
        self.embed_dim = args.intermediate_dim
        self.num_heads = args.num_heads  # 5
        self.layers = args.fusion_layers
        self.attn_dropout = args.attn_dropout  # 0.1
        self.relu_dropout = args.relu_dropout  # 0.1
        self.res_dropout = args.res_dropout  # 0.1
        self.out_dropout = args.dropout  # 0
        self.embed_dropout = args.dropout  # 0.25
        self.attn_mask = True  # true
        self.frozen_mode = 'at'

        self.layerNorm = nn.LayerNorm(self.embed_dim)
        self.fc1 = Linear(self.embed_dim, self.embed_dim)  # The "Add & Norm" part in the paper
        self.fc2 = Linear(self.embed_dim, self.embed_dim)
        
        self.text_encoder = self.get_text_encoder(frozen_mode=self.frozen_mode)
        self.audio_encoder = self.get_audio_encoder(frozen_mode=self.frozen_mode)
        self.Conv1d = nn.Conv1d(in_channels=self.embed_dim, out_channels=self.embed_dim, kernel_size=64, stride=64, padding=0, bias=True)

        self.MRUblock = MRUEncoderBlock(self.embed_dim,
                                        num_heads=self.num_heads,
                                        attn_dropout=self.attn_dropout,
                                        relu_dropout=self.relu_dropout,
                                        res_dropout=self.res_dropout,
                                        attn_mask=self.attn_mask)

        self.transformer_all = TransformerEncoder(embed_dim=self.embed_dim,
                                                  num_heads=self.num_heads,
                                                  layers=1,
                                                  attn_dropout=self.attn_dropout,
                                                  relu_dropout=self.relu_dropout,  # 0.1
                                                  res_dropout=self.res_dropout,  # 0.1
                                                  embed_dropout=self.embed_dropout,  # 0、25
                                                  attn_mask=self.attn_mask)  # true

        self.linear_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.trans_proj = nn.Linear(self.embed_dim, 1, bias=False) 
        self.proj1 = nn.Linear(self.embed_dim, self.embed_dim)
        self.proj2 = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_layer = nn.Linear(self.embed_dim, self.output_dim)
        

    def MUMblock(self, text_features, audio_features):
        '''
            text_features: [seq_len_t, batch_size, embed_dim]
            audio_features: [seq_len_a, batch_size, embed_dim]
        '''
        text_feature = torch.mean(text_features, dim=0) # [batch_size, embed_dim]
        audio_feature = torch.mean(audio_features, dim=0) # [batch_size, embed_dim]

        single_l = self.trans_proj(torch.tanh(self.linear_proj(text_features))) # [batch_size, 1]
        single_a = self.trans_proj(torch.tanh(self.linear_proj(audio_features))) # [batch_size, 1]

        sum_atten = torch.exp(single_l) + torch.exp(single_a) # [batch_size, 1]
        atten_l = torch.exp(single_l) / sum_atten
        atten_a = torch.exp(single_a) / sum_atten

        # todo: check whether the following code is correct
        common_message = atten_l * text_feature + atten_a * audio_feature # [batch_size, embed_dim]

        residual = common_message
        common_message = self.layerNorm(common_message)
        common_message = F.relu(self.fc1(common_message))
        common_message = F.dropout(common_message, p=self.relu_dropout, training=self.training)
        common_message = self.fc2(common_message)
        common_message = F.dropout(common_message, p=self.res_dropout, training=self.training)
        common_message = residual + common_message

        return common_message # [seq_len, batch_size, embed_dim]


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

        emos_out, fusion_feature = self.forward_features(t_embedding, a_embedding) # shape: (batch_size, seq_len, output_size)

        if labels is not None:
            loss = ce_loss.get_loss(emos_out, labels)
            logits = emos_out
        
        if self.train_mode:
            pass
        
        else:
            return_dict['fusion_feature'] = fusion_feature

        return {"loss": loss, 
                "logits": logits, 
                "return_dict": return_dict}

    def forward_features(self, text_features, audio_features, ):
        """
        text, audio, and vision should have dimension [batch_size, seq_len, n_features]
        """
        text_features = text_features.permute(1, 0, 2)  # [seq_len, batch_size, n_features]
        audio_features =audio_features.permute(1, 0, 2)  # [seq_len, batch_size, n_features]

        common_m = torch.cat((text_features,audio_features), dim=0) # seq=2022
        common_m = self.Conv1d(common_m.permute(1, 2, 0)).permute(2,0,1) # [batch_size, embed_dim, seq_len]
        
        for layer in range(self.layers):
            audio_features = self.MRUblock(audio_features, common_m, common_m)
            text_features = self.MRUblock(text_features, common_m, common_m)

            a_t_oc = self.MRUblock(common_m, audio_features, audio_features)
            l_t_oc = self.MRUblock(common_m, text_features, text_features)
            common_m = self.MUMblock(l_t_oc, a_t_oc)

        combined_f = torch.cat([common_m, text_features, audio_features], dim=0) # [seq_len, batch_size, embed_dim]
        combined_f = self.transformer_all(combined_f)
        # import pdb; pdb.set_trace()
        if type(combined_f) == tuple:
            h_ls = combined_f[0]
        # {batch_size, feature_dim}
        last_hs = combined_f[-1]  # Take the last output for prediction

        # A residual block
        last_hs_proj = self.proj2(F.dropout(F.relu(self.proj1(last_hs)), p=self.out_dropout, training=self.training))
        last_hs_proj += last_hs

        output = self.out_layer(last_hs_proj)
        return output, last_hs
