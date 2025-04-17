
import torch
import torch.nn as nn
from utils.loss import LabelSmoothingCELoss, CrossEntropyLoss
from modules.transformer import TransformerEncoder, GatedTransformerEncoder
from modules.adapter import BottleneckAdapter as Adapter
from torch.nn import TransformerEncoderLayer
from .model import MyModel

#TODO

class MSE(nn.Module):
    def __init__(self):
        super(MSE, self).__init__()

    def forward(self, pred, real):
        diffs = torch.add(real, -pred)
        n = torch.numel(diffs.data)
        mse = torch.sum(diffs.pow(2)) / n

        return mse


class DiffLoss(nn.Module):

    def __init__(self):
        super(DiffLoss, self).__init__()

    def forward(self, input1, input2):

        batch_size = input1.size(0)
        input1 = input1.view(batch_size, -1)
        input2 = input2.view(batch_size, -1)

        # Zero mean
        input1_mean = torch.mean(input1, dim=0, keepdims=True)
        input2_mean = torch.mean(input2, dim=0, keepdims=True)
        input1 = input1 - input1_mean
        input2 = input2 - input2_mean

        input1_l2_norm = torch.norm(input1, p=2, dim=1, keepdim=True).detach()
        input1_l2 = input1.div(input1_l2_norm.expand_as(input1) + 1e-6)
        

        input2_l2_norm = torch.norm(input2, p=2, dim=1, keepdim=True).detach()
        input2_l2 = input2.div(input2_l2_norm.expand_as(input2) + 1e-6)

        diff_loss = torch.mean((input1_l2.t().mm(input2_l2)).pow(2))

        return diff_loss


class CMD(nn.Module):
    """
    Adapted from https://github.com/wzell/cmd/blob/master/models/domain_regularizer.py
    """
    def __init__(self):
        super(CMD, self).__init__()

    def forward(self, x1, x2, n_moments):
        mx1 = torch.mean(x1, 0)
        mx2 = torch.mean(x2, 0)
        sx1 = x1-mx1
        sx2 = x2-mx2
        dm = self.matchnorm(mx1, mx2)
        scms = dm
        for i in range(n_moments - 1):
            scms += self.scm(sx1, sx2, i + 2)
        return scms

    def matchnorm(self, x1, x2):
        power = torch.pow(x1-x2,2)
        summed = torch.sum(power)
        sqrt = summed**(0.5)
        return sqrt
        # return ((x1-x2)**2).sum().sqrt()

    def scm(self, sx1, sx2, k):
        ss1 = torch.mean(torch.pow(sx1, k), 0)
        ss2 = torch.mean(torch.pow(sx2, k), 0)
        return self.matchnorm(ss1, ss2)


class MISA(MyModel):
    def __init__(self,args, main_test=False):
        super(MISA, self).__init__(args)

        self.intermediate_dim = args.intermediate_dim
        self.num_heads = args.num_heads
        self.fusion_layers = 1
        self.gated = args.gated
        self.dropout = args.dropout
        self.attn_dropout = args.attn_dropout
        self.relu_dropout = args.relu_dropout
        self.res_dropout = args.res_dropout

        self.adapter_dim = args.adapter_dim
        self.bottleneck_len = args.bottleneck_len
        self.sim_weight =  1.0
        self.diff_weight =  0.3
        self.recon_weight =  1.0
        
        if not main_test:
            self.text_encoder = self.get_text_encoder(self.adapter_mode, self.frozen_mode)
            self.audio_encoder = self.get_audio_encoder(self.adapter_mode, self.frozen_mode)
        else:
            self.text_encoder = None
            self.audio_encoder = None

        self.attn_mask = True
        self.layers = args.fusion_layers 
        self.dropout = args.attn_dropout
        self.num_heads = args.num_heads 
        self.hidden_dim = args.intermediate_dim   

        # map into a common space
        self.project_t = nn.Sequential()
        self.project_t.add_module('project_t', nn.Linear(in_features=self.hidden_dim, out_features=self.hidden_dim))
        self.project_t.add_module('project_t_activation', nn.ReLU())
        self.project_t.add_module('project_t_layer_norm', nn.LayerNorm(self.hidden_dim))

        # self.project_v = nn.Sequential()
        # self.project_v.add_module('project_v', nn.Linear(in_features=hidden_dim, out_features=hidden_dim))
        # self.project_v.add_module('project_v_activation', nn.ReLU())
        # self.project_v.add_module('project_v_layer_norm', nn.LayerNorm(hidden_dim))

        self.project_a = nn.Sequential()
        self.project_a.add_module('project_a', nn.Linear(in_features=self.hidden_dim, out_features=self.hidden_dim))
        self.project_a.add_module('project_a_activation', nn.ReLU())
        self.project_a.add_module('project_a_layer_norm', nn.LayerNorm(self.hidden_dim))

        # private encoders
        self.private_t = nn.Sequential()
        self.private_t.add_module('private_t_1', nn.Linear(in_features=self.hidden_dim, out_features=self.hidden_dim))
        self.private_t.add_module('private_t_activation_1', nn.Sigmoid())
        
        # self.private_v = nn.Sequential()
        # self.private_v.add_module('private_v_1', nn.Linear(in_features=hidden_dim, out_features=hidden_dim))
        # self.private_v.add_module('private_v_activation_1', nn.Sigmoid())
        
        self.private_a = nn.Sequential()
        self.private_a.add_module('private_a_3', nn.Linear(in_features=self.hidden_dim, out_features=self.hidden_dim))
        self.private_a.add_module('private_a_activation_3', nn.Sigmoid())
        
        # shared encoder
        self.shared = nn.Sequential()
        self.shared.add_module('shared_1', nn.Linear(in_features=self.hidden_dim, out_features=self.hidden_dim))
        self.shared.add_module('shared_activation_1', nn.Sigmoid())

        # reconstruct
        self.recon_t = nn.Sequential()
        self.recon_t.add_module('recon_t_1', nn.Linear(in_features=self.hidden_dim, out_features=self.hidden_dim))
        # self.recon_v = nn.Sequential()
        # self.recon_v.add_module('recon_v_1', nn.Linear(in_features=hidden_dim, out_features=hidden_dim))
        self.recon_a = nn.Sequential()
        self.recon_a.add_module('recon_a_1', nn.Linear(in_features=self.hidden_dim, out_features=self.hidden_dim))

        # fusion + cls
        self.fusion = Adapter(self.hidden_dim*4, self.hidden_dim)
        self.transformer_encoder = TransformerEncoderLayer(d_model=self.hidden_dim, nhead=self.num_heads)

        self.fc_out = nn.Linear(self.hidden_dim*4, self.output_dim)

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
        
    def shared_private(self, utterance_t, utterance_a):
        # Projecting to same sized space
        self.utt_t_orig = utterance_t = self.project_t(utterance_t)
        self.utt_a_orig = utterance_a = self.project_a(utterance_a)

        # Private-shared components
        self.utt_private_t = self.private_t(utterance_t)
        self.utt_private_a = self.private_a(utterance_a)

        self.utt_shared_t = self.shared(utterance_t)
        self.utt_shared_a = self.shared(utterance_a)

    def reconstruct(self):
        self.utt_t = (self.utt_private_t + self.utt_shared_t)
        self.utt_a = (self.utt_private_a + self.utt_shared_a)

        self.utt_t_recon = self.recon_t(self.utt_t)
        self.utt_a_recon = self.recon_a(self.utt_a)

    def get_recon_loss(self):
        inter_loss =  MSE()(self.utt_t_recon, self.utt_t_orig)
        # loss += MSE()(self.utt_v_recon, self.utt_v_orig)
        inter_loss += MSE()(self.utt_a_recon, self.utt_a_orig)
        inter_loss = inter_loss / 2.0
        return inter_loss

    def get_diff_loss(self):
        shared_t = self.utt_shared_t
        # shared_v = self.utt_shared_v
        shared_a = self.utt_shared_a
        private_t = self.utt_private_t
        # private_v = self.utt_private_v
        private_a = self.utt_private_a

        # Between private and shared
        inter_loss =  DiffLoss()(private_t, shared_t)
        # loss += DiffLoss()(private_v, shared_v)
        inter_loss += DiffLoss()(private_a, shared_a)

        # Across privates
        inter_loss += DiffLoss()(private_a, private_t)
        # loss += DiffLoss()(private_a, private_v)
        # loss += DiffLoss()(private_t, private_v)
        inter_loss = inter_loss/2.0
        return inter_loss

    def get_cmd_loss(self):
        # losses between shared states
        # loss =  CMD()(self.utt_shared_t, self.utt_shared_v, 5)
        inter_loss = CMD()(self.utt_shared_t, self.utt_shared_a, 5)
        # loss += CMD()(self.utt_shared_a, self.utt_shared_v, 5)
        return inter_loss
    
    def forward(self, audio, input_ids, attention_mask, queries, text=None, labels=None):
        loss_fct = CrossEntropyLoss()
        return_dict = {}
        utterance_audio = self.audio_encoder(audio).last_hidden_state # [batch, seq_len, hidden]
        utterance_text  = self.text_encoder(input_ids, attention_mask=attention_mask).last_hidden_state # [batch, seq_len, hidden]
        # forward features
        emos_out, features = self.forward_features(utterance_audio, utterance_text)
        
        if labels is not None:
            ce_loss = loss_fct.get_loss(emos_out, labels)
            inter_loss = self.diff_weight  * self.get_diff_loss() + \
                    self.sim_weight   * self.get_cmd_loss()  + \
                    self.recon_weight * self.get_recon_loss()
            loss = ce_loss + inter_loss
            logits = emos_out
            
        if self.train_mode:
            return_dict['ce_loss'] = ce_loss
            return_dict['interloss'] = inter_loss
        
        else:
            return_dict['fusion_logits'] = logits
            return_dict['fusion_features'] = features

        return {
            'loss': loss,
            'logits': logits,
            'return_dict': return_dict
        }

    def forward_features(self, text_features, audio_features):
        # mean pooling
        utterance_audio = torch.mean(audio_features, dim=1) # [batch, hidden]
        utterance_text  = torch.mean(text_features, dim=1) # [batch, hidden]

        # shared-private encoders
        self.shared_private(utterance_text, utterance_audio)

        # For reconstruction
        self.reconstruct()
        
        # 1-LAYER TRANSFORMER FUSION
        h = torch.stack((
            self.utt_private_t, 
            self.utt_private_a, 
            self.utt_shared_t, 
            self.utt_shared_a), dim=0) # [4, batch, hidden]

        h = self.transformer_encoder(h)
        # h = torch.cat((h[0], h[1], h[2], h[3], h[4], h[5]), dim=1)
        h = torch.cat((h[0], h[1], h[2], h[3]), dim=1) # [batch, hidden*4]
        features = self.fusion(h) # [batch, output_dim]

        emos_out  = self.fc_out(features)
        
        return emos_out, features