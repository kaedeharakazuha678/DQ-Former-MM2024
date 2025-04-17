import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.nn import TransformerEncoderLayer
from modules.transformer import TransformerEncoderLayer as Cross_model_encoder
from modules.ConNet import ConNet
from utils.helper import PredictLayer
from .model import MyModel

class DQ_TAV(MyModel):
    def __init__(self,args):
        super(DQ_TAV, self).__init__(args)   
        self.num_heads = args.num_heads
        self.fusion_layers = args.num_layers
        self.dropout = args.dropout
        self.relu_dropout = args.relu_dropout
        self.res_dropout = args.res_dropout
        self.adapter_dim = args.adapter_dim
        self.bottleneck_len = args.bottleneck_len
        self.tau = args.tau
        self.t_loss_weight = args.t_loss_weight
        self.a_loss_weight = args.a_loss_weight
        self.v_loss_weight = args.v_loss_weight
        self.f_loss_weight = args.f_loss_weight
    
        self.text_encoder = self.get_text_encoder(adapter_mode=False, frozen_mode=False)
        self.audio_encoder = self.get_audio_encoder(adapter_mode=False, frozen_mode=False)
        self.vision_encoder = self.get_vision_encoder(adapter_mode=False, frozen_mode=False)
        self.bottleneck_embedding = nn.Embedding(self.bottleneck_len, self.intermediate_dim)
        
        self.query_encoder = nn.ModuleList([
            TransformerEncoderLayer(d_model=self.intermediate_dim, nhead=self.num_heads, batch_first=False, dropout=self.dropout, activation='gelu')
            for _ in range(self.fusion_layers)])
        
        # cross modal encoder
        self.cme_qt = nn.ModuleList([
            Cross_model_encoder(embed_dim=self.intermediate_dim, num_heads=self.num_heads, attn_dropout=self.dropout, relu_dropout=self.relu_dropout, res_dropout=self.res_dropout)
            for _ in range(self.fusion_layers)])
        self.cme_qa = nn.ModuleList([
            Cross_model_encoder(embed_dim=self.intermediate_dim, num_heads=self.num_heads, attn_dropout=self.dropout, relu_dropout=self.relu_dropout, res_dropout=self.res_dropout)
            for _ in range(self.fusion_layers)])
        self.cme_qv = nn.ModuleList([
            Cross_model_encoder(embed_dim=self.intermediate_dim, num_heads=self.num_heads, attn_dropout=self.dropout, relu_dropout=self.relu_dropout, res_dropout=self.res_dropout)
            for _ in range(self.fusion_layers)])
        
        # add unimodal fusion weight to adaptively fuse the unimodal representations 
        self.conf_net = ConNet(self.intermediate_dim)
    
        # prediciton head
        self.pred_head_t = PredictLayer(self.intermediate_dim, self.output_dim, dropout=self.dropout, residual=True)
        self.pred_head_a = PredictLayer(self.intermediate_dim, self.output_dim, dropout=self.dropout, residual=True)
        self.pred_head_v = PredictLayer(self.intermediate_dim, self.output_dim, dropout=self.dropout, residual=True)
        self.pred_head_union = PredictLayer(self.intermediate_dim, self.output_dim, dropout=self.dropout, residual=True)


    def get_confidence_score_train(self, logits, labels):
        # change the shape of labels to one-hot vector
        labels_one_hot = torch.zeros_like(logits)
        labels_one_hot.scatter_(1, labels.unsqueeze(1), 1)
        # calculate the confidence score
        conf = (F.softmax(logits, dim=-1) * labels_one_hot).sum(dim=-1)
        return conf
    

    def get_confidence_score_eval(self, logits):
        conf = F.softmax(logits, dim=-1).max(dim=-1)[0]
        return conf


    def forward(self, audio_value, audio_attention_mask, input_ids, attention_mask, queries=None, text=None, labels=None, pixel_values=None):
        return_dict = {}

        t_output = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        a_output = self.audio_encoder(audio_value, output_hidden_states=True)
        v_output = self.vision_encoder(pixel_values, output_hidden_states=True)

        batch_size = input_ids.shape[0]
        text_seq_len, text_hidden_size = t_output.hidden_states[0].shape[1], t_output.hidden_states[0].shape[2]
        audio_seq_len, audio_hidden_size = a_output.hidden_states[0].shape[1], a_output.hidden_states[0].shape[2]
        vision_seq_len, vision_hidden_size = v_output.hidden_states[0].shape[1], v_output.hidden_states[0].shape[2]

        assert text_hidden_size == audio_hidden_size == vision_hidden_size, "text_hidden_size, audio_hidden_size, vision_hidden_size are not equal!"

        # shared_neck = self.bottleneck_embedding(self.bottleneck.expand(batch_size, self.bottleneck_len))
        shared_neck = self.bottleneck_embedding(queries)  # (batch_size, bottleneck_len, hidden_size)
        
        a_confs = []
        t_confs = []
        v_confs = []
        for i in range(self.fusion_layers, 0, -1):
            shared_neck = self.query_encoder[-i](shared_neck.permute(1,0,2))  # (len, batch_size, hidden_size)
            resicual_neck = shared_neck  # (len, batch_size, hidden_size)
            output_t = t_output.hidden_states[-i].permute(1,0,2)  # (len, batch_size, hidden_size)
            output_a = a_output.hidden_states[-i].permute(1,0,2)  # (len, batch_size, hidden_size)
            output_v = v_output.hidden_states[-i].permute(1,0,2)

            # cross modal encoder
            qt = self.cme_qt[-i](shared_neck, output_t, output_t)  # (len, batch_size, hidden_size)
            qa = self.cme_qa[-i](shared_neck, output_a, output_a)  # (len, batch_size, hidden_size)
            qv = self.cme_qv[-i](shared_neck, output_v, output_v)  

            # get modality weight
            a_conf = self.conf_net(torch.mean(qa, dim=0))
            t_conf = self.conf_net(torch.mean(qt, dim=0))
            v_conf = self.conf_net(torch.mean(qv, dim=0))
            # 归一化
            sum_conf = a_conf + t_conf + v_conf
            a_conf = a_conf / sum_conf
            t_conf = t_conf / sum_conf
            v_conf = v_conf / sum_conf

            a_conf_expanded = a_conf.expand(shared_neck.shape[0], -1, shared_neck.shape[2])  # (len, batch_size, hidden_size)
            # if a_confidence is too low , we don't add it to the fusion feature
            a_conf_expanded = torch.where(a_conf_expanded > self.tau, a_conf_expanded, torch.zeros_like(a_conf_expanded))
            
            t_conf_expanded = t_conf.expand(shared_neck.shape[0], -1, shared_neck.shape[2])  # (len, batch_size, hidden_size)
            # if a_confidence is too low , we don't add it to the fusion feature
            t_conf_expanded = torch.where(t_conf_expanded > self.tau, t_conf_expanded, torch.zeros_like(t_conf_expanded))
            
            v_conf_expanded = v_conf.expand(shared_neck.shape[0], -1, shared_neck.shape[2])  # (len, batch_size, hidden_size)
            # if a_confidence is too low , we don't add it to the fusion feature
            v_conf_expanded = torch.where(v_conf_expanded > self.tau, v_conf_expanded, torch.zeros_like(v_conf_expanded))
            

            shared_neck = a_conf_expanded * qa + t_conf_expanded * qt + v_conf_expanded * qv  # (len, batch_size, hidden_size)
            shared_neck = shared_neck + resicual_neck  # (len, batch_size, hidden_size)
            shared_neck = shared_neck.permute(1,0,2)  # (batch_size, len, hidden_size)
            a_conf_expanded = a_conf_expanded.permute(1,0,2)  # (batch_size, len, hidden_size) 
            t_conf_expanded = t_conf_expanded.permute(1,0,2)
            v_conf_expanded = v_conf_expanded.permute(1,0,2)
            # shared_neck = self.a_weight*z_fsn_a + self.t_weight*z_fsn_t
            # save a_conf with the layer number
            a_confs.append(a_conf.squeeze().detach().cpu().numpy())
            t_confs.append(t_conf.squeeze().detach().cpu().numpy())
            v_confs.append(v_conf.squeeze().detach().cpu().numpy())
            
        fusion_pred = torch.mean(shared_neck, dim=1) # (batch_size, bottleneck_len, output_dim)
        fusion_logits = self.pred_head_union(fusion_pred) # (batch_size, bottleneck_len, output_dim)

        output_a = a_output.last_hidden_state
        a_pred = torch.mean(output_a, dim=1)
        a_logits = self.pred_head_a(a_pred)
        
        output_t = t_output.last_hidden_state
        t_pred = torch.mean(output_t, dim=1)
        t_logits = self.pred_head_t(t_pred)

        output_v = v_output.last_hidden_state
        v_pred = torch.mean(output_v, dim=1)
        v_logits = self.pred_head_v(v_pred)

        if labels is not None:
            t_loss = self.calculate_task_loss(t_logits, labels)
            a_loss = self.calculate_task_loss(a_logits, labels)
            v_loss = self.calculate_task_loss(v_logits, labels)
            f_loss = self.calculate_task_loss(fusion_logits, labels)
            loss = self.t_loss_weight * t_loss + self.a_loss_weight * a_loss + self.v_loss_weight * v_loss + self.f_loss_weight * f_loss
            logits = fusion_logits
        else:
            loss = None
            logits = fusion_logits

        if self.train_mode:
            return_dict["t_loss"] = t_loss
            return_dict["a_loss"] = a_loss
            return_dict["v_loss"] = v_loss
            return_dict["f_loss"] = f_loss
        else:
            return_dict['a_logits'] = a_logits
            return_dict['t_logits'] = t_logits
            return_dict['v_logits'] = v_logits
            return_dict['t_feature'] = t_pred
            return_dict['a_feature'] = a_pred
            return_dict['v_feature'] = v_pred
            return_dict['fusion_feature'] = fusion_pred
            return_dict['a_confs'] = a_confs
            return_dict['t_confs'] = t_confs
            return_dict['v_confs'] = v_confs

        return {"loss": loss, 
                "logits": logits, 
                "return_dict": return_dict}