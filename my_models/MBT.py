import torch
import torch.nn as nn
from torch.nn import TransformerEncoderLayer
from utils.loss import CrossEntropyLoss
from modules.adapter import BottleneckAdapter as Adapter
from .model import MyModel
from utils.helper import PredictLayer

class MBT(MyModel):
    def __init__(self,args):
        super(MBT, self).__init__(args)
        self.num_heads = args.num_heads
        self.dropout = args.dropout
        self.relu_dropout = args.relu_dropout
        self.res_dropout = args.res_dropout
 
        self.adapter_dim = args.adapter_dim
        self.bottleneck_len = args.bottleneck_len
    
        self.text_encoder = self.get_text_encoder()
        self.audio_encoder = self.get_audio_encoder()

        self.bottleneck_embedding = nn.Embedding(self.bottleneck_len, self.intermediate_dim)
        self.bottleneck = torch.LongTensor([i for i in range(self.bottleneck_len)]).to("cuda")
        self.b_t_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model=self.intermediate_dim, nhead=self.num_heads, batch_first=True).to("cuda") 
            for _ in range(self.fusion_layers)])
        self.b_a_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model=self.intermediate_dim, nhead=self.num_heads, batch_first=True).to("cuda") 
            for _ in range(self.fusion_layers)])

        self.pred_head_t = PredictLayer(self.intermediate_dim, self.output_dim, self.dropout, residual=True)
        self.pred_head_a = PredictLayer(self.intermediate_dim, self.output_dim, self.dropout, residual=True)
        self.pred_head_union = PredictLayer(self.intermediate_dim, self.output_dim, self.dropout, residual=True)


    def forward(self, audio_value, audio_attention_mask, input_ids, attention_mask, queries, text=None, labels=None, pixel_values=None):
        return_dict = {}
        ce_loss = CrossEntropyLoss()
        mse_loss = nn.MSELoss()

        t_output = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        a_output = self.audio_encoder(audio_value, output_hidden_states=True)

        batch_size = input_ids.shape[0]
        text_seq_len, text_hidden_size = t_output.hidden_states[0].shape[1], t_output.hidden_states[0].shape[2]
        audio_seq_len, audio_hidden_size = a_output.hidden_states[0].shape[1], a_output.hidden_states[0].shape[2]

        assert text_hidden_size == audio_hidden_size, "text_hidden_size != audio_hidden_size"

        shared_neck = self.bottleneck_embedding(self.bottleneck.expand(batch_size, self.bottleneck_len))

        a_conf_targets = []
        a_confs=[]
        
        output_a = a_output.hidden_states[-self.fusion_layers]
        output_t = t_output.hidden_states[-self.fusion_layers]

        for i in range(self.fusion_layers, 0, -1):
            b_t_layer = self.b_t_layers[-i]           
            b_a_layer = self.b_a_layers[-i]

            text_neck = torch.cat((output_t, shared_neck), dim=1) # (batch_size, seq_len, hidden_size)
            audio_neck = torch.cat((output_a, shared_neck), dim=1)

            b_t = b_t_layer(text_neck)  # (batch_size, seq_len, hidden_size)
            b_a = b_a_layer(audio_neck)

            z_fsn_t = b_t[:, text_seq_len:, :]
            z_fsn_a = b_a[:, audio_seq_len:, :]
            
            output_a = b_a[:, :audio_seq_len, :]
            output_t = b_t[:, :text_seq_len, :]
      
            shared_neck = 0.5*z_fsn_a + 0.5*z_fsn_t

        fusion_pred = torch.mean(shared_neck, dim=1) # (batch_size, bottleneck_len, output_dim)
        fusion_logits = self.pred_head_union(fusion_pred) # (batch_size, bottleneck_len, output_dim)

        a_pred = torch.mean(output_a, dim=1)
        a_logits = self.pred_head_a(a_pred)

        t_pred = torch.mean(output_t, dim=1)
        t_logits = self.pred_head_t(t_pred)

        if labels is not None:
            t_loss = self.calculate_task_loss(t_logits, labels)
            a_loss = self.calculate_task_loss(a_logits, labels)
            f_loss = self.calculate_task_loss(fusion_logits, labels)
            loss = t_loss + a_loss + f_loss
            # t_ce = ce_loss.get_loss(t_logits, labels)
            # a_ce = ce_loss.get_loss(a_logits, labels)
            # fusion_ce = ce_loss.get_loss(fusion_logits, labels)
            # loss = t_ce + a_ce + fusion_ce 
            logits = fusion_logits
        else:
            loss = None
            logits = fusion_logits

        if self.train_mode:
            return_dict["t_loss"] = t_loss
            return_dict["a_loss"] = a_loss
            return_dict["f_loss"] = f_loss
        else:
            return_dict['a_logits'] = a_logits
            return_dict['t_logits'] = t_logits
            return_dict['t_feature'] = t_pred
            return_dict['a_feature'] = a_pred
            return_dict['fusion_feature'] = fusion_pred

        return {"loss": loss, 
                "logits": logits, 
                "return_dict": return_dict}