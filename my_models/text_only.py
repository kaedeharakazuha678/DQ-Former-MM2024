import torch
import torch.nn as nn
from .model import MyModel
from utils.helper import PredictLayer

class TextOnly(MyModel):
    def __init__(self,args):
        super(TextOnly, self).__init__(args)
        self.text_encoder = self.get_text_encoder()
        self.pred_head_t = PredictLayer(self.text_encoder.config.hidden_size, self.output_dim, dropout=self.dropout, residual=True)

    def forward(self, audio_value = None, audio_attention_mask=None, input_ids=None, attention_mask=None, queries=None, text=None, labels=None, pixel_values=None):
        return_dict = {}
        t_output = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        t_last_hidden_state = t_output.last_hidden_state
        t_pred = torch.mean(t_last_hidden_state, dim=1)
        t_logits = self.pred_head_t(t_pred)
        
        if labels is not None:
            logits = t_logits
            loss = self.calculate_task_loss(logits, labels)
        else:
            loss = None
            logits = t_logits
            
        return {"loss": loss, 
                "logits": logits, 
                "return_dict": return_dict}
