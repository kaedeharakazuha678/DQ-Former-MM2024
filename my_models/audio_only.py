import torch
import torch.nn as nn
import torch.nn.functional as F
from .model import MyModel
from utils.helper import PredictLayer

class AudioOnly(MyModel):
    def __init__(self,args):
        super(AudioOnly, self).__init__(args)
        self.audio_encoder = self.get_audio_encoder(adapter_mode=False, frozen_mode=False)
        self.pred_head_a = PredictLayer(self.audio_encoder.config.hidden_size, self.output_dim, dropout=self.dropout, residual=True)

    def forward(self, audio_value, audio_attention_mask, input_ids, attention_mask, queries=None, text=None, labels=None, pixel_values=None):
        return_dict = {}
        a_output = self.audio_encoder(audio_value, output_hidden_states=True)
        a_last_hidden_state = a_output.last_hidden_state
        a_pred = torch.mean(a_last_hidden_state, dim=1)
        a_logits = self.pred_head_a(a_pred)

        if labels is not None:
            logits = a_logits
            loss = self.calculate_task_loss(logits, labels)
        else:
            loss = None
            logits = a_logits

        if not self.train_mode:
            return_dict['logits'] = logits
            return_dict['loss'] = loss
            
        return {"loss": loss, "logits": logits, "return_dict": return_dict}