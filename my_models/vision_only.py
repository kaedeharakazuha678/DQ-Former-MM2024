import torch
import torch.nn as nn
import torch.nn.functional as F
from .model import MyModel
from utils.helper import PredictLayer

class VisionOnly(MyModel):
    def __init__(self,args):
        super(VisionOnly, self).__init__(args)
        self.vision_encoder = self.get_vision_encoder(adapter_mode=False, frozen_mode=False)
        self.pred_head_v = PredictLayer(self.vision_encoder.config.hidden_size, self.output_dim, dropout=self.dropout, residual=True)

    def forward(self, audio_value, input_ids, attention_mask, queries=None, text=None, labels=None, pixel_values=None):
        return_dict = {}
        v_output = self.vision_encoder(pixel_values, output_hidden_states=True)
        v_last_hidden_state = v_output.last_hidden_state
        v_pred = torch.mean(v_last_hidden_state, dim=1)
        v_logits = self.pred_head_v(v_pred)

        if labels is not None:
            logits = v_logits
            loss = self.calculate_task_loss(logits, labels)
        else:
            loss = None
            logits = v_logits

        if not self.train_mode:
            return_dict['logits'] = logits
            return_dict['loss'] = loss
            
        return {"loss": loss, "logits": logits, "return_dict": return_dict}
