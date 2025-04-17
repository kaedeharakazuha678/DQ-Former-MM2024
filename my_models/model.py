import torch
import torch.nn as nn
import torch.utils.checkpoint
from transformers import AutoModel, HubertModel, Wav2Vec2Model, WavLMModel, WhisperModel
from transformers.models.whisper.modeling_whisper import WhisperEncoder
from transformers import CLIPVisionModel
from modules import Adapter
from modules.transformer import TransformerEncoder, GatedTransformerEncoder
from utils.loss import CrossEntropyLoss as MyCrossEntropyLoss
from utils.loss import MSELoss as MyMSELoss
from utils.loss import KLDivergenceLoss, MMD, InfoNCE_KDLoss
from utils.helper import PredictLayer
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

class MyModel(nn.Module):
    def __init__(self, args):
        super(MyModel, self).__init__()
        self.args = args
        self.dataset_name = args.dataset_name
        self.label_type = args.label_type
        self.class_type = args.class_type
        self.output_dim = self.get_output_dim()
        self.problem_type = None
        self.num_labels = args.num_labels
        self.num_layers = args.num_layers
        self.fusion_layers = args.fusion_layers
        self.intermediate_dim = args.intermediate_dim
        self.dropout = args.dropout
        self.text_model = args.text_model
        self.audio_model = args.audio_model
        self.vision_model = args.vision_model
        self.train_mode = args.do_train
        self.ce_loss = MyCrossEntropyLoss()
        self.mse_loss = MyMSELoss()


    def get_output_dim(self):
        output_dim_mapping = {'IEMOCAP': 6, 'MELD': 7, 'CMU_MOSI': 3, 'CMU_MOSEI':3, }
        if self.label_type == 'discrete':
            if self.class_type == '4_ways':
                output_dim = 4
            elif self.class_type == 'sentiment':
                output_dim = 3
            else: 
                output_dim = output_dim_mapping.get(self.dataset_name, 3)
        elif self.label_type == 'continuous':
            output_dim = 1
        else:
            raise ValueError('Invalid label_type')
        return output_dim
    
    def get_vision_encoder(self, adapter_mode=False, frozen_mode=False):
        vision_encoder = CLIPVisionModel.from_pretrained(self.vision_model)

        if frozen_mode:
            # freeze all the parameters of the model
            for param in vision_encoder.parameters():
                param.requires_grad = False
        elif adapter_mode:
            for i in range(0, vision_encoder.config.num_hidden_layers):
                hidden_size = vision_encoder.config.hidden_size
                adapter = Adapter(hidden_size, hidden_size//4)
                setattr(vision_encoder.encoder.layer[i].attention.output.dense, 'adapter', adapter)
                setattr(vision_encoder.encoder.layer[i].output.dense, 'adapter', adapter)

            for param_name, param in vision_encoder.named_parameters():
                # can only tune the adapter parameters and LayerNorm parameters 
                if 'adapter' in param_name or 'LayerNorm' in param_name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        
        return vision_encoder
    

    def get_text_encoder(self, adapter_mode=False, frozen_mode=False):
        text_encoder = AutoModel.from_pretrained(self.text_model)

        if frozen_mode:
            # freeze all the parameters of the model
            for param in text_encoder.parameters():
                param.requires_grad = False
        elif adapter_mode:
            for i in range(0, text_encoder.config.num_hidden_layers):
                hidden_size = text_encoder.config.hidden_size
                adapter = Adapter(hidden_size, hidden_size//4)
                setattr(text_encoder.encoder.layer[i].attention.output.dense, 'adapter', adapter)
                setattr(text_encoder.encoder.layer[i].output.dense, 'adapter', adapter)

            for param_name, param in text_encoder.named_parameters():
                # can only tune the adapter parameters and LayerNorm parameters 
                if 'adapter' in param_name or 'LayerNorm' in param_name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        
        return text_encoder


    def get_audio_encoder(self, adapter_mode=False, frozen_mode=False):
        MODEL_MAP = {
            "hubert": HubertModel,
            "wav2vec2": Wav2Vec2Model,
            "wavlm": WavLMModel,
            "whisper": WhisperModel,
        }

        for key, model_class in MODEL_MAP.items():
            if key in self.audio_model:
                audio_encoder = model_class.from_pretrained(self.audio_model)
                if key == 'whisper':
                    # audio_encoder = audio_encoder.encoder
                    audio_encoder = WhisperEncoder.from_pretrained(self.audio_model)
                break
                     
        else:
            available_models = ", ".join(MODEL_MAP.keys())
            raise ValueError(f"Unknown audio model '{self.audio_model}'. Available models are: {available_models}.")
        
        for param_name, param in audio_encoder.named_parameters():
            if "feature_projection" in param_name or "feature_extractor" in param_name or 'conv' in param_name or "masked_spec_embed" in param_name:
            # if 'feature_projection' in param_name or 'feature_extractor' in param_name:
                param.requires_grad = False
            if 'pos_conv_embed' in param_name:
                param.requires_grad = False
            else:
                param.requires_grad = True

        #region frozen or adapter
        if frozen_mode:
            for param in audio_encoder.parameters():
                param.requires_grad = False
        
        elif adapter_mode:
            for i in range(0, audio_encoder.config.num_hidden_layers):
                hidden_size = audio_encoder.config.hidden_size
                adapter = Adapter(hidden_size, hidden_size//4)
                setattr(audio_encoder.encoder.layers[i].attention, 'adapter', adapter)
                # setattr(audio_encoder.encoder.layers[i].feed_forward, 'adapter', adapter)

            for param_name, param in audio_encoder.named_parameters():
                if 'adapter' in param_name or 'LayerNorm' in param_name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False                   
        #endregion
        
        return audio_encoder


    def get_kd_loss(self, teacher_logits, student_logits, teacher_feature, student_feature, labels):
        # set loss functions
        kl_loss = KLDivergenceLoss()
        mmd = MMD(kernel=self.args.kernel, kernel_params=self.args.kernel_params)
        info_nce = InfoNCE_KDLoss(info_temp=self.args.info_temp)
        
        # knowledge distillation method, select NST_KD, CRD_KD or KL_KD
        if self.args.kd_mode == 'nst': # NST_KD
            kd_loss = mmd.get_loss(teacher_feature, student_feature)
        elif self.args.kd_mode == 'crd': # CRD_KD
            kd_loss = info_nce.get_loss(teacher_feature, student_feature) 
        elif self.args.kd_mode == 'kl': # KL_KD
            kd_loss = kl_loss.get_loss(student_logits, teacher_logits)
        elif self.args.kd_mode == 'nst_kl': # NST_KD + KL_KD
            kd_loss = mmd.get_loss(teacher_feature, student_feature) + kl_loss.get_loss(student_logits, teacher_logits)
        elif self.args.kd_mode == 'crd_kl': # CRD_KD + KL_KD
            kd_loss = info_nce.get_loss(teacher_feature, student_feature) + kl_loss.get_loss(student_logits, teacher_logits)
        else:  # no knowledge distillation
            kd_loss = 0.0
        
        return kd_loss


    def calculate_task_loss(self, logits, labels):
        loss = None
        if labels is not None:
            if self.label_type == 'discrete':
                loss = self.ce_loss.get_loss(logits, labels.view(-1))
            elif self.label_type == 'continuous':
                loss = self.mse_loss.get_loss(logits.view(-1), labels.view(-1))
            else:
                raise ValueError('Invalid label_type')
        else:
            raise ValueError('labels is None')
        return loss
    

    def get_loss(self, logits, labels):
        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            if self.problem_type is None:
                if self.num_labels == 1:
                    self.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.problem_type = "single_label_classification"
                else:
                    self.problem_type = "multi_label_classification"

            if self.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.view(-1), labels.view(-1))
                else:
                    loss = loss_fct(logits, labels)
            elif self.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
