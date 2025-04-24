import numpy as np
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils.constant_utils import *
from my_models import *
from my_datasets import *


def my_args(parser):
    #region basic settings
    parser.add_argument("--modalities", type=str, default='avt', help="modalities, choice from ['a','v','t','av','at','vt','avt']")
    parser.add_argument("--text_model", type=str, default='bert-base-uncased',help="Path to text model")
    parser.add_argument("--audio_model", type=str, default='facebook/wav2vec2-base-960h',help="Path to audio model")
    parser.add_argument("--audio_processor", type=str, default='facebook/wav2vec2-base',help="if audio_model is wavlm, audio_processor should be facebook/wav2vec2-base")
    parser.add_argument("--vision_model", type=str, default='clip-vit-base-patch-24', help="Path to vision model")
    parser.add_argument("--output_dir", type=str, default='../outputs/', help="Path to output folder, checkpoints will be saved here")

    # -------------dataset settings----------------
    parser.add_argument("--dataset_root_dir", type=str, default='/ERC', help="Root directory of ERC dataset")
    parser.add_argument("--dataset_name", type=str, default='IEMOCAP', choices=['IEMOCAP', 'MELD', 'CMU_MOSEI', 'CMU_MOSI', 'CH_SIMS', 'CH_SIMS_v2'], help="Name of the dataset")
    parser.add_argument("--class_type", type=str, default='emotion', 
                        choices=['emotion', 'sentiment', '4_ways'], help="4_ways is only enable for iemocap dataset")
    parser.add_argument("--cv_fold", type=int, default=1, help="cross validation fold for iemocap dataset",
                        choices=[1, 2, 3, 4, 5])
    parser.add_argument("--split_type", type=str, default='session', help="split type for iemocap dataset",
                        choices=['session', 'dialog', 'random'])    
    parser.add_argument("--label_type", type=str, default='discrete',
                        choices=['discrete', 'continuous'], help="only enable for iemocap and cmu dataset")
    
    # -------------common settings----------------
    parser.add_argument("--context_window_size", type=int, default=0, help="history context window size")
    parser.add_argument("--max_seq_len", type=int, default=128, help="Max text input sequence length")  
    parser.add_argument("--time_durations", type=float, default=5.0, help="max time duration for audio and video")
    parser.add_argument("--need_weights", type=bool, default=False, help="whether to return attention weights of single stream transformer")
    parser.add_argument("--seed", type=int, default=10, help="Random seed")
    #endregion

    #region model settings
    parser.add_argument("--num_labels", type=int, default=7, help="output_dim")
    parser.add_argument("--num_heads", type=int, default=4, help="Attention heads")
    parser.add_argument("--num_layers", type=int, default=6, help="Attention layers")  
    parser.add_argument("--fusion_layers", type=int, default=4, help="fusion layers")
    parser.add_argument("--attn_dropout", type=float, default=0.5, help="Attention dropout")
    parser.add_argument("--attn_dropout_a", type=float, default=0.5)
    parser.add_argument("--attn_dropout_v", type=float, default=0.5)
    parser.add_argument("--res_dropout", type=float, default=0.3, help="Residual dropout")
    parser.add_argument("--relu_dropout", type=float, default=0.3, help="Relu dropout")
    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout rate")
    parser.add_argument("--adapter_dim", type=int, default=128, help="adapter dim")
    parser.add_argument("--intermediate_dim", type=int, default=768, help="x gated attention dim")
    parser.add_argument("--attn_mask", type=bool, default=True, help="Use attention mask for self-attention")
    parser.add_argument("--gated", type=bool, default=True, help="Use gated attention")
    parser.add_argument("--tau", type=float, default=0.1, help="tau is used to control the contribution of the two attention mechanisms")
    parser.add_argument("--num_audio_queries", type=int, default=128, help="Number of audio queries")
    parser.add_argument("--model_name", type=str, default=None, 
                        help="'audio_only', 'text_only','single_stream', 'single_stream_mask_text', 'single_stream_mask_audio', 'dq_former', 'dq_former_mask_audio', 'dq_former_mask_text', 'lmnb'")
    parser.add_argument("--bottleneck_len", type=int, default=4, help="bottleneck length")
    parser.add_argument("--enable_modality_embedding", type=bool, default=False, help="enable modality embedding, only for single_stream")
    #endregion

    #region lora settings
    parser.add_argument("--pooling_strategy", type=str, default='mean')
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    #endregion

    #region loss settings
    parser.add_argument("--kd_mode", type=str, default= None, 
                        choices=['crd', 'nst', 'kl', None, 'crd_kl', 'nst_kl', 'none'], help="Knowledge Distillation mode")
    parser.add_argument("--kernel", type=str, default='polynomial', help="kernel function,chioce from ['gaussian','polynomial']")
    parser.add_argument("--kernel_params", type=dict, default={'degree':2, 'coef':1, 'const':0}, 
                        help="kernel parameters [gamma,degree,coef,const]")
    parser.add_argument("--info_temp", type=float, default=0.1, help="the temperature of infoNCE loss, contrastive learning")
    # loss weights
    parser.add_argument("--kd_loss_weight", type=float, default=0.0, help="control the contribution of Knowledge Distillation loss")
    parser.add_argument("--ce_loss_weight", type=float, default=1.0, help="control the contribution of CE loss")
    parser.add_argument("--t_loss_weight", type=float, default=1.0, help="control the contribution of textual modality loss")
    parser.add_argument("--a_loss_weight", type=float, default=1.0, help="control the contribution of audio modality loss")
    parser.add_argument("--v_loss_weight", type=float, default=1.0, help="control the contribution of visual modality loss")
    parser.add_argument("--f_loss_weight", type=float, default=1.0, help="control the contribution of fusion loss")
    #endregion

    return parser
