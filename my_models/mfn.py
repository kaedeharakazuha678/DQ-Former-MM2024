"""
paper: Memory Fusion Network for Multi-View Sequential Learning
From: https://github.com/pliang279/MFN
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from utils.loss import LabelSmoothingCELoss, CrossEntropyLoss
from transformers import Wav2Vec2Model, WhisperModel, HubertModel
from transformers import WavLMConfig, WavLMModel
from modules.transformer import TransformerEncoder, GatedTransformerEncoder
from modules.adapter import BottleneckAdapter as Adapter
from .model import MyModel

#TODO

class MFN(MyModel):
    def __init__(self, args):
        super(MFN, self).__init__(args)
        self.dropout = args.dropout

        self.adapter_mode = args.adapter_mode # whether to add adapter to the model, 't': text, 'a': audio, 'ta': text and audio
        self.frozen_mode = args.frozen_mode  # whether to freeze the parameters of the model
        assert self.adapter_mode in ['t', 'a', 'ta', 'tv', 'av', 'tav', 'v', ''], "Unknown adapter mode"
        assert self.frozen_mode in ['t', 'a', 'ta', 'tv', 'av', 'tav', 'v', ''], "Unknown frozen mode"
        self.text_encoder = self.get_text_encoder(self.adapter_mode, self.frozen_mode)
        self.text_dim = self.text_encoder.config.hidden_size
        self.audio_encoder = self.get_audio_encoder(self.adapter_mode, self.frozen_mode)
        self.audio_dim = self.audio_encoder.config.hidden_size

        self.mem_dim = 128 # as MERBench
        self.window_dim = 2 # as MERBench
        self.hidden_dim = args.intermediate_dim # 128

        # params: intermedia
        media_type_num = 2
        total_h_dim =  self.hidden_dim * media_type_num
        attInShape = total_h_dim * self.window_dim
        gammaInShape = attInShape + self.mem_dim
        final_out = total_h_dim + self.mem_dim
        feature_output_dim = self.hidden_dim // 2

        # each modality has one lstm cell
        self.lstm_text = nn.LSTMCell(self.text_dim,  self.hidden_dim)
        self.lstm_audio = nn.LSTMCell(self.audio_dim, self.hidden_dim)

        self.attention_fc1 = nn.Linear(attInShape, self.hidden_dim)
        self.attention_fc2 = nn.Linear(self.hidden_dim, attInShape)
        self.attention_dropout = nn.Dropout(self.dropout)

        self.attention2_fc1 = nn.Linear(attInShape, self.hidden_dim)
        self.attention2_fc2 = nn.Linear(self.hidden_dim, self.mem_dim)
        self.attention2_dropout = nn.Dropout(self.dropout)

        self.gamma1_fc1 = nn.Linear(gammaInShape, self.hidden_dim)
        self.gamma1_fc2 = nn.Linear(self.hidden_dim, self.mem_dim)
        self.gamma1_dropout = nn.Dropout(self.dropout)

        self.gamma2_fc1 = nn.Linear(gammaInShape, self.hidden_dim)
        self.gamma2_fc2 = nn.Linear(self.hidden_dim, self.mem_dim)
        self.gamma2_dropout = nn.Dropout(self.dropout)

        self.output_fc1 = nn.Linear(final_out, self.hidden_dim)
        self.output_fc2 = nn.Linear(self.hidden_dim, feature_output_dim)
        self.output_dropout = nn.Dropout(self.dropout)

        # output results
        self.fc_out = nn.Linear(feature_output_dim, self.output_dim)

    
    
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

        features, emos_out = self.forward_features(t_embedding, a_embedding)

        ce_loss_value = ce_loss(emos_out, labels)
        if labels is not None:
            loss = ce_loss_value
            return_dict['loss'] = loss
            logits = emos_out
        
        if self.train_mode:
            return_dict['ce_loss'] = ce_loss_value
        
        else:
            return_dict['logits'] = logits
            return_dict['features'] = features
            
        return {
            'loss': loss,
            'logits': logits,
            'return_dict': return_dict,
        }
        
    # MFN needs aligned multimodal features
    def forward_features(self, text_features, audio_features,):
        '''
        simulating word-align network (for seq_len_T == seq_len_A == seq_len_V)
        audio_x: tensor of shape (batch, seqlen, audio_in)
        video_x: tensor of shape (batch, seqlen, video_in)
        text_x: tensor of shape  (batch, seqlen, text_in)
        '''

        # text_x  = batch['texts'].permute(1,0,2)  # [seqlen, batch, dim]
        # audio_x = batch['audios'].permute(1,0,2) # [seqlen, batch, dim]
        # video_x = batch['videos'].permute(1,0,2) # [seqlen, batch, dim]
        text_x = text_features.permute(1,0,2) # [seqlen, batch, dim]
        audio_x = audio_features.permute(1,0,2) # [seqlen, batch, dim]

        seq_len, batch_size, dim = text_x.size()

        # Initializing hidden states and memory cells for text and audio
        self.hidden_text = torch.zeros(batch_size, self.hidden_dim)
        self.hidden_audio = torch.zeros(batch_size, self.hidden_dim)
        self.cell_text = torch.zeros(batch_size, self.hidden_dim)
        self.cell_audio = torch.zeros(batch_size, self.hidden_dim)
        self.memory = torch.zeros(batch_size, self.mem_dim)

        # Lists for storing outputs at each time step
        all_hidden_texts = []
        all_hidden_audios = []
        all_cell_texts = []
        all_cell_audios = []
        all_memories = []

        for i in range(seq_len):
            # Previous time step
            prev_cell_text = self.cell_text
            prev_cell_audio = self.cell_audio

            # Current time step
            new_hidden_text, new_cell_text = self.lstm_text(text_x[i], (self.hidden_text, self.cell_text))
            new_hidden_audio, new_cell_audio = self.lstm_audio(audio_x[i], (self.hidden_audio, self.cell_audio))
            
            # Concatenating previous and new cell states
            prev_cells = torch.cat([prev_cell_text, prev_cell_audio], dim=1) # shape: [batch, 2*hidden_dim]
            new_cells  = torch.cat([new_cell_text, new_cell_audio],  dim=1) # shape: [batch, 2*hidden_dim]

            cStar = torch.cat([prev_cells, new_cells], dim=1)
            attention = F.softmax(self.attention_fc2(self.attention_dropout(F.relu(self.attention_fc1(cStar)))), dim=1)
            attended = attention * cStar
            cHat = torch.tanh(self.attention2_fc2(self.attention2_dropout(F.relu(self.attention2_fc1(attended)))))
            both = torch.cat([attended, self.memory], dim=1)
            gamma1 = torch.sigmoid(self.gamma1_fc2(self.gamma1_dropout(F.relu(self.gamma1_fc1(both)))))
            gamma2 = torch.sigmoid(self.gamma2_fc2(self.gamma2_dropout(F.relu(self.gamma2_fc1(both)))))
            self.memory = gamma1 * self.memory + gamma2 * cHat

            all_memories.append(self.memory)

            # Update LSTM states
            self.hidden_text, self.cell_text = new_hidden_text, new_cell_text
            self.hidden_audio, self.cell_audio = new_hidden_audio, new_cell_audio

            all_hidden_texts.append(self.hidden_text)
            all_hidden_audios.append(self.hidden_audio)
            all_cell_texts.append(self.cell_text)
            all_cell_audios.append(self.cell_audio)

        # Final hidden state
        last_hidden_text = all_hidden_texts[-1]
        last_hidden_audio = all_hidden_audios[-1]
        last_memory = all_memories[-1]
        last_hidden_states = torch.cat([last_hidden_text, last_hidden_audio, last_memory], dim=1)
        
        features = self.output_fc2(self.output_dropout(F.relu(self.output_fc1(last_hidden_states))))
        self.last_hidden_states = last_hidden_states # for external access

        emos_out = self.fc_out(features)

        return features, emos_out