import torch.nn as nn
import torch
from typing import Any, Optional, Tuple, Union, List
from torch.nn import functional as F
from transformers.activations import get_activation
import math

class MappingNetwork(nn.Module):
    def __init__(self, constant_len, input_dim, project_dim):
        super(MappingNetwork, self).__init__()

        self.linear1 = nn.Linear(768, project_dim)
        self.constant_len = constant_len
        self.constant = nn.Parameter(torch.randn(1, self.constant_len, project_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=project_dim,
            nhead=8,
            dim_feedforward = 2 * project_dim,
            dropout=0.,
            activation = F.relu,
            batch_first=True,
            norm_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer,num_layers=4)
        self.linear2 = nn.Linear(project_dim, input_dim)

    def forward(self, image_features):
      
        image_features_len = image_features.size(1)
        image_features_input = self.linear1(image_features) 
        constant_tokens = self.constant.expand(image_features_input.shape[0], -1, -1)
        feature_input = torch.cat([image_features_input, constant_tokens], dim=1)  
        out = self.transformer(feature_input)[:, image_features_len:]  
        out_features = self.linear2(out)  

        return out_features