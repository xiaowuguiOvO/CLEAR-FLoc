import torch
import torch.nn as nn
import numpy as np
import math
import torch.nn.functional as F

class F3MlpDecoder(nn.Module):
    def __init__(self, d_min=0.1, d_max=20, d_hyp=-0.2, D=128, input_dim=128):
        super().__init__()
        
        self.head = nn.Linear(input_dim, D)
        self.d_min, self.d_max, self.d_hyp, self.D, self.input_dim = d_min, d_max, d_hyp, D, input_dim
        
    def forward(self, features): # [B, 40, input_dim] -> [B, 40, 1]
        d_vals = torch.linspace(
            self.d_min ** self.d_hyp, self.d_max ** self.d_hyp, self.D,
            device=features.device
        ) ** (1 / self.d_hyp)
        logits = self.head(features)
        prob = F.softmax(logits, dim=-1)
        pred_d = torch.sum(prob * d_vals, dim=-1)
        return pred_d
    
