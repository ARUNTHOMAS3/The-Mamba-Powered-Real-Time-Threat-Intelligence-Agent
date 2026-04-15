
import torch
import torch.nn as nn
from models.mamba_backbone import MambaBackbone

class MambaClassifier(nn.Module):
    def __init__(self, input_dim, d_model=128, n_layers=2):
        super().__init__()
        self.backbone = MambaBackbone(d_input=input_dim, d_model=d_model, d_state=16)
        # Note: MambaBackbone usually has 1 layer in current impl, we can stack them if needed.
        # But for fairness with LSTM (2 layers), we might want to ensure depth is similar.
        # The MambaBackbone in mamba_backbone.py seems to be a single block. 
        # Let's Stack them for true deep mamba.
        
        self.layers = nn.ModuleList([
            MambaBackbone(d_input=d_model if i > 0 else input_dim, d_model=d_model, d_state=16)
            for i in range(n_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)
        
    def forward(self, x):
        # x: (B, L, D)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        # Classification on last token
        return self.head(x[:, -1, :])

class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, d_model=128, n_layers=2):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        # Use n_layers+1 to compensate for fewer params per layer vs Mamba
        self.lstm = nn.LSTM(d_model, d_model, num_layers=n_layers + 1, batch_first=True)
        self.proj = nn.Linear(d_model, d_model)  # Extra projection for capacity parity
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)
        
    def forward(self, x):
        # x: (B, L, D)
        x = self.embedding(x)
        out, _ = self.lstm(x)
        out = self.proj(out)
        out = self.norm(out)
        return self.head(out[:, -1, :])
