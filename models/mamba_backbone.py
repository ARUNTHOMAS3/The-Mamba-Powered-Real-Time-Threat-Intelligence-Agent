# models/mamba_backbone.py
"""
Mamba backbone with automatic mamba-ssm acceleration.

When mamba-ssm is installed (pip install mamba-ssm), uses CUDA-optimized
selective scan kernels for 5-10x speedup. Falls back to pure PyTorch
implementation when mamba-ssm is not available (e.g., Kaggle, CPU-only).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Try to import mamba-ssm for CUDA-accelerated selective scan
try:
    from mamba_ssm import Mamba as MambaSSM
    HAS_MAMBA_SSM = True
    print("[Mamba] Using CUDA-accelerated mamba-ssm kernels ⚡")
except Exception as e:
    HAS_MAMBA_SSM = False
    print(f"[Mamba] mamba-ssm import failed: {e}")
    print("[Mamba] falling back to pure PyTorch (slower)")


class MambaConfig:
    def __init__(self, d_model=128, d_state=16, d_conv=4, expand=2):
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = self.expand * self.d_model


class PureMambaBlock(nn.Module):
    """
    A pure PyTorch implementation of the Mamba block (S6) for CPU usage.
    It simulates the selective scan mechanism.
    """
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = expand * d_model

        # Projects input to hidden state (x) and gate (z)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        # 1D Conv for local context
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=True,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
        )

        # Projection for x to B, C, delta
        self.dt_rank = math.ceil(d_model / 16)
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + d_state * 2, bias=False)
        
        # Projection for delta
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # S4D / HiPPO matrix A (approximated as diagonal for Mamba)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.act = nn.SiLU()

    def forward(self, x):
        # x: (B, L, D)
        (b, l, d) = x.shape
        
        # 1. Project to x and z
        xz = self.in_proj(x) # (B, L, 2*d_inner)
        x_branch, z_branch = xz.chunk(2, dim=-1) # (B, L, d_inner) each

        # 2. Conv1d
        x_branch = x_branch.transpose(1, 2) # (B, d_inner, L)
        x_branch = self.conv1d(x_branch)[:, :, :l] # Causal padding
        x_branch = x_branch.transpose(1, 2) # (B, L, d_inner)
        x_branch = self.act(x_branch)

        # 3. Discretization & SSM (Scan)
        x_dbl = self.x_proj(x_branch)  # (B, L, dt_rank + 2*d_state)
        
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        
        dt = F.softplus(self.dt_proj(dt))  # (B, L, d_inner)
        
        # Selective scan — vectorized for GPU efficiency
        A = -torch.exp(self.A_log)  # (d_inner, d_state)
        
        # Pre-compute all discretized params as full tensors
        dA = torch.exp(A.unsqueeze(0).unsqueeze(0) * dt.unsqueeze(-1))  # (B, L, d_inner, d_state)
        dB = dt.unsqueeze(-1) * B.unsqueeze(2)  # (B, L, d_inner, d_state)
        dBx = dB * x_branch.unsqueeze(-1)       # (B, L, d_inner, d_state)
        
        # Sequential scan (required for correctness of recurrence)
        h = torch.zeros(b, self.d_inner, self.d_state, device=x.device, dtype=x.dtype)
        ys = []
        for t in range(l):
            h = dA[:, t] * h + dBx[:, t]
            y_t = torch.sum(h * C[:, t].unsqueeze(1), dim=-1) + self.D * x_branch[:, t]
            ys.append(y_t)
        
        y = torch.stack(ys, dim=1)  # (B, L, d_inner)
        
        # 4. Gating
        y = y * self.act(z_branch)
        out = self.out_proj(y)
        
        return out


class MambaBackbone(nn.Module):
    """
    Mamba backbone that auto-selects CUDA-accelerated or pure PyTorch block.
    
    When mamba-ssm is installed: uses MambaSSM (fused CUDA selective scan)
    When not installed:          uses PureMambaBlock (Python loop fallback)
    """
    def __init__(self, d_input, d_model=128, d_state=16):
        super().__init__()
        self.d_model = d_model
        
        # Project input to d_model first if needed
        self.embedding = nn.Linear(d_input, d_model)
        
        # Choose backend based on availabile
        if HAS_MAMBA_SSM:
            # CUDA-optimized Mamba block from mamba-ssm library
            self.mamba_block = MambaSSM(
                d_model=d_model,
                d_state=d_state,
                d_conv=4,
                expand=2,
            )
        else:
            # Pure PyTorch fallback
            self.mamba_block = PureMambaBlock(d_model, d_state=d_state)
        
        self.layernorm = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: (b, seq_len, d_input)
        x_emb = self.embedding(x)
        y = self.mamba_block(x_emb)
        return self.layernorm(y)
