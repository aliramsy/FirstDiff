import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0)) 

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.frequency_embedding_size = frequency_embedding_size

    def forward(self, t):
        half_dim = self.frequency_embedding_size // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=1)
        return self.mlp(emb)

class DiTJointBlock(nn.Module):
    r"""
    Joint Block: Concatenates x and c along the sequence dimension for joint self-attention.
    """
    def __init__(self, hidden_dim, num_heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        
        # AdaLN for the combined sequence
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 6 * hidden_dim, bias=True)
        )

    def forward(self, x, c, t_emb):
        # x shape: (B, 96, D), c shape: (B, 32, D)
        L_x = x.size(1)
        
        # 1. Concatenate along sequence dimension -> z shape: (B, 128, D)
        z = torch.cat([x, c], dim=1)
        
        # 2. AdaLN Modulation
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(t_emb).chunk(6, dim=1)
        
        # 3. Joint Self-Attention
        z_norm1 = self.norm1(z) * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        attn_out, _ = self.attn(z_norm1, z_norm1, z_norm1)
        z = z + gate_msa.unsqueeze(1) * attn_out
        
        # 4. Joint MLP
        z_norm2 = self.norm2(z) * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        mlp_out = self.mlp(z_norm2)
        z = z + gate_mlp.unsqueeze(1) * mlp_out
        
        # 5. Split back into x and c
        x_out = z[:, :L_x, :]
        c_out = z[:, L_x:, :]
        
        return x_out, c_out

class TimeSeriesDiT(nn.Module):
    def __init__(self, target_seq_len=96, history_seq_len=32, num_sensors=51, hidden_dim=256, num_heads=8, num_layers=4):
        super().__init__()
        self.target_seq_len = target_seq_len
        self.hidden_dim = hidden_dim
        
        # Project noisy target window to hidden dim
        self.x_proj = nn.Linear(num_sensors, hidden_dim)
        self.x_pos_embed = PositionalEncoding(hidden_dim, max_len=target_seq_len)
        
        # History is already projected by your Patching module, so we only need Positional Embedding
        self.c_pos_embed = PositionalEncoding(hidden_dim, max_len=history_seq_len)
        
        self.t_embedder = TimestepEmbedder(hidden_dim)
        
        # Learnable null token for CFG (replaces history condition when dropped)
        self.null_cond = nn.Parameter(torch.zeros(1, history_seq_len, hidden_dim))
        
        # Joint Transformer Blocks
        self.blocks = nn.ModuleList([
            DiTJointBlock(hidden_dim, num_heads) for _ in range(num_layers)
        ])
        
        # Output Head
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, num_sensors)

    def forward(self, x_curr, t, c_hist=None, drop_prob=0.0):
        r"""
        x_curr: Noisy target window X (B, 96, 51)
        t: Diffusion timesteps (B)
        c_hist: Pre-embedded history condition C (B, 32, 256). Can be None for unconditional generation.
        """
        B = x_curr.shape[0]
        
        # Prepare Target Window X
        x = self.x_proj(x_curr) # (B, 96, D)
        x = self.x_pos_embed(x)
        
        # Prepare History Condition (or Null Condition for CFG)
        if c_hist is None:
            # Unconditional pass: Use the learned null token for the whole batch
            c = self.null_cond.expand(B, -1, -1)
            c = self.c_pos_embed(c)
        else:
            # Conditional pass: Use provided history
            c = self.c_pos_embed(c_hist) # (B, 32, D)
            
            # CFG Dropout during training
            if drop_prob > 0.0:
                drop_mask = torch.rand(B, device=x.device) < drop_prob
                c = torch.where(drop_mask.view(B, 1, 1), self.null_cond.expand(B, -1, -1), c)
        
        # Prepare Timestep
        t_emb = self.t_embedder(t) # (B, D)
            
        # Pass through Joint Blocks
        for block in self.blocks:
            x, c = block(x, c, t_emb)
            
        # Predict Noise (or X_0)
        x = self.final_norm(x)
        out = self.head(x) # (B, 96, 51)
        
        return out

