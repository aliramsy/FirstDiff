import torch
import torch.nn as nn
import math
import torch.fft

class SpectralSignatureEncoder(nn.Module):
    def __init__(self, num_sensors, hidden_dim, target_seq_length):
        super().__init__()
        # We only need the positive frequencies (n/2 + 1)
        self.mlp = nn.Sequential(
            nn.Linear(num_sensors * (target_seq_length // 2 + 1), hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, x):
        # x shape: (B, T, C) -> (B, 96, 51)
        
        # 1. Compute rFFT along the time dimension
        # Result shape: (B, 51, 49)
        fft_res = torch.fft.rfft(x, dim=1).transpose(1, 2)
        
        # 2. Compute PSD (Magnitude Squared)
        psd = torch.abs(fft_res)**2
        
        # 3. Flatten and Encode
        psd_flat = psd.reshape(psd.shape[0], -1)
        return self.mlp(torch.log(psd_flat + 1e-8)) # Log for numerical stability

# =========================================================
# 1. SAFE POSITIONAL ENCODING
# =========================================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        seq_len = x.size(1)
        if seq_len > self.pe.size(1):
            raise ValueError(f"Sequence length {seq_len} exceeds max_len")

        return x + self.pe[:, :seq_len, :]


# =========================================================
# 2. STABLE TIMESTEP EMBEDDING
# =========================================================
class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()

        self.frequency_embedding_size = frequency_embedding_size

        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

        # Stable init
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, t):
        t = t.float()

        half = self.frequency_embedding_size // 2
        device = t.device

        freqs = torch.exp(
            torch.arange(half, device=device)
            * (-math.log(10000.0) / (half - 1))
        )

        args = t[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

        return self.mlp(emb)


# =========================================================
# 3. COARSE STYLE ENCODER
# =========================================================
class CoarseStyleEncoder(nn.Module):
    def __init__(self, num_sensors, hidden_dim):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv1d(num_sensors, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),

            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.SiLU(),

            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )

    def forward(self, x):
        return self.encoder(x.transpose(1, 2))


# =========================================================
# 4. STABLE JOINT DiT BLOCK
# =========================================================
class DiTJointBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super().__init__()

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            batch_first=True
        )

        self.norm2 = nn.LayerNorm(hidden_dim)

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

        # AdaLN modulation
        self.adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 6 * hidden_dim)
        )

        # =====================================================
        # CRITICAL FIX: zero init for diffusion stability
        # =====================================================
        nn.init.zeros_(self.adaLN[-1].weight)
        nn.init.zeros_(self.adaLN[-1].bias)

    def forward(self, x, t_emb):
        B = x.size(0)

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN(t_emb).chunk(6, dim=1)

        # -------------------------
        # Attention (stabilized)
        # -------------------------
        x_norm = self.norm1(x)

        # clamp prevents explosion
        #scale_msa = torch.tanh(scale_msa)
        #shift_msa = torch.tanh(shift_msa)

        x_norm = x_norm * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)

        attn_out, _ = self.attn(x_norm, x_norm, x_norm)

        gate_msa = torch.sigmoid(gate_msa)

        x = x + gate_msa.unsqueeze(1) * attn_out

        # -------------------------
        # MLP (stabilized)
        # -------------------------
        x_norm2 = self.norm2(x)

        #scale_mlp = torch.tanh(scale_mlp)
        #shift_mlp = torch.tanh(shift_mlp)

        x_norm2 = x_norm2 * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)

        mlp_out = self.mlp(x_norm2)

        gate_mlp = torch.sigmoid(gate_mlp)

        x = x + gate_mlp.unsqueeze(1) * mlp_out

        return x


# =========================================================
# 5. FULL DiT MODEL (FIXED)
# =========================================================
class TimeSeriesDiT(nn.Module):
    def __init__(
        self,
        target_seq_len=96,
        history_seq_len=32,
        num_sensors=51,
        hidden_dim=256,
        num_heads=8,
        num_layers=4,
    ):
        super().__init__()

        self.target_seq_len = target_seq_len
        self.history_seq_len = history_seq_len
        self.hidden_dim = hidden_dim

        # -------------------------
        # Fine stream
        # -------------------------
        self.x_proj = nn.Linear(num_sensors, hidden_dim)
        self.x_pos = PositionalEncoding(hidden_dim, max_len=target_seq_len)

        # -------------------------
        # Coarse stream
        # -------------------------
        #self.coarse_encoder = SpectralSignatureEncoder(num_sensors, hidden_dim, target_seq_len)

        # -------------------------
        # timestep
        # -------------------------
        self.t_embedder = TimestepEmbedder(hidden_dim)
		#self.condition_fusion = nn.Sequential(
        #    nn.Linear(hidden_dim, hidden_dim),
        #    nn.SiLU(),
        #    nn.Linear(hidden_dim, hidden_dim)
        #)
        # -------------------------
        # blocks
        # -------------------------
        self.blocks = nn.ModuleList([
            DiTJointBlock(hidden_dim, num_heads)
            for _ in range(num_layers)
        ])

        # -------------------------
        # output
        # -------------------------
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, num_sensors)

    # =========================================================
    def forward(self, x_curr, t, drop_prob=0.0):

        B = x_curr.size(0)

        # -------------------------
        # fine
        # -------------------------
        x = self.x_proj(x_curr)
        x = self.x_pos(x)

        # -------------------------
        # coarse + timestep fusion (FIXED scaling)
        # -------------------------
		style = self.coarse_encoder(x_curr)
        #t_emb_raw = self.t_embedder(t)
        t_emb = self.t_embedder(t) + style

        # 2. FUSE THEM through the layer instead of simple addition
        # This provides a much richer 'c' vector for the AdaLN
        #t_emb = self.condition_fusion(t_emb_raw + style)       

        # -------------------------
        # transformer
        # -------------------------
        for blk in self.blocks:
            x = blk(x, t_emb)

        # -------------------------
        # output
        # -------------------------
        x = self.final_norm(x)
        return self.head(x)
