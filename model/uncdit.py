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


class StandardDiTBlock(nn.Module):
    """
    Standard DiT block for unconditional generation.
    Modulates LayerNorm with the timestep embedding.
    """
    def __init__(self, hidden_dim, num_heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        # Modulation layers for timestep t
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 6 * hidden_dim, bias=True)
        )

    def forward(self, x, t_emb):
        # Shift, scale, and gates for AdaLN from timestep embedding
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(t_emb).chunk(6, dim=1)
        
        # Self-Attention
        x_norm1 = self.norm1(x) * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        attn_out, _ = self.attn(x_norm1, x_norm1, x_norm1)
        x = x + gate_msa.unsqueeze(1) * attn_out
        
        # MLP
        x_norm2 = self.norm2(x) * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        mlp_out = self.mlp(x_norm2)
        x = x + gate_mlp.unsqueeze(1) * mlp_out
        
        return x

class UnconditionalTimeSeriesDiT(nn.Module):
    """
    Pure Unconditional DiT for baseline testing. No history conditioning.
    """
    def __init__(self, target_seq_len=96, num_sensors=51, hidden_dim=256, num_heads=8, num_layers=4):
        super().__init__()
        self.target_seq_len = target_seq_len
        self.hidden_dim = hidden_dim
        
        # Project noisy target window
        self.x_proj = nn.Linear(num_sensors, hidden_dim)
        self.x_pos_embed = PositionalEncoding(hidden_dim, max_len=target_seq_len)
        
        # Timestep embedder
        self.t_embedder = TimestepEmbedder(hidden_dim)
        
        # Standard DiT Blocks
        self.blocks = nn.ModuleList([
            StandardDiTBlock(hidden_dim, num_heads) for _ in range(num_layers)
        ])
        
        # Final Layer Norm and Output Head
        self.final_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 2 * hidden_dim, bias=True)
        )
        self.head = nn.Linear(hidden_dim, num_sensors)

    def forward(self, x_curr, t):
        r"""
        x_curr: Noisy target window X (B, 96, 51)
        t: Diffusion timesteps (B)
        """
        # Prepare Target Window X
        x = self.x_proj(x_curr) # (B, 96, D)
        x = self.x_pos_embed(x)
        
        # Prepare Timestep
        t_emb = self.t_embedder(t) # (B, D)
            
        # Pass through Blocks
        for block in self.blocks:
            x = block(x, t_emb)
            
        # Predict Noise
        shift_final, scale_final = self.adaLN_modulation(t_emb).chunk(2, dim=1)
        x = self.final_norm(x) * (1 + scale_final.unsqueeze(1)) + shift_final.unsqueeze(1)
        out = self.head(x) # (B, 96, 51)
        
        return out

#if __name__ == "__main__":
#  
#    def count_parameters(model):
#      """Returns total, trainable, and non-trainable parameter counts."""
#      total = sum(p.numel() for p in model.parameters())
#      trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
#      non_trainable = total - trainable
#      return total, trainable, non_trainable
#
#
#    def print_layer_breakdown(model):
#      """Prints learnable parameter counts for top-level modules and their sum."""
#      print(f"{'Module Name':<20} | {'Learnable Params':<16}")
#      print("-" * 41)
#
#      total_learnable = 0
#      for name, module in model.named_children():
#        # Sum only parameters that require gradients
#        params = sum(p.numel() for p in module.parameters() if p.requires_grad)
#        total_learnable += params
#        print(f"{name:<20} | {params:>16,}")
#
#      print("-" * 41)
#      print(f"{'TOTAL SUM':<20} | {total_learnable:>16,}")
#
#    model = UnconditionalTimeSeriesDiT(
#        target_seq_len=96,
#        num_sensors=51,
#        hidden_dim=256,
#        num_heads=8,
#        num_layers=6,
#    )   
#    # Calculate parameter counts
#    total, trainable, non_trainable = count_parameters(model)   
#    print("=" * 45)
#    print("         MODEL PARAMETER SUMMARY         ")
#    print("=" * 45)
#    print(f"Total Parameters:         {total:>12,}")
#    print(f"Trainable Parameters:     {trainable:>12,}")
#    print(f"Non-Trainable Parameters: {non_trainable:>12,}")
#    print("=" * 45)
#    print("\nLayer-by-Layer Breakdown:")
#    print_layer_breakdown(model)    
#    # Verify execution with a dummy input batch
#    batch_size = 4
#    dummy_x = torch.randn(batch_size, 96, 51)  # (Batch, Seq_Len, Sensors)
#    dummy_t = torch.randint(0, 1000, (batch_size,))  # Random timesteps 
#    output = model(dummy_x, dummy_t)
#    print("\n" + "=" * 45)
#    print(f"Forward Pass Output Shape: {output.shape}")