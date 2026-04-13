import torch
import torch.nn as nn

class HistoryEncoderPerceiver(nn.Module):
    """
    Compresses a history window of shape (B, 1024, 51) down to (B, 32, D)
    using Cross-Attention with learnable latent queries.
    """
    def __init__(self, seq_len=1024, in_channels=51, num_latents=32, d_model=256, nhead=8):
        super().__init__()
        self.d_model = d_model
        self.num_latents = num_latents
        
        # 1. Latent Queries ($Q \in \mathbb{R}^{32 \times D}$)
        # These are the 32 "detectives" that will summarize the sequence.
        self.latents = nn.Parameter(torch.randn(num_latents, d_model))
        nn.init.trunc_normal_(self.latents, std=0.02)
        
        # 2. Encode Input: Map 51 sensors to D dimensions
        self.input_proj = nn.Linear(in_channels, d_model)
        
        # Learnable Positional Encodings for the 1024 steps
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        # 3. Cross-Attention Layer
        # batch_first=True expects tensors of shape (Batch, Seq_Len, Features)
        self.cross_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, batch_first=True)
        
        # Optional but highly recommended: LayerNorm for stability
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)

    def forward(self, x):
        # Input x shape: (B, 1024, 51)
        B = x.shape[0]
        
        # Prepare Keys and Values ($K, V \in \mathbb{R}^{B \times 1024 \times D}$)
        kv = self.input_proj(x) + self.pos_embed
        kv = self.norm_kv(kv)
        
        # Prepare Queries ($Q \in \mathbb{R}^{B \times 32 \times D}$)
        # Expand the learnable latents to match the batch size
        q = self.latents.unsqueeze(0).expand(B, -1, -1)
        q = self.norm_q(q)
        
        # Cross-Attention: Q attends to K, V
        # Output shape will be strictly (B, 32, D)
        attn_output, _ = self.cross_attn(query=q, key=kv, value=kv)
        
        return attn_output

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# --- Quick Test & Parameter Print ---
if __name__ == "__main__":
    # Simulate a batch of history data: Batch Size = 8, Seq Len = 1024, Features = 51
    dummy_history = torch.randn(8, 1024, 51)
    
    # Initialize the Perceiver-style encoder
    encoder_perceiver = HistoryEncoderPerceiver(seq_len=1024, in_channels=51, num_latents=32, d_model=256, nhead=8)
    
    # Forward pass
    output = encoder_perceiver(dummy_history)
    
    print(f"Input shape: {dummy_history.shape}") 
    print(f"Output shape: {output.shape}") 
    print(f"Total Trainable Parameters: {count_parameters(encoder_perceiver):,}")
