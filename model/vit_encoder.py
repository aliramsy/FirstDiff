import torch
import torch.nn as nn

class HistoryEncoderViT(nn.Module):
    """
    Encodes a history window of shape (B, 1024, 51) down to (B, 32, D)
    using ViT-style patching and linear projection.
    """
    def __init__(self, seq_len=1024, in_channels=51, num_patches=32, d_model=256):
        super().__init__()
        self.seq_len = seq_len
        self.in_channels = in_channels
        self.num_patches = num_patches
        self.d_model = d_model
        
        # 1. Calculate patch size
        assert seq_len % num_patches == 0, "Sequence length must be divisible by number of patches"
        self.patch_size = seq_len // num_patches  # e.g., 1024 / 32 = 32
        
        # 2. Calculate flattened dimension: 32 * 51 = 1632
        self.patch_dim = self.patch_size * in_channels
        
        # 3. Linear Projection: W \in \mathbb{R}^{1632 \times D}, b \in \mathbb{R}^{D}
        self.proj = nn.Linear(self.patch_dim, d_model)
        
        # 4. Learnable Positional Encodings
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02) # Standard ViT initialization

    def forward(self, x):
        # Input x shape: (B, 1024, 51)
        B, L, C = x.shape
        
        # Reshape into patches: (B, num_patches, patch_size, C) -> (B, 32, 32, 51)
        x = x.view(B, self.num_patches, self.patch_size, C)
        
        # Flatten each patch: (B, num_patches, patch_size * C) -> (B, 32, 1632)
        x = x.view(B, self.num_patches, -1)
        
        # Linear Projection: (B, 32, D)
        x = self.proj(x)
        
        # Add Positional Encoding
        x = x + self.pos_embed
        
        return x

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# --- Quick Test & Parameter Print ---
if __name__ == "__main__":
    # Simulate a batch of history data: Batch Size = 8, Seq Len = 1024, Features = 51
    dummy_history = torch.randn(8, 1024, 51)
    
    # Initialize the ViT-style encoder
    encoder_vit = HistoryEncoderViT(seq_len=1024, in_channels=51, num_patches=32, d_model=256)
    
    # Forward pass
    output = encoder_vit(dummy_history)
    
    print(f"Input shape: {dummy_history.shape}") 
    print(f"Output shape: {output.shape}") 
    print(f"Total Trainable Parameters: {count_parameters(encoder_vit):,}")
