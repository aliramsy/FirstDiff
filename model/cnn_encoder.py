import torch
import torch.nn as nn

class HistoryEncoderCNN(nn.Module):
    """
    Encodes a history window of shape (B, 1024, 51) down to (B, 32, D)
    using 5 layers of strided 1D convolutions.
    """
    def __init__(self, in_channels=51, d_model=256):
        super().__init__()
        self.in_channels = in_channels
        self.d_model = d_model
        
        # We use a downsampling factor of 2 at each step: 1024 -> 512 -> 256 -> 128 -> 64 -> 32
        self.encoder = nn.Sequential(
            # Layer 1: H_1 \in \mathbb{R}^{512 \times 64}
            nn.Conv1d(in_channels=in_channels, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            
            # Layer 2: H_2 \in \mathbb{R}^{256 \times 128}
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            
            # Layer 3: H_3 \in \mathbb{R}^{128 \times 256}
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            
            # Layer 4: H_4 \in \mathbb{R}^{64 \times D}
            nn.Conv1d(in_channels=256, out_channels=self.d_model, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            
            # Layer 5: H_5 \in \mathbb{R}^{32 \times D}
            nn.Conv1d(in_channels=self.d_model, out_channels=self.d_model, kernel_size=3, stride=2, padding=1),
            nn.GELU()
        )

    def forward(self, x):
        # Input x is expected to be: (B, 1024, 51)
        # Conv1d expects: (Batch, Channels, Sequence_Length)
        x = x.permute(0, 2, 1) # Now: (B, 51, 1024)
        
        # Pass through the CNN to downsample
        x = self.encoder(x) # Now: (B, D, 32)
        
        # Permute back to: (Batch, Sequence_Length, Hidden_Dim) for the Transformer
        x = x.permute(0, 2, 1) # Now: (B, 32, D)
        
        return x
    
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# --- Quick Test ---
if __name__ == "__main__":
    # Simulate a batch of history data: Batch Size = 8, Seq Len = 1024, Features = 51
    dummy_history = torch.randn(8, 1024, 51)
    
    # Initialize the encoder (assuming your DiT uses a hidden dimension D=256)
    encoder = HistoryEncoderCNN(in_channels=51, d_model=256)
    
    # Forward pass
    output = encoder(dummy_history)
    
    print(f"Input shape: {dummy_history.shape}") 
    print(f"Output shape: {output.shape}") 
    print(f"Total Trainable Parameters: {count_parameters(encoder):,}")
    # Expected output shape: torch.Size([8, 32, 256])
