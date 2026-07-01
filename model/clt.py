import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
import math

# =========================================================
# 1. ENHANCED SPECTRAL ENCODER (Residual Architecture)
# =========================================================
class SpectralSignatureEncoder(nn.Module):
    def __init__(self, num_sensors, hidden_dim, window_size):
        super().__init__()
        self.freq_bins = (window_size // 2) + 1
        self.input_dim = num_sensors * self.freq_bins
        self.hidden_dim = hidden_dim

        # Initial Projection
        self.input_proj = nn.Linear(self.input_dim, hidden_dim)
        
        # Residual Blocks for deeper feature extraction
        self.res_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim)
            ) for _ in range(3)
        ])
        
        self.final_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        # x shape: (B, T, C)
        # 1. Compute rFFT and Power Spectral Density
        fft_res = torch.fft.rfft(x, dim=1).transpose(1, 2)
        psd = torch.abs(fft_res)**2
        
        # 2. Flatten and Log-scale
        psd_flat = psd.reshape(psd.shape[0], -1)
        x = torch.log(psd_flat + 1e-8)
        
        # 3. Pass through Residual MLP
        x = self.input_proj(x)
        for block in self.res_blocks:
            x = x + block(x) # Residual skip connection
            
        return self.final_norm(x)

# =========================================================
# 2. CONTRASTIVE PRE-TRAINING LOGIC
# =========================================================
def pretrain_spectral_encoder(encoder, train_loader, device, epochs=25):
    """
    Self-Supervised Contrastive Pre-training (SimCLR style).
    Forces the encoder to learn invariant physical features.
    """
    # Projector head (Only used for training, discarded after)
    projector = nn.Sequential(
        nn.Linear(encoder.hidden_dim, encoder.hidden_dim),
        nn.SiLU(),
        nn.Linear(encoder.hidden_dim, 128) 
    ).to(device)

    encoder.to(device)
    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(projector.parameters()), 
        lr=5e-4, 
        weight_decay=1e-2
    )

    # InfoNCE Loss (Contrastive Loss)
    def info_nce_loss(z1, z2, temperature=0.1):
        # Normalize to unit hypersphere
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        
        # Combined representations
        features = torch.cat([z1, z2], dim=0)
        logits = torch.matmul(features, features.T) / temperature
        
        # Mask out self-similarities
        batch_size = z1.shape[0]
        mask = torch.eye(2 * batch_size, device=device).bool()
        logits = logits.masked_fill(mask, -1e9)
        
        # Targets: z1 should match z2 and vice versa
        labels = torch.arange(batch_size, device=device)
        labels = torch.cat([labels + batch_size, labels], dim=0)
        
        return F.cross_entropy(logits, labels)

    print(f"Starting Phase 1: Contrastive Spectral Learning ({epochs} epochs)...")
    encoder.train()
    
    for epoch in range(epochs):
        total_loss = 0
        for data in train_loader:
            x = data[0].to(device) if isinstance(data, list) else data.to(device)
            
            # --- AUGMENTATION: Create two "Views" of the same data ---
            # View 1: Random Sensor Masking (Dropout 20% sensors)
            mask1 = (torch.rand_like(x) > 0.2).float()
            x1 = x * mask1
            
            # View 2: Spectral Jitter / Scaling
            noise = torch.randn_like(x) * 0.05
            x2 = (x + noise) * (0.8 + torch.rand(1).item() * 0.4)

            # --- Forward Pass ---
            h1 = encoder(x1)
            h2 = encoder(x2)
            
            z1 = projector(h1)
            z2 = projector(h2)
            
            loss = info_nce_loss(z1, z2)
            
            # --- Backward ---
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"Contrastive Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(train_loader):.6f}")

    # Save and Cleanup
    torch.save(encoder.state_dict(), "spectral_encoder_contrastive.pth")
    print("Phase 1 Complete. Strong Contrastive weights saved.")
    return encoder
