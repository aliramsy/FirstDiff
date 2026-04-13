from torch import nn

class AutoEncoder(nn.Module):
    """
    AutoEncoder model used to reconstruct time series and calculate
    reconstruction loss that is utilized to mask potential anomalies
    """
    def __init__(self, window_size=100, channels=38, h_dim=256, h_dim_2=64, emb_dim=128, device='cpu'):
        super(AutoEncoder, self).__init__()
        self.window_size = window_size
        self.channels = channels
        self.emb_dim = emb_dim
        self.device = device
        kernel_size = 5
        padding = 2
        self.encoder = nn.Sequential(
            nn.Conv1d(channels, h_dim_2, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(h_dim_2, h_dim, kernel_size=kernel_size, padding=padding),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Conv1d(h_dim, h_dim_2, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.ConvTranspose1d(h_dim_2, channels, kernel_size=kernel_size, padding=padding, stride=2, output_padding=1)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded.permute(0, 2, 1)