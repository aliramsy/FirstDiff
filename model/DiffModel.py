import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, GCNConv


class AttentionBlock(nn.Module):
    def __init__(self, channels, size):
        super(AttentionBlock, self).__init__()
        self.channels = channels
        self.size = size
        self.ln = nn.LayerNorm([channels])
        self.sp_conv = GATv2Conv(size, size)
        # self.gcn_conv = GCNConv(size, size)
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels)
        )
        self.default_edge_index = None
        self.B = None
        self.expanded_edge_index = None
    
    def forward_GAT(self, x, edge_index=None):
        x = x.permute(0, 2, 1) # B, L, D -> B, D, L
        B, D, L = x.shape[0], x.shape[1], x.shape[2]
        if self.default_edge_index is None:
            self.default_edge_index = torch.full((D, D), 1).nonzero().t().contiguous().to(x.get_device())
        edge_index = self.default_edge_index
        if self.expanded_edge_index is None or self.B != B:
            self.expanded_edge_index = (torch.arange(0, B).repeat(2).reshape(2, B, 1).expand((2, B, edge_index.shape[1])).reshape(2, -1) * D).to(x.get_device())
            self.B = B
        expanded_edge_index = self.expanded_edge_index
        x = x.reshape(B * D, L)
        output = self.sp_conv(x, edge_index.repeat((1, B)) + expanded_edge_index)
        return output.reshape(B, D, L).permute(0, 2, 1) # B, D, L -> B, L, D

    def forward_GCN(self, x, edge_index=None):
        x = x.permute(0, 2, 1) # B, L, D -> B, D, L
        D = x.shape[1]
        if self.default_edge_index is None:
            self.default_edge_index = torch.full((D, D), 1).nonzero().t().contiguous().to(x.get_device())
        edge_index = self.default_edge_index
        output = self.gcn_conv(x, edge_index)
        return output.permute(0, 2, 1) # B, D, L -> B, L, D

    def forward(self, x):
        x = x.permute(0, 2, 1) #B, D, L -> B, L, D
        x_ln = self.ln(x)

        attention_value = x_ln 
        attention_value = self.forward_GAT(attention_value) + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.permute(0, 2, 1) #B, L, D -> B, D, L


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv1d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            nn.Conv1d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels)
        )

    def forward(self, x):
        if self.residual:
            return F.gelu(x + self.double_conv(x))
        else:
            return self.double_conv(x) 


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=128):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool1d(2),
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, t):
        x = self.maxpool_conv(x)
        emb = self.emb_layer(t)[:, :, None].repeat(1, 1, x.shape[-1])
        return x + emb


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=128):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="linear", align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels, in_channels // 2),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, skip_x, t):
        x = self.up(x)
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        emb = self.emb_layer(t)[:, :, None].repeat(1, 1, x.shape[-1])
        return x + emb


class DiffModel(nn.Module):
    def __init__(self, c_in=3, c_out=3, time_dim=128, device="cuda:1"):
        super().__init__()
        self.device = device
        self.time_dim = time_dim
        self.inc = DoubleConv(c_in, 128)
        self.down1 = Down(128, 64)
        self.sa1 = AttentionBlock(64, 48)
        self.down2 = Down(64, 128)
        self.sa2 = AttentionBlock(128, 24)
        self.down3 = Down(128, 128)
        self.sa3 = AttentionBlock(128, 12)

        self.bot1 = DoubleConv(128, 256)
        self.bot2 = DoubleConv(256, 256)
        self.bot3 = DoubleConv(256, 128)

        self.up1 = Up(256, 64)
        self.sa4 = AttentionBlock(64, 24)
        self.up2 = Up(128, 32)
        self.sa5 = AttentionBlock(32, 48)
        self.up3 = Up(160, 64)
        self.sa6 = AttentionBlock(64, 96)
        self.outc = nn.Conv1d(64, c_out, kernel_size=1)

        self.B = None
        self.expanded_edge_index = None

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, x, t, condition, edge_index=None):
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)
        x = x.permute(0, 2, 1) # B, L, D -> B, D, L
        x1 = self.inc(x)
        x2 = self.down1(x1, t)
        x2 = self.sa1(x2)
        x3 = self.down2(x2, t)
        x3 = self.sa2(x3)
        x4 = self.down3(x3, t)
        x4 = self.sa3(x4)
        
        x4 = self.bot1(x4)       
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)
        
        x = self.up1(x4, x3, t)
        x = self.sa4(x)
        x = self.up2(x, x2, t)
        x = self.sa5(x)
        x = self.up3(x, x1, t)
        x = self.sa6(x)
        output = self.outc(x)
        return output.permute(0, 2, 1)