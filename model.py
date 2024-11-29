import torch
import torch.nn as nn


class BandAttention(nn.Module):

    def __init__(self, num_bands):
        super(BandAttention, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=num_bands, out_channels=num_bands, kernel_size=(3, 3))
        self.max_pool_2d = nn.MaxPool2d(kernel_size=(4, 4))
        self.fc1 = nn.Linear(in_features=num_bands, out_features=32)
        self.fc2 = nn.Linear(in_features=32, out_features=num_bands)
        self.batch_norm = nn.BatchNorm1d(num_features=num_bands)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        band_weights_4d = self.conv2d(x)
        band_weights_4d = self.max_pool_2d(band_weights_4d)
        band_weights_3d = torch.squeeze(band_weights_4d, dim=3)
        band_weights_2d = torch.squeeze(band_weights_3d, dim=2)  # [batch-size, num-band_weights,sl0.1,sl0.1].
        band_weights_2d = self.fc1(band_weights_2d)
        band_weights_2d = self.sigmoid(band_weights_2d)
        band_weights_2d = self.fc2(band_weights_2d)
        band_weights_2d = self.batch_norm(band_weights_2d)
        w1 = self.sigmoid(band_weights_2d)
        w2 = w1 * 0.5 + 0.5  # 1_w2 = self.sigmoid(1_w1)
        band_weights_3d = torch.unsqueeze(w2, dim=2)
        band_weights_4d = torch.unsqueeze(band_weights_3d, dim=3)
        return torch.mul(x, band_weights_4d), w1, w2


class SpatialSpectrumAttention(nn.Module):

    def __init__(self, num_channel):
        super(SpatialSpectrumAttention, self).__init__()
        self.max_pooling_3d = nn.MaxPool3d(kernel_size=(num_channel, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1))
        self.avg_pooling_3d = nn.AvgPool3d(kernel_size=(num_channel, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1))
        self.conv_2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.batch_norm = nn.BatchNorm2d(num_features=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x1 = self.max_pooling_3d(x)
        x2 = self.avg_pooling_3d(x)
        weights_2channel = torch.cat((x1, x2), dim=1)
        weights_1channel = self.conv_2d(weights_2channel)
        weights_1channel = self.batch_norm(weights_1channel)
        weights_1channel = self.relu(weights_1channel)
        return torch.mul(x, weights_1channel), weights_1channel


class MultiScaleReconstruction(nn.Module):

    def __init__(self):
        super(MultiScaleReconstruction, self).__init__()

        self.conv3d_3x3 = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.PReLU()
        )

        self.conv3d_5x5 = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=64, kernel_size=(5, 5, 5), stride=(1, 1, 1), padding=(2, 2, 2)),
            nn.BatchNorm3d(64),
            nn.PReLU()
        )

        self.pool3d = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))

        self.conv3d_3x3_2 = nn.Sequential(
            nn.Conv3d(in_channels=128, out_channels=64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.PReLU()
        )

        self.de_conv3d_1 = nn.Sequential(
            nn.ConvTranspose3d(
                in_channels=128, out_channels=64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.PReLU()
        )
        self.de_conv3d_2 = nn.Sequential(
            nn.ConvTranspose3d(
                in_channels=64, out_channels=1, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(1)
        )

    def forward(self, x):
        x = torch.unsqueeze(x, dim=1)  # (batch_size, channel=sl0.1, depth=band_weights, h, w) add channel on dim1
        x_3x3 = self.conv3d_3x3(x)
        x_5x5 = self.conv3d_5x5(x)
        x = torch.cat([x_3x3, x_5x5], dim=1)
        x = self.pool3d(x)
        x = self.de_conv3d_1(x)
        x = self.de_conv3d_2(x)
        x = torch.squeeze(x, dim=1)  # Remove channel dim.
        return x


class SSANetBS(nn.Module):

    def __init__(self, num_bands, patch_size_=(7, 7)):
        super(SSANetBS, self).__init__()
        self.ss_att_hb = SpatialSpectrumAttention(num_channel=patch_size_[0])
        self.ss_att_wb = SpatialSpectrumAttention(num_channel=patch_size_[1])
        self.bs_att = BandAttention(num_bands=num_bands)
        self.ms_rec = MultiScaleReconstruction()

    def forward(self, x):
        x_b, w1, w2 = self.bs_att(x)  # Attention on dim: band.
        x_hb = torch.permute(x_b, (0, 2, 3, 1))  # Attention on dim: height - band.
        x_hb, att_hb = self.ss_att_hb(x_hb)
        x_hb = torch.permute(x_hb, (0, 3, 1, 2))
        x_wb = torch.permute(x_b, (0, 3, 2, 1))  # Attention on dim: width - band.
        x_wb, att_wb = self.ss_att_wb(x_wb)
        x_wb = torch.permute(x_wb, (0, 3, 2, 1))
        x = (x_hb + x_wb) / 2
        x_reconstructed = self.ms_rec(x)  # Multi-scale reconstruction.
        return x_reconstructed, w1, w2
