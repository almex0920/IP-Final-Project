import torch
import torch.nn as nn

# Double Convolution block for UNet
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        return self.double_conv(x)

# UNet Architecture
class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Encoder
        self.enc1 = DoubleConv(3, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)
        self.enc4 = DoubleConv(256, 512)
        
        # Bottleneck
        self.bottleneck = DoubleConv(512, 1024)
        
        # Decoder
        self.dec4 = DoubleConv(1024 + 512, 512)
        self.dec3 = DoubleConv(512 + 256, 256)
        self.dec2 = DoubleConv(256 + 128, 128)
        self.dec1 = DoubleConv(128 + 64, 64)
        
        # Final layer
        self.final_conv = nn.Conv2d(64, 1, kernel_size=1)
        
        # Pooling and upsampling
        self.pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        
        # Bottleneck
        b = self.bottleneck(self.pool(e4))
        
        # Decoder
        d4 = self.dec4(torch.cat([self.upsample(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.upsample(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.upsample(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.upsample(d2), e1], dim=1))
        
        # Final layer
        out = self.final_conv(d1)
        return torch.sigmoid(out)
    
class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        """
        Attention Gate for focusing on relevant features
        F_g: Feature dimension of gating signal
        F_l: Feature dimension of learning signal
        F_int: Intermediate feature dimension
        """
        super(AttentionBlock, self).__init__()
        
        # Weight layers for gating signal
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        # Weight layers for learning signal
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        # Combine and activate
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, g, x):
        """
        g: Gating signal
        x: Learning signal
        """
        # Gating signal downsampling
        g1 = self.W_g(g)
        
        # Learning signal downsampling
        x1 = self.W_x(x)
        
        # Combine signals
        psi = self.relu(g1 + x1)
        
        # Channel-wise attention
        psi = self.psi(psi)
        
        # Reweight input
        return x * psi

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels)
        )
        
        # Residual connection
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1),
                nn.BatchNorm2d(out_channels)
            )
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.conv2(out)
        
        # Add residual connection
        out += self.shortcut(residual)
        out = self.relu(out)
        
        return out

class WaterSegmentationUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(WaterSegmentationUNet, self).__init__()
        
        # Encoder (Downsampling)
        self.encoder1 = nn.Sequential(
            ResidualBlock(in_channels, 64),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.encoder2 = nn.Sequential(
            ResidualBlock(64, 128),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.encoder3 = nn.Sequential(
            ResidualBlock(128, 256),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.encoder4 = nn.Sequential(
            ResidualBlock(256, 512),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Bridge
        self.bridge = ResidualBlock(512, 1024)
        
        # Decoder (Upsampling)
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        
        # Attention and decoder block for level 4
        self.attention4 = AttentionBlock(512, 512, 256)
        self.decoder4 = ResidualBlock(1024, 512)
        
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        
        # Attention and decoder block for level 3
        self.attention3 = AttentionBlock(256, 256, 128)
        self.decoder3 = ResidualBlock(512, 256)
        
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        
        # Attention and decoder block for level 2
        self.attention2 = AttentionBlock(128, 128, 64)
        self.decoder2 = ResidualBlock(256, 128)
        
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        
        # Attention and decoder block for level 1
        self.attention1 = AttentionBlock(64, 64, 32)
        self.decoder1 = ResidualBlock(128, 64)
        
        # Final Convolution
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)
        
        # Sigmoid activation for binary segmentation
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Encoder
        enc1 = self.encoder1[0](x)
        enc1_pool = self.encoder1[1](enc1)
        
        enc2 = self.encoder2[0](enc1_pool)
        enc2_pool = self.encoder2[1](enc2)
        
        enc3 = self.encoder3[0](enc2_pool)
        enc3_pool = self.encoder3[1](enc3)
        
        enc4 = self.encoder4[0](enc3_pool)
        enc4_pool = self.encoder4[1](enc4)
        
        # Bridge
        bridge = self.bridge(enc4_pool)
        
        # Decoder
        dec4 = self.upconv4(bridge)
        
        # Apply attention and concatenate
        dec4_att = self.attention4(dec4, enc4)
        dec4 = torch.cat([dec4_att, enc4], dim=1)
        dec4 = self.decoder4(dec4)
        
        dec3 = self.upconv3(dec4)
        
        # Apply attention and concatenate
        dec3_att = self.attention3(dec3, enc3)
        dec3 = torch.cat([dec3_att, enc3], dim=1)
        dec3 = self.decoder3(dec3)
        
        dec2 = self.upconv2(dec3)
        
        # Apply attention and concatenate
        dec2_att = self.attention2(dec2, enc2)
        dec2 = torch.cat([dec2_att, enc2], dim=1)
        dec2 = self.decoder2(dec2)
        
        dec1 = self.upconv1(dec2)
        
        # Apply attention and concatenate
        dec1_att = self.attention1(dec1, enc1)
        dec1 = torch.cat([dec1_att, enc1], dim=1)
        dec1 = self.decoder1(dec1)
        
        # Final Convolution and Sigmoid
        output = self.final_conv(dec1)
        output = self.sigmoid(output)
        
        return output