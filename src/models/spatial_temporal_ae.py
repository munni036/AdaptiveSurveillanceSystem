import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialTemporalAutoencoder(nn.Module):
    """
    Convolutional Autoencoder for Spatial-Temporal Normality Modeling.
    Trained exclusively on normal baseline footage to minimize reconstruction loss.
    Statistical deviations (anomalies) produce high reconstruction MSE loss.
    """
    def __init__(self, in_channels=1):
        super(SpatialTemporalAutoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),  # (B, 32, 64, 64)
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),          # (B, 64, 32, 32)
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),         # (B, 128, 16, 16)
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1), # (B, 64, 32, 32)
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # (B, 32, 64, 64)
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(32, in_channels, kernel_size=3, stride=2, padding=1, output_padding=1), # (B, 1, 128, 128)
            nn.Sigmoid()
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        return reconstruction, latent

    def compute_frame_error(self, x):
        """
        Calculates per-frame Mean Squared Error (MSE) reconstruction loss score.
        """
        reconstruction, _ = self.forward(x)
        mse_per_pixel = (reconstruction - x) ** 2
        # Average MSE error per frame
        frame_error = torch.mean(mse_per_pixel, dim=[1, 2, 3])
        return frame_error, reconstruction
