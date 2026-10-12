import torch
import torch.nn as nn

class SpatialAutoencoder(nn.Module):
    """
    Convolutional Autoencoder baseline for One-Class Video Anomaly Detection.
    Learns spatial-temporal normality patterns via reconstruction loss.
    """
    def __init__(self, in_channels=1):
        super(SpatialAutoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),  # (B, 32, H/2, W/2)
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),          # (B, 64, H/4, W/4)
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),         # (B, 128, H/8, W/8)
            nn.BatchNorm2d(128),
            nn.ReLU(True)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, in_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()  # Normalize outputs to [0, 1]
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        return reconstruction, latent

    def compute_reconstruction_error(self, x):
        """
        Calculates per-frame Mean Squared Error (MSE) reconstruction loss.
        High error indicates statistical deviation from learned normality.
        """
        reconstruction, _ = self.forward(x)
        mse_loss = nn.functional.mse_loss(reconstruction, x, reduction='none')
        # Average over C, H, W dimensions per frame
        frame_errors = torch.mean(mse_loss, dim=[1, 2, 3])
        return frame_errors, reconstruction
