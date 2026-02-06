import sys
sys.path.append("../")
import torch
import torch.nn as nn
from eclayr.vr.ripseclayr import RipsEcc


class AutoEncoder(nn.Module):
    def __init__(self, in_dim=101, latent_dim=2, hidden_dim=16):
        """
        Args:
            in_dim (int, optional): Dimension of input point cloud. Defaults to 101.
            latent_dim (int, optional): Dimension of latent space. Defaults to 2.
        """
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, in_dim)
        )
        self.loss = nn.MSELoss()

    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, x):
        return self.decoder(x)

    def forward(self, x, *args, **kwargs):
        """_summary_

        Args:
            x (torch.Tensor): Point cloud. Shape: [B, D]

        Returns:
            tuple: (reconstructed point cloud, latent embedding, loss)
        """
        z = self.encoder(x)
        x_recon = self.decoder(z)
        loss = self.loss(x, x_recon)
        return x_recon, z, loss, (loss.item(), 0)


class TopoAutoEncoder(nn.Module):
    def __init__(self, in_dim=101, latent_dim=2, hidden_dim=16,     # AE params
                 max_edge_length=2, max_dim=1, steps=32, beta=0.01, # ECC params
                 lam=0.1):
        """_summary_

        Args:
            in_dim (int, optional): Dimension of input point cloud. Defaults to 101.
            latent_dim (int, optional): Dimension of latent space. Defaults to 2.
            lam (float, optional): Controls the weight of topological loss. Defaults to .1.
        """
        super().__init__()
        self.lam = lam
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, in_dim)
        )
        # self.eclay_1 = ECLay(interval, steps, hidden_features, scale, type="Rips")
        # self.eclay_2 = ECLay(interval, steps, hidden_features, scale, type="Rips")
        self.eclay_1 = RipsEcc(max_edge_length, max_dim, steps, beta)
        self.eclay_2 = RipsEcc(max_edge_length, max_dim, steps, beta)
        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()

    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, x):
        return self.decoder(x)

    def forward(self, x):
        """_summary_

        Args:
            x (torch.Tensor): Point cloud. Shape: [B, D]

        Returns:
            torch.Tensor: (reconstructed point cloud, latent embedding, loss)
        """
        # autoencoder
        z = self.encoder(x)
        x_recon = self.decoder(z)
        recon_loss = self.mse_loss(x, x_recon)

        # topo layer
        x_centered = x - x.mean(0, keepdim=True)
        z_centered = z - z.mean(0, keepdim=True)
        x_max_norm = torch.sqrt(x_centered.square().sum(1)).max()
        z_max_norm = torch.sqrt(z_centered.square().sum(1)).max()
        ecc_1 = self.eclay_1((x_centered/x_max_norm).unsqueeze(0)).squeeze(0)
        ecc_2 = self.eclay_2((z_centered/z_max_norm).unsqueeze(0)).squeeze(0)

        # topo_loss = self.mse_loss(ecc_1, ecc_2)
        topo_loss = self.mae_loss(ecc_1, ecc_2)
        loss = recon_loss + self.lam*topo_loss
        
        return x_recon, z, loss, (recon_loss.item(), self.lam*topo_loss.item())