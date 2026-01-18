import os
import torch
import torch.nn as nn
import torch.nn.functional as F


class ClusteringAutoencoder(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        latent_dim: int,
        hidden_dim: int,
        cluster_dim: int,
        n_clusters: int,
        freeze_encoder: bool = False,
        device: str = "cuda"
    ):
        super().__init__()

        self.device = device
        self.checkpoint_path = 'checkpoints/clustering_autoencoder.pt'
        self.n_clusters = n_clusters

        self.encoder = encoder
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

        self.projection = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, cluster_dim)
        )

        self.cluster_centers = nn.Parameter(
            torch.randn(n_clusters, cluster_dim)
        )

        self.to(device)

        if os.path.exists(self.checkpoint_path):
            self.load_checkpoint()
            print(f"[INFO] Loaded checkpoint from {self.checkpoint_path}")

    def forward(self, x):
        z = self.encoder(x)
        if isinstance(z, (list, tuple)):
            z = z[0]  
        z = self.projection(z)
        return z

    def soft_assign(self, z):
        dist = torch.cdist(z, self.cluster_centers)
        q = 1.0 / (1.0 + dist.pow(2))
        q = q / q.sum(dim=1, keepdim=True)
        return q

    @staticmethod
    def target_distribution(q):
        weight = q.pow(2) / q.sum(0)
        return (weight.t() / weight.sum(1)).t()

  
    def compute_loss(self, x):
        z = self.forward(x)
        q = self.soft_assign(z)
        p = self.target_distribution(q).detach()

        loss = F.kl_div(
            q.log(),
            p,
            reduction="batchmean"
        )
        return loss


    def train_epoch(self, dataloader, optimizer):
        self.train()
        total_loss = 0.0

        for batch in dataloader:
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            x = x.to(self.device)

            optimizer.zero_grad()
            loss = self.compute_loss(x)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        return total_loss / len(dataloader)

    def fit(
        self,
        dataloader,
        epochs: int = 200,
        lr: float = 1e-3,
        patience: int = 15,
        verbose: bool = True
    ):
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=lr
        )

        best_loss = float("inf")
        patience_ctr = 0

        for epoch in range(epochs):
            loss = self.train_epoch(dataloader, optimizer)

            if verbose:
                print(f"[Epoch {epoch+1:03d}] loss = {loss:.6f}")

            if loss < best_loss:
                best_loss = loss
                patience_ctr = 0
                self.save_checkpoint(best_loss)
            else:
                patience_ctr += 1

            if patience_ctr >= patience:
                if verbose:
                    print("[INFO] Early stopping")
                break

        self.load_checkpoint()
        if verbose:
            print("[INFO] Best model restored")


    def save_checkpoint(self, loss_value):
        torch.save(
            {
                "encoder": self.encoder.state_dict(),
                "projection": self.projection.state_dict(),
                "centers": self.cluster_centers.data,
                "loss": loss_value
            },
            self.checkpoint_path
        )

    def load_checkpoint(self):
        ckpt = torch.load(self.checkpoint_path, map_location=self.device)
        self.encoder.load_state_dict(ckpt["encoder"])
        self.projection.load_state_dict(ckpt["projection"])
        self.cluster_centers.data.copy_(ckpt["centers"])

    @torch.no_grad()
    def predict(self, x):
        self.eval()
        x = x.to(self.device)
        z = self.forward(x)
        q = self.soft_assign(z)
        return torch.argmax(q, dim=1)

    @torch.no_grad()
    def predict_soft(self, x):
        self.eval()
        x = x.to(self.device)
        z = self.forward(x)
        return self.soft_assign(z)
