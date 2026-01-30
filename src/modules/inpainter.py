import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from pathlib import Path
import numpy as np
from tqdm import tqdm

class FiLMLayer(nn.Module):
    def __init__(self, channels, embedding_dim):
        super().__init__()
        self.scale = nn.Linear(embedding_dim, channels)
        self.shift = nn.Linear(embedding_dim, channels)

    def forward(self, x, embedding):
        s = self.scale(embedding).unsqueeze(2).unsqueeze(3)
        h = self.shift(embedding).unsqueeze(2).unsqueeze(3)
        return x * (1 + s) + h

class SelfAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=channels, num_heads=4, batch_first=True)
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        b, c, h, w = x.size()
        x_flat = x.view(b, c, -1).permute(0, 2, 1)
        
        attn_out, _ = self.mha(x_flat, x_flat, x_flat)
        x_flat = self.norm(x_flat + attn_out)
        
        return x_flat.permute(0, 2, 1).view(b, c, h, w)

class ConditionalResidualBlock(nn.Module):
    def __init__(self, channels, embedding_dim, dilation=1):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.film1 = FiLMLayer(channels, embedding_dim)
        
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.film2 = FiLMLayer(channels, embedding_dim)
        
        self.relu = nn.GELU()

    def forward(self, x, style_embedding):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.film1(out, style_embedding) 
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.film2(out, style_embedding)
        
        return self.relu(out + residual)

class LatentInpainter(nn.Module):
    def __init__(self, 
                 latent_channels=64,   
                 num_clusters=12,
                 embedding_dim=64,
                 hidden_channels=128,
                 learning_rate=0.001,
                 use_amp=True,
                 load_best=False
                ):
        super().__init__()
        
        self.num_clusters = num_clusters
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_amp = use_amp and torch.cuda.is_available()
        self.save_path = Path('checkpoints/best_inpainter.pth')
        
        self.cluster_styles = nn.Parameter(torch.randn(num_clusters, embedding_dim))
        
        self.classifier = nn.Sequential(
            nn.Conv2d(latent_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, num_clusters)
        )

        self.encoder = nn.Sequential(
            nn.Conv2d(latent_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.GELU()
        )
        
        self.mid1 = ConditionalResidualBlock(hidden_channels, embedding_dim, dilation=1)
        self.mid2 = ConditionalResidualBlock(hidden_channels, embedding_dim, dilation=2)
        self.attn = SelfAttention(hidden_channels) 
        self.mid3 = ConditionalResidualBlock(hidden_channels, embedding_dim, dilation=2)
        self.mid4 = ConditionalResidualBlock(hidden_channels, embedding_dim, dilation=1)

        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.GELU(),
            nn.Conv2d(hidden_channels, latent_channels, kernel_size=3, padding=1)
        )

        self.mse_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss() 
        
        self.optimizer = optim.AdamW(self.parameters(), lr=learning_rate, weight_decay=1e-4)
        self.scaler = torch.amp.GradScaler(self.device.__str__(), enabled=self.use_amp)
        
        self.to(self.device)

        if load_best and self.save_path.exists():
            self.load()

    def forward(self, x, return_logits=False):
        logits = self.classifier(x) 
        
        probs = F.softmax(logits, dim=1) 
        
        style_embedding = torch.matmul(probs, self.cluster_styles)
        
        h = self.encoder(x)
        h = self.mid1(h, style_embedding)
        h = self.mid2(h, style_embedding)
        h = self.attn(h)
        h = self.mid3(h, style_embedding)
        h = self.mid4(h, style_embedding)
        
        correction = self.decoder(h)
        
        repaired = x + correction
        
        if return_logits:
            return repaired, logits
        return repaired

    def fit(self, train_loader, epochs=10):
        best_loss = float('inf')
        
        for epoch in range(epochs):
            self.train()
            train_loss = 0.0
            recon_loss_acc = 0.0
            class_loss_acc = 0.0
            
            loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
            
            for clean_batch, corr_batch, labels in loop:
                clean = clean_batch.to(self.device, non_blocking=True)
                corr = corr_batch.to(self.device, non_blocking=True)
                labels = labels.to(self.device, dtype=torch.long)

                self.optimizer.zero_grad()

                with torch.amp.autocast(self.device.__str__(), enabled=self.use_amp):
                    repaired, predicted_logits = self.forward(corr, return_logits=True)
                    
                    l_rec = self.mse_loss(repaired, clean)
                    
                    l_cls = self.ce_loss(predicted_logits, labels)
                    
                    loss = l_rec + 0.2 * l_cls

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
                train_loss += loss.item()
                recon_loss_acc += l_rec.item()
                class_loss_acc += l_cls.item()
                
                loop.set_postfix(
                    mse=f"{l_rec.item():.4f}", 
                    cls=f"{l_cls.item():.4f}"
                )

            avg_loss = train_loss / len(train_loader)
            print(f"Epoch {epoch+1} done. Avg Loss: {avg_loss:.5f} (Rec: {recon_loss_acc/len(train_loader):.4f}, Cls: {class_loss_acc/len(train_loader):.4f})")
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                self.save(self.save_path)

    def predict(self, corrupted_latent):
        self.eval()
        with torch.no_grad():
            if corrupted_latent.dim() == 3:
                corrupted_latent = corrupted_latent.unsqueeze(0)
            
            corrupted_latent = corrupted_latent.to(self.device)
            
            repaired = self.forward(corrupted_latent)
            
            return repaired

    def save(self, path):
        torch.save(self.state_dict(), path)
        print(f"Model saved to {path}")

    def load(self, path=None):
        self.load_state_dict(torch.load(self.save_path, map_location=self.device))
        self.eval()
        print(f"Model loaded from {path}")