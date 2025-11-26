#%% Imports
import torch
import torch.nn as nn
import torch.nn.functional as F

#%% Check device - Cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#%% Simple Conditional U-Net Definition
class SimpleConditionalUNet(nn.Module):
    def __init__(self, in_channels=3, base_channels=64, num_clusters=10):
        super().__init__()
        self.embed_c = nn.Embedding(num_clusters, base_channels)
        self.conv1 = nn.Conv2d(in_channels + base_channels, base_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(base_channels, base_channels*2, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(base_channels*2, base_channels*4, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(base_channels*4, base_channels*4, 3, padding=1)
        self.deconv1 = nn.ConvTranspose2d(base_channels*4, base_channels*2, 4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(base_channels*2, base_channels, 4, stride=2, padding=1)
        self.final = nn.Conv2d(base_channels, in_channels, 3, padding=1)
        self.time_mlp = nn.Sequential(nn.Linear(1, base_channels*4), nn.ReLU(), nn.Linear(base_channels*4, base_channels*4))

    def forward(self, z_t, t, c):
        B, C, H, W = z_t.shape
        c_emb = self.embed_c(c).view(B, -1, 1, 1).expand(B, -1, H, W)
        x = torch.cat([z_t, c_emb], dim=1)
        t = t.float().view(B, 1)
        t_emb = self.time_mlp(t).view(B, -1, 1, 1)
        x = x + t_emb
        d1 = F.relu(self.conv1(x))
        d2 = F.relu(self.conv2(d1))
        d3 = F.relu(self.conv3(d2))
        b = F.relu(self.conv4(d3))
        u1 = F.relu(self.deconv1(b) + d2)
        u2 = F.relu(self.deconv2(u1) + d1)
        out = self.final(u2)
        return out

#%% Conditional DDPM Wrapper
class ConditionalDDPM(nn.Module):
    def __init__(self, in_channels=3, num_clusters=10):
        super().__init__()
        self.network = SimpleConditionalUNet(in_channels=in_channels, num_clusters=num_clusters)

    def forward(self, z_t, t, c):
        return self.network(z_t, t, c)

#%% Latent Repaint Inpainting Function
def latent_repaint_inpainting(z0, m, c, T=250, r=10, j=10):
    z0 = z0.to(device)
    m = m.to(device)
    c = c.to(device)
    ddpm = ConditionalDDPM(in_channels=z0.shape[1], num_clusters=torch.max(c).item() + 1).to(device)

    betas = torch.linspace(1e-4, 0.02, T, device=device)  
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    sigmas = torch.sqrt(betas)

    z_t = torch.randn_like(z0)

    for t in range(T, 0, -1): 
        for u in range(r):  
            epsilon_known = torch.randn_like(z0) if t > 1 else torch.zeros_like(z0)
            sqrt_alpha_bar = torch.sqrt(alpha_bars[t - 1])
            sqrt_one_minus_alpha_bar = torch.sqrt(1 - alpha_bars[t - 1])
            z_known = sqrt_alpha_bar * z0 + sqrt_one_minus_alpha_bar * epsilon_known

            epsilon_unknown = torch.randn_like(z0) if t > 1 else torch.zeros_like(z0)
            epsilon_theta = ddpm(z_t, torch.tensor([t], device=device), c)
            alpha_t = alphas[t - 1]
            beta_t = betas[t - 1]
            sigma_t = sigmas[t - 1]
            z_unknown = (1 / torch.sqrt(alpha_t)) * (z_t - (beta_t / sqrt_one_minus_alpha_bar) * epsilon_theta) + sigma_t * epsilon_unknown

            z_next = m * z_unknown + (1 - m) * z_known

            if u < r and t > 1:
                z_jump = z_next
                t_jump = t
                for _ in range(j):
                    beta_fwd = betas[max(t_jump - 2, 0)]
                    sqrt_one_minus_beta = torch.sqrt(1 - beta_fwd)
                    noise_fwd = torch.randn_like(z0)
                    z_jump = sqrt_one_minus_beta * z_jump + torch.sqrt(beta_fwd) * noise_fwd
                    t_jump = max(t_jump - 1, 1)
                z_t = z_jump
            else:
                z_t = z_next

    return z_t
