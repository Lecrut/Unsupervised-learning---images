"""
Funkcje do obliczania metryk jakości obrazów (SSIM, PSNR, MSE).

Używane do ewaluacji modeli inpainting i super-resolution.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Union
from pytorch_msssim import ssim, ms_ssim


def calculate_psnr(img1: torch.Tensor, img2: torch.Tensor, max_val: float = 1.0) -> float:
    """
    Oblicza Peak Signal-to-Noise Ratio (PSNR) między dwoma obrazami.
    
    PSNR = 20 * log10(MAX) - 10 * log10(MSE)
    
    Wyższa wartość = lepsza jakość (typowo > 25 dB jest akceptowalne)
    
    Args:
        img1: Pierwszy obraz [B, C, H, W] lub [C, H, W]
        img2: Drugi obraz (ta sama forma)
        max_val: Maksymalna wartość piksela (1.0 dla znormalizowanych, 255 dla uint8)
        
    Returns:
        PSNR w dB
    """
    mse = F.mse_loss(img1, img2)
    
    if mse == 0:
        return float('inf')
    
    psnr = 20 * torch.log10(torch.tensor(max_val)) - 10 * torch.log10(mse)
    return psnr.item()


def calculate_ssim(img1: torch.Tensor, 
                   img2: torch.Tensor, 
                   data_range: float = 1.0,
                   size_average: bool = True) -> float:
    """
    Oblicza Structural Similarity Index (SSIM) między dwoma obrazami.
    
    SSIM mierzy podobieństwo strukturalne - lepsze od MSE/PSNR dla percepcji ludzkiej.
    
    Wartości:
    - 1.0 = identyczne obrazy
    - > 0.85 = bardzo dobra jakość
    - 0.5-0.85 = akceptowalna jakość
    - < 0.5 = słaba jakość
    
    Args:
        img1: Pierwszy obraz [B, C, H, W]
        img2: Drugi obraz [B, C, H, W]
        data_range: Zakres wartości pikseli
        size_average: Czy uśrednić po batch
        
    Returns:
        SSIM score
    """
    # Upewnij się że mamy 4D tensor i odpowiedni typ
    if img1.dim() == 3:
        img1 = img1.unsqueeze(0)
    if img2.dim() == 3:
        img2 = img2.unsqueeze(0)
    
    # Konwersja do float32
    img1 = img1.float()
    img2 = img2.float()
    
    ssim_val = ssim(img1, img2, data_range=data_range, size_average=size_average)
    return ssim_val.item()


def calculate_ms_ssim(img1: torch.Tensor, 
                      img2: torch.Tensor,
                      data_range: float = 1.0) -> float:
    """
    Oblicza Multi-Scale SSIM - bardziej zaawansowana wersja SSIM.
    
    Analizuje obraz na różnych skalach - lepiej chwyta szczegóły.
    
    Args:
        img1: Pierwszy obraz [B, C, H, W]
        img2: Drugi obraz [B, C, H, W]
        data_range: Zakres wartości pikseli
        
    Returns:
        MS-SSIM score
    """
    if img1.dim() == 3:
        img1 = img1.unsqueeze(0)
    if img2.dim() == 3:
        img2 = img2.unsqueeze(0)
    
    ms_ssim_val = ms_ssim(img1, img2, data_range=data_range)
    return ms_ssim_val.item()


def calculate_mse(img1: torch.Tensor, img2: torch.Tensor) -> float:
    """
    Oblicza Mean Squared Error między obrazami.
    
    Args:
        img1: Pierwszy obraz
        img2: Drugi obraz
        
    Returns:
        MSE value
    """
    mse = F.mse_loss(img1, img2)
    return mse.item()


def calculate_mae(img1: torch.Tensor, img2: torch.Tensor) -> float:
    """
    Oblicza Mean Absolute Error między obrazami.
    
    Args:
        img1: Pierwszy obraz
        img2: Drugi obraz
        
    Returns:
        MAE value
    """
    mae = F.l1_loss(img1, img2)
    return mae.item()


def evaluate_reconstruction(model: torch.nn.Module,
                           dataloader: torch.utils.data.DataLoader,
                           device: torch.device,
                           max_batches: int = None) -> dict:
    """
    Ewaluuje model na danych testowych obliczając wszystkie metryki.
    
    Args:
        model: Model do ewaluacji
        dataloader: DataLoader z danymi (input, target)
        device: Urządzenie
        max_batches: Maksymalna liczba batchy do ewaluacji
        
    Returns:
        Słownik z metrykami: {
            'psnr': float,
            'ssim': float, 
            'ms_ssim': float,
            'mse': float,
            'mae': float
        }
    """
    model.eval()
    
    total_psnr = 0.0
    total_ssim = 0.0
    total_ms_ssim = 0.0
    total_mse = 0.0
    total_mae = 0.0
    num_batches = 0
    
    print("🔍 Ewaluacja modelu...")
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            if max_batches and batch_idx >= max_batches:
                break
            
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # Forward pass
            if hasattr(model, 'forward') and 'mask' in model.forward.__code__.co_varnames:
                # Model wymaga maski (np. PartialConvUNet)
                outputs = model(inputs, mask=None)
            else:
                outputs = model(inputs)
                
            # Jeśli model zwraca tuple (output, latent), weź tylko output
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            
            # Oblicz metryki
            total_psnr += calculate_psnr(outputs, targets)
            total_ssim += calculate_ssim(outputs, targets)
            
            # MS-SSIM wymaga większych obrazów (>= 160x160)
            if outputs.shape[-1] >= 160 and outputs.shape[-2] >= 160:
                total_ms_ssim += calculate_ms_ssim(outputs, targets)
            
            total_mse += calculate_mse(outputs, targets)
            total_mae += calculate_mae(outputs, targets)
            
            num_batches += 1
            
            if (batch_idx + 1) % 10 == 0:
                print(f"  Przetworzono {batch_idx + 1} batchy...")
    
    # Uśrednij
    metrics = {
        'psnr': total_psnr / num_batches,
        'ssim': total_ssim / num_batches,
        'ms_ssim': total_ms_ssim / num_batches if outputs.shape[-1] >= 160 else None,
        'mse': total_mse / num_batches,
        'mae': total_mae / num_batches
    }
    
    print("\n📊 Wyniki ewaluacji:")
    print(f"  PSNR: {metrics['psnr']:.2f} dB")
    print(f"  SSIM: {metrics['ssim']:.4f}")
    if metrics['ms_ssim'] is not None:
        print(f"  MS-SSIM: {metrics['ms_ssim']:.4f}")
    print(f"  MSE: {metrics['mse']:.6f}")
    print(f"  MAE: {metrics['mae']:.6f}")
    
    return metrics


def compare_models(models_dict: dict,
                  dataloader: torch.utils.data.DataLoader,
                  device: torch.device) -> dict:
    """
    Porównuje kilka modeli na tych samych danych.
    
    Args:
        models_dict: Słownik {nazwa_modelu: model}
        dataloader: DataLoader z danymi testowymi
        device: Urządzenie
        
    Returns:
        Słownik z wynikami dla każdego modelu
    """
    results = {}
    
    print(f"🔬 Porównanie {len(models_dict)} modeli...\n")
    
    for name, model in models_dict.items():
        print(f"📈 Ewaluacja: {name}")
        print("-" * 50)
        metrics = evaluate_reconstruction(model, dataloader, device, max_batches=20)
        results[name] = metrics
        print()
    
    # Podsumowanie
    print("\n" + "=" * 70)
    print("📊 PODSUMOWANIE PORÓWNANIA")
    print("=" * 70)
    print(f"{'Model':<25} {'PSNR (dB)':<12} {'SSIM':<10} {'MSE':<12}")
    print("-" * 70)
    
    for name, metrics in results.items():
        print(f"{name:<25} {metrics['psnr']:>10.2f}  {metrics['ssim']:>8.4f}  {metrics['mse']:>10.6f}")
    
    print("=" * 70)
    
    # Znajdź najlepszy model dla każdej metryki
    best_psnr = max(results.items(), key=lambda x: x[1]['psnr'])
    best_ssim = max(results.items(), key=lambda x: x[1]['ssim'])
    
    print(f"\n🏆 Najlepszy PSNR: {best_psnr[0]} ({best_psnr[1]['psnr']:.2f} dB)")
    print(f"🏆 Najlepszy SSIM: {best_ssim[0]} ({best_ssim[1]['ssim']:.4f})")
    
    return results


class PerceptualLoss(torch.nn.Module):
    """
    Perceptual Loss używający pre-trenowanego VGG do porównania features.
    
    Często lepszy niż MSE dla zadań generowania obrazów.
    """
    
    def __init__(self, layers=['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3']):
        super().__init__()
        from torchvision.models import vgg16
        
        vgg = vgg16(pretrained=True).features
        self.layers = layers
        
        # Zamroź wagi VGG
        for param in vgg.parameters():
            param.requires_grad = False
        
        # Podziel VGG na bloki
        self.blocks = torch.nn.ModuleList()
        current_block = []
        layer_names = []
        
        layer_mapping = {
            4: 'relu1_2',
            9: 'relu2_2', 
            16: 'relu3_3',
            23: 'relu4_3'
        }
        
        for i, layer in enumerate(vgg):
            current_block.append(layer)
            if i in layer_mapping:
                self.blocks.append(torch.nn.Sequential(*current_block))
                layer_names.append(layer_mapping[i])
                current_block = []
        
        self.layer_names = layer_names
        
    def forward(self, x, y):
        """
        Oblicza perceptual loss między x i y.
        """
        loss = 0.0
        
        for block in self.blocks:
            x = block(x)
            y = block(y)
            loss += F.mse_loss(x, y)
        
        return loss


class CombinedLoss(torch.nn.Module):
    """
    Kombinacja różnych funkcji straty dla lepszych wyników.
    
    Typowo: MSE + SSIM + Perceptual Loss
    """
    
    def __init__(self, 
                 mse_weight=0.3,    # Zmniejszony wkład MSE
                 ssim_weight=1.0,   # SSIM jako główna metryka
                 perceptual_weight=0.4,  # Zwiększony wkład perceptual loss
                 edge_weight=0.3,   # Nowa waga dla edge loss
                 use_perceptual=True):  # Domyślnie włączony perceptual loss
        super().__init__()
        self.mse_weight = mse_weight
        self.ssim_weight = ssim_weight
        self.perceptual_weight = perceptual_weight
        self.edge_weight = edge_weight
        self.use_perceptual = use_perceptual
        
        if use_perceptual:
            self.perceptual = PerceptualLoss()
            
        # Kernel do detekcji krawędzi
        self.register_buffer('edge_kernel', torch.tensor([
            [-1, -1, -1],
            [-1,  8, -1],
            [-1, -1, -1]
        ]).unsqueeze(0).unsqueeze(0).float())
        
    def detect_edges(self, x):
        """Wykrywa krawędzie w obrazie używając filtru Laplace'a"""
        b, c, h, w = x.shape
        edges = []
        for i in range(c):
            channel = x[:, i:i+1]
            edge = F.conv2d(channel, self.edge_kernel, padding=1)
            edges.append(edge)
        return torch.cat(edges, dim=1)
        
    def forward(self, pred, target):
        """
        Oblicza kombinowaną stratę z większym naciskiem na szczegóły i krawędzie
        """
        # Normalizacja do [-1,1] dla lepszej stabilności
        pred = 2 * pred - 1
        target = 2 * target - 1
        
        # MSE Loss na znormalizowanych danych
        mse_loss = F.mse_loss(pred, target)
        
        # SSIM Loss
        ssim_loss = 1 - ssim(pred, target, data_range=2.0, size_average=True)
        
        # Edge Loss - zachowanie krawędzi
        pred_edges = self.detect_edges(pred)
        target_edges = self.detect_edges(target)
        edge_loss = F.l1_loss(pred_edges, target_edges)
        
        # Łączna strata
        total_loss = (self.mse_weight * mse_loss + 
                     self.ssim_weight * ssim_loss +
                     self.edge_weight * edge_loss)
        
        # Perceptual loss
        if self.use_perceptual:
            # Denormalizacja do [0,1] dla VGG
            pred_vgg = (pred + 1) / 2
            target_vgg = (target + 1) / 2
            perceptual_loss = self.perceptual(pred_vgg, target_vgg)
            total_loss += self.perceptual_weight * perceptual_loss
        
        return total_loss
