"""
Funkcje do trenowania modeli autoencoderów.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from typing import Optional, List, Tuple, Dict
import time
import numpy as np
from .metrics import calculate_ssim, calculate_psnr


def train_autoencoder(model: nn.Module,
                     train_loader: DataLoader,
                     criterion: nn.Module,
                     optimizer: torch.optim.Optimizer,
                     device: torch.device,
                     epochs: int = 5,
                     experiment = None,
                     verbose: bool = True,
                     save_checkpoint_every: int = None,
                     checkpoint_path: str = "checkpoint.pth") -> List[float]:
    """
    Trenuje autoencoder na danych treningowych.
    
    Args:
        model: model autoencodera
        train_loader: DataLoader z danymi treningowymi (masked, target)
        criterion: funkcja straty (np. MSELoss)
        optimizer: optymalizator (np. Adam)
        device: urządzenie (cuda/cpu)
        epochs: liczba epok
        experiment: obiekt CometML do logowania (opcjonalny)
        verbose: czy wyświetlać postęp
        save_checkpoint_every: co ile epok zapisywać checkpoint
        checkpoint_path: ścieżka do zapisywania checkpointów
        
    Returns:
        Lista strat z każdej epoki
    """
    model.train()
    losses = []
    start_time = time.time()
    
    if verbose:
        print(f"🚀 Rozpoczynam trenowanie na {epochs} epok(ach)")
        print(f"📱 Urządzenie: {device}")
        print(f"📊 Rozmiar batcha: {train_loader.batch_size}")
        print(f"🔄 Liczba batchy: {len(train_loader)}")
        print("-" * 50)
    
    for epoch in range(epochs):
        epoch_start_time = time.time()
        running_loss = 0.0
        num_batches = 0
        
        for batch_idx, (masked_imgs, target_imgs) in enumerate(train_loader):
            # Przenieś dane na device
            masked_imgs = masked_imgs.to(device)
            target_imgs = target_imgs.to(device)

            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs, latent = model(masked_imgs)
            
            # Calculate loss
            loss = criterion(outputs, target_imgs)
            
            # Backward pass
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            num_batches += 1
            
            # Progress w trakcie epoki
            if verbose and (batch_idx + 1) % 50 == 0:
                print(f"  Batch {batch_idx + 1}/{len(train_loader)}, "
                      f"Loss: {loss.item():.4f}")

        # Średnia strata w epoce
        avg_loss = running_loss / num_batches
        losses.append(avg_loss)
        
        epoch_time = time.time() - epoch_start_time
        
        # Logowanie do Comet ML
        if experiment:
            experiment.log_metric("train_loss", avg_loss, step=epoch)
            experiment.log_metric("epoch_time", epoch_time, step=epoch)
            
        if verbose:
            print(f"📈 Epoka [{epoch+1}/{epochs}] - "
                  f"Loss: {avg_loss:.4f}, "
                  f"Czas: {epoch_time:.1f}s")

        # Logowanie przykładowych obrazów
        if experiment and epoch % max(1, epochs // 5) == 0:
            with torch.no_grad():
                # Weź pierwszy batch do wizualizacji
                sample_masked = masked_imgs[:4].cpu()
                sample_outputs = outputs[:4].detach().cpu()
                sample_targets = target_imgs[:4].cpu()
                
                for i in range(min(4, sample_masked.size(0))):
                    experiment.log_image(
                        sample_masked[i].permute(1,2,0).numpy(), 
                        name=f"masked_epoch_{epoch}_sample_{i}"
                    )
                    experiment.log_image(
                        sample_outputs[i].permute(1,2,0).numpy(), 
                        name=f"reconstructed_epoch_{epoch}_sample_{i}"
                    )
                    experiment.log_image(
                        sample_targets[i].permute(1,2,0).numpy(), 
                        name=f"target_epoch_{epoch}_sample_{i}"
                    )
        
        # Zapisywanie checkpointów
        if save_checkpoint_every and (epoch + 1) % save_checkpoint_every == 0:
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'losses_history': losses
            }
            torch.save(checkpoint, f"{checkpoint_path.split('.')[0]}_epoch_{epoch+1}.pth")
            if verbose:
                print(f"💾 Checkpoint zapisany: epoch {epoch+1}")
    
    total_time = time.time() - start_time
    if verbose:
        print("-" * 50)
        print(f"✅ Trenowanie zakończone!")
        print(f"⏱️  Całkowity czas: {total_time:.1f}s")
        print(f"📉 Finalna strata: {losses[-1]:.4f}")
        print(f"📊 Najniższa strata: {min(losses):.4f} (epoka {losses.index(min(losses))+1})")
    
    return losses


def validate_autoencoder(model: nn.Module,
                        val_loader: DataLoader,
                        criterion: nn.Module,
                        device: torch.device,
                        max_batches: Optional[int] = None) -> Tuple[float, dict]:
    """
    Waliduje model autoencodera na danych walidacyjnych.
    
    Args:
        model: model autoencodera
        val_loader: DataLoader z danymi walidacyjnymi
        criterion: funkcja straty
        device: urządzenie
        
    Returns:
        Tuple: (średnia_strata, słownik_z_metrykami)
    """
    model.eval()
    total_loss = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch_idx, (masked_imgs, target_imgs) in enumerate(val_loader):
            if max_batches and batch_idx >= max_batches:
                break
                
            masked_imgs = masked_imgs.to(device)
            target_imgs = target_imgs.to(device)
            
            # Forward pass bez mixed precision
            outputs, _ = model(masked_imgs)
            loss = criterion(outputs, target_imgs)
            
            # Oblicz metryki jakości
            psnr = calculate_psnr(outputs, target_imgs)
            ssim = calculate_ssim(outputs, target_imgs)
            
            total_loss += loss.item()
            total_psnr += psnr
            total_ssim += ssim
            num_batches += 1
    
    # Średnie wartości
    avg_loss = total_loss / num_batches
    avg_psnr = total_psnr / num_batches
    avg_ssim = total_ssim / num_batches
    
    metrics = {
        'validation_loss': avg_loss,
        'psnr': avg_psnr,
        'ssim': avg_ssim,
        'num_batches': num_batches
    }
    
    return avg_loss, metrics


def train_with_validation(model: nn.Module,
                         train_loader: DataLoader,
                         val_loader: DataLoader,
                         criterion: nn.Module,
                         optimizer: torch.optim.Optimizer,
                         device: torch.device,
                         epochs: int = 100,  # Zwiększona liczba epok
                         experiment = None,
                         patience: int = 15,  # Zwiększona cierpliwość
                         min_delta: float = 1e-4,
                         scheduler_type: str = 'cosine',  # Typ schedulera
                         grad_clip: float = 1.0,  # Clipping gradientów
                         mixup_alpha: float = 0.2,  # Parametr mixup augmentacji
                         calculate_ssim: bool = False  # Czy obliczać SSIM podczas treningu
                         ) -> Dict[str, any]:
    """
    Trenuje model z walidacją i early stoppingiem.
    
    Args:
        patience: liczba epok bez poprawy przed zatrzymaniem
        min_delta: minimalna poprawa uznawana za znaczącą
        
    Returns:
        Słownik z historią trenowania
    """
    # Inicjalizacja metryk i schedulera
    train_losses = []
    val_losses = []
    train_psnr = []
    val_psnr = []
    train_ssim = []
    val_ssim = []
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    # Inicjalizacja schedulera
    if scheduler_type == 'plateau':
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    else:  # cosine
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    
    print(f"🎯 Trenowanie z walidacją i early stopping (patience={patience})")
    print(f"📋 Scheduler: {scheduler_type}, Grad Clip: {grad_clip}, Mixup: {mixup_alpha}")
    
    def mixup_data(x, y, alpha=0.2):
        """Wykonuje mixup augmentację na batch'u"""
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1

        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(x.device)

        mixed_x = lam * x + (1 - lam) * x[index]
        mixed_y = lam * y + (1 - lam) * y[index]
        return mixed_x, mixed_y
    
    for epoch in range(epochs):
        # Trenowanie
        model.train()
        train_loss = 0.0
        batch_psnr = []
        batch_ssim = []
        
        for masked_imgs, target_imgs in train_loader:
            masked_imgs = masked_imgs.to(device)
            target_imgs = target_imgs.to(device)
            
            # Mixup augmentacja
            if mixup_alpha > 0:
                masked_imgs, target_imgs = mixup_data(masked_imgs, target_imgs, mixup_alpha)
            
            optimizer.zero_grad()
            
            # Forward pass bez mixed precision
            outputs, _ = model(masked_imgs)
            loss = criterion(outputs, target_imgs)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            
            optimizer.step()
            
            train_loss += loss.item()
            
            # Oblicz tylko PSNR podczas treningu (SSIM wymaga float32)
            with torch.no_grad():
                batch_psnr.append(calculate_psnr(outputs, target_imgs))
            
            train_loss /= len(train_loader)
            train_losses.append(train_loss)
            train_psnr.append(np.mean(batch_psnr))
            train_ssim.append(0.0)  # Tymczasowo wyłączone        # Walidacja
        val_loss, val_metrics = validate_autoencoder(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_psnr.append(val_metrics['psnr'])
        val_ssim.append(val_metrics['ssim'])
        
        # Aktualizacja schedulera
        if scheduler_type == 'plateau':
            scheduler.step(val_loss)
        else:
            scheduler.step()
        
        # Early stopping z uwzględnieniem SSIM
        current_metric = val_loss - val_metrics['ssim']  # Kombinowana metryka
        if current_metric < best_val_loss - min_delta:
            best_val_loss = current_metric
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            print(f"✨ Epoka {epoch+1}: Nowy najlepszy wynik!")
            print(f"   Val Loss: {val_loss:.4f}, SSIM: {val_metrics['ssim']:.4f}, PSNR: {val_metrics['psnr']:.2f} dB")
        else:
            patience_counter += 1
            
        # Wyświetl postęp
        print(f"📊 Epoka {epoch+1}:")
        print(f"   Train - Loss: {train_loss:.4f}, PSNR: {train_psnr[-1]:.2f} dB, SSIM: {train_ssim[-1]:.4f}")
        print(f"   Val   - Loss: {val_loss:.4f}, PSNR: {val_metrics['psnr']:.2f} dB, SSIM: {val_metrics['ssim']:.4f}")
        print(f"   LR: {optimizer.param_groups[0]['lr']:.2e}")
        
        # Logowanie do experiment trackera
        if experiment:
            metrics = {
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_psnr": train_psnr[-1],
                "val_psnr": val_metrics['psnr'],
                "train_ssim": train_ssim[-1],
                "val_ssim": val_metrics['ssim'],
                "learning_rate": optimizer.param_groups[0]['lr']
            }
            experiment.log_metrics(metrics, step=epoch)
            
            # Log sample images
            if epoch % 5 == 0:
                with torch.no_grad():
                    sample_outputs = outputs[:4].cpu()
                    sample_targets = target_imgs[:4].cpu()
                    for i in range(4):
                        experiment.log_image(
                            sample_outputs[i].permute(1,2,0).numpy(),
                            name=f"epoch_{epoch}_sample_{i}_output"
                        )
                        experiment.log_image(
                            sample_targets[i].permute(1,2,0).numpy(),
                            name=f"epoch_{epoch}_sample_{i}_target"
                        )
        
        if patience_counter >= patience:
            print(f"🛑 Early stopping po {epoch+1} epokach")
            break
    
    # Przywróć najlepszy model
    if best_model_state:
        model.load_state_dict(best_model_state)
        print(f"🔄 Przywrócono najlepszy model (Val Loss: {best_val_loss:.4f})")
    
    # Przywróć najlepszy model i zbierz finalne metryki
    if best_model_state:
        model.load_state_dict(best_model_state)
        _, final_metrics = validate_autoencoder(model, val_loader, criterion, device)
        print("\n🏁 Finalne metryki po przywróceniu najlepszego modelu:")
        print(f"   PSNR: {final_metrics['psnr']:.2f} dB")
        print(f"   SSIM: {final_metrics['ssim']:.4f}")
        print(f"   Loss: {final_metrics['validation_loss']:.4f}")
    
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_psnr': train_psnr,
        'val_psnr': val_psnr,
        'train_ssim': train_ssim,
        'val_ssim': val_ssim,
        'best_val_loss': best_val_loss,
        'final_metrics': final_metrics if best_model_state else None,
        'epochs_trained': len(train_losses)
    }