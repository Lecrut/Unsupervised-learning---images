"""
Funkcje do trenowania modeli autoencoderów.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional, List, Tuple
import time


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
                        device: torch.device) -> Tuple[float, dict]:
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
    num_batches = 0
    
    with torch.no_grad():
        for masked_imgs, target_imgs in val_loader:
            masked_imgs = masked_imgs.to(device)
            target_imgs = target_imgs.to(device)
            
            outputs, _ = model(masked_imgs)
            loss = criterion(outputs, target_imgs)
            
            total_loss += loss.item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches
    
    metrics = {
        'validation_loss': avg_loss,
        'num_batches': num_batches
    }
    
    return avg_loss, metrics


def train_with_validation(model: nn.Module,
                         train_loader: DataLoader,
                         val_loader: DataLoader,
                         criterion: nn.Module,
                         optimizer: torch.optim.Optimizer,
                         device: torch.device,
                         epochs: int = 10,
                         experiment = None,
                         patience: int = 5,
                         min_delta: float = 1e-4) -> dict:
    """
    Trenuje model z walidacją i early stoppingiem.
    
    Args:
        patience: liczba epok bez poprawy przed zatrzymaniem
        min_delta: minimalna poprawa uznawana za znaczącą
        
    Returns:
        Słownik z historią trenowania
    """
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    print(f"🎯 Trenowanie z walidacją i early stopping (patience={patience})")
    
    for epoch in range(epochs):
        # Trenowanie
        model.train()
        train_loss = 0.0
        for masked_imgs, target_imgs in train_loader:
            masked_imgs, target_imgs = masked_imgs.to(device), target_imgs.to(device)
            
            optimizer.zero_grad()
            outputs, _ = model(masked_imgs)
            loss = criterion(outputs, target_imgs)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Walidacja
        val_loss, _ = validate_autoencoder(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            print(f"✨ Epoka {epoch+1}: Nowy najlepszy wynik! Val Loss: {val_loss:.4f}")
        else:
            patience_counter += 1
            
        print(f"📊 Epoka {epoch+1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        if experiment:
            experiment.log_metric("train_loss", train_loss, step=epoch)
            experiment.log_metric("val_loss", val_loss, step=epoch)
        
        if patience_counter >= patience:
            print(f"🛑 Early stopping po {epoch+1} epokach (patience={patience})")
            break
    
    # Przywróć najlepszy model
    if best_model_state:
        model.load_state_dict(best_model_state)
        print(f"🔄 Przywrócono najlepszy model (Val Loss: {best_val_loss:.4f})")
    
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val_loss': best_val_loss,
        'epochs_trained': len(train_losses)
    }