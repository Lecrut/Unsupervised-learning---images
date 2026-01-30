import torch
import gc

# Sprawdź dostępność CUDA
if torch.cuda.is_available():
    # Opróżnij cache pamięci CUDA
    torch.cuda.empty_cache()
    # Synchronizuj (opcjonalne, dla pewności)
    torch.cuda.synchronize()
    print("Cache CUDA wyczyszczony.")
else:
    print("CUDA niedostępne.")

# Dodatkowe czyszczenie pamięci Pythona
gc.collect()
print("Pamięć Pythona wyczyszczona.")