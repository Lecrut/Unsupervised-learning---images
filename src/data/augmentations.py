#%% Imports
import torch
import torchvision.transforms.functional as TF

#%% Check device - Cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#%% Rotate 
def rotate_left(image: torch.Tensor):
    return TF.rotate(image, 90).to(device)

def rotate_right(image: torch.Tensor):
    return TF.rotate(image, -90).to(device)

#%% Reverse
def reverse(image: torch.Tensor):
    return TF.hflip(image).to(device)
