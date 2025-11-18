import torch
import torchvision.transforms.functional as TF

#%% Rotate 
def rotate_left(image: torch.Tensor):
    return TF.rotate(image, 90)

def rotate_right(image: torch.Tensor):
    return TF.rotate(image, -90)

#%% Reverse
def reverse(image: torch.Tensor):
    return TF.hflip(image)

