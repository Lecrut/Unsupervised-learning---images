
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

#%% VGG loss function
class VGGPerceptualLoss(nn.Module):
    def __init__(self, feature_layers=[0, 5, 10, 19, 28], use_l1=True):
        super().__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
        
        self.blocks = nn.ModuleList()
        current_layer = 0
        for target_layer in feature_layers:
            block = nn.Sequential()
            for i in range(current_layer, target_layer + 1):
                block.add_module(str(i), vgg[i])
            self.blocks.append(block)
            current_layer = target_layer + 1
            
        for param in self.parameters():
            param.requires_grad = False
            
        self.use_l1 = use_l1
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, input, target):
        if input.shape[1] == 4:
            input = input[:, :3, :, :]
            target = target[:, :3, :, :]
            
        # Normalizacja obrazów (zakładamy input w range [0, 1])
        input = (input - self.mean) / self.std
        target = (target - self.mean) / self.std
        
        loss = 0.0
        x = input
        y = target
        
        for block in self.blocks:
            x = block(x)
            y = block(y)
            
            if self.use_l1:
                loss += torch.nn.functional.l1_loss(x, y)
            else:
                loss += torch.nn.functional.mse_loss(x, y)
                
        return loss
