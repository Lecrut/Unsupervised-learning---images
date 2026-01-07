import torch.nn as nn

class Decoder(nn.Module):
    def __init__(self, latent_dim=768, output_channels=4, image_size=256):
        super().__init__()
        self.latent_dim = latent_dim
        self.image_size = image_size
        
        self.fc = nn.Linear(latent_dim, 512 * 8 * 8)
        self.unflatten = nn.Unflatten(1, (512, 8, 8))
        
        self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(512)
        self.act1 = nn.GELU()
        
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(256)
        self.act2 = nn.GELU()
        
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(128)
        self.act3 = nn.GELU()
        
        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(64)
        self.act4 = nn.GELU()
        
        self.final_conv = nn.Conv2d(64, output_channels * 4, kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(2) 
        
        self.out_act = nn.Sigmoid()
    
    def forward(self, latent):
        x = self.unflatten(self.fc(latent))
        
        x = self.act1(self.bn1(self.deconv1(x)))
        x = self.act2(self.bn2(self.deconv2(x)))
        x = self.act3(self.bn3(self.deconv3(x)))
        x = self.act4(self.bn4(self.deconv4(x)))
        
        x = self.pixel_shuffle(self.final_conv(x))
        x = self.out_act(x)
        
        return x