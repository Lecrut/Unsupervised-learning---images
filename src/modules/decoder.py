import torch.nn as nn

class Decoder(nn.Module):
    def __init__(self, latent_dim=768, output_channels=4, image_size=256):
        super().__init__()
        self.latent_dim = latent_dim
        self.image_size = image_size
        
        self.fc = nn.Linear(latent_dim, 512 * 8 * 8)
        self.unflatten = nn.Unflatten(1, (512, 8, 8))
        
        def decoder_block(in_c, out_c):
            return nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'), 
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False), 
                nn.BatchNorm2d(out_c),
                nn.GELU()
            )

        # 8 -> 16
        self.block1 = decoder_block(512, 512)
        # 16 -> 32
        self.block2 = decoder_block(512, 256)
        # 32 -> 64
        self.block3 = decoder_block(256, 128)
        # 64 -> 128
        self.block4 = decoder_block(128, 64)
        # 128 -> 256
        self.block5 = decoder_block(64, 32)
        
        # Output
        self.final_conv = nn.Conv2d(32, output_channels, kernel_size=3, padding=1)
        self.out_act = nn.Sigmoid()
    
    def forward(self, latent):
        x = self.unflatten(self.fc(latent))
        
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        
        x = self.out_act(self.final_conv(x))
        return x