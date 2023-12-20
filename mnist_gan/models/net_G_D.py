import torch
import torch.nn as nn
import os

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.zeros_(m.bias)

class MNIST_Generator_CNN(nn.Module):
    """
    GAN generator for MNIST dataset, based on transposed convolutional layers.
    """
    def __init__(
        self,
        img_channels: int = 1,
        hidden_dim: int = 64,
        latent_dim: int = 16,
    ):
        super().__init__()
        self.img_channels = img_channels
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=self.latent_dim,
                out_channels=self.hidden_dim * 4,
                kernel_size=4,
                stride=1,
                padding=0,
            ),
            nn.BatchNorm2d(self.hidden_dim * 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                in_channels=self.hidden_dim * 4,
                out_channels=self.hidden_dim * 2,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm2d(self.hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                in_channels=self.hidden_dim * 2,
                out_channels=self.hidden_dim,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm2d(self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                in_channels=self.hidden_dim,
                out_channels=self.img_channels,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.Tanh(),
        )
        self.apply(init_weights)
        
    def forward(self, x):
        return self.decoder(x)
    
    def restore(self, ckpt_dir):
        try:
            if os.path.exists(os.path.join(ckpt_dir, 'generator.pth')):
                path = os.path.join(ckpt_dir, 'generator.pth')
            else:
                path = os.path.join(ckpt_dir, str(max(int(name) for name in os.listdir(ckpt_dir))), 'generator.bin')
        except:
            return None
        self.load_state_dict(torch.load(path))
        return os.path.split(path)[0]

    def save(self, ckpt_dir, global_step):
        os.makedirs(os.path.join(ckpt_dir, str(global_step)), exist_ok=True)
        path = os.path.join(ckpt_dir, str(global_step), 'generator.pth')
        torch.save(self.state_dict(), path)
        return os.path.split(path)[0]
    
class MNIST_Discriminator_CNN(nn.Module):
    """
    GAN discriminator for MNIST dataset, based on convolutional layers.
    """
    def __init__(
        self,
        img_channels: int = 1,
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.img_channels = img_channels
        self.hidden_dim = hidden_dim
        
        self.encoder = nn.Sequential(
            # input is (num_channels) x 32 x 32
            nn.Conv2d(
                in_channels=self.img_channels,
                out_channels=self.hidden_dim,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (hidden_dim) x 16 x 16
            nn.Conv2d(
                in_channels=self.hidden_dim,
                out_channels=self.hidden_dim * 2,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm2d(self.hidden_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (hidden_dim*2) x 8 x 8
            nn.Conv2d(
                in_channels=self.hidden_dim * 2,
                out_channels=self.hidden_dim * 4,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm2d(self.hidden_dim * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (hidden_dim*4) x 4 x 4
            nn.Conv2d(
                in_channels=self.hidden_dim * 4,
                out_channels=1,
                kernel_size=4,
                stride=1,
                padding=0,
            ),
            # state size. 1 x 1 x 1
            nn.Sigmoid(),
        )
        self.apply(init_weights)
        
    def forward(self, x):
        return self.encoder(x).view(-1, 1).squeeze(1)
    
    def restore(self, ckpt_dir):
        try:
            if os.path.exists(os.path.join(ckpt_dir, 'discriminator.pth')):
                path = os.path.join(ckpt_dir, 'discriminator.pth')
            else:
                path = os.path.join(ckpt_dir, str(max(int(name) for name in os.listdir(ckpt_dir))), 'discriminator.bin')
        except:
            return None
        self.load_state_dict(torch.load(path))
        return os.path.split(path)[0]

    def save(self, ckpt_dir, global_step):
        os.makedirs(os.path.join(ckpt_dir, str(global_step)), exist_ok=True)
        path = os.path.join(ckpt_dir, str(global_step), 'discriminator.pth')
        torch.save(self.state_dict(), path)
        return os.path.split(path)[0]
    
