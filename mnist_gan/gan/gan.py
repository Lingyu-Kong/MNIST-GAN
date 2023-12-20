import os
import time
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn
from torchvision.utils import save_image
from mnist_gan.mnist_loader.data_loader import load_MNIST
from mnist_gan.models.net_G_D import MNIST_Generator_CNN, MNIST_Discriminator_CNN
import wandb

class GAN(object):
    """
    Generative Adversarial Network (GAN) class.
    """
    def __init__(
        self,
        generator: nn.Module,
        discriminator: nn.Module,
        criterion: nn.Module = nn.BCELoss(),
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        lr: float = 0.0002,
        z_dim: int = 64,
        n_epochs: int = 20,
        batch_size: int = 64,
        save_interval: int = None,
        output_dir: str = "results",
        name: str = "gan",
        wandb = None,
        save_ckpt: bool = True,
    ):
        ## Generator, discriminator, criterion, optimizers, device
        self.generator = generator
        self.discriminator = discriminator
        self.optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(0.5, 0.999))
        self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
        self.criterion = criterion
        self.device = device
        self.generator.to(self.device)
        self.discriminator.to(self.device)
        self.criterion.to(self.device)
        ## z_dim, n_epochs, batch_size
        self.z_dim = z_dim
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        ## Logging information
        self.save_interval = save_interval
        self.output_dir = os.path.join(output_dir, name)
        self.name = name
        self.wandb = wandb
        self.save_ckpt = save_ckpt
        if os.path.exists(self.output_dir):
            os.system("rm -r {}".format(self.output_dir))
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "checkpoints"), exist_ok=True)
        
    
    def train(self, train_loader: torch.utils.data.DataLoader, test_loader: torch.utils.data.DataLoader = None):
        """
        Train the GAN model.
        :param train_loader: the training dataloader
        """
        self.generator.train()
        self.discriminator.train()
        min_FID = None
        for epoch in range(self.n_epochs):
            g_losses = []
            d_losses = []
            start_time = time.time()
            for i, (imgs, _) in enumerate(tqdm(train_loader)):
                batch_size = imgs.size(0)
                real_images = imgs.to(self.device)
                
                ## Train the discriminator
                self.optimizer_D.zero_grad()

                # Real images
                output_real = self.discriminator(real_images)
                loss_real = self.criterion(output_real, torch.ones_like(output_real).to(self.device))
                loss_real = loss_real.mean()
                loss_real.backward()

                # Generated images
                noise = torch.randn(batch_size, self.z_dim).to(self.device)
                noise = noise.view(noise.size(0), self.z_dim, 1, 1)
                fake_images = self.generator(noise)
                output_fake = self.discriminator(fake_images.detach())
                loss_fake = self.criterion(output_fake, torch.zeros_like(output_fake).to(self.device))
                loss_fake = loss_fake.mean()
                loss_fake.backward()
                
                # Discriminator loss backward
                self.optimizer_D.step()
                loss_D = loss_real + loss_fake

                ## Train the generator
                self.optimizer_G.zero_grad()
                output_G = self.discriminator(fake_images)
                loss_G = self.criterion(output_G, torch.ones_like(output_G).to(self.device))
                loss_G = loss_G.mean()
                loss_G.backward()
                self.optimizer_G.step()
                
                g_losses.append(loss_G.item())
                d_losses.append(loss_D.item())
            
            ## Logging
            if self.save_interval is not None and epoch % self.save_interval == 0:
                self.sample_image(n_row=8, epoch=epoch)
                if self.save_ckpt:
                    self.generator.save(ckpt_dir=os.path.join(self.output_dir, "checkpoints"), global_step=epoch)
                    self.discriminator.save(ckpt_dir=os.path.join(self.output_dir, "checkpoints"), global_step=epoch)

            g_loss = np.mean(g_losses).item()
            d_loss = np.mean(d_losses).item()
            if self.wandb is not None:
                self.wandb.log({
                    "Losses/Generator": g_loss,
                    "Losses/Discriminator": d_loss,
                    "Losses/Total": g_loss+d_loss,
                })
            print("[Epoch %d/%d] [D loss: %f] [G loss: %f] [Time: %f]" % (epoch, self.n_epochs, d_loss, g_loss, time.time()-start_time))
        
    def sample_image(self, n_row: int, epoch: int):
        """
        Sample the images from the generator.
        :param n_row: the number of rows of the sampled images
        :param batches_done: the number of batches done
        """
        # Sample noise
        z = torch.randn(n_row ** 2, self.z_dim).to(self.device)
        z = z.view(z.size(0), self.z_dim, 1, 1)
        gen_imgs = self.generator(z) * 0.5 + 0.5
        save_image(gen_imgs.data, "{}/images/epoch-{}.png".format(self.output_dir, epoch), nrow=n_row, normalize=True)
    
        
if __name__ == "__main__":
    batch_size = 128
    train_loader, test_loader = load_MNIST(batch_size=batch_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   
    z_dim = 16
    hidden_dim = 64
    generator = MNIST_Generator_CNN(
        img_channels=1,
        hidden_dim=hidden_dim,
        latent_dim=z_dim,
    )
    discriminator = MNIST_Discriminator_CNN(
        img_channels=1,
        hidden_dim=hidden_dim,
    )
    # wandb.login(key="37f3de06380e350727df28b49712f8b7fe5b14aa")
    # wandb.init(project="gan-mnist")
    gan = GAN(
        generator = generator,
        discriminator = discriminator,
        device = device,
        lr = 0.0002,
        n_epochs = 20,
        z_dim = z_dim,
        # wandb = wandb,
        save_interval = 1,
        batch_size = batch_size,
        output_dir="results",
        name="gan-mnist",
    )
    gan.train(train_loader=train_loader, test_loader=test_loader)
    z = torch.randn(64, z_dim).to(device)
    gen_imgs = gan.generator(z)
    save_image(gen_imgs.data, "images/final.png", nrow=8, normalize=True)
    