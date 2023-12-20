import torch
from torchvision.utils import save_image
from mnist_gan.mnist_loader.data_loader import load_MNIST
from mnist_gan.models.net_G_D import MNIST_Generator_CNN, MNIST_Discriminator_CNN
from mnist_gan.gan.gan import GAN
import wandb


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
    z = z.view(z.size(0), z_dim, 1, 1)
    gen_imgs = gan.generator(z)
    save_image(gen_imgs.data, "images/final.png", nrow=8, normalize=True)