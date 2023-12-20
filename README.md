# Generative Adversial Networks

A generative adversarial network (GAN) example implemented on the MNIST dataset.

## Env Setup

```
conda create -n gan_mnist python=3.10 -y
conda activate gan_mnist
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# Network error, try this: pip install torch torchvision torchaudio -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install matplotlib
pip install numpy
pip install tqdm
pip install wandb
```