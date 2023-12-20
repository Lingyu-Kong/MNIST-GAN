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

## Loss Function

Here we used the simplest loss function with BCE(binary cross entropy)

$$
\mathcal{L}_{\text{D}} = -\frac{1}{N} \sum_{i=1}^{N} \left[ \log(D(x_i)) + \log(1 - D(G(z_i))) \right]
$$

$$
\mathcal{L}_{\text{G}} = -\frac{1}{N} \sum_{i=1}^{N} \log(D(G(z_i)))
$$