import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from mnist_gan.mnist_loader.data_loader import load_MNIST

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.zeros_(m.bias)

class General_CNN(nn.Module):
    """
    A general multi level CNN model
    """
    def __init__(
        self,
        input_channel: int,
        conv_channels: List[int],
        kernel_sizes: List[int],
        strides: List[int] = None,
        paddings: List[int] = None,
        pooling_sizes: List[int] = None,
        pooling_types: List[str] = None,
        batch_norm: bool = True,
        bias: bool = True,
    ):
        super().__init__()
        
        ## Check and Set the parameters
        assert len(conv_channels) == len(kernel_sizes), "The number of conv channels and kernel sizes should be the same."
        if strides is None:
            strides = [1] * len(conv_channels)
            print("The strides are set to 1 by default.")
        elif isinstance(strides, int):
            strides = [strides] * len(conv_channels)
        else:
            assert len(conv_channels) == len(strides), "The number of conv channels and strides should be the same."
        if paddings is None:
            paddings = [0] * len(conv_channels)
            print("The paddings are set to 0 by default.")
        elif isinstance(paddings, int):
            paddings = [paddings] * len(conv_channels)
        else:
            assert len(conv_channels) == len(paddings), "The number of conv channels and paddings should be the same."
        if pooling_sizes is None:
            pooling_sizes = [2] * len(conv_channels)
            print("The pooling sizes are set to 2 by default.")
        elif isinstance(pooling_sizes, int):
            pooling_sizes = [pooling_sizes] * len(conv_channels)
        else:
            assert len(conv_channels) == len(pooling_sizes), "The number of conv channels and pooling sizes should be the same."
        if pooling_types is None:
            pooling_types = ["max"] * len(conv_channels)
            print("The pooling types are set to \"max\" by default.")
        elif isinstance(pooling_types, str):
            pooling_types = [pooling_types] * len(conv_channels)
        else:
            assert len(conv_channels) == len(pooling_types), "The number of conv channels and pooling types should be the same."
            
        ## Construct the layers
        layers = []
        in_channel = input_channel
        for conv_channel, kernel_size, stride, padding, pooling_type, pooling_size in zip(conv_channels, kernel_sizes, strides, paddings, pooling_types, pooling_sizes):
            ## conv2d -> pooling -> batchnorm -> relu
            layers.append(nn.Conv2d(in_channel, conv_channel, kernel_size, stride, padding, bias=bias))
            if pooling_type == "max":
                layers.append(nn.MaxPool2d(pooling_size))
            elif pooling_type == "avg":
                layers.append(nn.AvgPool2d(pooling_size))
            elif pooling_type == "nan":
                pass
            else:
                raise NotImplementedError("The pooling type is not implemented.")
            if batch_norm:
                layers.append(nn.BatchNorm2d(conv_channel))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            in_channel = conv_channel
        layers.append(nn.Flatten())
        layers = nn.Sequential(*layers)
        self.layers = layers
        
    def forward(self, x):
        return self.layers(x)
                    
class General_CNN_trans(nn.Module):
    """
    A general transposed convolutional neural network model
    """
    def __init__(
        self,
        input_channel: int,
        conv_channels: List[int],
        kernel_sizes: List[int],
        strides: List[int] = None,
        paddings: List[int] = None,
        batch_norm: bool = True,
    ):
        super().__init__()

        assert len(conv_channels) == len(kernel_sizes), "The number of conv channels and kernel sizes should be the same."
        if strides is None:
            strides = [1] * len(conv_channels)
        elif isinstance(strides, int):
            strides = [strides] * len(conv_channels)
        else:
            assert len(conv_channels) == len(strides), "The number of conv channels and strides should be the same."
        if paddings is None:
            paddings = [0] * len(conv_channels)
        elif isinstance(paddings, int):
            paddings = [paddings] * len(conv_channels)
        else:
            assert len(conv_channels) == len(paddings), "The number of conv channels and paddings should be the same."


        ## Construct the layers
        layers = []
        in_channel = input_channel
        for conv_channel, kernel_size, stride, padding in zip(conv_channels, kernel_sizes, strides, paddings):
            ## transposed conv2d -> batchnorm -> relu
            layers.append(nn.ConvTranspose2d(
                in_channels = in_channel, 
                out_channels = conv_channel, 
                kernel_size = kernel_size, 
                stride = stride, 
                padding = padding
            ))
            if batch_norm:
                layers.append(nn.BatchNorm2d(conv_channel))
            layers.append(nn.ReLU())
            in_channel = conv_channel
        layers.append(nn.Tanh())  # Tanh activation for pixel values between -1 and 1
        layers = nn.Sequential(*layers)
        self.layers = layers

    def forward(self, x):
        return self.layers(x)
                    
class MNIST_Classifier_CNN(nn.Module):
    """
    CNN model for MNIST Classification
    We add an MLP behind the General_CNN
    """
    def __init__(
        self,
        input_channel: int = 1,
        flatten_dim: int = 3136,
        output_dim: int = 10,
        conv_channels: List[int] = [32, 64],
        kernel_sizes: List[int] = [3, 3],
        strides: List[int] = [2, 2],
        paddings: List[int] = [1, 1],
        pooling_sizes: List[int] = [2, 2],
        pooling_types: List[str] = ["max", "max"],
        batch_norm: bool = True,
        hidden_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        ## Construct the layers
        self.cnn = General_CNN(
            input_channel=input_channel,
            conv_channels=conv_channels,
            kernel_sizes=kernel_sizes,
            strides=strides,
            paddings=paddings,
            pooling_sizes=pooling_sizes,
            pooling_types=pooling_types,
            batch_norm=batch_norm,
        )
        self.mlp = nn.Sequential(
            nn.Linear(flatten_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.Softmax(dim=1),
        )
        self.apply(init_weights)
        
    def forward(self, x):
        x = self.cnn(x)
        x = self.mlp(x)
        return x
    

if __name__=="__main__":
    train_loader, test_loader = load_MNIST(batch_size=64)
    model = MNIST_Classifier_CNN()
    print(model)

    ## Start training
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    epochs = 5
    for epoch in range(epochs):
        model.train()
        for i, (images, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            if (i+1) % 100 == 0:
                print("Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}".format(epoch+1, epochs, i+1, len(train_loader), loss.item()))
    ## Test the model
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted==labels).sum().item()
        print("Accuracy of the model on the 10000 test images: {} %".format(100 * correct / total))
        