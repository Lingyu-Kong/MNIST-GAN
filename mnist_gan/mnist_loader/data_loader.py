import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def load_MNIST(data_path="./contents", batch_size=64):
    """
    Load the MNIST dataset and return the dataloaders.
    :param data_path: the path to save the dataset
    :param batch_size: the batch size of the dataloader
    """
    
    ## Define the transformation
    ## 1. ToTensor(): convert the image to PyTorch tensor
    ## 2. Normalize((0.5,), (0.5,)): normalize the image to [-1, 1]
    transform = transforms.Compose([transforms.Resize(32), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    
    ## Load or download the training dataset
    train_dataset = datasets.MNIST(root=data_path, train=True, transform=transform, download=True)
    
    ## Load or download the test dataset
    test_dataset = datasets.MNIST(root=data_path, train=False, transform=transform, download=True)
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_dataloader, test_dataloader
    
if __name__ == "__main__":
    train_loader, test_loader = load_MNIST()
    print("Number of data in the MNIST training dataloader: ", len(train_loader.dataset))
    print("Number of data in the MNIST test dataloader: ", len(test_loader.dataset))