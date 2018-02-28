import os
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST, CIFAR10, ImageFolder
from torchvision import transforms


def data_loader(args):
    root = args.data_dir
    dataset = args.dataset
    batch_size = args.batch_size
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
    ])

    if dataset == 'MNIST':
        data = MNIST(root=root,
                     train=True,
                     transform=transform,
                     download=True)
        loader = DataLoader(data,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=4,
                            drop_last=False)

    elif dataset == 'CIFAR10':
        data = CIFAR10(root=root,
                       train=True,
                       transform=transform,
                       download=True)
        loader = DataLoader(data,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=16,
                            drop_last=False)

    elif dataset == 'CIFAR10_test':
        data = CIFAR10(root=root,
                       train=False,
                       transform=transform,
                       download=True)
        loader = DataLoader(data,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=4,
                            drop_last=False)



    return data, loader
