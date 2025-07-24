import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np

def get_data_loaders(train_dir, valid_dir, test_dir, img_width=224, img_height=224, batch_size=32, seed=None):
    #Define transformations
    train_transform = transforms.Compose([
        transforms.Resize((img_width, img_height)),
        transforms.CenterCrop(img_width),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize((img_width, img_height)),
        transforms.CenterCrop(img_width),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    #Load datasets
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    valid_dataset = datasets.ImageFolder(valid_dir, transform=val_test_transform)
    test_dataset = datasets.ImageFolder(test_dir, transform=val_test_transform)

    #Create data loaders
    
    if seed == None:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    else:
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,  
            generator=torch.Generator().manual_seed(seed),   #Add deterministic generator
            worker_init_fn=lambda worker_id: np.random.seed(seed + worker_id)
        )

    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, valid_loader, test_loader