import os
import torch
from torchvision import datasets, transforms
# mean = [0.4914, 0.4822, 0.4465]
# std = [0.2471, 0.2435, 0.2616]
def get_loader(args):
    train_transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.RandomCrop(args.img_size, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandAugment(),
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465],
                             [0.2470, 0.2435, 0.2616])
    ])
    train_dataset = datasets.CIFAR10(
        root='/data1/nliu/2025_homework/Pro_002_ViT_CIFAR10/data/',
        train=True,
        download=True,
        transform=train_transform
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        drop_last=True
    )

    test_transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465],
                             [0.2470, 0.2435, 0.2616])
    ])
    test_dataset = datasets.CIFAR10(
        root='/data1/nliu/2025_homework/Pro_002_ViT_CIFAR10/data/',
        train=False,
        download=True,
        transform=test_transform
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        drop_last=False
    )
    return train_loader, test_loader, len(train_dataset), len(test_dataset)
