import torch
import torchvision
import torchvision.transforms as transforms
import os
import argparse
import shutil
try:
    from generate_txt import generate_txt
except:
    from data.generate_txt import generate_txt

    
def generate_dataloader(args):
    # Data loading code
    # the dataloader for the target dataset.
    train_dir_source = os.path.join('data/M3SDA', args.source_domain)
    train_dir_target = os.path.join('data/M3SDA', args.target_domain)
    val_dir_target = os.path.join('data/M3SDA', args.target_domain)

    
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])  # the mean and std of ImageNet
                                     # 后面可以考虑重新计算Office31数据集的均值跟方差然后再填进来
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
        ])
    train_dataset_source = torchvision.datasets.ImageFolder(train_dir_source, transform=transform_train)
    train_loader_source = torch.utils.data.DataLoader(
        train_dataset_source, batch_size=args.batch_size, shuffle=True, 
        num_workers=args.workers, pin_memory=True, sampler=None, drop_last=True)
    train_dataset_target = torchvision.datasets.ImageFolder(train_dir_target, transform=transform_train)
    train_loader_target = torch.utils.data.DataLoader(
        train_dataset_target, batch_size=args.batch_size, shuffle=True, 
        num_workers=args.workers, pin_memory=True, sampler=None, drop_last=True)
    transform_val = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])
    val_dataset_target = torchvision.datasets.ImageFolder(val_dir_target, transform=transform_val)
    val_loader_target = torch.utils.data.DataLoader(
        val_dataset_target, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=None)

    return train_loader_source, train_loader_target, val_loader_target
