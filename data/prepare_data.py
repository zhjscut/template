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

"""
directory
---------
classes

AverageMeter: Computes and stores the average and current value
My_dataset: Customized dataset used in torch class Dataset
My_dataset_v2: The 2nd version of My_dataset
My_dataset_v3: The 3nd version of My_dataset

functions

default_loader: Default data loader used in torch class DataLoader
default_loader_v2: The 2nd version of default_loader
generate_dataloader: Generate dataloader for training
"""
    
def default_loader(path):
#     # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)        
#     with open(path, 'rb') as f:
#         img = Image.open(f)
#         img_PIL = img.convert('RGB')
    img_PIL = Image.open(path)
    # 到时记得transform是要做标准化的，可能要补上一些语句
#     transform = transforms.Compose([
#         transforms.ToTensor(),
# #         normalize
#     ])  
    # 到时记得transform是要做标准化的，可能要补上一些语句
    transform = transforms.Compose([
        transforms.Resize(320),
        transforms.RandomCrop((299,299)),
        transforms.ToTensor(),
#         normalize
    ])      
    img_tensor = transform(img_PIL)
    if img_tensor.size(0) == 1: # expand the grayscale image to RGB image
        img_tensor = img_tensor.repeat(3, 1, 1)

    return img_tensor

def default_loader_v2(path):
#     # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)        
    with open(path, 'rb') as f:
        img = Image.open(f)
        img_PIL = img.convert('RGB')

    return img_PIL

class My_dataset_v1(torch.utils.data.Dataset):
    #     def __init__(self, img_paths, label_paths, loader=default_loader_valid):
    #     def __init__(self, s_filename, loader=default_loader_valid):
        def __init__(self, source_filename, target_filename):
    #         self.loader = loader
            self.source = np.load(source_filename)
            self.target = np.load(target_filename)

        def __getitem__(self, index):
            src = self.source[index, :].copy()
            tgt = self.target[index, :].copy()
            return src, tgt

        def __len__(self):
    #         return len(self.img_paths)
            return self.source.shape[0]

class My_dataset_v1_1(torch.utils.data.Dataset):
    """File is in .npy format, only contain input images."""
    def __init__(self, s_train_filename):
        self.s = np.load(s_train_filename)

    def __getitem__(self, index):
        image = self.s[index, :].copy()
        label = self.s[index, :].copy()

        return image, label

    def __len__(self):
        return self.s.shape[0]
    
class My_dataset_v2(torch.utils.data.Dataset):
    def __init__(self, source_folder, target_folder, fmt='jpg', loader=default_loader):
        self.loader = loader
        self.images_path_src = glob.glob(source_folder + '/*.' + fmt)
        self.images_path_tgt = glob.glob(target_folder + '/*.' + fmt)

    def __getitem__(self, index):
        src = self.loader(self.images_path_src[index])
        tgt = self.loader(self.images_path_tgt[index])
        return src, tgt

    def __len__(self):
        return len(self.images_path_src)

class My_dataset_v3(torch.utils.data.Dataset):
    def __init__(self, folder, fmt='jpg', loader=default_loader, transform=None):
        self.loader = loader
        self.images_path = glob.glob(folder + '/*.' + fmt)
        self.transform = transform
        
    def __getitem__(self, index):
        img = self.loader(self.images_path[index])
        if self.transform is not None:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.images_path)
    
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
    train_dataset_target = My_dataset(train_dir_target, transform=transform_train)
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
