import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch
import os

class CustomImageDatasets:
    def __init__(self,root='',batch_size=10,shuffle=True,num_workers=0,pin_memory=True):
        self.classes = sorted(os.listdir(os.path.join(root,'train')))
        #transforms
        self.train_transform = transforms.Compose([
                                transforms.Resize(224),
                                transforms.RandomHorizontalFlip(p=0.5),
                                transforms.RandomVerticalFlip(p=0.5),
                                transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
                                transforms.RandomRotation(degrees=(30, 70)),
                                transforms.ToTensor(),
                                transforms.Normalize(
                                mean=[0.5, 0.5, 0.5],
                                std=[0.5, 0.5, 0.5])])
        self.valid_transform = transforms.Compose([
                                transforms.Resize(224),
                                transforms.ToTensor(),
                                transforms.Normalize(
                                mean=[0.5, 0.5, 0.5],
                                std=[0.5, 0.5, 0.5])])

        self.train_loader,self.valid_loader = self.train_val_dataset(root,self.train_transform,self.valid_transform,
                                    batch_size,shuffle,num_workers,pin_memory)


    def train_val_dataset(self,root,train_transform,valid_transform,batch_size,shuffle,num_workers,pin_memory):

        train_dataset = datasets.ImageFolder(
            root=os.path.join(root,'train'),
            transform=train_transform)

        # validation dataset
        valid_dataset = datasets.ImageFolder(
            root=os.path.join(root,'valid'),
            transform=valid_transform)

        # training data loaders
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=shuffle,
            num_workers=num_workers, pin_memory=pin_memory)

        # validation data loaders
        valid_loader = DataLoader(
            valid_dataset, batch_size=batch_size, shuffle=shuffle,
            num_workers=num_workers, pin_memory=pin_memory)
        return train_loader,valid_loader

if __name__ == "__main__":
    dataset = CustomImageDatasets('/media/ngocthien/DATA/Python_basic/archive')
    print(len(dataset.train_loader.dataset))
    for batch, (X, y) in enumerate(dataset.train_loader):
        for idx, image in enumerate(X):
            image = torch.transpose(image,0,1)
            image = torch.transpose(image,1,2)
            # image = torch.transpose(image,0,2)
            print(y[idx].item())
            print(dataset.classes[y[idx].item()])
            plt.imshow(image)
            plt.show()
            
