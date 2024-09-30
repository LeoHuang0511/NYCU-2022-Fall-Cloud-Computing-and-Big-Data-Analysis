import glob
import os

import torchvision
from torch.utils.data import Dataset, Subset
from PIL import Image
import sys


class TrainDataset(Dataset):
    def __init__(self, transform, root='./data/unlabeled' ):
        self.paths = sorted(
            glob.glob(os.path.join(root, "*.jpg")))
        print("number of training data: ", len(self.paths))
        self.transform = transform
        

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        #label = self.labels[idx]
        #frames = torchvision.io.read_video(self.paths[idx])[0]
        #frames = frames.permute(3, 0, 1, 2)  # THWC -> CTHW
        img = Image.open(self.paths[idx])
        img_1, img_2 = self.transform(img), self.transform(img)

        return img_1, img_2

class ValDataset(Dataset):
    def __init__(self, transform, root='./data/test',):
        self.transform = transform
        self.paths = sorted(glob.glob(os.path.join(root, "**/*.jpg"), recursive=True))
        print("number of validation data: ",len(self.paths))
        

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx])
        img1, img2 = self.transform(img),self.transform(img)
        return img1, img2  

class TestDataset(Dataset):
    def __init__(self, transform, root='./data/test',):
        self.transform = transform
        self.paths = sorted(glob.glob(os.path.join(root, "**/*.jpg"), recursive=True))
        print("number of testing data: ",len(self.paths))
        
        
        self.labels = [
            int(os.path.basename(os.path.dirname(path)))
            for path in self.paths
        ]
        
        self.num_classes = len(set(self.labels))
        print("number of classes: ",self.num_classes )
        print("length of label list", len(self.labels))
        

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        label = self.labels[idx]
        img = Image.open(self.paths[idx])
        #frames = torchvision.io.read_video(self.paths[idx])[0]
        #frames = frames.permute(3, 0, 1, 2)  # THWC -> CTHW
        img = self.transform(img)
        return img, label        

class UnlabeledDataset(Dataset):
    def __init__(self, transform, root='./data/unlabeled' ):
        self.paths = sorted(
            glob.glob(os.path.join(root, "*.jpg")))
        print("number of training data: ", len(self.paths))
        self.transform = transform
        

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        
        #print("file number: ",filename)
        if idx < len(self.paths)-1:
            filename = os.path.split(self.paths[idx])[1].strip('.jpg')
            next_file = os.path.split(self.paths[idx+1])[1].strip('.jpg')
            if int(filename)!= int(next_file)-1:
                sys.exit("The order of unlabeled data is wrong!")
        img = Image.open(self.paths[idx])
        img = self.transform(img)

        return img