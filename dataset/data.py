import pickle

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.utils
from torchvision import models
import torchvision.datasets as datasets

import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

from PIL import Image
import os, glob

def loadPickle(fileName):
    with open(fileName, mode="rb") as f:
        return pickle.load(f)

class TwitterDataset(Dataset):
    def __init__(self, split, use_account, image_transform=None, text_transform=None, data_dir='collect_data/data'):
        self.split = split
        self.image_transform = image_transform
        self.text_transform = text_transform

        self.data_dir =  data_dir
        self.imgs = glob.glob(os.path.join(self.data_dir, "images/*.png"))

        # 使うtwitterアカウントのアノテーションだけ読み込む
        self.annos = []
        for user in use_account:
            ann_path = os.path.join(self.data_dir, f"annos/{user}.pickle")
            ann = loadPickle(ann_path)
            self.annos += ann
        
    def __getitem__(self, idx):
        ann = self.annos[idx]
        image_file = os.path.join(self.data_dir, f'images/{ann["filename"]}')
        text = ann['text']

        img = Image.open(image_file).convert("RGB")
        if self.image_transform:
            img = self.image_transform(img)

        if self.text_transform:
            text = self.text_transform(text)

        data = {'image': img, 'text': text, 'screen_name': ann['screen_name']}
        
        return data

    def __len__(self):
        return len(self.annos)


if __name__=='__main__':

    split = 'test'
    use_account = ['mofumofu_cn']
    
    
    dataset = TwitterDataset(split, use_account ,data_dir='../collect_data/data')


    print(dataset.__getitem__(0))

