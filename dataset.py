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
import numpy as np

def loadPickle(fileName):
    with open(fileName, mode="rb") as f:
        return pickle.load(f)

class TwitterDataset(Dataset):
    def __init__(self, split, use_account, image_transform=None, text_transform=None, data_dir='data'):
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

        print(f'Created {self.split} Dataset of Len: {len(self.annos)}')
        
    def __getitem__(self, idx):
        ann = self.annos[idx]
        image_file = os.path.join(self.data_dir, f'images/{ann["filename"]}')

        orig_text = ann['text']
        orig_img = Image.open(image_file).convert("RGB")

        img = self.image_transform[self.split](orig_img)
        # text = self.text_transform[self.split](orig_text)
        text = orig_text

        data = {'image': img, 'text': text, 'orig_img': orig_img, 'orig_text': orig_text ,'screen_name': ann['screen_name']}
        
        return data

    def __len__(self):
        return len(self.annos)


if __name__=='__main__':

    split = 'train'

    from collect_twitter_data.data_info import data_info
    use_account = data_info['animal']

    from img_transform import *
    
    dataset = TwitterDataset(split, use_account, image_transform=image_transform, data_dir='data')

    # print(dataset.__getitem__(0)['image'])

    for i in range(10):
        img = dataset.__getitem__(i)['image']
        img = unnorm(img)
        img = img.numpy().transpose(1,2,0)
        img = np.array(img*255, dtype='uint8')
        img = Image.fromarray(img)
        img.save(f"vis/sample_{i}.jpg") 
        

