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

from PIL import Image, ImageDraw, ImageFont
import os, glob
import numpy as np
import cv2

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
            print(f'{user}: {len(ann)}')
            self.annos += ann

        print(f'Created {self.split} Dataset of Len: {len(self.annos)}')
        
    def __getitem__(self, idx):
        ann = self.annos[idx]
        image_file = os.path.join(self.data_dir, f'images/{ann["filename"]}')

        orig_text = ann['text']
        orig_img = Image.open(image_file).convert("RGB")
        # orig_img = cv2.imread(image_file)[:,:,[2,1,0]]

        img = self.image_transform[self.split](orig_img)
        # text = self.text_transform[self.split](orig_text)
        text = orig_text

        data = {'image': img, 'text': text, 'orig_img': orig_img, 'orig_text': orig_text ,'screen_name': ann['screen_name'], 'image_file': image_file}
        
        return data

    def __len__(self):
        return len(self.annos)


if __name__=='__main__':

    split = 'val'

    from collect_twitter_data.data_info import data_info
    use_account = data_info['animal']

    from img_transform import *
    
    dataset = TwitterDataset(split, use_account, image_transform=image_transform, data_dir='data')

    # print(dataset.__getitem__(0)['image'])
    # annos = dataset.annos

    text2file = {}

    texts = ''
    
    for i in range(len(dataset)):
        if i % 100 == 0:
            print(i)
        # ann = annos[i]
        # image_file = os.path.join(dataset.data_dir, f'images/{ann["filename"]}')

        data = dataset.__getitem__(i)

        if i % 50 == 0:
            img = data['image']
            img = unnorm(img)
            img = img.numpy().transpose(1,2,0)
            img = np.clip(img, 0, 1)
            img = np.array(img*255, dtype='uint8')

            img = Image.fromarray(img)
            filename = f"vis/{split}_{i}.jpg"
            img.save(filename)
        

        text = data['orig_text']
        image_file = data['image_file']

        texts += f'{image_file} {text}\n'

        text2file[text] = image_file


    f = open(f'vis/{split}_captions.txt', 'w')
    f.write(texts)
    f.close()
    
    with open('text2file.pickle', 'wb') as f:
        pickle.dump(text2file, f)
        

