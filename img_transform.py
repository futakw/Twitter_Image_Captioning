
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import PIL
from imgaug import augmenters as iaa
import imgaug as ia
import numpy as np

# https://colab.research.google.com/drive/109vu3F1LTzD1gdVV6cho9fKGx7lzbFll#scrollTo=8q8a2Ha9pnaz


im_size = (224, 224)
mean, std = [0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]
normalize = transforms.Normalize(mean = mean, std = std)


class Train_ImgAugTransform:
  def __init__(self):
    self.aug = iaa.Sequential([
        # affine
        iaa.PadToSquare(),
        iaa.Resize(im_size),
        iaa.Fliplr(0.5),
        iaa.Affine(rotate=(-20, 20), mode='symmetric'),
        
        # color
        iaa.Sometimes(0.5, iaa.GaussianBlur(sigma=(0, 3.0))),
        iaa.Sometimes(0.5, iaa.AdditiveGaussianNoise(scale=(0, 0.1*255))),
        iaa.AddToHueAndSaturation(value=(-10, 10), per_channel=True)
    ])
      
  def __call__(self, img):
    img = np.array(img)
    return self.aug.augment_image(img)


class Test_ImgAugTransform:
  def __init__(self):
    self.aug = iaa.Sequential([
        iaa.PadToSquare(),
        iaa.Resize(im_size),
    ])
      
  def __call__(self, img):
    img = np.array(img)
    return self.aug.augment_image(img)


image_transform =  {
    'train': transforms.Compose([
        Train_ImgAugTransform(),
        lambda x: PIL.Image.fromarray(x),
        transforms.ToTensor(),
        normalize
    ]),
    'val': transforms.Compose([
        Test_ImgAugTransform(),
        lambda x: PIL.Image.fromarray(x),
        transforms.ToTensor(),
        normalize
    ]),
    'test': transforms.Compose([
        Test_ImgAugTransform(),
        lambda x: PIL.Image.fromarray(x),
        transforms.ToTensor(),
        normalize
    ])
}


# 可視化用のunnormalize関数
class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor
unnorm = UnNormalize(mean, std)