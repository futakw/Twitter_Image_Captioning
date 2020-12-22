import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torch.utils.data as data
from torch.utils.data import Dataset, DataLoader
import os
import pickle
import numpy as np
import nltk
from PIL import Image
from build_vocab import Vocabulary
import os, glob
import numpy as np

# from dataset import TwitterDataset
from preprocess.japanese_tokenizer import JapaneseTokenizer

from pycocotools2.coco import COCO


class CocoDataset(data.Dataset):
    """COCO Custom Dataset compatible with torch.utils.data.DataLoader."""

    def __init__(self, root, json, vocab, transform=None, text_tokenizer=None):
        """Set the path for images, captions and vocabulary wrapper.

        Args:
            root: image directory.
            json: coco annotation file path.
            vocab: vocabulary wrapper.
            transform: image transformer.
        """
        self.root = root
        self.coco = COCO(json)
        self.ids = list(self.coco.anns.keys())
        self.vocab = vocab
        self.transform = transform
        self.text_tokenizer = text_tokenizer

    def __getitem__(self, index):
        """Returns one data pair (image and caption)."""
        coco = self.coco
        vocab = self.vocab
        ann_id = self.ids[index]
        caption = coco.anns[ann_id]["caption"]
        # caption = coco.anns[ann_id]["tokenized_caption"]
        img_id = coco.anns[ann_id]["image_id"]
        path = coco.loadImgs(img_id)[0]["file_name"]

        image = Image.open(os.path.join(self.root, path)).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        # Convert caption (string) to word ids.
        # tokens = nltk.tokenize.word_tokenize(str(caption).lower())
        # tokens = str(caption).split()
        tokens = self.text_tokenizer.tokenize(caption, return_str=True).split()
        caption = []
        caption.append(vocab("<start>"))
        caption.extend([vocab(token) for token in tokens])
        caption.append(vocab("<end>"))
        target = torch.Tensor(caption)
        return image, target

    def __len__(self):
        return len(self.ids)


def loadPickle(fileName):
    with open(fileName, mode="rb") as f:
        return pickle.load(f)


class TwitterDataset(Dataset):
    def __init__(
        self,
        split,
        use_account,
        vocab,
        image_transform=None,
        text_tokenizer=None,
        data_dir="data",
    ):
        self.split = split
        self.image_transform = image_transform
        self.text_tokenizer = text_tokenizer
        self.vocab = vocab

        self.data_dir = data_dir
        self.imgs = glob.glob(os.path.join(self.data_dir, "images/*.png"))

        # 使うtwitterアカウントのアノテーションだけ読み込む
        self.annos = []
        for user in use_account:
            ann_path = os.path.join(self.data_dir, f"annos/{user}.pickle")
            ann = loadPickle(ann_path)
            if self.split == 'train':
                self.annos += ann[:-20]
            elif self.split == 'val':
                self.annos += ann[-20:]

        print(f"Created {self.split} Dataset of Len: {len(self.annos)}")

    def __getitem__(self, idx):
        ann = self.annos[idx]
        image_file = os.path.join(self.data_dir, f'images/{ann["filename"]}')

        orig_text = ann["text"]
        orig_img = Image.open(image_file).convert("RGB")

        img = self.image_transform[self.split](orig_img)
        tokens = self.text_tokenizer.tokenize(orig_text, return_str=True).split()
        caption = []
        caption.append(self.vocab("<start>"))
        caption.extend([self.vocab(token) for token in tokens])
        caption.append(self.vocab("<end>"))
        target = torch.Tensor(caption)

        #         data = {'image': img, 'text': text, 'orig_img': orig_img, 'orig_text': orig_text ,'screen_name': ann['screen_name']}

        return img, target

    def __len__(self):
        return len(self.annos)


def collate_fn(data):
    """Creates mini-batch tensors from the list of tuples (image, caption).

    We should build custom collate_fn rather than using default collate_fn,
    because merging caption (including padding) is not supported in default.

    Args:
        data: list of tuple (image, caption).
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.

    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length (descending order).
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions = zip(*data)

    # Merge images (from tuple of 3D tensor to 4D tensor).
    images = torch.stack(images, 0)

    # Merge captions (from tuple of 1D tensor to 2D tensor).
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]
    return images, targets, lengths


def get_loader(
    root,
    json,
    vocab,
    transform,
    batch_size,
    shuffle,
    num_workers,
    mode,
    mecab_dict_path,
):
    """Returns torch.utils.data.DataLoader for dataset."""

    text_tokenizer = JapaneseTokenizer(splitter="MeCab", model=mecab_dict_path)
    # COCO caption dataset
    if mode == "coco":
        dataset = CocoDataset(
            root=root,
            json=json,
            vocab=vocab,
            transform=transform,
            text_tokenizer=text_tokenizer,
        )

    if mode == "twitter":
        # TODO 読み込み方
        from collect_twitter_data.data_info import data_info

        use_account = data_info["animal"]
        dataset = TwitterDataset(
            "train",
            use_account,
            vocab,
            image_transform=transform,
            text_tokenizer=text_tokenizer,
            data_dir="data",
        )

    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    return data_loader
