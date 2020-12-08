from sklearn.utils import shuffle
import pickle
from torch.autograd import Variable
import numpy as np
import random
import re
from collections import Counter
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from gensim.models import KeyedVectors
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader

PAD_TOKEN = "<PAD>"


class Dataset(object):
    def __init__(self, name, instances, label_names):
        self.name = name
        self.instances = instances
        self.label_names = label_names

    def __iter__(self):
        for instance in self.instances:
            yield instance

    def __len__(self):
        return len(self.instances)

    def get_instances(self, fold=None):
        if fold is None:
            return self.instances
        else:
            return [ins for ins in self.instances if ins.fold == fold]


class DatasetInstance(object):
    def __init__(self, text, label, fold):
        self.text = text
        self.label = label
        self.fold = fold


def generate_features(dataset, tokenizer, min_count, max_word_length, min_line_length):
    def create_numpy_sequence(source_sequence, length, dtype):
        ret = np.zeros(length, dtype=dtype)
        source_sequence = source_sequence[:length]
        ret[: len(source_sequence)] = source_sequence
        return ret

    word_counter = Counter()
    for instance in tqdm(dataset):
        sentence = instance.text.lower()
        tokenized = tokenizer.tokenize(sentence, return_str=True)
        tokenized_list = tokenized.split()
        word_counter.update(token for token in tokenized_list)

    words = [word for word, count in word_counter.items() if count >= min_count]
    word_vocab = {word: index for index, word in enumerate(words, 1)}
    word_vocab[PAD_TOKEN] = 0

    ret = dict(train=[], dev=[], test=[], word_vocab=word_vocab)
    ret_text = dict(train=[], dev=[], test=[])

    for fold in ("train", "dev", "test"):
        for instance in dataset.get_instances(fold):
            sentence = instance.text.lower()
            tokenized = tokenizer.tokenize(sentence, return_str=True)
            tokenized_list = tokenized.split()
            if len(tokenized_list) < min_line_length:
                continue
            word_ids = [
                word_vocab[token] for token in tokenized_list if token in word_vocab
            ]
            ret_text[fold].append(sentence)
            ret[fold].append(
                dict(
                    word_ids=create_numpy_sequence(word_ids, max_word_length, np.int),
                    label=instance.label,
                )
            )
    return ret, ret_text


def load_dataset(dataset_path, categories):
    instances = []
    categories_index = {t: i for i, t in enumerate(categories)}

    def read(mode, instances):
        x, y = [], []
        with open(dataset_path, "r", encoding="utf-8") as f:
            for line in f:
                if line[-1] == "\n":
                    line = line[:-1]
                try:
                    label = line.split("\t")[0]
                    text = line.split("\t")[1]
                    y.append(categories_index[label])
                    x.append(text)
                except:
                    pass
        x_test, y_test = x[-50:], y[-50:]
        x_dev, y_dev = x[-100:-50], y[-100:-50]
        x, y = x[:-100], y[:-100]

        instances += [
            DatasetInstance(text, label, "train") for (text, label) in zip(x, y)
        ]
        instances += [
            DatasetInstance(text, label, "dev") for (text, label) in zip(x_dev, y_dev)
        ]
        instances += [
            DatasetInstance(text, label, "test")
            for (text, label) in zip(x_test, y_test)
        ]

    read(None, instances)
    return Dataset("akuma", instances, categories)


def shuffle_samples(seed=1, *args):
    np.random.seed(seed)
    zipped = list(zip(*args))
    np.random.shuffle(zipped)
    shuffled = list(zip(*zipped))
    result = []
    for ar in shuffled:
        result.append(ar)
    #         result.append(np.asarray(ar))
    return result


def get_embedding(vec_path, word_vocab):
    embedding = KeyedVectors.load_word2vec_format(vec_path, binary=False)
    cnt = 0
    dim_size = len(embedding.wv["りんご"])
    word_embedding = np.random.uniform(
        low=-0.05, high=0.05, size=(len(word_vocab), dim_size)
    )
    word_embedding[0] = np.zeros(dim_size)
    for word, index in word_vocab.items():
        try:
            word_embedding[index] = embedding.wv[word]
            cnt += 1
        except KeyError:
            continue
    print(cnt)
    return word_embedding


def get_text_list(path):
    with open(path) as f:
        categories = [s.strip() for s in f.readlines()]
    return categories
