import nltk
import pickle
import argparse
from collections import Counter
from pycocotools2.coco import COCO
import os
from preprocess.japanese_tokenizer import JapaneseTokenizer


## python3 build_vocab.py --caption_path /home/smg/nishikawa/STAIR-captions/stair_captions_v1.2_train_tokenized.json --vocab_path ./data/vocab.pkl --use_twitter

def loadPickle(fileName):
    with open(fileName, mode="rb") as f:
        return pickle.load(f)

class Vocabulary(object):
    """Simple vocabulary wrapper."""

    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx["<unk>"]
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)


def build_vocab(json, threshold, use_twitter=False, mecab_dict_path=None):
    """Build a simple vocabulary wrapper."""
    text_tokenizer = JapaneseTokenizer(splitter="MeCab", model=mecab_dict_path)
    coco = COCO(json)
    counter = Counter()
    ids = coco.anns.keys()
    for i, id in enumerate(ids):
        caption = str(coco.anns[id]['caption'])
        # tokens = nltk.tokenize.word_tokenize(caption.lower())
        tokens = text_tokenizer.tokenize(caption, return_str=True).split()
        counter.update(tokens)

        if (i + 1) % 1000 == 0:
            print("[{}/{}] Tokenized the captions.".format(i + 1, len(ids)))

    if use_twitter == True:
        # TODO 読み込み方
        from collect_twitter_data.data_info import data_info

        use_account = data_info["animal"]
        print("start build vocab for twitter corpus")
        annos = []
        for user in use_account:
            ann_path = os.path.join("data", f"annos/{user}.pickle")
            ann = loadPickle(ann_path)
            annos += ann

        for ann in annos:
            tokens = text_tokenizer.tokenize(ann["text"], return_str=True).split()
            counter.update(tokens)

    # If the word frequency is less than 'threshold', then the word is discarded.
    words = [word for word, cnt in counter.items() if cnt >= threshold]

    # Create a vocab wrapper and add some special tokens.
    vocab = Vocabulary()
    vocab.add_word("<pad>")
    vocab.add_word("<start>")
    vocab.add_word("<end>")
    vocab.add_word("<unk>")

    # Add the words to the vocabulary.
    for i, word in enumerate(words):
        vocab.add_word(word)
    return vocab


def main(args):
    vocab = build_vocab(
        json=args.caption_path, threshold=args.threshold, use_twitter=args.use_twitter, mecab_dict_path=args.mecab_dict_path
    )
    vocab_path = args.vocab_path
    with open(vocab_path, "wb") as f:
        pickle.dump(vocab, f)
    print("Total vocabulary size: {}".format(len(vocab)))
    print("Saved the vocabulary wrapper to '{}'".format(vocab_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--caption_path",
        type=str,
        # default="data/annotations/captions_train2014.json",
        default="data/stair_captions_v1.2_train.json",
        help="path for train annotation file",
    )
    parser.add_argument(
        "--vocab_path",
        type=str,
        default="./data/vocab.pkl",
        help="path for saving vocabulary wrapper",
    )
    parser.add_argument(
        "--threshold", type=int, default=4, help="minimum word count threshold"
    )

    parser.add_argument("--use_twitter", action="store_true")
    parser.add_argument(
        "--mecab_dict_path", default="/home/smg/nishikawa/src/lib/mecab/dic/ipadic"
    )
    args = parser.parse_args()
    main(args)
