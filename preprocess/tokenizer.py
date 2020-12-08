# usage
# python3 tokenizer.py --lang ja --input sample/ja.txt --output sample/ja.tok

# requirement
# mojimoji
# kytea

from moses import MosesTokenizer
from japanese_tokenizer import JapaneseTokenizer
from tqdm import tqdm
import argparse


def tokenize(input_filename, output_filename, lang):
    with open(input_filename, "r") as f:
        seqs = f.readlines()

    if lang == "ja":
        tokenizer = JapaneseTokenizer(
            splitter="Kytea", model="/home/ubuntu/kytea/data/model.bin"
        )
    else:
        tokenizer = MosesTokenizer(lang=lang)
    tokenized_seqs = []

    for sentence in tqdm(seqs, desc="tokenizing..."):
        # 全て小文d字
        sentence = sentence.lower()
        tokenized = tokenizer.tokenize(sentence, return_str=True)
        tokenized_seqs.append(tokenized)

    with open(output_filename, "w") as f:
        f.writelines("\n".join(tokenized_seqs))

    print("Successfully tokenized!")


def main():
    parser = argparse.ArgumentParser(description="Tokenize corpora")
    parser.add_argument("--lang", help="language of corpus")
    parser.add_argument("--data_path", default="../data/", help="language of corpus")
    parser.add_argument("--input", help="input filename")
    parser.add_argument("--output", help="output filename")
    args = parser.parse_args()

    tokenize(input_filename=args.input, output_filename=args.output, lang=args.lang)


if __name__ == "__main__":
    main()
