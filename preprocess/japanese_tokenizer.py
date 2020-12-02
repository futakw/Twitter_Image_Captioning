import unicodedata
from mojimoji import zen_to_han, han_to_zen
import MeCab


class JapaneseSplitter:
    def __init__(self, type, model=None):
        self.type = type
        if type == "Kytea":
            if model is None:
                splitter = Mykytea.Mykytea("-wsconst D")
            else:
                splitter = Mykytea.Mykytea(f"-wsconst D -model {model}")
        elif type == "MeCab":
            splitter = MeCab.Tagger(f"-Owakati -r/dev/null -d {model}")

        else:
            raise ValueError("Spliter type should be in ['MeCab' or 'Kytea']. ")

        self.splitter = splitter

    def split_text(self, text):
        if self.type == "MeCab":
            split_text = self.splitter.parse(text)
        elif self.type == "Kytea":
            split_text = " ".join(list(self.splitter.getWS(text)))
        else:
            raise ValueError
        return split_text


class JapaneseTokenizer:
    def __init__(
        self,
        splitter="MeCab",
        unicode=True,
        half2full=True,
        full2half=True,
        lower_case=True,
        model=None,
    ):
        self.split = splitter in ("MeCab", "Kytea")
        self.unicode = unicode
        self.half2full = half2full
        self.full2half = full2half
        self.lower_case = lower_case
        if splitter is not None:
            self.spliter = JapaneseSplitter(type=splitter, model=model)

    def tokenize(self, text, return_str=True):
        if self.unicode:
            text = self._normalize_unicode(text)

        if self.half2full:
            text = self._normalize_kana(text)

        if self.full2half:
            text = self._normalize_num_alphabet(text)

        if self.lower_case:
            text = self._lower_text(text)

        if self.split:
            text = self._split_words(text)

        return text.replace("\n", "")

    # ユニコード正規化
    def _normalize_unicode(self, text, form="NFKC"):
        return unicodedata.normalize(form, text)

    # 小文字
    def _lower_text(self, text):
        return text.lower()

    # 全角英数字を半角英数字へ
    def _normalize_num_alphabet(self, text):
        return zen_to_han(text, kana=False)

    # 半角カタカナを全角カタカナへ
    def _normalize_kana(self, text):
        return han_to_zen(text, digit=False, ascii=False)

    # 半角スペースで単語ごとに分割
    def _split_words(self, text):
        return self.spliter.split_text(text)


def main():
    text = "私は９００．０円をＡＢＣﾏｰﾄから盗みました。まじですみません。😉"
    tokenizer = JapaneseTokenizer(splitter="Kytea", model="~/kytea/data/model.bin")
    tokenized = tokenizer.tokenize(text)

    print(text)
    print(tokenized)


if __name__ == "__main__":
    main()
