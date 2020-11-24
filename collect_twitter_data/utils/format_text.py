import re
def format_text(text):
    """
    不要なテキストを削除するなどの前処理
    """

    text=re.sub(r'https?://[\w/:%#\$&\?\(\)~\.=\+\-…]+', "", text)
    text=re.sub('RT', "", text)
    text=re.sub('お気に入り', "", text)
    text=re.sub('まとめ', "", text)
    #text=re.sub(r'[!-~]', "", text)#半角記号,数字,英字
    #text=re.sub(r'[︰-＠]', "", text)#全角記号
    #text=re.sub('\n', " ", text)#改行文字

    return text