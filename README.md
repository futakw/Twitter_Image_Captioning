# install
~~~
pip install -r requirements.txt
~~~


# データ収集

##### twitterのデータ
1. collect_twitter_data/data_info.pyに、収集したいアカウントを追加する
2. 以下を実行
```bash
cd collect_twitter_data
python collect_data.py
```

##### cocodataset
```bash
chmod +x download.sh
./download.sh
```

##### STAIR-captions

https://github.com/STAIR-Lab-CIT/STAIR-captions.git
からgit cloneして、tar.gzファイルを解凍
`stair_captions_v1.2_train.json`をdata以下にmv

# MeCabのインストール

https://qiita.com/ragzboned/items/834c0bc3caaa494fc906

に従ってローカル環境にインストール

`{local}/lib/mecab/dic/ipadic`

をmecab_dict_pathの引数で与える


# データの確認
```
python dataset.py
```

# 学習

##### 語彙の辞書を作成

```bash
python3 build_vocab.py --use_twitter 
```

##### リサイズ（いる？）
```bash
python3 resize.py
```

##### train
```bash
python3 train.py
```

##### further train with twitter corpus
```bash
python3 train.py --mode twitter --save_step 20 --batch_size 16 --do_further_train
```

# 検証
事前に、画像の重複をチェックする！！
```bash
cd collect_twitter_data
python check_duplicated_images.py
```

sample.pyの中でモデル、データセットを指定してから、
```bash
python3 sample.py
```
<!--
# データセット
dataset.pyにて作成。
preprocessのpythonファイルは画像用とテキスト用で分けて作成してdataset.pyで読み込むようにする。

-->

