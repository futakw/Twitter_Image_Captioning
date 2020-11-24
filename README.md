# install
~~~
pip install requests requests_oauthlib
pip install python-twitter
pip install imgaug
~~~


# データ収集
1. collect_twitter_data/data_info.pyに、収集したいアカウントを追加する
2. 以下を実行
~~~
cd collect_twitter_data
python collect_data.py
~~~
 
# データセット
dataset.pyにて作成。
preprocessのpythonファイルは画像用とテキスト用で分けて作成してdataset.pyで読み込むようにする。



