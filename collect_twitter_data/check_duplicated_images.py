# https://news.mynavi.jp/article/zeropython-25/

import os, glob, hashlib
import pickle

# 重複ファイルがあるかどうかを調べるディレクトリ
target_dir = '../data/images'
body_dict = {}

# ファイルの内容を返す関数 --- (*1)
def get_body(fname):
    with open(fname, "rb") as f:
        return f.read()

duplicated_images_path = []

# ファイルの一覧を得て重複があるか調べる --- (*2)
files = glob.glob(target_dir + "/*")
for f in files:
    # ファイルを開いてハッシュ値を調べる --- (*3)
    body = get_body(f)
    v = hashlib.sha256(body).hexdigest()
    if v in body_dict: # 重複しているか --- (*4)
        f2 = body_dict[v]
        # 念のため実際に合致しているか調べる --- (*5)
        if body == get_body(f2):
            print(f, "==", f2)
            duplicated_images_path += [f, f2]
            # 実際に削除するなら以下のコメントを外す ---- (*5a)
            # os.remove(f)
    else:
        body_dict[v] = f # --- (*6)


duplicated_images_path = list(set(duplicated_images_path))
with open('duplicated_images_path.pickle', 'wb') as f:
    pickle.dump(duplicated_images_path, f)

print(len(duplicated_images_path))

print("ok")


