import json
import os, glob
import csv
import pandas as pd
import pickle
from tqdm import tqdm
import time

import shutil
import requests
from requests_oauthlib import OAuth1Session #OAuthのライブラリの読み込み
import twitter

import config
from utils.format_text import format_text

print(twitter)

#認証
CK = config.CONSUMER_KEY
CS = config.CONSUMER_SECRET
AT = config.ACCESS_TOKEN
ATS = config.ACCESS_TOKEN_SECRET
api = twitter.Api(consumer_key=CK,
                      consumer_secret=CS,
                      access_token_key=AT,
                      access_token_secret=ATS)


class TweetCollector(object):

    def __init__(self, screen_name, save_data_dir, max_num=10000):
        self.screen_name = screen_name
        print(f'Initialized collector: {self.screen_name}')

        self.save_data_dir = save_data_dir
        self.max_num = max_num


    def collectTweets(self):
        f_statuses = api.GetUserTimeline(
            screen_name=self.screen_name, 
            include_rts=False, exclude_replies=True, #リツイートとリプライを除く
            count=self.max_num) #直近max_numのツイートをみる

        f_0 = f_statuses[1]
        print('\nSample Data: \n',f_0)
        print('\n\n')

        data = []
        texts = []
        same_tweet_n = 0
        for f_s in tqdm(f_statuses):
            media = f_s.media
            if media is not None: #メディアツイートである場合
                this_media = media[0] #複数ある場合は初め
                is_photo = True if this_media.type=='photo' else False
                if is_photo: # 画像ツイートである場合
                    text = format_text(f_s.text)
                    if text not in texts:
                        media_id = this_media.id
                        media_url = this_media.media_url
                        filename = f'{self.screen_name}_{media_id}.png'
                        # save images
                        save_path = os.path.join(self.save_data_dir, 'images', filename)
                        if self.save_img(media_url, save_path): #画像取得成功したら、追加
                            data.append({
                                'screen_name':self.screen_name, 'text': text, 'media_url':media_url, 'media_id':media_id,
                                'filename': filename,
                            }) 
                            texts.append(text)
                    elif text in texts:
                        same_tweet_n += 1
            if same_tweet_n == 200:
                print('This Account is a Bot. Ended collecting.')
                break

        self.data = data
        self.save_annos()
        print(f'Collected data. Length = {len(data)}.')

    def save_img(self, image_url, save_path):
        if os.path.exists(save_path) == False:
            r = requests.get(image_url, stream = True)
            if r.status_code == 200:
                r.raw.decode_content = True
                with open(save_path,'wb') as f:
                    shutil.copyfileobj(r.raw, f)
                return True
            else:
                print('Image Couldn\'t be retreived', filename)
                return False
        return True

    def save_annos(self):
        anno_path = os.path.join(self.save_data_dir, 'annos', f'{self.screen_name}.pickle')
        with open(anno_path,'wb') as f:
            pickle.dump(self.data, f)

    def get_tweet_data(self):
        return self.data


if __name__=='__main__':

    save_data_dir = '../data'
    os.makedirs(save_data_dir + '/images', exist_ok=True)
    os.makedirs(save_data_dir + '/annos', exist_ok=True)

    ##########　使うアカウント　############
    from data_info import data_info
    #####################################

    for k in data_info:
        print(k)
        screen_name_list = data_info[k]
        for screen_name in screen_name_list:
            C = TweetCollector(screen_name, save_data_dir)
            C.collectTweets()


