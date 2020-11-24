import json
import os, glob
import csv
import pandas as pd
import pickle
from tqdm import tqdm

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


    def collectTweet(self):
        f_statuses = api.GetUserTimeline(
            screen_name=self.screen_name, 
            include_rts=False, exclude_replies=True, #リツイートとリプライを除く
            count=self.max_num) #直近200のツイートをみる

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
                        data.append({'screen_name':self.screen_name, 'text': text, 'media_url':media_url}) 
                        texts.append(text)
                    elif text in texts:
                        same_tweet_n += 1
            if same_tweet_n == 200:
                print('This Account is Bot. Ended collecting.')
                break

        self.data = data
        print(f'Collected data. Length = {len(data)}.')


    def save_data(self):
        invalid = 0
        for idx, d in tqdm(enumerate(self.data)):
            image_url = d['media_url']
            filename = f"{d['screen_name']}_{str(idx)}.png"
            d['filename'] = filename

            save_path = os.path.join(self.save_data_dir, 'images', filename)
            if os.path.exists(save_path) == False:
                r = requests.get(image_url, stream = True)
                if r.status_code == 200:
                    # Set decode_content value to True, otherwise the downloaded image file's size will be zero.
                    r.raw.decode_content = True
                    
                    # Open a local file with wb ( write binary ) permission.
                    with open(save_path,'wb') as f:
                        shutil.copyfileobj(r.raw, f)
                                
                    # print('Image sucessfully Downloaded: ',filename)
                else:
                    invalid += 1    
                    print('Image Couldn\'t be retreived', filename)
            # else:
            #     print('Image already exists: ',filename)

        # print(self.data)

        anno_path = os.path.join(self.save_data_dir, 'annos', f'{self.screen_name}.pickle')
        with open(anno_path,'wb') as f:
            pickle.dump(self.data, f)

        print(f'Saved Data. Len: {len(self.data)-invalid}')

    def get_tweet_data(self):
        return self.data


if __name__=='__main__':

    screen_name = 'mofumofu_cn'
    save_data_dir = 'data'

    C = TweetCollector(screen_name, save_data_dir)
    C.collectTweet()
    C.save_data()


