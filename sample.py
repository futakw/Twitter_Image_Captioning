import torch
import matplotlib.pyplot as plt
import numpy as np 
import argparse
import pickle 
import os
import glob
import cv2
from torchvision import transforms 
from build_vocab import Vocabulary
from model import EncoderCNN, DecoderRNN
from PIL import Image
# for unicode error
import io,sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

#  python sample.py --image='png/example.png' 
# python3 sample.py --image='png/example.png' --vocab_path data/vocab_ja.pkl --encoder_path models/encoder-5-20.twitter.ckpt --decoder_path models/decoder-5-20.twitter.ckpt

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def loadPickle(fileName):
    with open(fileName, mode="rb") as f:
        return pickle.load(f)

def load_image(image_path, transform=None):
    image = Image.open(image_path).convert('RGB')
    image = image.resize([224, 224], Image.LANCZOS)
    
    if transform is not None:
        image = transform(image).unsqueeze(0)
    
    return image

from PIL import ImageFont, ImageDraw, Image
def puttext(cv_image, text, point=(10,10), font_path='Osaka.ttc', font_size=20, color=(0,255,0)):
    font = ImageFont.truetype(font_path, font_size)

    cv_rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(cv_rgb_image)

    draw = ImageDraw.Draw(pil_image)
    draw.text(point, text, fill=color, font=font)

    cv_rgb_result_image = np.asarray(pil_image)
    cv_bgr_result_image = cv2.cvtColor(cv_rgb_result_image, cv2.COLOR_RGB2BGR)

    return cv_bgr_result_image

def resize_square_pad(im, desired_size=500):
    old_size = im.shape[:2] # old_size is in (height, width) format

    ratio = float(desired_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])

    # new_size should be in (width, height) format
    im = cv2.resize(im, (new_size[1], new_size[0]))

    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)

    color = [0, 0, 0]
    new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,
        value=color)

    return new_im


def main(args):
    # Image preprocessing
    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))])
    
    # Load vocabulary wrapper
    print('Loading vocab  ')
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    # Build models
    print('Building models')
    encoder = EncoderCNN(args.embed_size).eval()  # eval mode (batchnorm uses moving mean/variance)
    decoder = DecoderRNN(args.embed_size, args.hidden_size, len(vocab), args.num_layers)
    encoder = encoder.to(device)
    decoder = decoder.to(device)

    # Load the trained model parameters
    print('Loading models')
    encoder.load_state_dict(torch.load(args.encoder_path))
    decoder.load_state_dict(torch.load(args.decoder_path))


    frame_rate = 5.0 # フレームレート
    w, h = 500, 800
    size = (w, h) # 動画の画面サイズ
    fmt = cv2.VideoWriter_fourcc('m', 'p', '4', 'v') # ファイル形式(ここではmp4)
    writer = cv2.VideoWriter(args.out_file, fmt, frame_rate, size) # ライター作成

    accounts = [p.split('/')[-1].split('.')[0] for p in glob.glob('data/annos/*')]
    for user in accounts:
        print(user)
        ann_path = os.path.join(f"data/annos/{user}.pickle")
        annos = loadPickle(ann_path)

        for i in range(10): # 10まい
            ann = annos[i]

            image_path =  f'data/images/{ann["filename"]}'
            orig_text = ann['text']

            image = load_image(image_path, transform)
            # image = load_image(args.image, transform)
            image_tensor = image.to(device)
            
            # Generate an caption from the image
            feature = encoder(image_tensor)
            sampled_ids = decoder.sample(feature)
            sampled_ids = sampled_ids[0].cpu().numpy()          # (1, max_seq_length) -> (max_seq_length)
            
            # Convert word_ids to words
            sampled_caption = []
            for word_id in sampled_ids:
                word = vocab.idx2word[word_id]
                sampled_caption.append(word)
                if word == '<end>':
                    break
            sentence = ' '.join(sampled_caption)
            
            # Print out the image and the generated caption
            # print(type(sentence))
            print(sentence)
            # print(sentence.encode('utf_8'))
            # image = Image.open(args.image)
            # plt.imshow(np.asarray(image))

            img = cv2.imread(image_path)
            img = resize_square_pad(img)

            frame = np.zeros((h,w,3)).astype('uint8')
            frame[h-w:, :, :] = img
            frame = cv2.putText(frame, image_path, (10, h-20), cv2.FONT_HERSHEY_SIMPLEX ,  
                   0.5, (0,255,0), 1, cv2.LINE_AA) 

            n = 20
            s = 'GT: \n'
            for i in range(len(orig_text)//n + 1):
                s += orig_text[n*i:n*(i+1)] + '\n'
            frame = puttext(frame, s, point=(15,20*(i+1)), color=(255,255,255))

            s = 'Result: \n' 
            res = sentence.replace('<start>','').replace('<end>','').replace(' ','')
            for i in range(len(res)//n + 1):
                s += res[n*i:n*(i+1)] + '\n'
            frame = puttext(frame, s, point=(15,(h-w)/2+20*(i+1)), color=(255,0,0))

            writer.write(frame)
    
    writer.release()
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--image', type=str, required=True, help='input image for generating caption')

    mode = 'coco'
    if mode == 'coco':
        parser.add_argument('--encoder_path', type=str, default='models/encoder-5-3000.coco.ckpt', help='path for trained encoder')
        parser.add_argument('--decoder_path', type=str, default='models/decoder-5-3000.coco.ckpt', help='path for trained decoder')
    elif mode == 'twitter':
        parser.add_argument('--encoder_path', type=str, default='models/encoder-5-20.twitter.ckpt', help='path for trained encoder')
        parser.add_argument('--decoder_path', type=str, default='models/decoder-5-20.twitter.ckpt', help='path for trained decoder')

    parser.add_argument('--vocab_path', type=str, default='data/vocab.pkl', help='path for vocabulary wrapper')
    
    # Model parameters (should be same as paramters in train.py)
    parser.add_argument('--embed_size', type=int , default=256, help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int , default=512, help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int , default=1, help='number of layers in lstm')
    args = parser.parse_args()

    args.data_path = 'data/images'
    args.out_file = f'results_{mode}.mp4'
    main(args)
