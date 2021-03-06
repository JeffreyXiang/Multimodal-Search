import torch
import numpy as np
import clip
from PIL import Image
import json
from tqdm import tqdm, trange
import os

device = "cuda:0" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("data/fine_tune/005000.pt", device=device)

with open('./data/annotations/captions_val2014.json') as index_file:
    index = json.load(index_file)

text_num = len(index['annotations'])
text_encode = {}

for i in trange(text_num):
    text_info = index['annotations'][i]
    text = clip.tokenize([text_info['caption']]).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text)
        if int(text_info['image_id']) in text_encode:
            text_encode[int(text_info['image_id'])].append(text_features.squeeze().detach().cpu().numpy())
        else:
            text_encode[int(text_info['image_id'])] = [text_features.squeeze().detach().cpu().numpy()]
    if i > 10000:
        break

np.save('./data/text_encode_finetuned.npy', text_encode)
