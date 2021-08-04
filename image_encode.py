import torch
import numpy as np
import clip
from PIL import Image
import json
from tqdm import tqdm, trange
import os

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

with open('./data/annotations/captions_train2014.json') as index_file:
    index = json.load(index_file)

image_num = len(index['images'])
image_encode = {}

for i in trange(image_num):
    image_info = index['images'][i]
    image_path = os.path.join('./data/train2014', image_info['file_name'])
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image)
        image_encode[int(image_info['file_name'][-10:-4])] = image_features.squeeze().detach().cpu().numpy()

np.save('./data/image_encode.npy', image_encode)
