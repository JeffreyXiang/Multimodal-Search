import torch
from torch.utils import data
from torch.utils.data import DataLoader, Dataset
import numpy as np
import clip
from PIL import Image
import json
from tqdm import tqdm, trange
import os
import random

class TEST(Dataset):
    def __init__(self, num, **kwargs):
        super().__init__()

        self.data = list(range(num))
        self.num = num

    def __len__(self):
        return self.num

    def __getitem__(self, index):
        return self.data[index], 0

class MSCOCO2014(Dataset):
    def __init__(self, dir, preprocess, type='train', **kwargs):
        super().__init__()

        annotation_filepath = os.path.join(dir, f'annotations/captions_{type}2014.json')
        with open(annotation_filepath, 'r') as index_file:
            index = json.load(index_file)

        self.dir = dir
        self.type = type
        self.data = index['annotations']
        self.num = len(index['annotations'])
        self.preprocess = preprocess

    def __len__(self):
        return self.num

    def __getitem__(self, index):
        text_info = self.data[index]
        text = clip.tokenize([text_info['caption']]).squeeze()

        image_path = os.path.join(self.dir, f'{self.type}2014/COCO_{self.type}2014_%012d.jpg'%text_info['image_id'])
        image = self.preprocess(Image.open(image_path))

        return image, text


class MSCOCO2014_:
    def __init__(self, dir, batch_size, preprocess, type='train', keep_full=True, device='cuda'):
        annotation_filepath = os.path.join(dir, f'annotations/captions_{type}2014.json')
        with open(annotation_filepath, 'r') as index_file:
            index = json.load(index_file)

        self.dir = dir
        self.type = type
        self.data = index['annotations']
        self.num = len(index['annotations'])
        self.shuffle()

        self.batch_size = batch_size
        self.keep_full = keep_full
        self.epoch = 0
        self.batch = 0

        self.preprocess = preprocess
        self.device = device

    def shuffle(self):
        random.shuffle(self.data)

    def get(self):
        epoch, batch = self.epoch, self.batch
        start = self.batch * self.batch_size
        end = min((self.batch + 1) * self.batch_size, self.num)

        texts = []
        images = []

        for i in range(start, end):
            text_info = self.data[i]
            text = clip.tokenize([text_info['caption']]).to(self.device)
            texts.append(text)

            image_path = os.path.join(self.dir, f'{self.type}2014/COCO_{self.type}2014_%012d.jpg'%text_info['image_id'])
            image = self.preprocess(Image.open(image_path)).unsqueeze(0).to(self.device)
            images.append(image)

        texts = torch.cat(texts, dim=0)
        images = torch.cat(images, dim=0)

        self.batch += 1
        if not self.keep_full and end == self.num or\
                self.keep_full and end + self.batch_size > self.num:
            self.shuffle()
            self.epoch += 1
            self.batch = 0
        return epoch, batch, images, texts


def get_dataset(dataset, batch_size=1, shuffle=True, drop_last=True):
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        pin_memory=True,
        num_workers=0
    )
    return dataloader


def get_dataset_distributed(dataset, world_size, rank, batch_size, shuffle=False, drop_last=True):
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        pin_memory=True,
        num_workers=4,
    )
    return dataloader

            
if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load('ViT-B/32', device)
    dataset = MSCOCO2014('data', preprocess, type='train')
    # dataset = TEST(5)
    dataset = get_dataset(dataset, 128)

    step_bar = tqdm(dynamic_ncols=True)
    step_bar.reset(total=10000)
    step_bar.set_description("Total")
    global_step = 0
    
    for epoch in range(1000):
        for batch, (image, text) in enumerate(dataset):
            global_step += 1
            step_bar.update(1)
            tqdm.write(f'{global_step}({epoch}-{batch}) {image.shape} {text.shape}')

