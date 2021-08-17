import torch
import numpy as np
import nanopq
from datasets import *

@torch.no_grad()
def get_init_centriods(M=32, K=256):
    model, preprocess = clip.load("./notebooks/model.pt", "cuda:3")
    
    print("Initing centroids")
    dataset = MSCOCO2014('data', preprocess, type='train')
    dataset = get_dataset(dataset, 100)
    image_embeds = []
    text_embeds = []
    for batch, (image, text) in tqdm(enumerate(dataset)):
        if batch == 100: break

        image = image.to('cuda:3')
        text = text.to('cuda:3')

        # Calculate features
        image_embed = model.encode_image(image)
        text_embed = model.encode_text(text)
        image_embeds.append(image_embed)
        text_embeds.append(text_embed)

    image_embeds = torch.cat(image_embeds)
    text_embeds = torch.cat(text_embeds)

    img_vectors = image_embeds.detach().cpu().numpy().astype('float32')
    text_vectors= text_embeds.detach().cpu().numpy().astype('float32')
    print(img_vectors.shape, text_vectors.shape)
    word_vectors = np.concatenate((img_vectors, text_vectors))
    
    pq = nanopq.PQ(M=M, Ks=K, verbose=True)
    pq.fit(vecs=word_vectors, iter=20)
    print(pq.codewords.shape)
    np.save('./data/centriods_{}.npy'.format(M), pq.codewords)
    
get_init_centriods(32)