import nanopq
import numpy as np
import tqdm
import random
import torch
import clip
from datasets import *

def dist(a, b):
    import torch
    a_torch = torch.tensor(a).to("cuda")
    b_torch = torch.tensor(b).to("cuda")
    matmul_result = torch.mm(a_torch, b_torch.t())
    return np.sum(a ** 2, axis=1)[:, None] + np.sum(b ** 2, axis=1)[None, :] - matmul_result.cpu().numpy() * 2


def topk(matrix, k):
    return np.argpartition(matrix, k)[:, :k]


def orig_test(img_vectors, word_vectors, k_values):
    batch_size = 2000
    accuracy = np.zeros((len(k_values),))
    for i in tqdm(range(img_vectors.shape[0]//batch_size + 1)):
        dist_matrix = dist(img_vectors[i*2000: i*2000 + 2000,:], word_vectors)
        for j in range(len(k_values)):
            k = k_values[j]
            result = topk(dist_matrix, k)
            accuracy[j] += np.sum([(m+2000*i) in result[m, :] for m in range(result.shape[0])])
    print(word_vectors.shape[0])
    accuracy = accuracy/word_vectors.shape[0]

    print("Original Test:", accuracy)
    return accuracy


def pq_test1(img_vectors, word_vectors, k_values):
    vectors = np.concatenate((img_vectors, word_vectors))
    pq_params = [(32, 256), (64, 256), (128, 256)]
    for pq_param in pq_params:
        pq = nanopq.PQ(M=pq_param[0], Ks=pq_param[1], verbose=False)
        pq.fit(vecs=word_vectors, iter=20)
        word_vectors_coded = pq.encode(vecs=word_vectors)

        accuracy = np.zeros((len(k_values),))
        for i in tqdm(range(len(img_vectors))):
            dt = pq.dtable(query=img_vectors[i])
            dists = dt.adist(codes=word_vectors_coded)
            for j in range(len(k_values)):
                if i in topk(dists[None,:], k_values[j])[0]:
                    accuracy[j] += 1

        accuracy = accuracy/word_vectors.shape[0]
        print("PQ_Test:", pq_param, accuracy)



def opq_test1(img_vectors, word_vectors, k_values):
    vectors = np.concatenate((img_vectors, word_vectors))
    pq_params = [(32, 256), (64, 256)]
    for pq_param in pq_params:
        opq = nanopq.OPQ(M=pq_param[0], Ks=pq_param[1], verbose=False)
        opq.fit(vecs=word_vectors, pq_iter=20, rotation_iter=20)
        word_vectors_coded = opq.encode(vecs=word_vectors)

        accuracy = np.zeros((len(k_values),))
        for i in tqdm(range(len(img_vectors))):
            dt = opq.dtable(query=img_vectors[i])
            dists = dt.adist(codes=word_vectors_coded)
            for j in range(len(k_values)):
                if i in topk(dists[None,:], k_values[j])[0]:
                    accuracy[j] += 1

        accuracy = accuracy/word_vectors.shape[0]
        print("OPQ_Test1:", pq_param, accuracy)

        
device = "cuda:3" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("data/fine_tune_quant_a/002000.pt", device=device)
dataset = MSCOCO2014('data', preprocess, type='val')
dataset = get_dataset(dataset, batch_size=100)
image_embeds = []
text_embeds = []

with torch.no_grad():
    for batch, (image, text) in tqdm(enumerate(dataset)):
        if batch == 20: break

        image = image.to(device)
        text = text.to(device)

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

k_values = [1, 5, 10, 50, 100]

orig_test(img_vectors, text_vectors, k_values)
pq_test1(img_vectors, text_vectors, k_values)
opq_test1(img_vectors, text_vectors, k_values)

orig_test(text_vectors, img_vectors, k_values)
pq_test1(text_vectors, img_vectors, k_values)
opq_test1(text_vectors, img_vectors, k_values)


