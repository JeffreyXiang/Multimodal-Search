from tqdm import trange, tqdm
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1,2'


import torch
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from clipQuantized import *

import clip
from datasets import *


'''=============== CONFIG ==============='''
log_path = './data/fine_tune_quant_a'

learning_rate = 1e-4
batch_size = 128

epochs = 1000
iterations = 10000

i_print = 5
i_val = 250
i_save = 250

centroids = []
logs = {
    "loss": [],
    "accuracy": [],
}


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group('gloo', rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def train(rank, world_size):
    torch.manual_seed(0)

    setup(rank, world_size)
    device = torch.device(rank)
    training_process(rank, world_size, device)
    cleanup()
    

@torch.no_grad()
def val(model):
    dataset = MSCOCO2014('data', model.get_preprocess(), type='val')
    dataset = get_dataset(dataset, 100)
    image_embeds = []
    text_embeds = []
    for batch, (image, text) in enumerate(dataset):
        if batch == 20: break

        image = image.to('cuda')
        text = text.to('cuda')

        # Calculate features
        image_embed, _ = model.encode_image(image)
        text_embed, _ = model.encode_text(text)
        image_embeds.append(image_embed)
        text_embeds.append(text_embed)
        
    image_embeds = torch.cat(image_embeds)
    text_embeds = torch.cat(text_embeds)

    similarity = image_embeds @ text_embeds.T
    _, image_pred_1   = torch.topk(similarity, 1  , dim=1)
    _, image_pred_5   = torch.topk(similarity, 5  , dim=1)
    _, image_pred_10  = torch.topk(similarity, 10 , dim=1)
    _, image_pred_50  = torch.topk(similarity, 50 , dim=1)
    _, image_pred_100 = torch.topk(similarity, 100, dim=1)
    _, text_pred_1    = torch.topk(similarity, 1  , dim=0)
    _, text_pred_5    = torch.topk(similarity, 5  , dim=0)
    _, text_pred_10   = torch.topk(similarity, 10 , dim=0)
    _, text_pred_50   = torch.topk(similarity, 50 , dim=0)
    _, text_pred_100  = torch.topk(similarity, 100, dim=0)

    accuracy = {
        'image to text': {'top1': 0, 'top5': 0, 'top10': 0, 'top50': 0, 'top100': 0},
        'text to image': {'top1': 0, 'top5': 0, 'top10': 0, 'top50': 0, 'top100': 0}
    }
    
    for i in range(2000):
        accuracy['image to text']['top1']   += (i in image_pred_1[i])/2000
        accuracy['image to text']['top5']   += (i in image_pred_5[i])/2000
        accuracy['image to text']['top10']  += (i in image_pred_10[i])/2000
        accuracy['image to text']['top50']  += (i in image_pred_50[i])/2000
        accuracy['image to text']['top100'] += (i in image_pred_100[i])/2000
        accuracy['text to image']['top1']   += (i in text_pred_1[:, i])/2000
        accuracy['text to image']['top5']   += (i in text_pred_5[:, i])/2000
        accuracy['text to image']['top10']  += (i in text_pred_10[:, i])/2000
        accuracy['text to image']['top50']  += (i in text_pred_50[:, i])/2000
        accuracy['text to image']['top100'] += (i in text_pred_100[:, i])/2000

    return accuracy


def training_process(rank, world_size, device):
    global centroids
    global logs
    alpha = 1
    beta = 1
    M = 32
    
    if rank==0:
        print("Start training..")
    # Load the model
    # model = ClipQuantizedI2T("ViT-B/32", device, M=M, alpha=alpha, beta=beta)
    model = ClipQuantizedI2T("notebooks/model.pt", device, M=M, alpha=alpha, beta=beta)

    centroids = np.load("./data/centriods_{}.npy".format(M))

    model.set_centroids(centroids)
    model.cuda(device)
    
    global_step = 0
    for filename in os.listdir(log_path):
        if 'pt' in filename and 'log' not in filename:
            temp = int(filename[0:6])
            if temp > global_step: global_step = temp
    if global_step > 0:
        model.load_state_dict(torch.load(os.path.join(log_path, '%06d.pt'%global_step)))
        logs = torch.load(os.path.join(log_path, 'log%06d.pt'%global_step))
    model = model.train()
    model_ddp = DDP(model, device_ids=[rank], find_unused_parameters=True)
    model = model_ddp.module

    # Dataset
    dataset = MSCOCO2014('data', model.get_preprocess(), type='train')
    dataset = get_dataset_distributed(dataset, world_size, rank, batch_size)

    # Optimizer
    optimizer = torch.optim.SGD(params=model_ddp.parameters(), lr=learning_rate)
    target = torch.tensor(list(range(batch_size)), dtype=torch.long, device=device)

    step_bar = tqdm(dynamic_ncols=True)
    step_bar.reset(total=iterations)
    step_bar.set_description('Total')
    
    for epoch in range(epochs):
        for batch, (image, text) in enumerate(dataset):
            # Calculate features
            optimizer.zero_grad()
            similarity_i2t, similarity_t2i, quant_loss = model_ddp(image, text)
            loss = F.cross_entropy(similarity_i2t, target) + F.cross_entropy(similarity_t2i, target) + quant_loss
            loss.backward()
            optimizer.step()
            logs['loss'].append(loss.item())
            
            if rank == 0:
                if global_step % i_print == 0:
                    tqdm.write(f'[Train] Iter: {global_step}({epoch}-{batch}) loss: {loss.item()-quant_loss.item()}, quant_loss: {quant_loss.item()/(alpha+beta)}')
                if global_step % i_val == 0:
                    accuracy = val(model)
                    tqdm.write(f'[Validation] accuracy: {accuracy}')
                    accuracy['global_step'] = global_step
                    logs['accuracy'].append(accuracy)
                if global_step % i_save == 0:
                    torch.save(model.state_dict(), os.path.join(log_path, '%06d.pt'%global_step))
                    torch.save(logs, os.path.join(log_path, 'log%06d.pt'%global_step))
                global_step += 1
                step_bar.update(1)
            dist.barrier()


if __name__ == '__main__':
    os.makedirs(log_path, exist_ok=True)
    num_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    mp.spawn(train, args=(num_gpus, ), nprocs=num_gpus, join=True)
    