from tqdm import trange, tqdm
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '4,5,6,7'

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

import clip
from datasets import *

'''=============== CONFIG ==============='''
log_path = './data/fine_tune'

learning_rate = 1e-4
batch_size = 256

epochs = 1000
iterations = 10000

i_print = 1
i_save = 2000

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

def training_process(rank, world_size, device):
    # Logs
    log = {'loss': []}

    # Load the model
    model, preprocess = clip.load('ViT-B/32', device)
    global_step = 0
    for filename in os.listdir(log_path):
        if 'pt' in filename:
            temp = int(filename[0:6])
            if temp > global_step: global_step = temp
    if global_step > 0:
        model.load_state_dict(torch.load(os.path.join(log_path, '%06d.pt'%global_step)))
    model = model.train()
    model_ddp = DDP(model, device_ids=[rank], find_unused_parameters=True)
    model = model_ddp.module

    # Dataset
    dataset = MSCOCO2014('data', preprocess, type='train')
    dataset = get_dataset_distributed(dataset, world_size, rank, batch_size)

    # Optimizer
    optimizer = torch.optim.SGD(params=model_ddp.parameters(), lr=learning_rate)
    target = torch.eye(batch_size, device=device)

    step_bar = tqdm(dynamic_ncols=True)
    step_bar.reset(total=iterations)
    step_bar.set_description('Total')
    
    for epoch in range(epochs):
        for batch, (image, text) in enumerate(dataset):
            # Calculate features
            optimizer.zero_grad()
            similarity = model_ddp(image, text)
            loss = torch.sum((similarity - target)**2) / batch_size
            loss.backward()
            optimizer.step()
            log['loss'].append(loss.item())
            
            if rank == 0:
                global_step += 1
                step_bar.update(1)
                if global_step % i_print == 0:
                    tqdm.write(f'[Train] Iter: {global_step}({epoch}-{batch}) loss: {loss.item()}')
                if global_step % i_save == 0:
                    torch.save(model.state_dict(), os.path.join(log_path, '%06d.pt'%global_step))
            dist.barrier()


if __name__ == '__main__':
    os.makedirs(log_path, exist_ok=True)
    num_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    mp.spawn(train, args=(num_gpus, ), nprocs=num_gpus, join=True)
    