from collections import OrderedDict
from copy import deepcopy
import os
import torch
from tqdm import tqdm
from time import time
import torch.nn as nn
import torch.nn.functional as F
import multiprocessing as mp
from torch.utils.data import DataLoader
import torch.distributed as dist
from src.cv_parser import get_parser
from src.gpu_devices import GPU_Support

from utils.global_params import Global

def findOptimalNumWorkers(dataset, phase, batch_size):

    Global.LOGGER.info(f"Calculating optimal number of workers with batch size 128 for {phase}")

    num_workers_time = {}
    for num_workers in range(2, mp.cpu_count(), 2):
        Global.LOGGER.info(f"Calculating time for num workers: {num_workers}")
        dataLoader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )
        start_time = time()
        for epoch in range(5):
            iterator = tqdm(enumerate(dataLoader, 0), desc=f"Epoch: {epoch+1}", unit="batches")
            for _, _ in iterator:
                pass
        end_time = time()

        num_workers_time[num_workers] = end_time - start_time
        Global.LOGGER.info(f"Num workers: {num_workers} | Time taken: {round(end_time - start_time, 4)} seconds")

    optimal_num_workers = dict(sorted(num_workers_time.items(), key=lambda x: x[1], reverse=False))
    return list(optimal_num_workers.keys())[0]

def setup_gpu_devices(log=True):
    args = get_parser().parse_args()
    GPU_Support.set_gpu_devices(args.gpu_devices, log=log)

def numpy2tensor(array):
    return torch.from_numpy(array)

def async_parallel_setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12368"
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["RANK"] = str(rank)

    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size
    )

def async_cleanup():
    dist.destroy_process_group()

class DropPath(nn.Module):
    
    def __init__(self, drop_prob, scale_by_keep=True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0.0 and self.scale_by_keep:
            random_tensor.div_(keep_prob)

        return x * random_tensor

class LayerScale(nn.Module):
    def __init__(self, num_channels, init_value):
        super(LayerScale, self).__init__()

        self.scale = nn.Parameter(
            init_value * torch.ones((1, num_channels, 1, 1))
        )

    def forward(self, x):
        return x * self.scale

class ConvLayerNorm(nn.Module):
    
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class EMA(nn.Module):

    def __init__(self, model: nn.Module, decay: float, warmup: int = 10000, decay_step=5000):

        super(EMA, self).__init__()

        self.shadow = deepcopy(model)
        self.decay = decay
        self.warmup = warmup

        self.warmup_steps = 0
        self.decay_step = decay_step

    @torch.no_grad()
    def update(self, model: nn.Module):

        self.warmup_steps += 1
        if self.warmup_steps > self.warmup and self.warmup_steps % self.decay_step == 0:
            model_params = OrderedDict(model.named_parameters())
            shadow_params = OrderedDict(self.shadow.named_parameters())

            assert model_params.keys() == shadow_params.keys()

            for name, param in model_params.items():
                shadow_params[name].lerp_(param, self.decay)

    def forward(self, x):
        return self.shadow(x)