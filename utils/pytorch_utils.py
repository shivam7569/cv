from collections import OrderedDict
from copy import deepcopy
import math
import os
import random
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
    os.environ["MASTER_PORT"] = "12366"
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["RANK"] = str(rank)

    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size
    )

def async_cleanup():
    dist.destroy_process_group()

def get_sinusoidal_embedding(max_seq_len, embedding_dim):

    if embedding_dim % 2 != 0:
        raise ValueError(f"Sinusoidal position embeddings cannot be applied to odd token embedding dimension")
    


    position = torch.arange(0, max_seq_len).unsqueeze_(1)
    denominators = torch.pow(10000.0, 2*torch.arange(0, embedding_dim // 2) / 2)

    sinusoidal_embedding = torch.zeros(max_seq_len, embedding_dim)
    
    sinusoidal_embedding[:, 0::2] = torch.sin(position / denominators)
    sinusoidal_embedding[:, 1::2] = torch.cos(position / denominators)
    
    return sinusoidal_embedding

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
    def __init__(self, num_channels, init_value, type_="conv"):
        super(LayerScale, self).__init__()

        if type_ == "conv":
            self.scale = nn.Parameter(
                init_value * torch.ones((1, num_channels, 1, 1))
            )
        elif type_ == "msa":
            self.scale = nn.Parameter(
                init_value * torch.ones((1, 1, num_channels))
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

class RepeatAugSampler(torch.utils.data.Sampler):

    def __init__(
            self,
            dataset,
            num_replicas=None,
            rank=None,
            shuffle=True,
            num_repeats=3,
            selected_round=256,
            selected_ratio=0,
    ):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.num_repeats = num_repeats
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * num_repeats / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        selected_ratio = selected_ratio or num_replicas  # ratio to reduce selected samples by, num_replicas if 0
        if selected_round:
            self.num_selected_samples = int(math.floor(
                 len(self.dataset) // selected_round * selected_round / selected_ratio))
        else:
            self.num_selected_samples = int(math.ceil(len(self.dataset) / selected_ratio))

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.epoch)
        if self.shuffle:
            indices = torch.randperm(len(self.dataset), generator=g)
        else:
            indices = torch.arange(start=0, end=len(self.dataset))

        if isinstance(self.num_repeats, float) and not self.num_repeats.is_integer():
            repeat_size = math.ceil(self.num_repeats * len(self.dataset))
            indices = indices[torch.tensor([int(i // self.num_repeats) for i in range(repeat_size)])]
        else:
            indices = torch.repeat_interleave(indices, repeats=int(self.num_repeats), dim=0)
        indices = indices.tolist()
        padding_size = self.total_size - len(indices)
        if padding_size > 0:
            indices += indices[:padding_size]
        assert len(indices) == self.total_size

        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices[:self.num_selected_samples])

    def __len__(self):
        return self.num_selected_samples

    def set_epoch(self, epoch):
        self.epoch = epoch

class TransformerSEBlock(nn.Module):

    def __init__(self, in_channels, r=16):
        super(TransformerSEBlock, self).__init__()

        self.squeeze = nn.Sequential(
            nn.AdaptiveAvgPool1d(output_size=1),
            nn.Flatten(start_dim=1),
            nn.Linear(in_features=in_channels, out_features=in_channels // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=in_channels // r, out_features=in_channels, bias=False),
            nn.Sigmoid()
        )

        self.in_channels = in_channels

    def forward(self, x):
        squeeze = self.squeeze(x)
        excitation = squeeze.view(-1, self.in_channels, 1).expand_as(x) * x

        return excitation