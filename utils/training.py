from imp import is_frozen
import math
import torch
import torch.nn as nn
from copy import deepcopy
import torch.distributed as dist
from collections import OrderedDict

from utils.global_params import Global

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


from copy import deepcopy
from functools import partial
import torch
from torch import nn, Tensor
from beartype import beartype
from beartype.typing import Set, Optional

class EMA(nn.Module):

    @staticmethod
    def exists(val):
        return val is not None
    
    @staticmethod
    def inplace_copy(target: Tensor, source: Tensor):
        target.copy_(source)

    @staticmethod
    def inplace_lerp(target: Tensor, source: Tensor, weight: float):
        target.lerp_(source, weight=weight)

    @beartype
    def __init__(
            self,
            model: nn.Module,
            ema_model: Optional[nn.Module] = None,
            beta: float = 0.9999,
            warmup_steps: int = 100,
            decay_period: int = 10,
            inv_gamma: float = 1.0,
            power: float = 2/3,
            min_value: float = 0.0
    ):
        
        super(EMA, self).__init__()

        self.beta = beta
        self.is_frozen = beta == 1.0

        self.ema_model = ema_model

        if not self.exists(self.ema_model):
            try:
                self.ema_model = deepcopy(model)
            except Exception as e:
                Global.LOGGER.error(f"Could not copy model to shadow")

        self.ema_model.requires_grad_(requires_grad=False)

        self.parameter_names = {
            name for name, param in self.ema_model.named_parameters() if param.dtype in [torch.float, torch.float16]
        }
        self.buffer_names = {
            name for name, buffer in self.ema_model.named_buffers() if buffer.dtype in [torch.float, torch.float16]
        }

        self.warmup_steps = warmup_steps
        self.decay_period = decay_period

        self.inv_gamma = inv_gamma
        self.power = power
        self.min_value = min_value

        self.register_buffer(
            "initted", torch.tensor(False)
        )
        self.register_buffer(
            "step", torch.tensor(0)
        )

    def update(self, network: nn.Module):
        step = self.step.item()
        self.step += 1

        if step % self.decay_period != 0: return
        if step <= self.warmup_steps:
            self.copy_params_from_model_to_ema(network)
            return
        
        if not self.initted.item():
            self.copy_params_from_model_to_ema(network)
            self.initted.data.copy_(torch.tensor(True))

        self.update_moving_average(self.ema_model, network)

    @torch.no_grad()
    def update_moving_average(self, ma_model, current_model):
        if self.is_frozen: return

        current_decay = self.get_current_decay()

        for (_, current_params), (_, ma_params) in zip(self.get_params_iter(current_model), self.get_params_iter(ma_model)):
            self.inplace_lerp(ma_params.data, current_params.data, 1.0 - current_decay)

        for (_, current_buffers), (_, ma_buffers) in zip(self.get_buffers_iter(current_model), self.get_buffers_iter(ma_model)):
            self.inplace_lerp(ma_buffers.data, current_buffers.data, 1.0 - current_decay)

    def get_params_iter(self, model: nn.Module):
        for name, param in model.named_parameters():
            if name not in self.parameter_names:
                continue

            yield name, param

    def get_buffers_iter(self, model: nn.Module):
        for name, buffer in model.named_buffers():
            if name not in self.buffer_names:
                continue

            yield name, buffer

    def copy_params_from_model_to_ema(self, model: nn.Module):
        
        for (_, ma_params), (_, current_params) in zip(self.get_params_iter(self.ema_model), self.get_params_iter(model)):
            self.inplace_copy(ma_params.data, current_params.data)

        for (_, ma_buffers), (_, current_buffers) in zip(self.get_buffers_iter(self.ema_model), self.get_buffers_iter(model)):
            self.inplace_copy(ma_buffers.data, current_buffers.data)

    def get_current_decay(self):
        epoch = (self.step - self.warmup_steps - 1).clamp(min=0.0)
        value = 1 - (1 + epoch / self.inv_gamma) ** -self.power

        if epoch.item() <= 0:
            return 0.0

        return value.clamp(min=self.min_value, max=self.beta).item()   

    def forward(self, x):
        return self.ema_model(x)

# class EMA(nn.Module):

#     def __init__(self, model: nn.Module, decay: float, warmup: int = 10000, decay_step=5000):

#         super(EMA, self).__init__()

#         self.shadow = deepcopy(model)
#         self.decay = decay
#         self.warmup = warmup

#         self.warmup_steps = 0
#         self.decay_step = decay_step

#     @torch.no_grad()
#     def update(self, model: nn.Module):

#         self.warmup_steps += 1
#         if self.warmup_steps > self.warmup and self.warmup_steps % self.decay_step == 0:
#             model_params = OrderedDict(model.named_parameters())
#             shadow_params = OrderedDict(self.shadow.named_parameters())

#             assert model_params.keys() == shadow_params.keys()

#             for name, param in model_params.items():
#                 shadow_params[name].lerp_(param, self.decay)

#     def forward(self, x):
#         return self.shadow(x)