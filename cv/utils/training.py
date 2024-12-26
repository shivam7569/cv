from __future__ import annotations

import math
import torch
import torch.nn as nn
from copy import deepcopy
from typing import Callable
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp

from cv.utils import Global
from cv.src.crf import DenseCRF
from cv.utils import MetaWrapper
from cv.datasets import ClassificationDataset

class RepeatAugSampler(torch.utils.data.Sampler, metaclass=MetaWrapper):

    @classmethod
    def __class_repr__(cls):
        return "Class implementing repeated augmentated sample across multiple GPUs"

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


class ConditionalRandomFields(metaclass=MetaWrapper):

    @classmethod
    def __class_repr__(cls):
        return "Class implementing conditional random fields"

    def __init__(self, interpolation_params, crf_params):
        self.crf = DenseCRF(**crf_params)
        self.interpolation_params = interpolation_params

    def process(self, img_batch, output_batch):

        B, _, H, W = img_batch.shape

        img_batch = ClassificationDataset.INVERSE_TRANSFORM(img_batch)
        output_batch = F.interpolate(input=output_batch, size=(H, W), **self.interpolation_params)

        processed_probmap_batch = []

        for i in range(B):
            img = (img_batch[i] * 255).to(torch.uint8).permute(1, 2, 0).detach().cpu().numpy()
            prob = output_batch[i].detach().cpu().numpy()

            processed_probmap = self.crf(image=img, probmap=prob)
            processed_probmap_batch.append(torch.from_numpy(processed_probmap).to(output_batch.device))

        processed_probmap_batch = torch.stack(processed_probmap_batch, dim=0)
        
        return output_batch, processed_probmap_batch
    
    def process_one(self, args):
        idx, img, output = args
        _, _, H, W = img.shape
        img = ClassificationDataset.INVERSE_TRANSFORM(img)
        output = F.interpolate(input=output, size=(H, W), **self.interpolation_params)

        img = (img[0] * 255).to(torch.uint8).permute(1, 2, 0).numpy()
        prob = output[0].numpy()

        processed_probmap = self.crf(image=img, probmap=prob)
        processed_probmap = torch.from_numpy(processed_probmap)

        return idx, output, processed_probmap
    
    def mp_process(self, img_batch, output_batch):

        B, _, _, _ = img_batch.shape
        inputs = [(i, img_batch[i].unsqueeze(0).detach().cpu(), output_batch[0].unsqueeze(0).detach().cpu()) for i in range(B)]

        ctx = mp.get_context('spawn')
        with ctx.Pool(processes=mp.cpu_count()) as pool:
            results = pool.map(self.process_one, inputs)

        results = sorted(results, key=lambda x: x[0])
        output_batch = torch.cat([i[1] for i in results], dim=0).to(img_batch.device)
        processed_probmap_batch = torch.stack([i[2] for i in results], dim=0).to(img_batch.device)

        return output_batch, processed_probmap_batch
    

class EMA(nn.Module):
    
    UTIL_FUNCTIONS = {
        "exists": lambda x: x is not None,
        "divisible_by": lambda x, y: (x % y) == 0,
        "inplace_copy": lambda tgt, src: tgt.copy_(src),
        "inplace_lerp": lambda tgt, src, weight: tgt.lerp_(src, weight)
    }

    def __init__(
        self,
        model: nn.Module,
        ema_model: nn.Module | Callable[[], nn.Module] | None = None,
        beta=0.9999,
        update_after_step=100,
        update_every=10,
        inv_gamma=1.0,
        power=2/3,
        min_value=0.0,
        include_online_model=True,
        use_foreach=True,
        update_model_with_ema_every=None,
        update_model_with_ema_beta=0.,
    ):
        super().__init__()
        self.beta = beta
        self.is_frozen = beta == 1.
        self.include_online_model = include_online_model

        if include_online_model:
            self.online_model = model
        else:
            self.online_model = [model]

        if not isinstance(ema_model, nn.Module) and callable(ema_model):
            ema_model = ema_model()

        self.ema_model = None

        self.init_ema(ema_model)

        self.inplace_copy = EMA.UTIL_FUNCTIONS["inplace_copy"]
        self.inplace_lerp = EMA.UTIL_FUNCTIONS["inplace_lerp"]

        self.update_every = update_every
        self.update_after_step = update_after_step

        self.inv_gamma = inv_gamma
        self.power = power
        self.min_value = min_value

        # continual learning related
        self.update_model_with_ema_every = update_model_with_ema_every
        self.update_model_with_ema_beta = update_model_with_ema_beta

        if use_foreach:
            try:
                assert hasattr(torch, '_foreach_lerp_') and hasattr(torch, '_foreach_copy_'), 'your version of torch does not have the prerequisite foreach functions'
            except:
                use_foreach = False

        self.use_foreach = use_foreach

        self.register_buffer('initted', torch.tensor(False))
        self.register_buffer('step', torch.tensor(0))

    def init_ema(
        self,
        ema_model: nn.Module | None = None
    ):
        self.ema_model = ema_model

        if not EMA.UTIL_FUNCTIONS["exists"](self.ema_model):
            try:
                Global.LOGGER.info(f"Initializing EMA model")
                self.ema_model = deepcopy(self.model)
            except Exception as e:
                Global.LOGGER.error(f"Error: While trying to deepcopy model: {e}")
                Global.LOGGER.error("Model was not copyable. Please make sure any LazyLinear is not being used")
                exit()

        for p in self.ema_model.parameters():
            p.detach_()

        self.parameter_names = {name for name, param in self.ema_model.named_parameters() if torch.is_floating_point(param) or torch.is_complex(param)}
        self.buffer_names = {name for name, buffer in self.ema_model.named_buffers() if torch.is_floating_point(buffer) or torch.is_complex(buffer)}

    def add_to_optimizer_post_step_hook(self, optimizer):
        assert hasattr(optimizer, 'register_step_post_hook')

        def hook(*_):
            self.update()

        return optimizer.register_step_post_hook(hook)

    @property
    def model(self):
        return self.online_model if self.include_online_model else self.online_model[0]

    def eval(self):
        return self.ema_model.eval()

    def get_params_iter(self, model):
        for name, param in model.named_parameters():
            if name not in self.parameter_names:
                continue
            yield name, param

    def get_buffers_iter(self, model):
        for name, buffer in model.named_buffers():
            if name not in self.buffer_names:
                continue
            yield name, buffer

    def copy_params_from_model_to_ema(self):
        copy = EMA.UTIL_FUNCTIONS["inplace_copy"]

        for (_, ma_params), (_, current_params) in zip(self.get_params_iter(self.ema_model), self.get_params_iter(self.model)):
            copy(ma_params.data, current_params.data)

        for (_, ma_buffers), (_, current_buffers) in zip(self.get_buffers_iter(self.ema_model), self.get_buffers_iter(self.model)):
            copy(ma_buffers.data, current_buffers.data)

    def copy_params_from_ema_to_model(self):
        copy = EMA.UTIL_FUNCTIONS["inplace_copy"]

        for (_, ma_params), (_, current_params) in zip(self.get_params_iter(self.ema_model), self.get_params_iter(self.model)):
            copy(current_params.data, ma_params.data)

        for (_, ma_buffers), (_, current_buffers) in zip(self.get_buffers_iter(self.ema_model), self.get_buffers_iter(self.model)):
            copy(current_buffers.data, ma_buffers.data)

    def update_model_with_ema(self, decay = None):
        if not EMA.UTIL_FUNCTIONS["exists"](decay):
            decay = self.update_model_with_ema_beta

        if decay == 0.:
            return self.copy_params_from_ema_to_model()

        self.update_moving_average(self.model, self.ema_model, decay)

    def get_current_decay(self):
        epoch = (self.step - self.update_after_step - 1).clamp(min=0.)
        value = 1 - (1 + epoch / self.inv_gamma) ** - self.power

        if epoch.item() <= 0:
            return 0.

        return value.clamp(min=self.min_value, max=self.beta).item()

    def update(self, last_epoch):
        step = self.step.item()
        self.step += 1

        beta = self.get_current_decay()

        if not self.initted.item():
            if not EMA.UTIL_FUNCTIONS["exists"](self.ema_model):
                self.init_ema()

            self.copy_params_from_model_to_ema()
            self.initted.data.copy_(torch.tensor(True))
            
            return beta

        should_update = EMA.UTIL_FUNCTIONS["divisible_by"](step, self.update_every) or last_epoch

        if should_update and step <= self.update_after_step:
            self.copy_params_from_model_to_ema()
            
            return beta

        if should_update:
            Global.LOGGER.info(f"Updating EMA model")
            self.update_moving_average(self.ema_model, self.model)

        if EMA.UTIL_FUNCTIONS["exists"](self.update_model_with_ema_every) and EMA.UTIL_FUNCTIONS["divisible_by"](step, self.update_model_with_ema_every):
            self.update_model_with_ema()

        return beta

    @torch.no_grad()
    def update_moving_average(self, ma_model, current_model, current_decay = None):
        if self.is_frozen:
            return

        if not EMA.UTIL_FUNCTIONS["exists"](current_decay):
            current_decay = self.get_current_decay()

        tensors_to_copy = []
        tensors_to_lerp = []

        for (_, current_params), (_, ma_params) in zip(self.get_params_iter(current_model), self.get_params_iter(ma_model)):
            tensors_to_lerp.append((ma_params.data, current_params.data))

        for (_, current_buffer), (_, ma_buffer) in zip(self.get_buffers_iter(current_model), self.get_buffers_iter(ma_model)):
            tensors_to_lerp.append((ma_buffer.data, current_buffer.data))

        if not self.use_foreach:
            for tgt, src in tensors_to_copy:
                self.inplace_copy(tgt, src)

            for tgt, src in tensors_to_lerp:
                self.inplace_lerp(tgt, src, 1. - current_decay)
        else:
            if len(tensors_to_copy) > 0:
                tgt_copy, src_copy = zip(*tensors_to_copy)
                torch._foreach_copy_(tgt_copy, src_copy)

            if len(tensors_to_lerp) > 0:
                tgt_lerp, src_lerp = zip(*tensors_to_lerp)
                torch._foreach_lerp_(tgt_lerp, src_lerp, 1. - current_decay)

    def __call__(self, *args, **kwargs):
        return self.ema_model(*args, **kwargs)
