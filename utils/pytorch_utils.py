import os
import torch
from tqdm import tqdm
from time import time
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
    os.environ["MASTER_PORT"] = "12364"
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["RANK"] = str(rank)

    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size
    )

def async_cleanup():
    dist.destroy_process_group()
