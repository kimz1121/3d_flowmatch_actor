from datasets.rlbench import Peract2Dataset
from torch.utils.data import DataLoader

import random
import numpy as np
import torch
from torch.utils.data.distributed import DistributedSampler

def test():
    data_dir_path = "Peract2_zarr/train.zarr"
    instruction_file_path = "./instructions/peract2/instructions.json"
    train_dataset = Peract2Dataset(root=data_dir_path, instructions=instruction_file_path)
    print("Number of samples in Peract2Dataset:", len(train_dataset))

    print(train_dataset[1])

    # params
    batch_size = 8
    num_workers = 4
    chunk_size = 4

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    # train_sampler = DistributedSampler(train_dataset, drop_last=True)
    g = torch.Generator()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size // chunk_size,
        shuffle=False,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        collate_fn=None,
        pin_memory=True,
        # sampler=train_sampler,
        drop_last=True,
        # generator=g,
        prefetch_factor=4,
        persistent_workers=True
    )


if __name__ == "__main__":
    test()