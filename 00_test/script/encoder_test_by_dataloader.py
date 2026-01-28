import torch
import zarr
import random
import numpy as np

from modeling.encoder.multimodal.encoder3d import Encoder
from utils.depth2cloud.rlbench import RLBenchDepth2Cloud
from datasets.rlbench import Peract2Dataset
from torch.utils.data import DataLoader


class TesterEncoder:
    def __init__(self,
                 # Encoder arguments
                 backbone="clip",
                 finetune_backbone=False,
                 finetune_text_encoder=False,
                 num_vis_instr_attn_layers=2,
                 fps_subsampling_factor=5,
                 # Encoder and decoder arguments
                 embedding_dim=120,
                 num_attn_heads=8,
                 nhist=3,
                 nhand=1,
                 # Decoder arguments
                 num_shared_attn_layers=4,
                 relative=False,
                 rotation_format='quat_xyzw',
                 # Denoising arguments
                 denoise_timesteps=100,
                 denoise_model="ddpm",
                 # Training arguments
                 lv2_batch_size=1):
        
        self.encoder = Encoder(
            backbone=backbone,
            embedding_dim=embedding_dim,
            nhist=nhist * nhand,
            num_attn_heads=num_attn_heads,
            num_vis_instr_attn_layers=num_vis_instr_attn_layers,
            fps_subsampling_factor=fps_subsampling_factor,
            finetune_backbone=finetune_backbone,
            finetune_text_encoder=finetune_text_encoder
        )

    def encode_inputs(self, rgb3d, rgb2d, pcd, instruction, proprio):
        fixed_inputs = self.encoder(
            rgb3d, rgb2d, pcd, instruction,
            proprio.flatten(1, 2)
        )
        # Query trajectory (for relative trajectory prediction)
        query_trajectory = proprio[:, -1:]
        return (query_trajectory,) + fixed_inputs


def load_input_data():
    data_dir_path = "Peract2_zarr/train.zarr"
    instruction_file_path = "./instructions/peract2/instructions.json"
    train_dataset = Peract2Dataset(root=data_dir_path, instructions=instruction_file_path)
    print("Number of samples in Peract2Dataset:", len(train_dataset))

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

    def base_collate_fn(batch):
        """Custom collate_fn, measured to be faster than default."""
        _dict = {}

        # Values for these come as lists
        list_keys = ["task", "instr"]
        for key in list_keys:
            if key not in batch[0].keys():
                continue
            _dict[key] = []
            for item in batch:
                _dict[key].extend(item[key])

        # Treat rest as tensors
        _dict.update({
            k_: (
                torch.cat([item[k_] for item in batch])
                if batch[0][k_] is not None else None
            )
            for k_ in batch[0].keys() if k_ not in list_keys
        })

        return _dict

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size // chunk_size,
        shuffle=False,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        collate_fn=base_collate_fn,
        pin_memory=True,
        # sampler=train_sampler,
        drop_last=True,
        # generator=g,
        prefetch_factor=4,
        persistent_workers=True
    )
    sample = next(iter(train_loader))
    print(sample)

    @torch.no_grad()
    def prepare_batch(sample, augment=False):
        sample["action"] = self.preprocessor.process_actions(sample["action"])
        proprio = self.preprocessor.process_proprio(sample["proprioception"])
        rgbs, pcds = self.preprocessor.process_obs(
            sample["rgb"], sample["rgb2d"],
            sample["depth"], sample["extrinsics"], sample["intrinsics"],
            augment=augment
        )
        return (
            sample["action"],
            torch.zeros(sample["action"].shape[:-1], dtype=bool, device='cuda'),
            rgbs,
            None,
            pcds,
            sample["instr"],
            proprio
        )
    
    batch = prepare_batch(sample)
    
    return batch

def test_encoder(encoder: TesterEncoder):
    instruction = ["push the box to the red area"]
    rgb_img, pointcloud = load_input_data()

    encoder.encode_inputs(rgb_img, None, pointcloud, instruction, None)

if __name__ == "__main__":
    tester = TesterEncoder()
    print(tester)
    print("Encoder loaded successfully.")
    load_input_data()

    test_encoder(tester)
    