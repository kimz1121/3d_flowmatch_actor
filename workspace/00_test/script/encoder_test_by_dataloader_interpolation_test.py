import torch
import zarr
import random
import numpy as np

from modeling.encoder.multimodal.encoder3d import Encoder
from modeling.encoder.text import fetch_tokenizers
from utils.depth2cloud.rlbench import RLBenchDepth2Cloud
from datasets.rlbench import Peract2Dataset
from torch.utils.data import DataLoader
from utils.data_preprocessors import fetch_data_preprocessor
from utils.depth2cloud import fetch_depth2cloud


class EncoderDebug(Encoder):
    def __init__(self, backbone="clip", embedding_dim=60, nhist=1, num_attn_heads=9, num_vis_instr_attn_layers=2, fps_subsampling_factor=5, finetune_backbone=False, finetune_text_encoder=False):
        super().__init__(backbone, embedding_dim, nhist, num_attn_heads, num_vis_instr_attn_layers, fps_subsampling_factor, finetune_backbone, finetune_text_encoder)


class TesterEncoder:
    def __init__(self,
                 # Encoder arguments
                 device = "cuda",
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
        
        self.encoder = EncoderDebug(
            backbone=backbone,
            embedding_dim=embedding_dim,
            nhist=nhist * nhand,
            num_attn_heads=num_attn_heads,
            num_vis_instr_attn_layers=num_vis_instr_attn_layers,
            fps_subsampling_factor=fps_subsampling_factor,
            finetune_backbone=finetune_backbone,
            finetune_text_encoder=finetune_text_encoder
        ).to(device)

        self.toknizer = fetch_tokenizers(backbone)

    # rgb2d 입력은 이번 버전에서 구현 되지 않았음을 주석에서 확인할 수 있음.
    # modeling/encoder/multimodal/encoder3d.py
        # line 139~140
        # 2D camera features (don't support mixed cameras in this release)
        # rgb2d_feats = None

    def tokenize(self, instruction):
        tokens = self.toknizer(instruction)
        # GPU로 이동
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.to('cuda')
        elif hasattr(tokens, 'input_ids'):  # BatchEncoding인 경우
            tokens = tokens.to('cuda')
        return tokens

    def encode_inputs(self, rgb3d, rgb2d, pcd, instruction, proprio):
        instruction_token = self.tokenize(instruction)
        fixed_inputs = self.encoder(
            rgb3d, rgb2d, pcd, instruction_token,
            proprio.flatten(1, 2)
        )
        return fixed_inputs

    # 3D feature에, Clip을 통한 semantic feature를 입히는 과정
    def encode_clip(self, rgb3d, rgb2d, pcd, instruction):
        instruction_token = self.tokenize(instruction)
        rgb3d_feats, rgb2d_feats, pcd, instr_feats = self.encoder.encode_clip(
            rgb3d=rgb3d, rgb2d=rgb2d, pcd=pcd, text=instruction_token
        )
        return rgb3d_feats, rgb2d_feats, pcd, instr_feats


def load_input_data():
    data_dir_path = "Peract2_zarr/train.zarr"
    instruction_file_path = "./instructions/peract2/instructions.json"
    train_dataset = Peract2Dataset(root=data_dir_path, instructions=instruction_file_path)
    print("Number of samples in Peract2Dataset:", len(train_dataset))

    # definde dataset args
    from collections import namedtuple
    datset_args = namedtuple("args", ["dataset", "keypose_only", "num_history", 'custom_img_size', 'eval_only', 'log_dir'])
    
    args = datset_args(
        dataset="peract2",
        keypose_only=False,
        num_history=1,
        custom_img_size=None,
        eval_only=True,
        log_dir="./train_log"
    )
    # define preprocessor
    preprocessor_class = fetch_data_preprocessor(args.dataset)
    preprocessor = preprocessor_class(
            args.keypose_only,
            args.num_history,
            custom_imsize=args.custom_img_size,
            depth2cloud=fetch_depth2cloud(args.dataset)
        )

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
    # print(sample)

    if args.dataset == "peract2" or args.dataset == "rlbench":
        @torch.no_grad()
        def prepare_batch_rlbench(sample, augment=False):
            sample["action"] = preprocessor.process_actions(sample["action"])
            proprio = preprocessor.process_proprio(sample["proprioception"])
            rgbs, pcds = preprocessor.process_obs(
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
        
        batch = prepare_batch_rlbench(sample)

    elif args.dataset == "peract":
        @torch.no_grad()
        def prepare_batch_peract(sample, augment=False):
            sample["action"] = preprocessor.process_actions(sample["action"])
            proprio = preprocessor.process_proprio(sample["proprioception"])
            rgbs, pcds = preprocessor.process_obs(
                sample["rgb"], sample["pcd"],
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
        
        batch = prepare_batch_peract(sample)
    
    return batch

def test_encoder(encoder: TesterEncoder):
    instruction = ["push the box to the red area"]
    action, action_mask, rgbs, rgb2d, pcds, instr, prop = load_input_data()

    # rgb2d 는 2d 카메라용 입력 공간으로, depth cam 만으로 구성된 입력에 대해서는 값을 전달하지 않는다.
    # encoder.encode_inputs(rgbs, None, pcds, instr, prop)
    feature = encoder.encode_clip(rgbs, None, pcds, instr)
    return feature

if __name__ == "__main__":
    tester = TesterEncoder()
    print(tester)
    print("Encoder loaded successfully.")
    load_input_data()

    feature_tuple = test_encoder(tester)

    feature:torch.Tensor
    for feature in feature_tuple:
        print(feature.shape)

    