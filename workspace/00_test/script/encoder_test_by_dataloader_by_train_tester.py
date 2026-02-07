"""
Docstring for 00_test.script.encoder_test_by_dataloader_by_train_tester

메모
    원본코드는 객체지향을 통해 잘 구조화 된 편이나, 
    argument 대입에 대한 편의성이 아쉬운 구조이다. 

    개별 객체 요소를 활용하기위해 필요한 인자를 찾기 어렵고
    객체 활용에서의 어려움에 비해 더 이상의 활용 가치를 찾지 못하였다.
    
    BaseTrainTester 객체를 활용한 방법의 테스터 코드는 여기서 마친다.

"""


import torch
import zarr
import random
import numpy as np

from modeling.encoder.multimodal.encoder3d import Encoder
from utils.depth2cloud.rlbench import RLBenchDepth2Cloud
from utils.trainers.rlbench import BaseTrainTester
from datasets.rlbench import Peract2Dataset
from torch.utils.data import DataLoader
from utils.data_preprocessors import fetch_data_preprocessor


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

class TesterDataloader(BaseTrainTester):
    def __init__(self, args, dataset_cls):
        super().__init__(args, dataset_cls, None)

        pass



def load_input_data(tester_dataloader:TesterDataloader):
    
    
    batch = tester_dataloader.prepare_batch(sample)
    
    return batch

def test_encoder(encoder: TesterEncoder):
    instruction = ["push the box to the red area"]
    rgb_img, pointcloud = load_input_data()

    encoder.encode_inputs(rgb_img, None, pointcloud, instruction, None)

if __name__ == "__main__":
    # define dataset
    data_dir_path = "Peract2_zarr/train.zarr"
    instruction_file_path = "./instructions/peract2/instructions.json"
    train_dataset = Peract2Dataset(root=data_dir_path, instructions=instruction_file_path)

    # definde dataset args
    from collections import namedtuple
    datset_args = namedtuple("args", ["dataset", "keypose_only", "num_history", 'custom_img_size', 'eval_only', 'log_dir'])
    

    args = datset_args(
        dataset="peract",
        keypose_only=False,
        num_history=1,
        custom_img_size=None,
        eval_only=True,
        log_dir="./train_log"
    )

    tester_encoder = TesterEncoder()
    tester_dataloader = TesterDataloader(args, train_dataset)
    print(tester_encoder)
    print("Encoder loaded successfully.")
    load_input_data()

    test_encoder(tester_encoder, tester_dataloader)
