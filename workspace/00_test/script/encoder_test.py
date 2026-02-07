import torch
import zarr

from modeling.encoder.multimodal.encoder3d import Encoder
from utils.depth2cloud.rlbench import RLBenchDepth2Cloud

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
    # basic parameters
    frame_idx = 0
    camera_idx = 0
    height = 256
    width = 256
    batch_size = 1
    num_cameras = 3

    train = zarr.open("Peract2_zarr/train.zarr", mode='r')

    rgb = train['rgb']  # zarr array
    depth = train['depth']  # zarr array
    extrinsics = train['extrinsics']  # zarr array
    intrinsics = train['intrinsics']  # zarr array
    preprioception = train['proprioception']  # zarr array
    # x, y, z, qx, qy, qz, qw, gripper total 8 dimensions
    

    rgb_img = rgb[frame_idx]  # shape: (3, 256, 256)
    depth_img = depth[frame_idx]  # shape: (256, 256)
    extrinsics = extrinsics[frame_idx]  # shape: (4, 4)
    intrinsics = intrinsics[frame_idx]  # shape: (3, 3)
    preprioception = preprioception[frame_idx]  # shape: (8)
    
    rgb_img = torch.from_numpy(rgb_img).float().cuda(non_blocking=True)
    depth_img = torch.from_numpy(depth_img).float().cuda(non_blocking=True)
    extrinsics = torch.from_numpy(extrinsics).float().cuda(non_blocking=True)
    intrinsics = torch.from_numpy(intrinsics).float().cuda(non_blocking=True)
    preprioception = torch.from_numpy(preprioception).float().cuda(non_blocking=True)

    print("depth shape:", depth_img.shape)
    print("rgb shape:", rgb_img.shape)
    print("extrinsic shape:", extrinsics.shape)
    print("intrinsic shape:", intrinsics.shape)
    print("proprioception shape:", preprioception.shape)

    # batch dimension 추가
    rgb_img = rgb_img.reshape(batch_size, num_cameras, 3, height, width)
    depth_img = depth_img.reshape(batch_size, num_cameras, height, width)
    extrinsics = extrinsics.reshape(batch_size, num_cameras, 4, 4)
    intrinsics = intrinsics.reshape(batch_size, num_cameras, 3, 3)
    preprioception = preprioception.reshape(batch_size, 1, -1)

    # convert to pointcloud
    depth2cloud = RLBenchDepth2Cloud((256, 256))
    pointcloud = depth2cloud(depth=depth_img, extrinsics=extrinsics, intrinsics=intrinsics)

    print("pointcloud shape:", pointcloud.shape)

    return rgb_img, pointcloud, preprioception

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
    