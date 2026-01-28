import torch
import zarr
import matplotlib.pyplot as plt
import os
from utils.depth2cloud.rlbench import RLBenchDepth2Cloud

# Zarr 데이터 로드
train = zarr.open("Peract2_zarr/train.zarr", mode='r')
val = zarr.open("Peract2_zarr/val.zarr", mode='r')
print(train.tree())

depth = train['depth']  # zarr array
extrinsics = train['extrinsics']  # zarr array
intrinsics = train['intrinsics']  # zarr array
print("depth shape:", depth.shape)
print("extrinsic shape:", extrinsics.shape)
print("intrinsic shape:", intrinsics.shape)
# 첫 번째 프레임, 첫 번째 카메라
freame_idx = 0
camera_idx = 0
height = 256
width = 256
batch_size = 1
num_cameras = 3

depth_img = depth[freame_idx]  # shape: (256, 256)
extrinsics = extrinsics[freame_idx]  # shape: (256, 256)
intrinsics = intrinsics[freame_idx]  # shape: (256, 256)

depth_img = depth_img.reshape(batch_size, num_cameras, height, width)
extrinsics = extrinsics.reshape(batch_size, num_cameras, 4, 4)
intrinsics = intrinsics.reshape(batch_size, num_cameras, 3, 3)

print("depth_img shape:", depth_img.shape)
print("extrinsic shape:", extrinsics.shape)
print("intrinsic shape:", intrinsics.shape)

depth2cloud = RLBenchDepth2Cloud((256, 256))

# convret numpy arrays to torch tensors
depth_img = torch.from_numpy(depth_img).float().cuda(non_blocking=True)
extrinsics = torch.from_numpy(extrinsics).float().cuda(non_blocking=True)
intrinsics = torch.from_numpy(intrinsics).float().cuda(non_blocking=True)

pointcloud = depth2cloud(depth=depth_img, extrinsics=extrinsics, intrinsics=intrinsics)

print("pointcloud shape:", pointcloud.shape)


