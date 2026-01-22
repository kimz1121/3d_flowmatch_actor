import zarr
import matplotlib.pyplot as plt
import os
from utils.depth2cloud.rlbench import RLBenchDepth2Cloud

# Zarr 데이터 로드
train = zarr.open("Peract2_zarr/train.zarr", mode='r')
val = zarr.open("Peract2_zarr/val.zarr", mode='r')
print(train.tree())

depth = train['depth']  # zarr array
extrinsic = train['extrinsics']  # zarr array
intrinsic = train['intrinsics']  # zarr array
print("depth shape:", depth.shape)
print("extrinsic shape:", extrinsic.shape)
print("intrinsic shape:", intrinsic.shape)
# 첫 번째 프레임, 첫 번째 카메라
freame_idx = 0
camera_idx = 0
height = 256
width = 256
batch_size = 1
num_cameras = 3

depth_img = depth[freame_idx]  # shape: (256, 256)
extrinsic = extrinsic[freame_idx]  # shape: (256, 256)
intrinsic = intrinsic[freame_idx]  # shape: (256, 256)

depth_img = depth_img.reshape(batch_size, num_cameras, height, width)
extrinsic = extrinsic.reshape(batch_size, num_cameras, 4, 4)
intrinsic = intrinsic.reshape(batch_size, num_cameras, 3, 3)

print("depth_img shape:", depth_img.shape)
print("extrinsic shape:", extrinsic.shape)
print("intrinsic shape:", intrinsic.shape)

depth2cloud = RLBenchDepth2Cloud((256, 256))

pointcloud = depth2cloud(depth=depth_img, extrinsic=extrinsic, intrinsic=intrinsic)

print("pointcloud shape:", pointcloud.shape)
