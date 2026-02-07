import torch
import zarr
import os
import numpy as np
from utils.depth2cloud.rlbench import RLBenchDepth2Cloud

# PLY 저장 함수
def save_pointcloud_as_ply(points, filename="../data/cloud.ply"):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    N = points.shape[0]
    with open(filename, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {N}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")
        for i in range(N):
            f.write(f"{points[i,0]} {points[i,1]} {points[i,2]}\n")
    print(f"[+] Saved point cloud to: {filename}")

# Zarr 데이터 로드
train = zarr.open("Peract2_zarr/train.zarr", mode='r')
val = zarr.open("Peract2_zarr/val.zarr", mode='r')
print(train.tree())

depth = train['depth']
extrinsics = train['extrinsics']
intrinsics = train['intrinsics']

print("depth shape:", depth.shape)
print("extrinsic shape:", extrinsics.shape)
print("intrinsic shape:", intrinsics.shape)

# 첫 번째 프레임
frame_idx = 0
height = 256
width = 256
batch_size = 1
num_cameras = 3

# 데이터 슬라이스
depth_img = depth[frame_idx]        # (Nc, H, W)
extrinsics_img = extrinsics[frame_idx]  # (Nc, 4, 4)
intrinsics_img = intrinsics[frame_idx]  # (Nc, 3, 3)

# reshape to (B, Nc, ...)
depth_img = depth_img.reshape(batch_size, num_cameras, height, width)
extrinsics_img = extrinsics_img.reshape(batch_size, num_cameras, 4, 4)
intrinsics_img = intrinsics_img.reshape(batch_size, num_cameras, 3, 3)

print("depth_img shape:", depth_img.shape)
print("extrinsics_img shape:", extrinsics_img.shape)
print("intrinsics_img shape:", intrinsics_img.shape)

# RLBench Depth2Cloud
depth2cloud = RLBenchDepth2Cloud((256, 256))

# convert numpy arrays to torch tensors
depth_img = torch.from_numpy(depth_img).float().cuda(non_blocking=True)
extrinsics_img = torch.from_numpy(extrinsics_img).float().cuda(non_blocking=True)
intrinsics_img = torch.from_numpy(intrinsics_img).float().cuda(non_blocking=True)

pointcloud = depth2cloud(depth=depth_img, extrinsics=extrinsics_img, intrinsics=intrinsics_img)

print("pointcloud shape:", pointcloud.shape)  # (B, Nc, 3, H, W)

# === 포인트클라우드 저장 ===
# 여기서는 카메라 0번만 저장 — 필요하면 for문 돌리면 됨
pc = pointcloud[0, 0].permute(1, 2, 0).reshape(-1, 3).cpu().numpy()  # (H*W, 3)

save_pointcloud_as_ply(pc, "00_test/data/cloud_cam0.ply")
