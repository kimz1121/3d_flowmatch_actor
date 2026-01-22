import zarr
import matplotlib.pyplot as plt
import os

# 데이터 저장 경로 준비
save_dir = "00_test/data"
os.makedirs(save_dir, exist_ok=True)

# Zarr 데이터 로드
train = zarr.open("Peract2_zarr/train.zarr", mode='r')
val = zarr.open("Peract2_zarr/val.zarr", mode='r')
print(train.tree())

# <RGB 데이터 로드>-------------------------------
rgb = train['rgb']  # zarr array
print("rgb shape:", rgb.shape)
# 첫 번째 프레임, 첫 번째 카메라
img = rgb[0, 0]  # (3, 256, 256)
# C,H,W -> H,W,C
img = img.transpose(1, 2, 0)  # (256, 256, 3)

plt.imshow(img)
# plt.title("RGB[0, camera0]")
plt.axis('off')

# 이미지 저장
save_path = os.path.join(save_dir, "rgb_0_cam0.png")
plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
plt.close()

print(f"저장 완료: {save_path}")


# <Depth 데이터 로드>-------------------------------
depth = train['depth']  # zarr array
print("depth shape:", depth.shape)

# 첫 번째 프레임, 첫 번째 카메라
depth_img = depth[0, 0]  # shape: (256, 256)

plt.imshow(depth_img, cmap='YlGnBu')
plt.axis('off')

# 이미지 저장
save_path = os.path.join(save_dir, "depth_0_cam0.png")
plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
plt.close()

print(f"저장 완료: {save_path}")