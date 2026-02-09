"""
Docstring for 01_test.stem


입력 데이터, 이미지, depth, intrinsic, extrinsic(proprioception) 을 모두 처리하면서도 HPT Stem과 같은 구조로 정리하기


Lerobot 데이터를 받을 수 있도록 하기

일단은 기존의 Zarr 데이터 셋을 활용할 것.

"""

import torch
from torch import Tensor, einsum, nn
from einops import rearrange, repeat
import torchvision
from torchvision import transforms

from typing import Callable, List, Optional

INIT_CONST = 0.02


class CrossAttention(nn.Module):
    """
    CrossAttention module used in the Perceiver IO model.

    Args:
        query_dim (int): The dimension of the query input.
        heads (int, optional): The number of attention heads. Defaults to 8.
        dim_head (int, optional): The dimension of each attention head. Defaults to 64.
        dropout (float, optional): The dropout probability. Defaults to 0.0.
    """

    def __init__(self, query_dim: int, heads: int = 8, dim_head: int = 64, dropout: float = 0.0):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = query_dim
        self.scale = dim_head**-0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, query_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, context: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """
        Forward pass of the CrossAttention module.

        Args:
            x (Tensor): The query input tensor.
            context (Tensor): The context input tensor.
            mask (Tensor, optional): The attention mask tensor. Defaults to None.

        Returns:
            Tensor: The output tensor.
        """
        h = self.heads
        q = self.to_q(x)
        k, v = self.to_kv(context).chunk(2, dim=-1)
        q = rearrange(q, "b n (h d) -> (b h) n d", h=h)
        k = rearrange(k, "b n (h d) -> (b h) n d", h=h)
        v = rearrange(v, "b n (h d) -> (b h) n d", h=h)

        sim = einsum("b i d, b j d -> b i j", q, k) * self.scale

        if mask is not None:
            # fill in the masks with negative values
            mask = rearrange(mask, "b ... -> b (...)")
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, "b j -> (b h) () j", h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)

        # dropout
        attn = self.dropout(attn)
        out = einsum("b i j, b j d -> b i d", attn, v)
        out = rearrange(out, "(b h) n d -> b n (h d)", h=h)
        return self.to_out(out)


class PolicyStem(nn.Module):
    """policy stem"""

    def __init__(self, **kwargs):
        super().__init__()

    def init_cross_attn(self, stem_spec, modality: str):
        """initialize cross attention module and the learnable tokens"""
        token_num = getattr(stem_spec, modality + "_crossattn_latent")
        self.tokens = nn.Parameter(torch.randn(1, token_num, stem_spec.embed_dim) * INIT_CONST)

        self.cross_attention = CrossAttention(
            stem_spec.embed_dim,
            heads=stem_spec.crossattn_heads,
            dim_head=stem_spec.crossattn_dim_head,
            dropout=stem_spec.crossattn_modality_dropout,
        )

    @property
    def device(self):
        return next(self.parameters()).device

    def compute_latent(self, x: Tensor) -> Tensor:
        """
        Computes the latent representations of input data by attention.

        Args:
            Input tensor with shape [32, 3, 1, 49, 512] representing the batch size,
            horizon, instance (e.g. num of views), number of features, and feature dimensions respectively.

        Returns:
            Output tensor with latent tokens, shape [32, 16, 128], where 16 is the number
            of tokens and 128 is the dimensionality of each token.

        Examples for vision features from ResNet:
        >>> x = np.random.randn(32, 3, 1, 49, 512)
        >>> latent_tokens = model.compute_latent(x)
        >>> print(latent_tokens.shape)
        (32, 16, 128)

        Examples for proprioceptive features:
        >>> x = np.random.randn(32, 3, 1, 7)
        >>> latent_tokens = model.compute_latent(x)
        >>> print(latent_tokens.shape)
        (32, 16, 128)
        """
        # Initial reshape to adapt to token dimensions
        stem_feat = self(x)  # (32, 3, 1, 49, 128)
        stem_feat = stem_feat.reshape(stem_feat.shape[0], -1, stem_feat.shape[-1])  # (32, 147, 128)

        # Replicating tokens for each item in the batch and computing cross-attention
        stem_tokens = self.tokens.repeat(len(stem_feat), 1, 1)  # (32, 16, 128)
        stem_tokens = self.cross_attention(stem_tokens, stem_feat)  # (32, 16, 128)
        return stem_tokens


from einops import rearrange, repeat
from torch.nn import functional as F
from collections import OrderedDict
from encoder.vision.clip import load_clip
from encoder.vision.fpn import EfficientFeaturePyramidNetwork
from depth2cloud.depth2cloud import Depth2Cloud
from embedding.position_encodings import RotaryPositionEncoding3D
from encoder.dendsity_based_sampler import density_based_sampler

class RoPESelfAttention(nn.Module):
    """
    Scaled Dot Product Self Attention with RoPE

    핵심 포인트:
    - q, k, v를 먼저 만든 다음
    - q, k에만 RoPE 회전 적용
    - v는 절대 건드리지 않음!
    """

    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        self.scale = self.head_dim ** -0.5

        # q, k, v를 한번에 생성하는 linear
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, rotary_pos: Optional[Tensor] = None) -> Tensor:
        """
        Args:
            x:          [B, N, D]    - feature (RGB feature 등)
            rotary_pos: [B, N, D, 2] - RoPE cos/sin (point cloud 좌표에서 생성)
                        [..., 0] = cos, [..., 1] = sin

        Returns:
            output: [B, N, D]
        """
        B, N, D = x.shape

        # ===== Step 1: feature에서 q, k, v 생성 =====
        # 이 시점에서는 위치 정보가 전혀 없음!
        qkv = self.qkv_proj(x)                         # [B, N, 3*D]
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)               # [3, B, H, N, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]               # 각각 [B, H, N, head_dim]

        # ===== Step 2: q, k에만 RoPE 회전 적용 =====
        # v는 건드리지 않는다!
        if rotary_pos is not None:
            cos = rotary_pos[..., 0]  # [B, N, D]
            sin = rotary_pos[..., 1]  # [B, N, D]

            # head 차원에 맞게 reshape: [B, N, D] → [B, H, N, head_dim]
            cos = cos.view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            sin = sin.view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

            # q 회전 (위치 m에 따라 회전)
            q = RotaryPositionEncoding3D.embed_rotary(q, cos, sin)
            # k 회전 (위치 n에 따라 회전)
            k = RotaryPositionEncoding3D.embed_rotary(k, cos, sin)
            # v는 회전하지 않음! ← 이것이 핵심

        # ===== Step 3: Scaled Dot Product Attention =====
        # q·k^T 의 결과는 위치 (m-n)에만 의존하게 됨 (RoPE의 마법)
        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout.p if self.training else 0.0
        )  # [B, H, N, head_dim]

        # ===== Step 4: 출력 =====
        attn_output = attn_output.permute(0, 2, 1, 3).reshape(B, N, D)  # [B, N, D]
        # 이부분의 피쳐의 차원을 바꿔주는 코드는 einops.rearrange 로 다시 작성하기
        attn_output = self.out_proj(attn_output)

        return attn_output


class SpatialStem_3DFA(PolicyStem):
    def __init__(
            self, 
            output_dim: int = 10,
            inner_feature_dim: int = 128,
            subsampling_factor: float = 5,
            # 내부 transformer Hyper Parameter
            num_heads: int = 8,
            num_self_attn_layers: int = 2,
            dropout: float = 0.0,
        ):
        super().__init__()
        # === 시스템 기본설정 ===
        self.subsampling_factor = subsampling_factor

        # === RGB 2d semantic feature embedding ===
        # CLIP
        self.backbone, self.normalize = load_clip()
        # backbone of clip from 3D-FA is 'ModifiedResNetFeatures'

        # Feature_pyramid
        # 차후 기본 feature_pyramid 구현과, 3D-FA 의 feature_pyramid 구현간의 차이점 비교하기
        # Postprocess scene features
        self.output_level = "res3"
        self.feature_pyramid = EfficientFeaturePyramidNetwork(
            [64, 256, 512, 1024, 2048],
            inner_feature_dim, output_level="res3"
        )

        # === Depth to PointCloud 변환 ===
        self.depth_to_point = Depth2Cloud((256, 256))

        # === 3D feature positonal embedding === 
        self.relative_pe_layer = RotaryPositionEncoding3D(inner_feature_dim)

        # === Define 3D Layer ===
                # ===== RoPE 위치 인코딩 생성기 =====
        # feature에 더하는 게 아니라, cos/sin 값만 만들어주는 역할
        self.relative_pe_layer = RotaryPositionEncoding3D(output_dim)

        # ===== RoPE Self Attention 레이어들 =====
        self.self_attn_layers = nn.ModuleList()
        self.self_attn_norms = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()
        self.ffn_norms = nn.ModuleList()

        for _ in range(num_self_attn_layers):
            # Self Attention (RoPE 포함)
            self.self_attn_layers.append(
                RoPESelfAttention(output_dim, num_heads, dropout)
            )
            self.self_attn_norms.append(nn.LayerNorm(output_dim))

            # Feed Forward Network
            self.ffn_layers.append(nn.Sequential(
                nn.Linear(output_dim, 4 * output_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(4 * output_dim, output_dim),
                nn.Dropout(dropout),
            ))
            self.ffn_norms.append(nn.LayerNorm(output_dim))

        # # ===== Cross Attention으로 고정 길이 latent token 생성 =====
        # self.init_cross_attn(
        #     token_num=token_num,
        #     embed_dim=output_dim,
        #     heads=num_heads,
        #     dim_head=output_dim // num_heads,
        #     dropout=dropout
        # )
        # 이 부분은 클래스 스스로 호출하는 함수가 아니라 외부에서 호출하도록 정의된 함수임.
        # HPT에서 이런 불편한 구조를 활용한 이유에 대해서 생각해 볼 필요가 있음.


    def _depth_inputs_to_point_cloud(self, depth_inputs: torch.Tensor, extrinsic: torch.Tensor, intrinsic: torch.Tensor)->torch.Tensor:
        point_cloud_egocentric_2d = self.depth_to_point(depth_inputs, extrinsic, intrinsic)
        return point_cloud_egocentric_2d

    def _vision_encoding(self, rgb_input: torch.Tensor):
        # 3D camera features
        num_cameras = rgb_input.shape[1]
        # Pass each view independently through backbone
        rgb_input = rearrange(rgb_input, "bt ncam c h w -> (bt ncam) c h w")
        rgb_input = self.normalize(rgb_input)# Resnet의 사전 조사된 통계 정보를 통해서 값을 표준화 처리해준다.
        rgb_feats = self.backbone(rgb_input)
        # Pass visual features through feature pyramid network
        rgb_feats:torch.Tensor = self.feature_pyramid(rgb_feats)[self.output_level]

        return rgb_feats

    def _language_encoding(self):
        # Attention from vision to language
        # # Encode language
        # instruction = self.text_encoder(text)
        # instr_feats = self.instruction_encoder(instruction)

        # rgb3d_feats = self.vl_attention(seq1=rgb3d_feats, seq2=instr_feats)[-1]
        # -> 일단 언어 피쳐는 제외한 체로 실험하자. 
        pass

    def _run_dps(self, features: torch.Tensor, pos: torch.Tensor):
        # features (B, Np, F)
        # context_pos (B, Np, 3)
        # outputs of analogous shape, with smaller Np
        if self.subsampling_factor == 1:
            return features, pos

        bs, npts, ch = features.shape
        sampled_inds = density_based_sampler(features, self.subsampling_factor)
        # 원래의 3D-FA의 코드에서 왜, PointCloud가 아닌, 3Drgb resnet Feature에 대해서 거리를 측정하고 Subsampling 하는거지? 
        # 그것도 코사인 유사도 기준이 아닌 유클리드 거리 기준으로 측정하는 거지?

        # Sample features
        expanded_inds = sampled_inds.unsqueeze(-1).expand(-1, -1, ch)  # B Np F
        sampled_features = torch.gather(features, 1, expanded_inds)

        # If positions are None, return
        if pos is None:
            return sampled_features, None

        # Else sample positions
        expanded_inds = sampled_inds.unsqueeze(-1).expand(-1, -1, 3)  # B Np 3
        sampled_pos = torch.gather(pos, 1, expanded_inds)
        return sampled_features, sampled_pos
    
    def forward_self_attention(self, rgb_feats: torch.Tensor, point_cloud: torch.Tensor):
        """
        Args:
            rgb_feats:    [B, N, D]  - RGB에서 추출한 feature
            point_cloud:  [B, N, 3]  - 대응하는 3D 좌표

        Returns:
            latent_tokens: [B, token_num, D]

        주의: rgb_feats와 point_cloud는 절대 합쳐지지 않는다!
        """

        # ===== 1. RoPE용 cos/sin 생성 =====
        # point_cloud 좌표 → cos/sin 값 (feature와 분리!)
        rotary_pos = self.relative_pe_layer(point_cloud)  # [B, N, D, 2]

        # ===== 2. Self Attention (RoPE 적용) =====
        x = rgb_feats
        for i in range(len(self.self_attn_layers)):
            # Pre-norm + Self Attention + Residual
            residual = x
            x = self.self_attn_norms[i](x)
            x = self.self_attn_layers[i](x, rotary_pos)  # ← feature와 pos를 분리해서 전달!
            x = residual + x

            # Pre-norm + FFN + Residual
            residual = x
            x = self.ffn_norms[i](x)
            x = self.ffn_layers[i](x)
            x = residual + x

        return x

    def forward(self, rgb_input: torch.Tensor, depth_inputs: torch.Tensor, extrinsic: torch.Tensor, intrinsic: torch.Tensor) -> torch.Tensor:
        # RGB Feature
        rgb_feats = self._vision_encoding(rgb_input)

        # Point cloud
        # dimension = (bt ncam c h w)
        point_cloud_2D_array = self._depth_inputs_to_point_cloud(depth_inputs, extrinsic, intrinsic)

        num_cameras = point_cloud_2D_array.shape[1]
        # Interpolate point cloud to get the corresponding locations of RGB feature
        feat_h, feat_w = rgb_feats.shape[-2:]
        point_cloud_2D_array = F.interpolate(
            rearrange(point_cloud_2D_array, "bt ncam c h w -> (bt ncam) c h w"),
            (feat_h, feat_w),
            mode='bilinear'
        )
        # interpolation하는 이유는 
        # pcd 포인트의 개수는 원본이미지의 화소와 같은 수를 갖지만,
        # rgb_feats는 Resnet의 Convolution과 Avgpooling에서의 Stride 에 의해 차원의 길이가 줄어들기 때문.
        # 피쳐의 수가 서로 달라 pcd 데이터에 rgb_feats를 입히는 과정에서 대응되지 않는 점을 제외시키는 역할을 함.
        # Merge different cameras

        rgb_feats = rearrange(
            rgb_feats,
            "(bt ncam) c h w -> bt (ncam h w) c", ncam=num_cameras
        )
        
        # Merge different cameras
        point_cloud = rearrange(
            point_cloud_2D_array,
            "(bt ncam) c h w -> bt (ncam h w) c", ncam=num_cameras
        )

        rgb_feature_subsampled, point_cloud_subsampled = self._run_dps(rgb_feature_subsampled, point_cloud)

        # point_cloud, point_cloud_subsampled 정보는 위치 임베딩의 역할을 한다. 
        # 각 피쳐가 3D 공간상에서 어떤 위치에 존재했는지 피쳐간의 공간적 위치 차이를 
        # 포인트 클라우드 점의 위치를 직접 활용한 positional embedding을 이용하여 표현한다.
        latent_tokens = self.forward_self_attention(rgb_feature_subsampled, point_cloud_subsampled)

        # 토큰을 한번더 프로젝션해서 feature로 변환해 주어야함...?
        # 아직 이상태는 어텐션 연산을 수행하였을 뿐 feadforward layer가 적용되지 않은 상태?
        # 원본 코드를 보니 토큰 차원이 포함된 데이터를 반환하는 것이 맞는 것으로 보임.

        # 차후 코드 검증을 Claud에 돌려보기
        return latent_tokens



if __name__ == "__main__":
    # B, T, N, D = 4, 3, 1024, 128
    # # Batch, time, Num_of_copy, feature_dim
    # # 입력 데이터 (실제로는 ResNet + FPN에서 나옴)
    # rgb_feats = torch.randn(B, T, N, D)       # RGB feature
    # point_cloud = torch.randn(B, T, N, 3)     # 3D 좌표 (x, y, z)
    import zarr

    # Zarr 데이터 로드
    train = zarr.open("Peract2_zarr/train.zarr", mode='r')
    val = zarr.open("Peract2_zarr/val.zarr", mode='r')
    print(train.tree())

    rgb = train['rgb']  # zarr array
    depth = train['depth']  # zarr array
    extrinsics = train['extrinsics']  # zarr array
    intrinsics = train['intrinsics']  # zarr array
    print("rgb shape:", rgb.shape)
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

    rgb_img = rgb[freame_idx]  # shape: (256, 256)
    depth_img = depth[freame_idx]  # shape: (256, 256)
    extrinsics = extrinsics[freame_idx]  # shape: (256, 256)
    intrinsics = intrinsics[freame_idx]  # shape: (256, 256)

    rgb_img = rgb_img.reshape(batch_size, num_cameras, 3, height, width)
    depth_img = depth_img.reshape(batch_size, num_cameras, height, width)
    extrinsics = extrinsics.reshape(batch_size, num_cameras, 4, 4)
    intrinsics = intrinsics.reshape(batch_size, num_cameras, 3, 3)

    print("rgb_img shape:", rgb_img.shape)
    print("depth_img shape:", depth_img.shape)
    print("extrinsic shape:", extrinsics.shape)
    print("intrinsic shape:", intrinsics.shape)


    # convret numpy arrays to torch tensors
    rgb_img = torch.from_numpy(rgb_img).float().cuda(non_blocking=True)
    depth_img = torch.from_numpy(depth_img).float().cuda(non_blocking=True)
    extrinsics = torch.from_numpy(extrinsics).float().cuda(non_blocking=True)
    intrinsics = torch.from_numpy(intrinsics).float().cuda(non_blocking=True)

    # 모델 생성
    stem =  SpatialStem_3DFA(
        output_dim=64,
        inner_feature_dim=128,
        subsampling_factor=5,
        # 내부 transformer Hyper Parameter
        num_heads=8,
        num_self_attn_layers=2,
        dropout=0.0,
    )
    
    output = stem(rgb_img, depth_img, extrinsics, intrinsics)
    
    print(f"Input  - rgb_img: {rgb_img.shape}, depth_img: {depth_img.shape}, extrinsics: {extrinsics.shape}, intrinsics: {intrinsics.shape}")
    print(f"Output - latent_tokens: {output.shape}")