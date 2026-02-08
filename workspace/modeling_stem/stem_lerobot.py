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


class MLPStem(PolicyStem):
    def __init__(
        self,
        input_dim: int = 10,
        output_dim: int = 10,
        widths: Tuple[int] = (512, 512),
        tanh_end: bool = False,
        ln: bool = True,
        num_of_copy: int = 1,
    ) -> None:
        """MLP Stem class"""
        super().__init__()
        modules = [nn.Linear(input_dim, widths[0]), nn.SiLU()]

        for i in range(len(widths) - 1):
            modules.extend([nn.Linear(widths[i], widths[i + 1])])
            if ln:
                modules.append(nn.LayerNorm(widths[i + 1]))
            modules.append(nn.SiLU())

        modules.append(nn.Linear(widths[-1], output_dim))
        if tanh_end:
            modules.append(nn.Tanh())
        self.net = nn.Sequential(*modules)
        self.num_of_copy = num_of_copy
        if self.num_of_copy > 1:
            self.net = nn.ModuleList([nn.Sequential(*modules) for _ in range(num_of_copy)])

    def forward(self, x: Tensor) -> Tensor:
        """
        Performs a forward pass of the model.
        Args:
            x: Image tensor with shape [B, T, N, 3, H, W]
        Returns:
            Flatten tensor with shape [B, M, 512]
        """
        if self.num_of_copy > 1:
            out = []
            iter_num = min(self.num_of_copy, x.shape[1])
            for idx in range(iter_num):
                input = x[:, idx]
                net = self.net[idx]
                out.append(net(input))
            y = torch.stack(out, dim=1)
        else:
            y = self.net(x)
        return y
    

class SpatialStem_3DFA(PolicyStem):
    def __init__(
            self, 
            output_dim: int = 10,
            num_of_copy: int = 1,# ? 중요한가? depth의 경우는 합쳐질 수있지 않나? 카메라 개수 차원을 추가해서 반환? 
        ):
        super().__init__()

        # 가져온 예시 코드
        input_dim = input_dim[0]
        self.input_dim = input_dim
        self.out_dim = output_dim
        self.point_num = point_num
        self.token_num = token_num

        layers = []
        for oc in widths:
            layers.extend(
                [
                    nn.Conv1d(input_dim, oc, 1, bias=False), 
                    nn.LayerNorm((oc, self.point_num)), 
                ]
            )
            input_dim = oc

        self.linear = nn.Linear(widths[-1], output_dim)
        self.net = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        B, T, D, N = inputs.shape
        inputs = inputs.view(-1, D, N)
        x = self.net(inputs)

        token_perpoint_num = self.point_num // self.token_num
        x = x[:, :, ::token_perpoint_num].transpose(-1, -2) 
        token_feat = self.linear(x).view(B, T, -1, self.out_dim)
        return token_feat


from encoder.vision.clip import load_clip
from einops import rearrange, repeat
from torch.nn import functional as F
from encoder.vision.fpn import EfficientFeaturePyramidNetwork
from collections import OrderedDict
from depth2cloud.depth2cloud import Depth2Cloud
from embedding.position_encodings import RotaryPositionEncoding3D
from encoder.dendsity_based_sampler import density_based_sampler
    
class SpatialStem_3DFA(PolicyStem):
    def __init__(
            self, 
            output_dim: int = 10,
            inner_feature_dim: int = 128,
            subsampling_factor: float = 5
        ):
        super().__init__()
        self.subsampling_factor = subsampling_factor

        self.backbone, self.normalize = load_clip()
        # backbone of clip from 3D-FA is 'ModifiedResNetFeatures'

        # 차후 기본 feature_pyramid 구현과, 3D-FA 의 feature_pyramid 구현간의 차이점 비교하기
        # Postprocess scene features
        self.output_level = "res3"
        self.feature_pyramid = EfficientFeaturePyramidNetwork(
            [64, 256, 512, 1024, 2048],
            inner_feature_dim, output_level="res3"
        )

        self.depth_to_point = Depth2Cloud(256, 256)

        self.relative_pe_layer = RotaryPositionEncoding3D(inner_feature_dim)

    def _depth_inputs_to_point_cloud(self, depth_inputs: torch.Tensor, extrinsic: torch.Tensor, intrinsic: torch.Tensor)->torch.Tensor:
        point_cloud_egocentric_2d = self.depth_to_point(depth_inputs, extrinsic, intrinsic)
        return point_cloud_egocentric_2d
    
    def _3d_point_cloud_encoding(self, poin):
        
        pass


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

        rgb_subsampled, point_cloud_subsampled = self._run_dps(rgb_subsampled, point_cloud)

        # get 3d positional embedding
        point_cloud_pos_embedded = self.relative_pe_layer(point_cloud)
        
        # TODO: rgb_feats 피쳐를 사용하고, point_cloud_pos_embedded 정보를 위치 임베딩으로 활용하는 코드를 만들고 싶어  
        rgb_feats
        point_cloud_pos_embedded


        pass




