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

    def forward(self, x: torch.Tensor, context: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of the CrossAttention module.

        Args:
            x (torch.Tensor): The query input tensor.
            context (torch.Tensor): The context input tensor.
            mask (torch.Tensor, optional): The attention mask tensor. Defaults to None.

        Returns:
            torch.Tensor: The output tensor.
        """
        h = self.heads
        q = self.to_q(x)
        k, v = self.to_kv(context).chunk(2, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> (b h) n d", h=h), (q, k, v))
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
        """ initialize cross attention module and the learnable tokens """
        token_num = getattr(stem_spec.crossattn_latent, modality)
        self.tokens = nn.Parameter(
            torch.randn(1, token_num, stem_spec.modality_embed_dim) * INIT_CONST
        )
        # INIT_CONST는 학습 초기에 learnable parameter가 너무 큰 영향을 주지 않도록 하는 도구인 듯.
        
        self.cross_attention = CrossAttention(
            stem_spec.modality_embed_dim,
            heads=stem_spec.crossattn_heads,
            dim_head=stem_spec.crossattn_dim_head,
            dropout=stem_spec.crossattn_modality_dropout,
        )

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True

    def save(self, path : str):
        torch.save(self.state_dict(), path)

    @property
    def device(self):
        return next(self.parameters()).device

    def compute_latent(self, x: torch.Tensor) -> torch.Tensor:
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
        # token과 feature에 cross attention을 적용하는 코드를 통해서 출력 피쳐의 차원을 항상 일정하게 유지시켜준다는 장점이 생긴다.
        # 내부의 learnable token 의 차원으로 cross attention의 결과가 결정되기 때문임.

        # Initial reshape to adapt to token dimensions
        # (32, 3, 1, 49, 128)
        stem_feat = self(x)  
        stem_feat = stem_feat.reshape(stem_feat.shape[0], -1, stem_feat.shape[-1])  # (32, 147, 128)
        # Replicating tokens for each item in the batch and computing cross-attention
        stem_tokens = self.tokens.repeat(len(stem_feat), 1, 1)  # (32, 16, 128)
        stem_tokens = self.cross_attention(stem_tokens, stem_feat)  # (32, 16, 128)
        return stem_tokens
    
class MLP(PolicyStem):
    def __init__(
        self,
        input_dim: int = 10,
        output_dim: int = 10,
        widths: List[int] = [512],
        tanh_end: bool = False,
        ln: bool = True,
        num_of_copy: int = 1,
        **kwargs,
    ) -> None:
        """vanilla MLP class"""
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass of the model.
        Args:
            x: Image tensor with shape [B, T, N, 3, H, W] representing the batch size, 
            horizon, instance (e.g. num of views)
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


class ResNet(PolicyStem):
    def __init__(
        self,
        output_dim: int = 10,
        weights: str = "DEFAULT",
        resnet_model: str = "resnet18",
        num_of_copy: int = 1,
        **kwargs,
    ) -> None:
        """ResNet Encoder for Images"""  
        super().__init__()
        pretrained_model = getattr(torchvision.models, resnet_model)(weights=weights)

        # by default we use a separate image encoder for each view in downstream evaluation
        self.num_of_copy = num_of_copy
        self.net = nn.Sequential(*list(pretrained_model.children())[:-2])

        if num_of_copy > 1:
            self.net = nn.ModuleList(
                [nn.Sequential(*list(pretrained_model.children())[:-2]) for _ in range(num_of_copy)]
            )
        self.input = input
        self.out_dim = output_dim
        self.to_tensor = transforms.ToTensor()
        self.proj = nn.Linear(512, output_dim)
        self.avgpool = nn.AvgPool2d(7, stride=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass of the model.
        Args:
            x: Image tensor with shape [B, T, N, 3, H, W] representing the batch size, 
            horizon, instance (e.g. num of views)
        Returns:
            Flatten tensor with shape [B, M, 512]     
        """
        B, *_, H, W = x.shape
        x = x.view(len(x), -1, 3, H, W)
        if self.num_of_copy > 1:
            # separate encoding for each view
            # resnet stem 에서는 MLP stem에서와 같이 self.num_of_copy를 활용하나,
            # 시간에 따른 모델 구분이 아닌, 카메라 개수에 따른 모델 구분을 위하여 사용된다.
            out = []
            iter_num = min(self.num_of_copy, x.shape[1])
            for idx in range(iter_num):
                input = x[:, idx]
                net = self.net[idx]
                out.append(net(input))
            feat = torch.stack(out, dim=1)
        else:
            x = x.view(-1, 3, H, W)
            feat = self.net(x)
        # concat along time
        feat = feat.reshape(B, feat.shape[1], -1).contiguous()
        feat = rearrange(feat, "B L T -> B T L") # (batchsize, number of tokens, resnet feature dimension=512)
        feat = self.proj(feat) # project to (batchsize, number of tokens, output_dim)
        return feat


class PointNet(PolicyStem):
    """Simple Pointnet-Like Network"""

    def __init__(
        self, output_dim: int = 3, input_dim: int = 4, 
        widths: List[int] = [64, 256, 512, 512], dim: int = 1, 
        token_num: int = 64, point_num: int = 1024, **kwargs
    ) -> None:
        super(PointNet, self).__init__()
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
