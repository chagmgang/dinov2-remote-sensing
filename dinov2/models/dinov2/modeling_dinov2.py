from typing import Callable, Optional, Tuple, Union, Dict, Any, List
from dataclasses import dataclass
from functools import partial

import math
import torch
import torch.nn as nn
from transformers import PreTrainedModel
from transformers.file_utils import ModelOutput

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.init import trunc_normal_
from torch.nn.utils import weight_norm
from timm.models.layers import to_2tuple, trunc_normal_, DropPath

from .configuration_dinov2 import DINOv2Config
from .param_groups import get_params_groups_with_decay


def drop_add_residual_stochastic_depth(
    x: torch.Tensor,
    residual_func: Callable[[torch.Tensor], torch.Tensor],
    sample_drop_ratio: float = 0.0,
) -> torch.Tensor:
    # 1) extract subset using permutation
    b, n, d = x.shape
    sample_subset_size = max(int(b * (1 - sample_drop_ratio)), 1)
    brange = (torch.randperm(b, device=x.device))[:sample_subset_size]
    x_subset = x[brange]

    # 2) apply residual_func to get residual
    residual = residual_func(x_subset)

    x_flat = x.flatten(1)
    residual = residual.flatten(1)

    residual_scale_factor = b / sample_subset_size

    # 3) add the residual
    x_plus_residual = torch.index_add(x_flat, 0, brange, residual.to(dtype=x.dtype), alpha=residual_scale_factor)
    return x_plus_residual.view_as(x)


def get_branges_scales(x, sample_drop_ratio=0.0):
    b, n, d = x.shape
    sample_subset_size = max(int(b * (1 - sample_drop_ratio)), 1)
    brange = (torch.randperm(b, device=x.device))[:sample_subset_size]
    residual_scale_factor = b / sample_subset_size
    return brange, residual_scale_factor


def add_residual(x, brange, residual, residual_scale_factor, scaling_vector=None):
    if scaling_vector is None:
        x_flat = x.flatten(1)
        residual = residual.flatten(1)
        x_plus_residual = torch.index_add(x_flat, 0, brange, residual.to(dtype=x.dtype), alpha=residual_scale_factor)
    else:
        x_plus_residual = scaled_index_add(
            x, brange, residual.to(dtype=x.dtype), scaling=scaling_vector, alpha=residual_scale_factor
        )
    return x_plus_residual


attn_bias_cache: Dict[Tuple, Any] = {}


def get_attn_bias_and_cat(x_list, branges=None):
    """
    this will perform the index select, cat the tensors, and provide the attn_bias from cache
    """
    batch_sizes = [b.shape[0] for b in branges] if branges is not None else [x.shape[0] for x in x_list]
    all_shapes = tuple((b, x.shape[1]) for b, x in zip(batch_sizes, x_list))
    if all_shapes not in attn_bias_cache.keys():
        seqlens = []
        for b, x in zip(batch_sizes, x_list):
            for _ in range(b):
                seqlens.append(x.shape[1])
        attn_bias = fmha.BlockDiagonalMask.from_seqlens(seqlens)
        attn_bias._batch_sizes = batch_sizes
        attn_bias_cache[all_shapes] = attn_bias

    if branges is not None:
        cat_tensors = index_select_cat([x.flatten(1) for x in x_list], branges).view(1, -1, x_list[0].shape[-1])
    else:
        tensors_bs1 = tuple(x.reshape([1, -1, *x.shape[2:]]) for x in x_list)
        cat_tensors = torch.cat(tensors_bs1, dim=1)

    return attn_bias_cache[all_shapes], cat_tensors


def drop_add_residual_stochastic_depth_list(
    x_list: List[torch.Tensor],
    residual_func: Callable[[torch.Tensor, Any], torch.Tensor],
    sample_drop_ratio: float = 0.0,
    scaling_vector=None,
) -> torch.Tensor:
    # 1) generate random set of indices for dropping samples in the batch
    branges_scales = [get_branges_scales(x, sample_drop_ratio=sample_drop_ratio) for x in x_list]
    branges = [s[0] for s in branges_scales]
    residual_scale_factors = [s[1] for s in branges_scales]

    # 2) get attention bias and index+concat the tensors
    attn_bias, x_cat = get_attn_bias_and_cat(x_list, branges)

    # 3) apply residual_func to get residual, and split the result
    residual_list = attn_bias.split(residual_func(x_cat, attn_bias=attn_bias))  # type: ignore

    outputs = []
    for x, brange, residual, residual_scale_factor in zip(x_list, branges, residual_list, residual_scale_factors):
        outputs.append(add_residual(x, brange, residual, residual_scale_factor, scaling_vector).view_as(x))
    return outputs


def named_apply(fn: Callable, module: nn.Module, name="", depth_first=True, include_root=False) -> nn.Module:
    if not depth_first and include_root:
        fn(module=module, name=name)
    for child_name, child_module in module.named_children():
        child_name = ".".join((name, child_name)) if name else child_name
        named_apply(fn=fn, module=child_module, name=child_name, depth_first=depth_first, include_root=True)
    if depth_first and include_root:
        fn(module=module, name=name)
    return module
    

def init_weights_vit_timm(module: nn.Module, name: str = ""):
    """ViT weight initialization, original timm impl (for reproducibility)"""
    if isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)

            
def _build_mlp(nlayers, in_dim, bottleneck_dim, hidden_dim=None, use_bn=False, bias=True):
    if nlayers == 1:
        return nn.Linear(in_dim, bottleneck_dim, bias=bias)
    else:
        layers = [nn.Linear(in_dim, hidden_dim, bias=bias)]
        if use_bn:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.GELU())
        for _ in range(nlayers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim, bias=bias))
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
        layers.append(nn.Linear(hidden_dim, bottleneck_dim, bias=bias))
        return nn.Sequential(*layers)


class DINOHead(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        use_bn=False,
        nlayers=3,
        hidden_dim=2048,
        bottleneck_dim=256,
        mlp_bias=True,
    ):
        super().__init__()
        nlayers = max(nlayers, 1)
        self.mlp = _build_mlp(nlayers, in_dim, bottleneck_dim, hidden_dim=hidden_dim, use_bn=use_bn, bias=mlp_bias)
        self.apply(self._init_weights)
        self.last_layer = nn.Linear(bottleneck_dim, out_dim, bias=False)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        eps = 1e-6 if x.dtype == torch.float16 else 1e-12
        x = nn.functional.normalize(x, dim=-1, p=2, eps=eps)
        x = self.last_layer(x)
        return x


class LayerScale(nn.Module):
    def __init__(
        self,
        dim: int,
        init_values: Union[float, torch.Tensor] = 1e-5,
        inplace: bool = False,
    ) -> None:
        super().__init__()
        self.inplace = inplace
        self.lambda1 = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mul_(self.lambda1) if self.inplace else x * self.lambda1


class PatchEmbed(nn.Module):

    def __init__(
        self,
        img_size: Union[int, Tuple[int, int]] = 224,
        patch_size: Union[int, Tuple[int, int]] = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        norm_layer: Optional[Callable] = None,
        flatten_embedding: bool = True,
    ) -> None:
        super().__init__()

        image_HW = to_2tuple(img_size)
        patch_HW = to_2tuple(patch_size)
        patch_grid_size = (
            image_HW[0] // patch_HW[0],
            image_HW[1] // patch_HW[1],
        )

        self.img_size = image_HW
        self.patch_size = patch_HW
        self.patches_resolution = patch_grid_size
        self.num_patches = patch_grid_size[0] * patch_grid_size[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.flatten_embedding = flatten_embedding

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_HW, stride=patch_HW)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, H, W = x.shape
        patch_H, patch_W = self.patch_size

        assert H % patch_H == 0, f"Input image height {H} is not a multiple of patch height {patch_H}"
        assert W % patch_W == 0, f"Input image width {W} is not a multiple of patch width: {patch_W}"

        x = self.proj(x)  # B C H W
        H, W = x.size(2), x.size(3)
        x = x.flatten(2).transpose(1, 2)  # B HW C
        x = self.norm(x)
        if not self.flatten_embedding:
            x = x.reshape(-1, H, W, self.embed_dim)  # B H W C
        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        qkv = qkv.contiguous()

        q, k, v = qkv.unbind()

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2)

        x = x.reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def get_attn(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        qkv = qkv.contiguous()

        q, k, v = qkv.unbind()

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2)

        x = x.reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


class Mlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        drop: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values=None,
        drop_path: float = 0.0,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        attn_class: Callable[..., nn.Module] = Attention,
        ffn_layer: Callable[..., nn.Module] = Mlp,
    ) -> None:
        super().__init__()
        # print(f"biases: qkv: {qkv_bias}, proj: {proj_bias}, ffn: {ffn_bias}")
        self.norm1 = norm_layer(dim)
        self.attn = attn_class(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = ffn_layer(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
            bias=ffn_bias,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.sample_drop_ratio = drop_path

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        def attn_residual_func(x: torch.Tensor) -> torch.Tensor:
            return self.ls1(self.attn(self.norm1(x)))

        def ffn_residual_func(x: torch.Tensor) -> torch.Tensor:
            return self.ls2(self.mlp(self.norm2(x)))

        if self.training and self.sample_drop_ratio > 0.1:
            # the overhead is compensated only for a drop path rate larger than 0.1
            x = drop_add_residual_stochastic_depth(
                x,
                residual_func=attn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
            )
            x = drop_add_residual_stochastic_depth(
                x,
                residual_func=ffn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
            )
        elif self.training and self.sample_drop_ratio > 0.0:
            x = x + self.drop_path1(attn_residual_func(x))
            x = x + self.drop_path1(ffn_residual_func(x))  # FIXME: drop_path2
        else:
            x = x + attn_residual_func(x)
            x = x + ffn_residual_func(x)
        return x

    def get_attention(self, x: torch.Tensor) -> torch.Tensor:

        def attn_residual_func(x: torch.Tensor) -> torch.Tensor:
            x = self.norm1(x)
            x, attn = self.attn.get_attn(x)
            x = self.ls1(x)
            return x, attn

        def ffn_residual_func(x: torch.Tensor) -> torch.Tensor:
            return self.ls2(self.mlp(self.norm2(x)))

        y, attn = attn_residual_func(x)
        x = x + y
        x = x + ffn_residual_func(x)
        return x, attn


@dataclass
class ViTModelOutput(ModelOutput):

    x_norm_clstoken: torch.FloatTensor = None
    x_norm_regtokens: torch.FloatTensor = None
    x_norm_patchtokens: torch.FloatTensor = None
    x_prenorm: torch.FloatTensor = None
    x_lastlayer_attn_cls: torch.FloatTensor = None
    x_lastlayer_attn_regtokens: torch.FloatTensor = None
    x_lastlayer_attn_patchtokens: torch.FloatTensor = None
    x_lastlayer_attn_alltokens: torch.FloatTensor = None
    

@dataclass
class DINOv2ModelOutput(ModelOutput):

    loss: torch.FloatTensor = None
    loss_dict: dict[torch.FloatTensor] = None


class ViTPretrainedModel(PreTrainedModel):

    config_class = DINOv2Config
    base_model_prefix = 'model'
    supports_gradient_checkpointing = True
    _no_split_modules = []

    def _init_weights(self, m):
        pass


class ViT(ViTPretrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.gradient_checkpointing = False

        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.num_features = self.embed_dim = config.embed_dim
        self.num_tokens = 1
        self.n_blocks = config.depth
        self.num_heads = config.num_heads
        self.patch_size = config.patch_size
        self.num_register_tokens = config.num_register_tokens
        self.interpolate_antialias = config.interpolate_antialias
        self.interpolate_offset = config.interpolate_offset

        self.patch_embed = PatchEmbed(img_size=config.img_size, patch_size=config.patch_size, in_chans=config.in_chans, embed_dim=config.embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, config.embed_dim))
        assert config.num_register_tokens >= 0
        self.register_tokens = (
            nn.Parameter(torch.zeros(1, config.num_register_tokens, config.embed_dim)) if config.num_register_tokens else None
        )

        if config.drop_path_uniform is True:
            dpr = [config.drop_path_rate] * config.depth
        else:
            dpr = [x.item() for x in torch.linspace(0, config.drop_path_rate, config.depth)]

        blocks_list = [
            Block(
                dim=config.embed_dim,
                num_heads=config.num_heads,
                mlp_ratio=config.mlp_ratio,
                qkv_bias=config.qkv_bias,
                proj_bias=config.proj_bias,
                ffn_bias=config.ffn_bias,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=nn.GELU,
                ffn_layer=Mlp,
                init_values=config.init_values,
            ) for i in range(config.depth)]

        self.blocks = nn.ModuleList(blocks_list)
        self.norm = norm_layer(config.embed_dim)
        self.head = nn.Identity()
        self.mask_token = nn.Parameter(torch.zeros(1, config.embed_dim))
        
        self.init_weights()
        
    def init_weights(self):
        trunc_normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.cls_token, std=1e-6)
        if self.register_tokens is not None:
            nn.init.normal_(self.register_tokens, std=1e-6)
        named_apply(init_weights_vit_timm, self)

    def interpolate_pos_encoding(self, x, w, h):
        previous_dtype = x.dtype
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        pos_embed = self.pos_embed.float()
        class_pos_embed = pos_embed[:, 0]
        patch_pos_embed = pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_size
        h0 = h // self.patch_size
        M = int(math.sqrt(N))  # Recover the number of patches in each dimension
        assert N == M * M
        kwargs = {}
        if self.interpolate_offset:
            # Historical kludge: add a small number to avoid floating point error in the interpolation, see https://github.com/facebookresearch/dino/issues/8
            # Note: still needed for backward-compatibility, the underlying operators are using both output size and scale factors
            sx = float(w0 + self.interpolate_offset) / M
            sy = float(h0 + self.interpolate_offset) / M
            kwargs["scale_factor"] = (sx, sy)
        else:
            # Simply specify an output size instead of a scale factor
            kwargs["size"] = (w0, h0)
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, M, M, dim).permute(0, 3, 1, 2),
            mode="bicubic",
            antialias=self.interpolate_antialias,
            **kwargs,
        )
        assert (w0, h0) == patch_pos_embed.shape[-2:]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1).to(previous_dtype)

    def prepare_tokens(self, x):

        B, nc, h, w = x.shape
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        
        x = x + self.interpolate_pos_encoding(x, h, w)

        if self.register_tokens is not None:
            x = torch.cat(
                (
                    x[:, :1],
                    self.register_tokens.expand(B, -1, -1),
                    x[:, 1:],
                ),
                dim=1,
            )
        return x        

    def forward(self, x):

        x = self.prepare_tokens(x)

        for idx, blk in enumerate(self.blocks):
            if self.gradient_checkpointing and self.training:
                x = self._gradient_checkpointing_func(
                    blk,
                    x,
                )
            else:
                x = blk(x)

        x_norm = self.norm(x)
        return ViTModelOutput(
            x_norm_clstoken=x_norm[:, 0],
            x_norm_regtokens=x_norm[:, 1 : self.num_register_tokens + 1],
            x_norm_patchtokens=x_norm[:, self.num_register_tokens + 1:],
            x_prenorm=x,
        )

    def get_last_attention(self, x):

        x = self.prepare_tokens(x)

        for idx, blk in enumerate(self.blocks):
            if self.gradient_checkpointing and self.training:
                x, attn = self._gradient_checkpointing_func(
                    blk.get_attention,
                    x,
                )
            else:
                x, attn = blk.get_attention(x)
        
        x_norm = self.norm(x)

        return ViTModelOutput(
            x_norm_clstoken=x_norm[:, 0],
            x_norm_regtokens=x_norm[:, 1 : self.num_register_tokens + 1],
            x_norm_patchtokens=x_norm[:, self.num_register_tokens + 1:],
            x_prenorm=x,
            x_lastlayer_attn_cls=attn[:, :, :, 0],
            x_lastlayer_attn_regtokens=attn[:, :, :, 1 : self.num_register_tokens + 1],
            x_lastlayer_attn_patchtokens=attn[:, :, self.num_register_tokens + 1:],
            x_lastlayer_attn_alltokens=attn,
        )

    def prepare_tokens_with_masks(self, x, masks):
        B, nc, h, w = x.shape
        x = self.patch_embed(x)
        x = torch.where(
            masks.unsqueeze(-1),
            self.mask_token.to(x.dtype).unsqueeze(0),
            x,
        )
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = x + self.interpolate_pos_encoding(x, h, w)

        if self.register_tokens is not None:
            x = torch.cat(
                (
                    x[:, :1],
                    self.register_tokens.expand(x.shape[0], -1, -1),
                    x[:, 1:],
                ),
                dim=1,
            )
        return x
                          
    def forward_mask(self, x, masks):
        
        x = self.prepare_tokens_with_masks(x, masks)

        for idx, blk in enumerate(self.blocks):
            if self.gradient_checkpointing and self.training:
                x = self._gradient_checkpointing_func(
                    blk,
                    x,
                )
            else:
                x = blk(x)

        x_norm = self.norm(x)
        return ViTModelOutput(
            x_norm_clstoken=x_norm[:, 0],
            x_norm_regtokens=x_norm[:, 1 : self.num_register_tokens + 1],
            x_norm_patchtokens=x_norm[:, self.num_register_tokens + 1:],
            x_prenorm=x,
        )


class DINOLoss(nn.Module):

    def __init__(
        self,
        out_dim,
        student_temp=0.1,
        center_momentum=0.9,
    ):
        super().__init__()

        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.register_buffer('center', torch.zeros(1, out_dim))
        self.updated = True
        self.reduce_handle = None
        self.len_teacher_output = None
        self.async_batch_center = None

    def forward(self, student_output_list, teacher_out_softmaxed_centered_list):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        # TODO: Use cross_entropy_distribution here
        total_loss = 0
        for s in student_output_list:
            lsm = F.log_softmax(s / self.student_temp, dim=-1)
            for t in teacher_out_softmaxed_centered_list:
                loss = torch.sum(t * lsm, dim=-1)
                total_loss -= loss.mean()
        return total_loss

    @torch.no_grad()
    def sinkhorn_knopp_teacher(self, teacher_output, teacher_temp, n_iterations=3):
        teacher_output = teacher_output.float()
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        Q = torch.exp(teacher_output / teacher_temp).t()  # Q is K-by-B for consistency with notations from our paper
        B = Q.shape[1] * world_size  # number of samples to assign
        K = Q.shape[0]  # how many prototypes

        # make the matrix sums to 1
        sum_Q = torch.sum(Q)
        if dist.is_initialized():
            dist.all_reduce(sum_Q)
        Q /= sum_Q

        for it in range(n_iterations):
            # normalize each row: total weight per prototype must be 1/K
            sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
            if dist.is_initialized():
                dist.all_reduce(sum_of_rows)
            Q /= sum_of_rows
            Q /= K

            # normalize each column: total weight per sample must be 1/B
            Q /= torch.sum(Q, dim=0, keepdim=True)
            Q /= B

        Q *= B  # the columns must sum to 1 so that Q is an assignment
        return Q.t()


class KoLeoLoss(nn.Module):
    """Kozachenko-Leonenko entropic loss regularizer from Sablayrolles et al. - 2018 - Spreading vectors for similarity search"""

    def __init__(self):
        super().__init__()
        self.pdist = nn.PairwiseDistance(2, eps=1e-8)

    def pairwise_NNs_inner(self, x):
        """
        Pairwise nearest neighbors for L2-normalized vectors.
        Uses Torch rather than Faiss to remain on GPU.
        """
        # parwise dot products (= inverse distance)
        dots = torch.mm(x, x.t())
        n = x.shape[0]
        dots.view(-1)[:: (n + 1)].fill_(-1)  # Trick to fill diagonal with -1
        # max inner prod -> min distance
        _, I = torch.max(dots, dim=1)  # noqa: E741
        return I

    def forward(self, student_output, eps=1e-8):
        """
        Args:
            student_output (BxD): backbone output of student
        """
        with torch.cuda.amp.autocast(enabled=False):
            student_output = F.normalize(student_output, eps=eps, p=2, dim=-1)
            I = self.pairwise_NNs_inner(student_output)  # noqa: E741
            distances = self.pdist(student_output, student_output[I])  # BxD, BxD -> B
            loss = -torch.log(distances + eps).mean()
        return loss


class iBOTPatchLoss(nn.Module):

    def __init__(
        self,
        patch_out_dim,
        student_temp=0.1,
        center_momentum=0.9,
    ):
        super().__init__()

        self.student_temp = student_temp

    @torch.no_grad()
    def sinkhorn_knopp_teacher(
        self, 
        teacher_output, 
        teacher_temp, 
        n_masked_patches_tensor, 
        n_iterations=3
    ):
        teacher_output = teacher_output.float()
        Q = torch.exp(teacher_output / teacher_temp).t()
        B = n_masked_patches_tensor
        if dist.is_initialized():
            dist.all_reduce(B)
        K = Q.shape[0]  # how many prototypes

        # make the matrix sums to 1
        sum_Q = torch.sum(Q)
        if dist.is_initialized():
            dist.all_reduce(sum_Q)
        Q /= sum_Q

        for it in range(n_iterations):
            # normalize each row: total weight per prototype must be 1/K
            sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
            if dist.is_initialized():
                dist.all_reduce(sum_of_rows)
            Q /= sum_of_rows
            Q /= K

            # normalize each column: total weight per sample must be 1/B
            Q /= torch.sum(Q, dim=0, keepdim=True)
            Q /= B

        Q *= B  # the columns must sum to 1 so that Q is an assignment
        return Q.t()

    def forward_masked(
        self,
        student_patch_tokens_masked,
        teacher_patch_tokens_masked,
        student_masks_flat,
        n_masked_patches=None,
        masks_weight=None,
    ):
        def lossfunc(t, s, temp):
            return torch.sum(t * F.log_softmax(s / temp, dim=-1), dim=-1)
        
        t = teacher_patch_tokens_masked
        s = student_patch_tokens_masked
        # loss = torch.sum(t * F.log_softmax(s / self.student_temp, dim=-1), dim=-1)
        loss = lossfunc(t, s, self.student_temp)
        if masks_weight is None:
            masks_weight = (
                (1 / student_masks_flat.sum(-1).clamp(min=1.0))
                .unsqueeze(-1)
                .expand_as(student_masks_flat)[student_masks_flat]
            )
        if n_masked_patches is not None:
            loss = loss[:n_masked_patches]
        loss = loss * masks_weight
        return -loss.sum() / student_masks_flat.shape[0]

        
class DINOv1(ViTPretrainedModel):

    def __init__(self, config):
        super().__init__(config)

        student_model_dict = dict()
        teacher_model_dict = dict()

        student_backbone = ViT(config)
        teacher_backbone = ViT(config)

        student_model_dict['backbone'] = student_backbone
        teacher_model_dict['backbone'] = teacher_backbone

        self.embed_dim = config.embed_dim
        self.dino_out_dim = config.dino_head_n_prototypes

        student_dino_head = DINOHead(
            in_dim=config.embed_dim,
            out_dim=config.dino_head_n_prototypes,
            hidden_dim=config.head_hidden_dim,
            bottleneck_dim=config.head_bottleneck_dim,
            nlayers=config.head_nlayers,
        )
        teacher_dino_head = DINOHead(
            in_dim=config.embed_dim,
            out_dim=config.dino_head_n_prototypes,
            hidden_dim=config.head_hidden_dim,
            bottleneck_dim=config.head_bottleneck_dim,
            nlayers=config.head_nlayers,
        )

        student_model_dict["dino_head"] = student_dino_head
        teacher_model_dict["dino_head"] = teacher_dino_head

        self.student = nn.ModuleDict(student_model_dict)
        self.teacher = nn.ModuleDict(teacher_model_dict)

        for param_src, param_dst in zip(self.student.parameters(),
                                        self.teacher.parameters()):
            param_dst.data.copy_(param_src.data)
            param_dst.requires_grad = False

        self.dino_loss = DINOLoss(
            out_dim=config.dino_head_n_prototypes,
            student_temp=config.student_temp,
            center_momentum=config.center_momentum,
        )
        self.koleo_loss = KoLeoLoss()

    def forward(self, collated_global_crops, collated_local_crops, upperbound, teacher_temp, **kwargs):
        n_global_crops = 2
        n_local_crops = self.config.local_crops_number
        assert n_global_crops == 2

        
        n_local_crops_loss_terms = max(n_local_crops * n_global_crops, 1)
        n_global_crops_loss_terms = (n_global_crops - 1) * n_global_crops

        global_crops = collated_global_crops.cuda(non_blocking=True)
        local_crops = collated_local_crops.cuda(non_blocking=True)

        teacher_dino_softmaxed_centered_list, _ = self.get_teacher_output(global_crops, teacher_temp)

        loss_dict = {}
        student_global_backbone_outputs = self.student.backbone(global_crops)
        student_local_backbone_outputs = self.student.backbone(local_crops)

        # 1a: local crops cls tokens
        student_local_cls_tokens = student_local_backbone_outputs.x_norm_clstoken
        student_local_cls_tokens_after_head = self.student.dino_head(student_local_cls_tokens)

        # 1b: global crops cls tokens
        student_global_cls_tokens = student_global_backbone_outputs.x_norm_clstoken
        student_global_cls_tokens_after_head = self.student.dino_head(student_global_cls_tokens)

        # student local & teacher global dino loss
        dino_local_crops_loss = self.dino_loss(
            student_output_list=student_local_cls_tokens_after_head.chunk(n_local_crops),
            teacher_out_softmaxed_centered_list=teacher_dino_softmaxed_centered_list,
        ) / (n_global_crops_loss_terms + n_local_crops_loss_terms)

        loss_dict['dino_local_crops_loss'] = dino_local_crops_loss * self.config.dino_loss_weight

        # student global & teacher global dino loss
        dino_global_crops_loss = self.dino_loss(
            student_output_list=[student_global_cls_tokens_after_head],
            teacher_out_softmaxed_centered_list=[teacher_dino_softmaxed_centered_list.flatten(0, 1)],
        ) * 2 / (n_global_crops_loss_terms + n_local_crops_loss_terms)

        loss_dict['dino_global_crops_loss'] = dino_global_crops_loss * self.config.dino_loss_weight

        # koleo loss
        koleo_loss = self.config.koleo_loss_weight * sum(
            self.koleo_loss(p) for p in student_global_cls_tokens.chunk(2)
        )
        loss_dict['koleo_loss'] = koleo_loss

        loss = 0
        for key in loss_dict.keys():
            loss += loss_dict[key]

        return DINOv2ModelOutput(
            loss=loss,
            loss_dict=loss_dict,
        )

    @torch.no_grad()
    def get_teacher_output(self, global_crops, teacher_temp):
        n_global_crops_teacher = 2
        teacher_backbone_outputs = self.teacher.backbone(global_crops)
        teacher_cls_tokens = teacher_backbone_outputs.x_norm_clstoken
        teacher_cls_tokens = teacher_cls_tokens.chunk(n_global_crops_teacher)
        teacher_cls_tokens = torch.cat((teacher_cls_tokens[1], teacher_cls_tokens[0]))
        _dim = teacher_cls_tokens.shape[-1]
        n_cls_tokens = teacher_cls_tokens.shape[0]

        teacher_cls_tokens_after_head = self.teacher.dino_head(teacher_cls_tokens)
        masked_teacher_ibot_softmaxed_centered = None

        teacher_dino_softmaxed_centered_list = self.dino_loss.sinkhorn_knopp_teacher(
            teacher_cls_tokens_after_head, teacher_temp=teacher_temp,
        ).view(n_global_crops_teacher, -1, *teacher_cls_tokens_after_head.shape[1:])

        return teacher_dino_softmaxed_centered_list, masked_teacher_ibot_softmaxed_centered

    def get_params_groups(self):
        all_params_groups = []
        for m in self.student.values():
            all_params_groups += get_params_groups_with_decay(
                model=m,
                lr_decay_rate=self.config.layerwise_decay,
                patch_embed_lr_mult=self.config.patch_embed_lr_mult,
            )
        return all_params_groups

        
class DINOv2(ViTPretrainedModel):

    def __init__(self, config):
        super().__init__(config)

        student_model_dict = dict()
        teacher_model_dict = dict()

        student_backbone = ViT(config)
        teacher_backbone = ViT(config)

        student_model_dict['backbone'] = student_backbone
        teacher_model_dict['backbone'] = teacher_backbone

        self.embed_dim = config.embed_dim
        self.dino_out_dim = config.dino_head_n_prototypes

        student_dino_head = DINOHead(
            in_dim=config.embed_dim,
            out_dim=config.dino_head_n_prototypes,
            hidden_dim=config.head_hidden_dim,
            bottleneck_dim=config.head_bottleneck_dim,
            nlayers=config.head_nlayers,
        )
        teacher_dino_head = DINOHead(
            in_dim=config.embed_dim,
            out_dim=config.dino_head_n_prototypes,
            hidden_dim=config.head_hidden_dim,
            bottleneck_dim=config.head_bottleneck_dim,
            nlayers=config.head_nlayers,
        )
        student_ibot_head = DINOHead(
            in_dim=config.embed_dim,
            out_dim=config.dino_head_n_prototypes,
            hidden_dim=config.head_hidden_dim,
            bottleneck_dim=config.head_bottleneck_dim,
            nlayers=config.head_nlayers,
        )
        teacher_ibot_head = DINOHead(
            in_dim=config.embed_dim,
            out_dim=config.dino_head_n_prototypes,
            hidden_dim=config.head_hidden_dim,
            bottleneck_dim=config.head_bottleneck_dim,
            nlayers=config.head_nlayers,
        )

        student_model_dict["dino_head"] = student_dino_head
        student_model_dict["ibot_head"] = student_ibot_head
        teacher_model_dict["dino_head"] = teacher_dino_head
        teacher_model_dict["ibot_head"] = teacher_ibot_head

        self.student = nn.ModuleDict(student_model_dict)
        self.teacher = nn.ModuleDict(teacher_model_dict)

        for param_src, param_dst in zip(self.student.parameters(),
                                        self.teacher.parameters()):
            param_dst.data.copy_(param_src.data)
            param_dst.requires_grad = False

        self.dino_loss = DINOLoss(
            out_dim=config.dino_head_n_prototypes,
            student_temp=config.student_temp,
            center_momentum=config.center_momentum,
        )
        self.ibot_patch_loss = iBOTPatchLoss(
            patch_out_dim=config.dino_head_n_prototypes,
            student_temp=config.student_temp,
            center_momentum=config.center_momentum,
        )
        self.koleo_loss = KoLeoLoss()

    def forward(
        self, 
        collated_global_crops, 
        collated_local_crops, 
        collated_masks,
        mask_indices_list,
        n_masked_patches,
        upperbound, 
        masks_weight,
        teacher_temp, 
        **kwargs
    ):
        n_global_crops = 2
        n_local_crops = self.config.local_crops_number
        assert n_global_crops == 2

        global_crops = collated_global_crops.cuda(non_blocking=True)
        local_crops = collated_local_crops.cuda(non_blocking=True)

        masks = collated_masks.cuda(non_blocking=True)
        mask_indices_list = mask_indices_list.cuda(non_blocking=True)
        n_masked_patches_tensor = n_masked_patches.cuda(non_blocking=True)
        n_masked_patches = mask_indices_list.shape[0]
        masks_weight = masks_weight.cuda(non_blocking=True)

        n_local_crops_loss_terms = max(n_local_crops * n_global_crops, 1)
        n_global_crops_loss_terms = (n_global_crops - 1) * n_global_crops

        ibot_loss_scale = 1.0 / n_global_crops

        teacher_dino_softmaxed_centered_list, masked_teacher_ibot_softmaxed_centered = self.get_teacher_output(
            global_crops, 
            n_global_crops,
            upperbound,
            mask_indices_list,
            n_masked_patches,
            n_masked_patches_tensor,
            teacher_temp,
        )

        loss_dict = {}
        student_global_backbone_output_dict = self.student.backbone.forward_mask(
            global_crops,
            masks,
        )
        student_local_backbone_output_dict = self.student.backbone(
            local_crops,
        )

        # do ibot
        _dim = student_global_backbone_output_dict.x_norm_clstoken.shape[-1]
        ibot_student_patch_tokens = student_global_backbone_output_dict.x_norm_patchtokens
        buffer_tensor_patch_tokens = ibot_student_patch_tokens.new_zeros(upperbound, _dim)
        buffer_tensor_patch_tokens[:n_masked_patches].copy_(
            torch.index_select(
                ibot_student_patch_tokens.flatten(0, 1),
                dim=0,
                index=mask_indices_list,
            ),
        )
        student_global_masked_patch_tokens_after_head = self.student.ibot_head(
            buffer_tensor_patch_tokens,
        )[:n_masked_patches]

        student_global_cls_tokens = student_global_backbone_output_dict.x_norm_clstoken
        student_local_cls_tokens = student_local_backbone_output_dict.x_norm_clstoken
        student_global_cls_tokens_after_head = self.student.dino_head(student_global_cls_tokens)
        student_local_cls_tokens_after_head = self.student.dino_head(student_local_cls_tokens)

        # student local & teacher global dino loss
        dino_local_crops_loss = self.dino_loss(
            student_output_list=student_local_cls_tokens_after_head.chunk(n_local_crops),
            teacher_out_softmaxed_centered_list=teacher_dino_softmaxed_centered_list,
        ) / (n_global_crops_loss_terms + n_local_crops_loss_terms)
        loss_dict['dino_local_crops_loss'] = dino_local_crops_loss * self.config.dino_loss_weight

        # student global & teacher global dino loss
        dino_global_crops_loss = self.dino_loss(
            student_output_list=[student_global_cls_tokens_after_head],
            teacher_out_softmaxed_centered_list=[teacher_dino_softmaxed_centered_list.flatten(0, 1)],
        ) * 2 / (n_global_crops_loss_terms + n_local_crops_loss_terms)
        loss_dict['dino_global_crops_loss'] = dino_global_crops_loss * self.config.dino_loss_weight

        # koleo loss
        koleo_loss = self.config.koleo_loss_weight * sum(
            self.koleo_loss(p) for p in student_global_cls_tokens.chunk(2)
        )
        loss_dict['koleo_loss'] = koleo_loss

        # ibot loss
        ibot_patch_loss = self.ibot_patch_loss.forward_masked(
            student_global_masked_patch_tokens_after_head,
            masked_teacher_ibot_softmaxed_centered,
            student_masks_flat=masks,
            n_masked_patches=n_masked_patches,
            masks_weight=masks_weight,
        ) * 2 * ibot_loss_scale
        loss_dict['ibot_patch_loss'] = ibot_patch_loss * self.config.ibot_loss_weight / 2

        loss = 0
        for key in loss_dict.keys():
            loss += loss_dict[key]

        return DINOv2ModelOutput(
            loss=loss,
            loss_dict=loss_dict,
        )

    @torch.no_grad()
    def get_teacher_output(
        self, 
        global_crops, 
        n_global_crops,
        upperbound,
        mask_indices_list,
        n_masked_patches,
        n_masked_patches_tensor,
        teacher_temp,
    ):
        x, n_global_crops_teacher = global_crops, n_global_crops
        teacher_backbone_output = self.teacher.backbone(x)
        teacher_cls_tokens = teacher_backbone_output.x_norm_clstoken
        teacher_cls_tokens = teacher_cls_tokens.chunk(n_global_crops_teacher)
        teacher_cls_tokens = torch.cat((teacher_cls_tokens[1], teacher_cls_tokens[0]))
        ibot_teacher_patch_tokens = teacher_backbone_output.x_norm_patchtokens
        _dim = ibot_teacher_patch_tokens.shape[-1]
        n_cls_tokens = teacher_cls_tokens.shape[0]

        buffer_tensor_teacher = ibot_teacher_patch_tokens.new_zeros(upperbound, _dim)
        torch.index_select(
            ibot_teacher_patch_tokens.flatten(0, 1),
            dim=0,
            index=mask_indices_list,
            out=buffer_tensor_teacher[:n_masked_patches],
        )
        teacher_cls_tokens_after_head = self.teacher.dino_head(teacher_cls_tokens)
        masked_teacher_patch_tokens_after_head = self.teacher.ibot_head(buffer_tensor_teacher)
        masked_teacher_patch_tokens_after_head = masked_teacher_patch_tokens_after_head[:n_masked_patches]

        teacher_dino_softmaxed_centered_list = self.dino_loss.sinkhorn_knopp_teacher(
            teacher_cls_tokens_after_head, teacher_temp=teacher_temp,
        ).view(n_global_crops_teacher, -1, *teacher_cls_tokens_after_head.shape[1:])

        masked_teacher_ibot_softmaxed_centered = self.ibot_patch_loss.sinkhorn_knopp_teacher(
            masked_teacher_patch_tokens_after_head,
            teacher_temp=teacher_temp,
            n_masked_patches_tensor=n_masked_patches_tensor,
        )
        
        return teacher_dino_softmaxed_centered_list, masked_teacher_ibot_softmaxed_centered

    def get_params_groups(self):
        all_params_groups = []
        for m in self.student.values():
            all_params_groups += get_params_groups_with_decay(
                model=m,
                lr_decay_rate=self.config.layerwise_decay,
                patch_embed_lr_mult=self.config.patch_embed_lr_mult,
            )
        return all_params_groups
