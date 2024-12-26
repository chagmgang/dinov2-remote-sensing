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
from .configuration_convvit_dinov2 import ConvViTDINOv2Config


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


class PatchEmbed(nn.Module):

    def __init__(
        self,
        img_size,
        patch_size,
        in_chans,
        embed_dim,
    ):
        super().__init__()

        self.img_size = (img_size, img_size)
        self.patch_size = (patch_size, patch_size)
        self.grid_size = (self.img_size[0] // self.patch_size[0],
                          self.img_size[1] // self.patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return self.act(x)
        

class CMlp(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class CBlock(nn.Module):

    def __init__(
        self,
        dim,
        mlp_ratio,
        qkv_bias,
        qk_scale,
        drop_path=0.,
        drop=0.,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.conv1 = nn.Conv2d(dim, dim, 1)
        self.conv2 = nn.Conv2d(dim, dim, 1)
        self.attn = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        # self.attn = nn.Conv2d(dim, dim, 1)

        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = CMlp(
            in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, x, mask=None):
        residual = x
        x = self.conv1(
            self.norm1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2))
        if mask is not None:
            x = self.attn(mask * x)
        else:
            x = self.attn(x)
        x = residual + self.drop_path(self.conv2(x))
        x = x + self.drop_path(
            self.mlp(self.norm2(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)))
        return x
            

class Attention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads,
                                  C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[
            2]  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):

    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.GELU,
                 drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


@dataclass
class ConvViTModelOutput(ModelOutput):

    x_norm_clstoken: torch.FloatTensor = None
    x_norm_regtokens: torch.FloatTensor = None
    x_norm_patchtokens: torch.FloatTensor = None
    x_prenorm: torch.FloatTensor = None
    x_lastlayer_attn_cls: torch.FloatTensor = None
    x_lastlayer_attn_regtokens: torch.FloatTensor = None
    x_lastlayer_attn_patchtokens: torch.FloatTensor = None
    x_lastlayer_attn_alltokens: torch.FloatTensor = None
    

@dataclass
class ConvViTDINOv2ModelOutput(ModelOutput):

    loss: torch.FloatTensor = None
    loss_dict: dict[torch.FloatTensor] = None


class ConvViTPretrainedModel(PreTrainedModel):

    config_class = ConvViTDINOv2Config
    base_model_prefix = 'model'
    supports_gradient_checkpointing = True
    _no_split_modules = []

    def _init_weights(self, m):
        pass


class ConvViT(ConvViTPretrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.gradient_checkpointing = False

        self.num_features = self.embed_dim = config.embed_dim
        self.num_tokens = 1
        self.num_register_tokens = config.num_register_tokens
        self.interpolate_antialias = config.interpolate_antialias
        self.interpolate_offset = config.interpolate_offset

        self.patch_embed1 = PatchEmbed(
            img_size=config.img_size[0],
            patch_size=config.patch_size[0],
            in_chans=config.in_chans,
            embed_dim=config.embed_dim[0],
        )

        self.patch_embed2 = PatchEmbed(
            img_size=config.img_size[1],
            patch_size=config.patch_size[1],
            in_chans=config.embed_dim[0],
            embed_dim=config.embed_dim[1],
        )

        self.patch_embed3 = PatchEmbed(
            img_size=config.img_size[2],
            patch_size=config.patch_size[2],
            in_chans=config.embed_dim[1],
            embed_dim=config.embed_dim[2],
        )

        self.patch_embed4 = nn.Linear(config.embed_dim[2], config.embed_dim[2])

        self.blocks1 = nn.ModuleList([
            CBlock(
                dim=config.embed_dim[0],
                mlp_ratio=config.mlp_ratio[0],
                qkv_bias=config.qkv_bias,
                qk_scale=None,
            ) for i in range(config.depth[0])
        ])

        self.blocks2 = nn.ModuleList([
            CBlock(
                dim=config.embed_dim[1],
                mlp_ratio=config.mlp_ratio[1],
                qkv_bias=config.qkv_bias,
                qk_scale=None,
            ) for i in range(config.depth[1])
        ])

        self.blocks3 = nn.ModuleList([
            Block(
                dim=config.embed_dim[2],
                num_heads=config.num_heads,
                mlp_ratio=config.mlp_ratio[2],
                qkv_bias=config.qkv_bias,
                qk_scale=None) for i in range(config.depth[2])
        ])
        self.norm = nn.LayerNorm(config.embed_dim[-1])

        num_patches = self.patch_embed3.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.embed_dim[-1]))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, config.embed_dim[-1]))
        assert config.num_register_tokens >= 0
        self.register_tokens = (
            nn.Parameter(torch.zeros(1, config.num_register_tokens, config.embed_dim[-1])) if config.num_register_tokens else None
        )

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
        w0 = w // (self.config.patch_size[0] * self.config.patch_size[1] * self.config.patch_size[2])
        h0 = h // (self.config.patch_size[0] * self.config.patch_size[1] * self.config.patch_size[2])
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
        x = self.patch_embed1(x)
        for blk in self.blocks1:
            if self.gradient_checkpointing and self.training:
                x = self._gradient_checkpointing_func(
                    blk,
                    x,
                    None,
                )
            else:
                x = blk(x, None)

        x = self.patch_embed2(x)
        for blk in self.blocks2:
            if self.gradient_checkpointing and self.training:
                x = self._gradient_checkpointing_func(
                    blk,
                    x,
                    None,
                )
            else:
                x = blk(x, None)

        x = self.patch_embed3(x)
        x = x.flatten(2).permute(0, 2, 1)
        x = self.patch_embed4(x)
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
        for idx, blk in enumerate(self.blocks3):
            if self.gradient_checkpointing and self.training:
                x = self._gradient_checkpointing_func(
                    blk,
                    x,
                )
            else:
                x = blk(x)

        x_norm = self.norm(x)
        return ConvViTModelOutput(
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


class ConvDINOv1(ConvViTPretrainedModel):

    def __init__(self, config):
        super().__init__(config)

        student_model_dict = dict()
        teacher_model_dict = dict()

        student_backbone = ConvViT(config)
        teacher_backbone = ConvViT(config)

        student_model_dict['backbone'] = student_backbone
        teacher_model_dict['backbone'] = teacher_backbone

        self.embed_dim = config.embed_dim
        self.dino_out_dim = config.dino_head_n_prototypes

        student_dino_head = DINOHead(
            in_dim=config.embed_dim[-1],
            out_dim=config.dino_head_n_prototypes,
            hidden_dim=config.head_hidden_dim,
            bottleneck_dim=config.head_bottleneck_dim,
            nlayers=config.head_nlayers,
        )
        teacher_dino_head = DINOHead(
            in_dim=config.embed_dim[-1],
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

        return ConvViTDINOv2ModelOutput(
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