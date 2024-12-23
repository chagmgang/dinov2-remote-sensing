from transformers.utils import logging
from transformers.configuration_utils import PretrainedConfig

logger = logging.get_logger(__name__)


class DINOv2Config(PretrainedConfig):

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        embed_dim=768,        
        depth=12,
        mlp_ratio=4.0,
        in_chans=3,
        num_heads=12,
        qkv_bias=True,
        ffn_bias=True,
        proj_bias=True,
        drop_path_rate=0.3,
        drop_path_uniform=False,
        init_values=None,
        num_register_tokens=4,
        interpolate_antialias=False,
        interpolate_offset=0.1,
        dino_head_n_prototypes=65536,
        head_bottleneck_dim=256,
        head_hidden_dim=2048,
        head_nlayers=3,
        global_crops_scale=(0.32, 1.0),
        local_crops_scale=(0.05, 0.32),
        local_crops_number=8,
        local_crops_size=96,
        student_temp=0.1,
        center_momentum=0.9,
        dino_loss_weight=1.0,
        ibot_loss_weight=1.0,
        koleo_loss_weight=0.1,
        layerwise_decay=0.9,
        patch_embed_lr_mult=0.2,
        base_lr=0.004,
        min_lr=1e-6,
        batch_size=1024,
        lr_warmup_percentile=0.1,
        weight_decay=0.04,
        weight_decay_end=0.4,
        momentum_teacher=0.992,
        final_momentum_teacher=1.0,
        teacher_temp=0.07,
        teacher_temp_warmup_percentile=0.3,
        warmup_teacher_temp=0.04,
        clip_grad=3.0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.depth = depth
        self.mlp_ratio = mlp_ratio
        self.in_chans = in_chans
        self.num_heads = num_heads
        self.qkv_bias = qkv_bias
        self.ffn_bias = ffn_bias
        self.proj_bias = proj_bias
        self.drop_path_rate = drop_path_rate
        self.drop_path_uniform = drop_path_uniform
        self.init_values = init_values
        self.num_register_tokens = num_register_tokens
        self.interpolate_antialias = interpolate_antialias
        self.interpolate_offset = interpolate_offset
        self.dino_head_n_prototypes = dino_head_n_prototypes
        self.head_bottleneck_dim = head_bottleneck_dim
        self.head_hidden_dim = head_hidden_dim
        self.head_nlayers = head_nlayers
        self.global_crops_scale = global_crops_scale
        self.local_crops_scale = local_crops_scale
        self.local_crops_number = local_crops_number
        self.local_crops_size = local_crops_size
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.dino_loss_weight = dino_loss_weight
        self.ibot_loss_weight = ibot_loss_weight
        self.koleo_loss_weight = koleo_loss_weight
        self.layerwise_decay = layerwise_decay
        self.patch_embed_lr_mult = patch_embed_lr_mult
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.batch_size = batch_size
        self.lr_warmup_percentile = lr_warmup_percentile
        self.weight_decay = weight_decay
        self.weight_decay_end = weight_decay_end
        self.momentum_teacher = momentum_teacher
        self.final_momentum_teacher = final_momentum_teacher
        self.teacher_temp = teacher_temp
        self.teacher_temp_warmup_percentile = teacher_temp_warmup_percentile
        self.warmup_teacher_temp = warmup_teacher_temp
        self.clip_grad = clip_grad