import argparse
import os
from functools import partial
from glob import glob

import math
import torch
import torch.nn as nn
import numpy as np
from transformers import TrainingArguments
from transformers import Trainer
from transformers import TrainerCallback

from dinov2.models import ConvViTDINOv2Config, ConvDINOv1
from dinov2.data import MaskingGenerator, DINOAugmentation, BaseDataset, collate_data_and_cast



class CosineScheduler(object):
    def __init__(self, base_value, final_value, total_iters, warmup_iters=0, start_warmup_value=0, freeze_iters=0):
        super().__init__()
        self.final_value = final_value
        self.total_iters = total_iters

        freeze_schedule = np.zeros((freeze_iters))

        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

        iters = np.arange(total_iters - warmup_iters - freeze_iters)
        schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))
        self.schedule = np.concatenate((freeze_schedule, warmup_schedule, schedule))

        assert len(self.schedule) == self.total_iters

    def __getitem__(self, it):
        if it >= self.total_iters:
            return self.final_value
        else:
            return self.schedule[it]

def build_schedulers(config, num_training_steps):

    lr = config.base_lr
    lr *= math.sqrt(config.batch_size / 1024)
    lr = dict(
        base_value=lr,
        final_value=config.min_lr,
        total_iters=num_training_steps,
        warmup_iters=int(config.lr_warmup_percentile * num_training_steps),
    )
    wd = dict(
        base_value=config.weight_decay,
        final_value=config.weight_decay_end,
        total_iters=num_training_steps,
    )
    momentum = dict(
        base_value=config.momentum_teacher,
        final_value=config.final_momentum_teacher,
        total_iters=num_training_steps,
    )
    teacher_temp = dict(
        base_value=config.teacher_temp,
        final_value=config.teacher_temp,
        total_iters=int(config.teacher_temp_warmup_percentile * num_training_steps),
        warmup_iters=int(config.teacher_temp_warmup_percentile * num_training_steps),
        start_warmup_value=config.warmup_teacher_temp,
    )

    lr_schedule = CosineScheduler(**lr)
    wd_schedule = CosineScheduler(**wd)
    momentum_schedule = CosineScheduler(**momentum)
    teacher_temp_schedule = CosineScheduler(**teacher_temp)
    last_layer_lr_schedule = CosineScheduler(**lr)

    return (
        lr_schedule,
        wd_schedule,
        momentum_schedule,
        teacher_temp_schedule,
        last_layer_lr_schedule,
    )


class ConvDINOv2Trainer(Trainer):

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        self.optimizer = torch.optim.AdamW(
            self.model.student.parameters(),
            betas=(0.9, 0.999),
        )
        (
            self.lr_schedule,
            self.wd_schedule,
            self.momentum_schedule,
            self.teacher_temp_schedule,
            self.last_layer_lr_schedule,
        ) = build_schedulers(self.model.config, num_training_steps)

        self.create_scheduler(num_training_steps)

    def update_teacher(self, m):
        student_param_list = []
        teacher_param_list = []
        with torch.no_grad():
            for src, dst in zip(self.model.student.parameters(),
                                self.model.teacher.parameters()):
                dst.data = dst.data * m + src.data * (1 - m)

    def create_optimizer(self, num_training_steps):

        class DummyScheduler(object):

            def __init__(self):
                super().__init__()

            def step(self):
                pass

        self.lr_scheduler = DummyScheduler()

    def training_step(
        self, 
        model, 
        inputs,
    ) -> torch.Tensor:

        lr = self.lr_schedule[self.state.global_step]
        wd = self.wd_schedule[self.state.global_step]
        mom = self.momentum_schedule[self.state.global_step]
        teacher_temp = self.teacher_temp_schedule[self.state.global_step]

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
            param_group['weight_decay'] = wd

        log_params = dict()

        log_params.update({
            'lr': lr,
            'wd': wd,
            'momentum': mom,
            'teacher_temp': teacher_temp,
        })

        self.log(log_params)

        inputs['teacher_temp'] = teacher_temp

        ret = super(ConvDINOv2Trainer, self).training_step(
            model=model,
            inputs=inputs,
        )

        self.update_teacher(mom)

        return ret

def read_filename(filename):

    lines = []
    f = open(filename, 'r')
    while True:
        line = f.readline()
        if not line: break
        line = line.replace('\n', '')
        lines.append(line)
    f.close()
    return lines

def main():

    config = ConvViTDINOv2Config(
        img_size=[224, 56, 28],
        embed_dim=[128, 256, 384],
        patch_size=[4, 2, 2],
        depth=[2, 2, 11],
        mlp_ratio=[4.0, 4.0, 4.0],
        num_heads=6,
        batch_size=2048,
        base_lr=0.002,
        lr_warmup_percentile=float(3 / 25),
        teacher_temp_warmup_percentile=float(8 / 25),
    )
    model = ConvDINOv1(config)

    filenames = list()
    for filename in [
        'data_lists/test.txt',
    ]:
        filenames.extend(read_filename(filename))
    
    image_size = config.img_size[0]
    patch_size = config.patch_size[0] * config.patch_size[1] * config.patch_size[2]
    n_tokens = (image_size // patch_size) ** 2
    mask_generator = MaskingGenerator(
        input_size=(image_size // patch_size, image_size // patch_size),
        max_num_patches=0.5 * image_size // patch_size * image_size // patch_size,
    )
    
    transforms = DINOAugmentation(
        global_crops_scale=config.global_crops_scale,
        local_crops_scale=config.local_crops_scale,
        local_crops_number=config.local_crops_number,
        global_crops_size=config.img_size[0],
        local_crops_size=config.local_crops_size,
    )
    
    dataset = BaseDataset(
        filenames,
        transforms=transforms,
    )
    
    collate_fn = partial(
        collate_data_and_cast,
        mask_ratio_tuple=(0.1, 0.5),
        mask_probability=0.5,
        n_tokens=n_tokens,
        mask_generator=mask_generator,
        dtype=torch.float32,
    )

    batch_size = model.config.batch_size
    per_device_train_batch_size = int(batch_size / int(os.environ['WORLD_SIZE']))

    training_args = TrainingArguments(
        output_dir='checkpoint-convvit-dinov1/model',
        logging_dir='checkpoint-convvit-dinov1/logs',
        logging_steps=5,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=1,
        save_strategy='epoch',
        report_to='tensorboard',
        do_train=True,
        num_train_epochs=25,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={
            'use_reentrant': False,
        },
        fp16=False,
        bf16=False,
        max_grad_norm=model.config.clip_grad,
        dataloader_num_workers=4,
        save_safetensors=True,
    )

    trainer = ConvDINOv2Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collate_fn,
    )

    trainer.train()
        

if __name__ == '__main__':
    main()
