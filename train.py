import os
import argparse
from functools import partial

import numpy as np
import math
import torch
import deepspeed

from transformers import TrainingArguments, Trainer
from dinov2.models import DINOv2, DINOv2Trainer
from dinov2.data import MaskingGenerator, DINOAugmentation, BaseDataset, collate_data_and_cast


parser = argparse.ArgumentParser()
parser.add_argument('--batch-size', type=int)
parser.add_argument('--output-dir', type=str, default='checkpoint')
parser.add_argument('--local-rank', '--local_rank', default=-1, type=int)
parser.add_argument('--model-path')
parser.add_argument('--hf-token')
parser.add_argument('--bf16', action='store_true')
parser.add_argument('--fp16', action='store_true')
parser = deepspeed.add_config_arguments(parser)
args = parser.parse_args()


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

    model = DINOv2.from_pretrained(
        args.model_path,
        token=args.hf_token,
    )

    filenames = list()
    for filename in [
        '/nas/k8s/dev/mlops/chagmgang/dinov2_corpus/corpus/Million-AID.txt',
    ]:
        filenames.extend(read_filename(filename))

    image_size = model.config.img_size
    patch_size = model.config.patch_size
    n_tokens = (image_size // patch_size) ** 2
    mask_generator = MaskingGenerator(
        input_size=(image_size // patch_size, image_size // patch_size),
        max_num_patches=0.5 * image_size // patch_size * image_size // patch_size,
    )
    transforms = DINOAugmentation(
        global_crops_scale=model.config.global_crops_scale,
        local_crops_scale=model.config.local_crops_scale,
        local_crops_number=model.config.local_crops_number,
        global_crops_size=model.config.img_size,
        local_crops_size=model.config.local_crops_size,
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

    batch_size = args.batch_size
    if batch_size is None:
        batch_size = model.config.batch_size
    per_device_train_batch_size = int(batch_size / int(os.environ['WORLD_SIZE']))

    if args.output_dir is None:
        output_dir = args.model_path
    else:
        output_dir = args.output_dir
    
    training_args = TrainingArguments(
        output_dir=os.path.join(output_dir, 'model'),
        logging_dir=os.path.join(output_dir, 'logs'),
        logging_steps=5,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=1,
        
        # save_strategy='epoch',
        save_strategy='steps',
        save_steps=10,
        
        report_to='tensorboard',
        do_train=True,
        
        num_train_epochs=100,

        deepspeed=args.deepspeed_config,
        
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={
            'use_reentrant': False,
        },
        fp16=args.fp16,
        bf16=args.bf16,
        max_grad_norm=model.config.clip_grad,
        dataloader_num_workers=4,
        save_safetensors=True,
    )
        
    trainer = DINOv2Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collate_fn,
    )

    trainer.train()
    
    

if __name__ == '__main__':
    main()
