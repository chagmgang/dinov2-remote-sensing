import math
import numpy as np
import torch

from transformers import Trainer


def cosine_scheduler(iteration, base_value, final_value, total_iters, warmup_iters=0, start_warmup_values=0, freeze_iters=0):

    if iteration >= total_iters:
        return final_value

    if iteration < freeze_iters:
        return 0.0

    if iteration < warmup_iters + freeze_iters:
        warmup_progress = (iteration - freeze_iters) / max(1, warmup_iters)
        return start_warmup_values + (base_value - start_warmup_values) * warmup_progress

    cosine_iteration = iteration - warmup_iters
    cosine_total = total_iters - freeze_iters - warmup_iters
    cosine_progress = cosine_iteration / max(1, cosine_total)
    return final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * cosine_progress))


class DINOv2Trainer(Trainer):

    def create_optimizer(self):
        params_groups = self.model.get_params_groups()
        self.optimizer = torch.optim.AdamW(
            params_groups,
            betas=(0.9, 0.999),
        )
        return self.optimizer

    def update_teacher(self, m):
        student_param_list = []
        teacher_param_list = []
        with torch.no_grad():
            for src, dst in zip(self.model.student.parameters(),
                                self.model.teacher.parameters()):
                dst.data = dst.data * m + src.data * (1 - m)

    def training_step(
        self,
        model,
        inputs,
        num_items_in_batch=None,
    ):
        total_batch_size = int(self.args.world_size * self.args.per_device_train_batch_size)
        max_lr = self.model.config.base_lr
        max_lr = max_lr * math.sqrt(total_batch_size / 1024)
        lr = cosine_scheduler(
            iteration=self.state.global_step,
            base_value=max_lr,
            final_value=self.model.config.min_lr,
            total_iters=self.state.max_steps,
            warmup_iters=int(self.model.config.lr_warmup_percentile * self.state.max_steps),
        )
        wd = cosine_scheduler(
            iteration=self.state.global_step,
            base_value=self.model.config.weight_decay,
            final_value=self.model.config.weight_decay_end,
            total_iters=self.state.max_steps,
        )
        momentum = cosine_scheduler(
            iteration=self.state.global_step,
            base_value=self.model.config.momentum_teacher,
            final_value=self.model.config.final_momentum_teacher,
            total_iters=self.state.max_steps,
        )
        teacher_temp = cosine_scheduler(
            iteration=self.state.global_step,
            base_value=self.model.config.teacher_temp,
            final_value=self.model.config.teacher_temp,
            total_iters=int(self.model.config.teacher_temp_warmup_percentile * self.state.max_steps),
            warmup_iters=int(self.model.config.teacher_temp_warmup_percentile * self.state.max_steps),
            start_warmup_values=self.model.config.warmup_teacher_temp,
        )
        last_layer_lr = cosine_scheduler(
            iteration=self.state.global_step,
            base_value=max_lr,
            final_value=self.model.config.min_lr,
            total_iters=self.state.max_steps,
            warmup_iters=int(self.model.config.lr_warmup_percentile * self.state.max_steps),
        )

        log_params = dict()
        log_params['momentum'] = momentum
        log_params['teacher_temp'] = teacher_temp

        for param_group in self.optimizer.param_groups:
            is_last_layer = param_group['is_last_layer']
            lr_multiplier = param_group["lr_multiplier"]
            wd_multiplier = param_group["wd_multiplier"]
            param_group["weight_decay"] = wd * wd_multiplier
            param_group["lr"] = (last_layer_lr if is_last_layer else lr) * lr_multiplier
            name = param_group['name']
            log_params.update({
                f'{name}.lr': param_group['lr'],
                f'{name}.wd': param_group['weight_decay'],
            })
        
        self.log(log_params)

        inputs['teacher_temp'] = teacher_temp
        ret = super(DINOv2Trainer, self).training_step(
            model=model,
            inputs=inputs,
            num_items_in_batch=num_items_in_batch,
        )
        self.update_teacher(momentum)
        return ret

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None,
    ):
        loss, outputs = super(
            DINOv2Trainer,
            self,
        ).compute_loss(
            model=model,
            inputs=inputs,
            return_outputs=True,
            num_items_in_batch=num_items_in_batch,
        )

        log_params = dict()
        loss_dict = dict(outputs.loss_dict)
        for key in loss_dict.keys():
            log_params[key] = float(loss_dict[key])

        self.log(log_params)
        return loss