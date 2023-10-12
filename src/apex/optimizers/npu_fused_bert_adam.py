# coding=utf-8
# Copyright (c) 2021 Huawei Technologies. All rights reserved.
# Copyright (c) 2019 NVIDIA CORPORATION. All rights reserved.
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from collections import defaultdict

import torch
from torch.optim import Optimizer
from torch.optim.optimizer import required

from ..contrib.combine_tensors import combine_npu

WARMUP_DEFAULT = 0.002
DEGREE_DEFAULT = 0.5


def warmup_cosine(x, warmup=WARMUP_DEFAULT):
    if x < warmup:
        return x / warmup
    return 0.5 * (1.0 + torch.cos(math.pi * x))


def warmup_constant(x, warmup=WARMUP_DEFAULT):
    if x < warmup:
        return x / warmup
    return 1.0


def warmup_linear(x, warmup=WARMUP_DEFAULT):
    if x < warmup:
        return x / warmup
    return max((x - 1.) / (warmup - 1.), 0.)


def warmup_poly(x, warmup=WARMUP_DEFAULT, degree=DEGREE_DEFAULT):
    if x < warmup:
        return x / warmup
    return (1.0 - x) ** degree


SCHEDULES = {
    'warmup_cosine': warmup_cosine,
    'warmup_constant': warmup_constant,
    'warmup_linear': warmup_linear,
    'warmup_poly': warmup_poly,
}


class NpuFusedBertAdam(Optimizer):
    """Implements BERT version of Adam algorithm with weight decay fix. This is the fused version on NPU
    Params:
        lr: learning rate
        warmup: portion of t_total for the warmup, -1  means no warmup. Default: -1
        t_total: total number of training steps for the learning
            rate schedule, -1  means constant learning rate. Default: -1
        schedule: schedule to use for the warmup (see above). Default: 'warmup_linear'
        b1: Adams b1. Default: 0.9
        b2: Adams b2. Default: 0.999
        e: Adams epsilon. Default: 1e-6
        weight_decay: Weight decay. Default: 0.01
        max_grad_norm: Maximum norm for the gradients (-1 means no clipping). Default: 1.0
    """

    def __init__(self, params, lr=required, warmup=-1, t_total=-1, schedule='warmup_linear',
                 b1=0.9, b2=0.99, e=1e-6, weight_decay=0.01,
                 max_grad_norm=-1):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if (warmup < 0.0 and warmup != -1) or warmup >= 1.0:
            raise ValueError("Invalid warmup: {}".format(warmup))
        if schedule not in SCHEDULES:
            raise ValueError("Invalid schedule parameter: {}".format(schedule))
        if b1 < 0.0 or b1 >= 1.0:
            raise ValueError("Invalid b1 parameter: {}".format(b1))
        if b2 < 0.0 or b2 >= 1.0:
            raise ValueError("Invalid b2 parameter: {}".format(b2))
        if e < 0.0:
            raise ValueError("Invalid epsilon value: {}".format(e))
        defaults = dict(lr=lr, schedule=schedule, warmup=warmup, t_total=t_total,
                        b1=b1, b2=b2, e=e, weight_decay=weight_decay,
                        max_grad_norm=max_grad_norm)
        self.is_npu_fused_optimizer = True
        self.max_grad_norm = max_grad_norm
        super(NpuFusedBertAdam, self).__init__(params, defaults)

    def _init_param_state(self, p):
        state = self.state[p]
        # state initialization
        if len(state) == 0:
            state['step'] = 0
            # Exponential moving average of gradient values
            state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
            # Exponential moving average of squared gradient values
            state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
        else:
            exp_avg_tmp = torch.zeros_like(p, memory_format=torch.preserve_format)
            exp_avg_tmp.copy_(state['exp_avg'])
            state['exp_avg'] = exp_avg_tmp

            exp_avg_sq_tmp = torch.zeros_like(p, memory_format=torch.preserve_format)
            exp_avg_sq_tmp.copy_(state['exp_avg_sq'])
            state['exp_avg_sq'] = exp_avg_sq_tmp

    def _combine_group_param_states(self, group_index):
        stash = self._amp_stash
        group_params_list = stash.params_lists_indexed_by_group[group_index]

        combined_param_states = []
        for params in group_params_list:
            step_list = []
            exp_avg_list = []
            exp_avg_sq_list = []

            for p in params:
                if p.grad is None:
                    continue

                self._init_param_state(p)
                state = self.state[p]
                step_list.append(state['step'])
                exp_avg_list.append(state['exp_avg'])
                exp_avg_sq_list.append(state['exp_avg_sq'])

            combined_step = 0
            combined_exp_avg = None
            combined_exp_avg_sq = None

            if len(exp_avg_list) > 0:
                combined_step = step_list[0]
                combined_exp_avg = combine_npu(exp_avg_list)
                combined_exp_avg_sq = combine_npu(exp_avg_sq_list)

            combined_state = defaultdict(dict)
            combined_state['step'] = combined_step
            combined_state['exp_avg'] = combined_exp_avg
            combined_state['exp_avg_sq'] = combined_exp_avg_sq

            combined_param_states.append(combined_state)
        stash.combined_param_states_indexed_by_group[group_index] = combined_param_states

    def _combine_param_states_by_group(self):
        stash = self._amp_stash
        if stash.param_states_are_combined_by_group:
            return

        stash.combined_param_states_indexed_by_group = []
        for _ in self.param_groups:
            stash.combined_param_states_indexed_by_group.append([])

        for i, _ in enumerate(self.param_groups):
            self._combine_group_param_states(i)
        stash.param_states_are_combined_by_group = True

    def _group_step(self, group_index):
        group = self.param_groups[group_index]

        beta1, beta2 = group['b1'], group['b2']

        stash = self._amp_stash
        combined_group_params = stash.combined_params_indexed_by_group[group_index]
        combined_group_grads = stash.combined_grads_indexed_by_group[group_index]
        combined_group_param_states = stash.combined_param_states_indexed_by_group[group_index]

        for combined_param, combined_grad, combined_param_state in zip(combined_group_params,
                                                                       combined_group_grads,
                                                                       combined_group_param_states):
            if combined_param is None or combined_grad is None:
                continue

            exp_avg, exp_avg_sq = combined_param_state['exp_avg'], combined_param_state['exp_avg_sq']

            if group['max_grad_norm'] > 0 and self.global_grad_norm != float('inf') and self.global_grad_norm > 1:
                combined_grad /= self.global_grad_norm

            exp_avg.mul_(beta1).add_(1 - beta1, combined_grad)
            exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, combined_grad, combined_grad)
            update = exp_avg / (exp_avg_sq.sqrt() + group['e'])

            if group['weight_decay'] > 0.0:
                update += group['weight_decay'] * combined_param.data

            if group['t_total'] != -1:
                schedule_fct = SCHEDULES[group['schedule']]
                lr_scheduled = group['lr'] * schedule_fct(combined_param_state['step'] / group['t_total'],
                                                          group['warmup'])
            else:
                lr_scheduled = group['lr']

            update_with_lr = lr_scheduled * update
            combined_param.data.add_(-update_with_lr)
            combined_param_state['step'] += 1

    def get_global_grad_norm(self):
        self.global_grad_norm = 0
        for i, _ in enumerate(self.param_groups):
            for combined_group_grads in self._amp_stash.combined_grads_indexed_by_group[i]:
                if combined_group_grads is not None:
                    self.global_grad_norm += combined_group_grads.pow(2).sum()
        self.global_grad_norm = self.global_grad_norm.sqrt().item()

    @torch.no_grad()
    def step(self, closure=None):
        self._check_already_combined_params_and_grads()
        self._combine_params_and_grads_by_group()
        self._combine_param_states_by_group()

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        if self.max_grad_norm > 0:
            self.get_global_grad_norm()

        for i, _ in enumerate(self.param_groups):
            self._group_step(i)

        return loss
