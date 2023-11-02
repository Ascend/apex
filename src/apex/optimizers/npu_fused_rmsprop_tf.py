# Copyright (c) 2020, Huawei Technologies.
# Copyright (c) 2019, Facebook CORPORATION.
# Copyright (c) 2020, Ross Wightman
# All rights reserved.
#
# Licensed under the BSD 3-Clause License  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from collections import defaultdict

import torch
from torch.optim.optimizer import Optimizer

from ..contrib.combine_tensors import combine_npu


class NpuFusedRMSpropTF(Optimizer):
    """Implements NpuFusedRMSpropTF algorithm.

    Currently NPU-only.  Requires Apex to be installed via
    ``pip install -v --no-cache-dir --global-option="--cpp_ext" ./``.

    Originally cut & paste from PyTorch RMSProp

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional, default: 1e-2): learning rate
        momentum (float, optional,, default: 0): momentum factor
        alpha (float, optional, default: 0.99): smoothing constant
        eps (float, optional, default: 1e-10): term added to the denominator to improve
            numerical stability
        centered (bool, optional, default: False) : if ``True``, compute the centered RMSProp,
            the gradient is normalized by an estimation of its variance
        weight_decay (float, optional, default: 0): weight decay (L2 penalty)
        decoupled_decay (bool, optional, default: False) : if ``True``, weight_decay will only be applied to param.
        lr_in_momentum (bool, optional, default: True) : if ``True``, lr is used when calculating momentum_buffer.
    """

    def __init__(self, params, lr=1e-2, alpha=0.9, eps=1e-10, weight_decay=0, momentum=0., centered=False,
                 decoupled_decay=False, lr_in_momentum=True):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if eps < 0.0:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if alpha < 0.0:
            raise ValueError("Invalid alpha value: {}".format(alpha))

        defaults = dict(lr=lr, momentum=momentum, alpha=alpha, eps=eps, centered=centered, weight_decay=weight_decay,
                        decoupled_decay=decoupled_decay, lr_in_momentum=lr_in_momentum)
        self.is_npu_fused_optimizer = True
        super(NpuFusedRMSpropTF, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(NpuFusedRMSpropTF, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('momentum', 0)
            group.setdefault('centered', False)

    def _init_param_state(self, p, momentum, centered):
        state = self.state[p]
        # State initialization
        if len(state) == 0:
            state['step'] = 0
            state['square_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
            if momentum > 0:
                state['momentum_buffer'] = torch.zeros_like(p, memory_format=torch.preserve_format)
            if centered:
                state['grad_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
        else:
            square_avg_tmp = torch.zeros_like(p, memory_format=torch.preserve_format)
            square_avg_tmp.copy_(state['square_avg'])
            state['square_avg'] = square_avg_tmp

            if momentum > 0:
                momentum_buffer_tmp = torch.zeros_like(p, memory_format=torch.preserve_format)
                momentum_buffer_tmp.copy_(state['momentum_buffer'])
                state['momentum_buffer'] = momentum_buffer_tmp
            if centered:
                grad_avg_tmp = torch.zeros_like(p, memory_format=torch.preserve_format)
                grad_avg_tmp.copy_(state['grad_avg'])
                state['grad_avg'] = grad_avg_tmp

    def _combine_group_param_states(self, group_index):
        group = self.param_groups[group_index]
        stash = self._amp_stash
        group_params_list = stash.params_lists_indexed_by_group[group_index]

        momentum = group['momentum']
        centered = group['centered']

        combined_param_states = []
        for params in group_params_list:
            step_list = []
            square_avg_list = []
            momentum_buffer_list = []
            grad_avg_list = []

            for p in params:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('NpuFusedRMSpropTF does not support sparse gradients.')

                self._init_param_state(p, momentum, centered)
                state = self.state[p]
                step_list.append(state['step'])
                square_avg_list.append(state['square_avg'])
                if momentum > 0:
                    momentum_buffer_list.append(state['momentum_buffer'])
                if centered:
                    grad_avg_list.append(state['grad_avg'])

            combined_step = 0
            combined_square_avg = None
            combined_momentum_buffer = None
            combined_grad_avg = None

            if len(square_avg_list) > 0:
                combined_step = step_list[0]
                combined_square_avg = combine_npu(square_avg_list)
                combined_momentum_buffer = combine_npu(momentum_buffer_list)
                combined_grad_avg = combine_npu(grad_avg_list)

            combined_state = defaultdict(dict)
            combined_state['step'] = combined_step
            combined_state['square_avg'] = combined_square_avg
            combined_state['momentum_buffer'] = combined_momentum_buffer
            combined_state['grad_avg'] = combined_grad_avg
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
        for p in group['params']:
            if p.grad is None:
                continue
            grad = p.grad
            if grad.is_sparse:
                raise RuntimeError('NpuFusedRMSpropTF does not support sparse gradients')

            state_p = self.state[p]
            state_p['step'] += 1

        one_minus_alpha = 1. - group['alpha']

        stash = self._amp_stash
        combined_group_params = stash.combined_params_indexed_by_group[group_index]
        combined_group_grads = stash.combined_grads_indexed_by_group[group_index]
        combined_group_param_states = stash.combined_param_states_indexed_by_group[group_index]

        for combined_param, combined_grad, combined_param_state in zip(combined_group_params,
                                                                       combined_group_grads,
                                                                       combined_group_param_states):
            if combined_param is None or combined_grad is None:
                continue

            square_avg = combined_param_state['square_avg']

            if group['weight_decay'] != 0:
                if 'decoupled_decay' in group and group['decoupled_decay']:
                    combined_param.add_(-group['weight_decay'], combined_param)
                else:
                    combined_grad = combined_grad.add(group['weight_decay'], combined_param)

            # Tensorflow order of ops for updating squared avg
            square_avg.add_(one_minus_alpha, combined_grad.pow(2) - square_avg)

            if group['centered']:
                grad_avg = combined_param_state['grad_avg']
                grad_avg.add_(one_minus_alpha, combined_grad - grad_avg)
                avg = square_avg.addcmul(-1, grad_avg, grad_avg).add(group['eps']).sqrt_()  # eps moved in sqrt
            else:
                avg = square_avg.add(group['eps']).sqrt_()  # eps moved in sqrt

            if group['momentum'] > 0:
                buf = combined_param_state['momentum_buffer']
                # Tensorflow accumulates the LR scaling in the momentum buffer
                if 'lr_in_momentum' in group and group['lr_in_momentum']:
                    buf.mul_(group['momentum']).addcdiv_(group['lr'], combined_grad, avg)
                    combined_param.add_(-buf)
                else:
                    # PyTorch scales the param update by LR
                    buf.mul_(group['momentum']).addcdiv_(combined_grad, avg)
                    combined_param.add_(-group['lr'], buf)
            else:
                combined_param.addcdiv_(-group['lr'], combined_grad, avg)

    @torch.no_grad()
    def step(self, closure=None):
        if not hasattr(self, "_amp_stash"):
            raise RuntimeError('apex.optimizers.NpuFusedRMSpropTF should be used with AMP.')

        self._check_already_combined_params_and_grads()
        # combine params and grads first
        self._combine_params_and_grads_by_group()
        # then combine param states
        self._combine_param_states_by_group()

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for i, _ in enumerate(self.param_groups):
            self._group_step(i)

        return loss
