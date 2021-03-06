# Copyright (c) 2020, Huawei Technologies.
# Copyright (c) 2019, Facebook CORPORATION.
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

import torch
from torch.optim.optimizer import Optimizer
from collections import defaultdict
from ..contrib.combine_tensors import combine_npu

class NpuFusedAdadelta(Optimizer):
    """Implements NpuFusedAdadelta algorithm.
    Currently NPU-only.  Requires Apex to be installed via
    ``pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--npu_float_status" ./``.

    This version of fused ADADELTA implements 1 fusions.

      * A combine-tensor apply launch that batches the elementwise updates applied to all the model's parameters into one or a few kernel launches.

    :class:`apex.optimizers.NpuFusedAdadelta` may be used as a drop-in replacement for ``torch.optim.Adadelta``::

        opt = apex.optimizers.NpuFusedAdadelta(model.parameters(), lr = ....)
        ...
        opt.step()

    :class:`apex.optimizers.NpuFusedAdadelta` should be used with Amp.  Currently, if you wish to use :class:`NpuFusedAdadelta` with Amp,
    only ``opt_level O2`` can be choosed::

        opt = apex.optimizers.NpuFusedAdadelta(model.parameters(), lr = ....)
        model, opt = amp.initialize(model, opt, opt_level="O2")
        ...
        opt.step()
    It has been proposed in `ADADELTA: An Adaptive Learning Rate Method`__.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        rho (float, optional): coefficient used for computing a running average
            of squared gradients (default: 0.9)
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-6)
        lr (float, optional): coefficient that scale delta before it is applied
            to the parameters (default: 1.0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)

    __ https://arxiv.org/abs/1212.5701
    """

    def __init__(self, params, lr=1.0, rho=0.9, eps=1e-6, weight_decay=0):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= rho <= 1.0:
            raise ValueError("Invalid rho value: {}".format(rho))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, rho=rho, eps=eps, weight_decay=weight_decay)
        self.is_npu_fused_optimizer = True
        super(NpuFusedAdadelta, self).__init__(params, defaults)

    def _init_param_state(self, p):
        state = self.state[p]
        # State initialization
        if len(state) == 0:
            state['step'] = 0
            state['square_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
            state['acc_delta'] = torch.zeros_like(p, memory_format=torch.preserve_format)
        else:
            square_avg_tmp = torch.zeros_like(p, memory_format=torch.preserve_format)
            square_avg_tmp.copy_(state['square_avg'])
            state['square_avg'] = square_avg_tmp

            acc_delta_tmp = torch.zeros_like(p, memory_format=torch.preserve_format)
            acc_delta_tmp.copy_(state['acc_delta'])
            state['acc_delta'] = acc_delta_tmp

    def _combine_group_param_states(self, group_index):
        group = self.param_groups[group_index]
        stash = self._amp_stash
        group_params_list = stash.params_lists_indexed_by_group[group_index]

        combined_param_states = []
        for params in group_params_list:
            step_list = []
            square_avg_list = []
            acc_delta_list = []

            for p in params:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('NpuFusedAdadelta does not support sparse gradients')
                
                self._init_param_state(p)
                state = self.state[p]
                step_list.append(state['step'])
                square_avg_list.append(state['square_avg'])
                acc_delta_list.append(state['acc_delta'])
            
            combined_step = 0
            combined_square_avg = None
            combined_acc_delta = None

            if len(square_avg_list) > 0:
                combined_step = step_list[0]
                combined_square_avg = combine_npu(square_avg_list)
                combined_acc_delta = combine_npu(acc_delta_list)
            
            combined_state = defaultdict(dict)
            combined_state['step'] = combined_step
            combined_state['square_avg'] = combined_square_avg
            combined_state['acc_delta'] = combined_acc_delta
            combined_param_states.append(combined_state)
        stash.combined_param_states_indexed_by_group[group_index] = combined_param_states

    def _combine_param_states_by_group(self):
        stash = self._amp_stash
        if stash.param_states_are_combined_by_group:
            return

        stash.combined_param_states_indexed_by_group = []
        for group in self.param_groups:
            stash.combined_param_states_indexed_by_group.append([])

        for i, group in enumerate(self.param_groups):
            self._combine_group_param_states(i)
        stash.param_states_are_combined_by_group = True

    def _group_step(self, group_index):
        group = self.param_groups[group_index]
        for p in group['params']:
            if p.grad is None:
                continue
            grad = p.grad
            if grad.is_sparse:
                raise RuntimeError('NpuFusedAdadelta does not support sparse gradients')
            state_p = self.state[p]
            state_p['step'] += 1

        rho, eps = group['rho'], group['eps']

        stash = self._amp_stash
        combined_group_params = stash.combined_params_indexed_by_group[group_index]
        combined_group_grads = stash.combined_grads_indexed_by_group[group_index]
        combined_group_param_states = stash.combined_param_states_indexed_by_group[group_index]

        for combined_param, combined_grad, combined_param_state in zip(combined_group_params, 
                                                                       combined_group_grads, 
                                                                       combined_group_param_states):
            if combined_param is None or combined_grad is None:
                continue

            square_avg, acc_delta = combined_param_state['square_avg'], combined_param_state['acc_delta']
            combined_param_state['step'] += 1

            if group['weight_decay'] != 0:
                combined_grad = combined_grad.add(combined_param, alpha=group['weight_decay'])

            square_avg.mul_(rho).addcmul_(combined_grad, combined_grad, value=1 - rho)
            std = square_avg.add(eps).sqrt_()
            delta = acc_delta.add(eps).sqrt_().div_(std).mul_(combined_grad)
            combined_param.add_(delta, alpha=-group['lr'])
            acc_delta.mul_(rho).addcmul_(delta, delta, value=1 - rho)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        if not hasattr(self, "_amp_stash"):
            raise RuntimeError('apex.optimizers.NpuFusedAdadelta should be used with AMP.')

        self._check_already_combined_params_and_grads()
        # combine params and grads first
        self._combine_params_and_grads_by_group()
        # then combine param states
        self._combine_param_states_by_group()

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        stash = self._amp_stash
        for i, group in enumerate(self.param_groups):
            self._group_step(i)

        return loss
