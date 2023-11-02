# Copyright (c) 2021, Huawei Technologies. All rights reserved.
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

"""Lamb optimizer."""

import torch
from torch.optim import Optimizer


class Lamb(Optimizer):
    r"""Implements Lamb algorithm.

    It has been proposed in `Large Batch Optimization for Deep Learning: Training BERT in 76 minutes`_.


    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional, default: 1e-3): learning rate
        betas (Tuple[float, float], optional, default=(0.9, 0.999)): coefficients used for computing
            running averages of gradient and its square
        eps (float, optional, default=1e-8): term added to the denominator to improve
            numerical stability
        weight_decay (float, optional, default=0): weight decay (L2 penalty)
        adam (bool, optional): always use trust ratio = 1, which turns this into
            Adam. Useful for comparison purposes.

    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-6,
                 weight_decay=0, adam=False):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if betas[0] < 0.0 or betas[0] >= 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if betas[1] < 0.0 or betas[1] >= 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if eps < 0.0:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay)
        self.adam = adam
        super(Lamb, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group['betas']
            step_size = group['lr']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('Lamb does not support sparse gradients, consider SparseAdam instad.')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                state['step'] += 1

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                adam_step = exp_avg / exp_avg_sq.sqrt().add(group['eps'])
                if group['weight_decay'] != 0:
                    adam_step.add_(p, alpha=group['weight_decay'])

                weight_norm = p.pow(2).sum().sqrt().clamp(0, 10)
                adam_norm = adam_step.pow(2).sum().sqrt()
                if weight_norm == 0 or adam_norm == 0:
                    trust_ratio = 1
                else:
                    trust_ratio = weight_norm / adam_norm
                state['weight_norm'] = weight_norm
                state['adam_norm'] = adam_norm
                state['trust_ratio'] = trust_ratio
                if self.adam:
                    trust_ratio = 1

                alpha = -step_size * trust_ratio
                adam_step.mul_(alpha)
                p.add_(adam_step)

        return loss
