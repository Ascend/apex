# Copyright 2020 Huawei Technologies Co., Ltd
# Copyright (c) 2019, NVIDIA CORPORATION.
#
# Licensed under the BSD 3-Clause License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://spdx.org/licenses/BSD-3-Clause.html
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import itertools
import torch

EPS = 0.05
err_count = 0


def compare_loss(loss_cmp, loss_base):
    temp = (loss_cmp - loss_base) / loss_base
    if (loss_cmp < loss_base) or temp < EPS:
        return True, temp
    else:
        return False, temp


parser = argparse.ArgumentParser(description='Compare')
parser.add_argument('--opt-level', type=str)
parser.add_argument('--loss-scale', type=str, default=None)
args = parser.parse_args()

base_file = "_" + str(args.opt_level) + "_" + \
            str(args.loss_scale) + "_"

file_names = []
file_descs = []
dict_datas = []

for comb in itertools.product(['True', 'False'], ['True', 'False']):
    file_name = comb[0] + base_file + comb[1]
    file_names.append(file_name)
    dict_datas.append(torch.load(file_name))
    file_desc = "combine_grad=" + comb[0] + " opt_level=" + args.opt_level + \
                " loss-scale=" + args.loss_scale + " npu_fused_sgd=" + comb[1]
    file_descs.append(file_desc)

torch.set_printoptions(precision=10)

print()
opt_str = "opt_level=" + args.opt_level + " loss-scale=" + args.loss_scale
print("Compare with baseline: combine_grad=False " + opt_str + " npu_fused_sgd=False EPS", EPS)
print()

for n, (i_e, i_p) in enumerate(zip(dict_datas[0]["Iteration"], dict_datas[1]["Iteration"])):
    print("step = {}:".format(i_e))
    if i_e != i_p:
        raise ValueError("i_e should equal to i_p. i_e = {}, i_p = {}".format(i_e, i_p))
    loss_base_n = dict_datas[3]["Loss"][n]
    for file_name, dict_data, file_desc in zip(file_names, dict_datas, file_descs):
        if file_name == file_names[3]:
            break
        else:
            loss_n = dict_data["Loss"][n]
            result = file_desc + " loss :{:.3f}".format(loss_n) + " loss base:{:.3f}".format(loss_base_n)
            res, ratio = compare_loss(loss_n, loss_base_n)
            if res == False:
                err_count = err_count + 1
                result = result + " ratio:{:.3f}, loss compare Failed".format(ratio)
            else:
                result = result + " ratio:{:.3f}, loss compare Ok".format(ratio)
            print(result)
if (err_count > 0):
    raise Exception("Loss compare failed!")
