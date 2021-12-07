# Copyright (c) 2021, Huawei Technologies.All rights reserved.
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

import sys
import torch
from torch import nn
import h5py
import numpy as np

torch.set_printoptions(profile="full")

class HDF5FileOperation():
    def __init__(self, file_path, is_cover=True):
        self.file_path = file_path
        if is_cover:
            with h5py.File(self.file_path, "w") as f:
                pass
        
    # read HDF5文件
    def read_data(self):
        print("######### HDF5 Data storage structure #########")
        with h5py.File(self.file_path, "r") as f:
            def print_name(name):
                print(name)
                dset = f[name]
                if isinstance(dset, h5py.Dataset):
                    print(dset.dtype)
                    print(dset.shape)
                    print()
            f.visit(print_name)

    # write HDF5文件
    def write_data(self, groupName, datasetName, data_dump):
        index = -1
        if type(data_dump) == tuple:
            for index, dd in enumerate(data_dump):
                self._write_dataset(groupName, datasetName, dd, index)
        else:
            self._write_dataset(groupName, datasetName, data_dump, index)
        print("write end.")

    def _write_dataset(self, groupName, datasetName, dataTensor, index):
        with h5py.File(self.file_path, "a") as f:
            grp = f[groupName]
            if dataTensor is not None:
                data = dataTensor.cpu().data.numpy()
                if index == -1:
                    print("writetoh5py create group:", groupName, "-", datasetName, "[shapes=", dataTensor.shape, "]")
                    grp.create_dataset(datasetName, data=data)
                else:
                    print("writetoh5py create group tuple:", groupName, "-", datasetName+str(index), "[shapes=", dataTensor.shape, "]")
                    grp.create_dataset(datasetName+str(index), data=data)
            else:
                if index == -1:
                    print("writetoh5py create group:", groupName, "-", datasetName, "[shapes=None]")
                    grp.create_dataset(datasetName, shape=(0,), dtype='f', data=None)
                else:
                    print("writetoh5py create group tuple:", groupName, "-", datasetName+str(index), "[shapes=None]")
                    grp.create_dataset(datasetName+str(index), shape=(0,), dtype='f', data=None)

class HookerAndDumper():
    def __init__(self, file_path=""):
        self.handles = []
        self.file_path = file_path
        self.G_WRITE_POINT_COUNT = 0

    def _create_group(self, module):
        print("modules:", type(module)) # 用于区分模块
        num_id_str = "%04d" % self.G_WRITE_POINT_COUNT
        self.G_WRITE_POINT_COUNT += 1
        group_name = num_id_str + "_" + module._get_name()
        with h5py.File(self.file_path, "a") as f:
            f.create_group(group_name)
        return group_name

    # dump前向数据 
    def forward_hook_fn(self, module, feat_input, feat_output):
        print("######## forward_hook_fn ########")
        group_name = self._create_group(module)
        self.hfo.write_data(group_name, "feat_input", feat_input)
        self.hfo.write_data(group_name, "feat_output", feat_output)

    # dump 后向数据
    def backward_hook_fn(self, module, grad_input, grad_output):
        print("######## backward_hook_fn ########")
        group_name = self._create_group(module)
        self.hfo.write_data(group_name, "grad_output", grad_output)
        self.hfo.write_data(group_name, "grad_input", grad_input)
        
    # epoch 和 step用来标志文件名称
    def register_hooks(self, model, epoch, step):
        # write HDF5文件
        self.file_path = self.file_path + str(epoch) + "-epoch" + "_" + str(step) + "-step" + "_dump.hdf5"
        self.hfo = HDF5FileOperation(self.file_path) 
        # 遍历所有 module，注册 forward hook 和 backward hook
        print("model:", type(model))
        self.model = model
        modules = list(self.model.named_children())
        print("register children model:")
        for name, module in modules:
            print("children_name:", name)
            forward_handle = module.register_forward_hook(self.forward_hook_fn)
            self.handles.append(forward_handle)
            backward_handle = module.register_backward_hook(self.backward_hook_fn)
            self.handles.append(backward_handle)
        print("register hook ok.")

    # remove 所有的handle
    def remove_hooks(self):
        for handle in self.handles:
            handle.remove()
        print("######## write path:", self.file_path, " ########")

    # read HDF5文件
    def read_h5py_data(self):
        self.hfor = HDF5FileOperation(file_path=self.file_path, is_cover=False)
        self.hfor.read_data()

