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

import unittest
import argparse
import functools as ft
import itertools as it
import sys
from apex import amp
import torch
from apex.contrib.combine_tensors import combine_npu
sys.path.append('../../../tests/L0')
import device

x_before_combine_des = 'x before combine: '
y_before_combine_des = 'y before combine: '
z_before_combine_des = 'z before combine: '
x_after_combine_des = 'x after combine: '
y_after_combine_des = 'y after combine: '
z_after_combine_des = 'z after combine: '
combine_tensor_des = 'combine tensor: '

def print_tensor_phy_info(des, tensor):
    print(des, tensor.data_ptr(), tensor.size(), tensor.device)

class TestCombineTensors(unittest.TestCase):

    def setUp(self):
        self.device = device.CALCULATE_DEVICE

    def basic_functionality(self, dtype = torch.float32):
        x = torch.zeros((2,2,2,2), device = self.device, dtype = dtype)
        y = torch.zeros((4,4), device = self.device, dtype = dtype)
        z = torch.zeros((3,3,3), device = self.device, dtype = dtype)

        print_tensor_phy_info(x_before_combine_des, x)
        print_tensor_phy_info(y_before_combine_des, y)
        print_tensor_phy_info(z_before_combine_des, z)

        lst = [x, y, z]

        combine_tensor = combine_npu(lst)
        print()
        print_tensor_phy_info(combine_tensor_des, combine_tensor)
        print_tensor_phy_info(x_after_combine_des, x)
        print_tensor_phy_info(y_after_combine_des, y)
        print_tensor_phy_info(z_after_combine_des, z)

        # test if combine_tensor is contiguous, and x,y,z are will moved into the combine_tensor.
        self.assertEqual(True, combine_tensor.is_contiguous())
        self.assertEqual(combine_tensor.data_ptr(), x.data_ptr())
        self.assertEqual(x.data_ptr() + x.storage().size() * x.element_size(), y.data_ptr())
        self.assertEqual(y.data_ptr() + y.storage().size() * y.element_size(), z.data_ptr())
        self.assertEqual(combine_tensor.storage().size(), x.storage().size() + y.storage().size() + z.storage().size())

    def test_basic_fp32(self):
        print('----------------------test basic functionality of fp32------------------------')
        self.basic_functionality(dtype=torch.float32)

    def test_large_functionality(self):
        print('----------------------test functionality with large tensors------------------------')
        x = torch.zeros((200,20,200,20), device = self.device)
        y = torch.zeros((4000,4000), device = self.device)
        z = torch.zeros((300,300,300), device = self.device)

        print_tensor_phy_info(x_before_combine_des, x)
        print_tensor_phy_info(y_before_combine_des, y)
        print_tensor_phy_info(z_before_combine_des, z)

        lst = [x, y, z]

        combine_tensor = combine_npu(lst)
        print()
        print_tensor_phy_info(combine_tensor_des, combine_tensor)
        print_tensor_phy_info(x_after_combine_des, x)
        print_tensor_phy_info(y_after_combine_des, y)
        print_tensor_phy_info(z_after_combine_des, z)

        # test for tensors with very large sizes.
        self.assertEqual(True, combine_tensor.is_contiguous())
        self.assertEqual(combine_tensor.data_ptr(), x.data_ptr())
        self.assertEqual(x.data_ptr() + x.storage().size() * x.element_size(), y.data_ptr())
        self.assertEqual(y.data_ptr() + y.storage().size() * y.element_size(), z.data_ptr())
        self.assertEqual(combine_tensor.storage().size(), x.storage().size() + y.storage().size() + z.storage().size())

    def test_computation(self):
        print('----------------------test computation------------------------')
        x = torch.zeros((2, 2, 2, 2), device=self.device)
        y = torch.zeros((4, 4), device=self.device)
        z = torch.zeros((3, 3, 3), device=self.device)

        print_tensor_phy_info(x_before_combine_des, x)
        print_tensor_phy_info(y_before_combine_des, y)
        print_tensor_phy_info(z_before_combine_des, z)

        lst = [x, y, z]

        combine_tensor = combine_npu(lst)

        print()
        print_tensor_phy_info(combine_tensor_des, combine_tensor)
        print_tensor_phy_info(x_after_combine_des, x)
        print_tensor_phy_info(y_after_combine_des, y)
        print_tensor_phy_info(z_after_combine_des, z)

        combine_tensor += 2

        self.assertEqual(32, x.sum())
        self.assertEqual(32, y.sum())
        self.assertEqual(54, z.sum())

        for tensor in lst:
            tensor.mul_(2)

        self.assertEqual(236, combine_tensor.sum())
        self.assertEqual(combine_tensor.sum(), x.sum() + y.sum() + z.sum())

    @unittest.skip("not stable test")
    def test_storage_reuse_and_memory_release(self):
        print('----------------------test storage reuse and memory release------------------------')
        x = torch.zeros((2, 2, 2, 2), device=self.device)
        y = torch.zeros((4, 4), device=self.device)

        print_tensor_phy_info(x_before_combine_des, x)
        print_tensor_phy_info(y_before_combine_des, y)

        store_x = x.data_ptr()
        store_y = y.data_ptr()

        lst = [x, y]

        combine_tensor = combine_npu(lst)
        print()
        print_tensor_phy_info(combine_tensor_des, combine_tensor)
        print_tensor_phy_info(x_after_combine_des, x)
        print_tensor_phy_info(y_after_combine_des, y)

        ### test for storage reuse ###
        c = torch.zeros((2, 2, 2, 2), device=self.device)
        d = torch.zeros((4, 4), device=self.device)

        print()
        print('new tensor c: ', c.data_ptr(), c.size(), c.device)
        print('new tensor d: ', d.data_ptr(), d.size(), d.device)

        self.assertEqual(store_x, c.data_ptr())
        self.assertEqual(store_y, d.data_ptr())

        ### test for memory release ###
        store_x_after_combine = x.data_ptr()
        x = None
        new_tensor = torch.zeros((2, 2, 2, 2), device=self.device)

        print('new_tensor with size of x: ', new_tensor.data_ptr(), new_tensor.size(), new_tensor.device)
        self.assertNotEqual(store_x_after_combine, new_tensor.data_ptr())

if __name__ == '__main__':
    unittest.main(argv=['test_combine_tensors.py'])
