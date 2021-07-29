#!/bin/bash

# Copyright (c) 2020, Huawei Technologies Co., Ltd
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

cd .. || exit

# mkdir src
mkdir -p temp/src/apex/contrib/test
mkdir -p temp/src/apex/optimizers

mkdir -p temp/src/csrc

mkdir -p temp/tests/L0/run_optimizers
mkdir -p temp/tests/L1/common

# move files
mv apex-npu/apex/contrib/combine_tensors temp/src/apex/contrib
mv apex-npu/apex/contrib/test/test_combine_tensors.py temp/src/apex/contrib/test
mv apex-npu/apex/dump temp/src/apex
mv apex-npu/apex/optimizers/lamb.py temp/src/apex/optimizers
mv apex-npu/apex/optimizers/npu* temp/src/apex/optimizers

mv apex-npu/csrc/combine_tensors temp/src/csrc
mv apex-npu/csrc/npu_float_status temp/src/csrc

mv apex-npu/tests/L0/run_optimizers/test_lamb_mnist.py temp/tests/L0/run_optimizers
mv apex-npu/tests/L0/device.py temp/tests/L0
mv apex-npu/tests/L1/common/compare_npu.py temp/tests/L1/common
mv apex-npu/tests/L1/common/main_amp_npu.py temp/tests/L1/common
mv apex-npu/tests/L1/common/run_test_npu.sh temp/tests/L1/common

mv apex-npu/README.zh.md temp
mv apex-npu/README.en.md temp
mv apex-npu/OWNERS temp
mv apex-npu/.gitee temp
mv apex-npu/说明.md temp

