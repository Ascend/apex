#!/bin/bash

# Copyright (c) 2020 Huawei Technologies Co., Ltd
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

SRC_DIR=apex-develop

# mkdir src
mkdir -p temp/src/apex/contrib/test
mkdir -p temp/src/apex/optimizers

mkdir -p temp/src/csrc

mkdir -p temp/tests/L0/run_optimizers
mkdir -p temp/tests/L1/common

# move files
mv $SRC_DIR/apex/contrib/combine_tensors temp/src/apex/contrib
mv $SRC_DIR/apex/contrib/test/test_combine_tensors.py temp/src/apex/contrib/test
mv $SRC_DIR/apex/dump temp/src/apex
mv $SRC_DIR/apex/optimizers/lamb.py temp/src/apex/optimizers
mv $SRC_DIR/apex/optimizers/npu* temp/src/apex/optimizers

mv $SRC_DIR/csrc/combine_tensors temp/src/csrc
mv $SRC_DIR/csrc/npu_float_status temp/src/csrc

mv $SRC_DIR/tests/L0/run_optimizers/test_lamb_mnist.py temp/tests/L0/run_optimizers
mv $SRC_DIR/tests/L0/device.py temp/tests/L0
mv $SRC_DIR/tests/L1/common/compare_npu.py temp/tests/L1/common
mv $SRC_DIR/tests/L1/common/run_test_npu.sh temp/tests/L1/common

mv $SRC_DIR/CONTRIBUTING.zh.md temp
mv $SRC_DIR/README.en.md temp
mv $SRC_DIR/OWNERS temp
mv $SRC_DIR/.gitee temp
mv $SRC_DIR/说明.md temp

diff -Nur '--exclude=.git' apex $SRC_DIR > npu.patch
