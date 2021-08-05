#!/bin/bash

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
mv $SRC_DIR/tests/L1/common/main_amp_npu.py temp/tests/L1/common
mv $SRC_DIR/tests/L1/common/run_test_npu.sh temp/tests/L1/common

mv $SRC_DIR/README.en.md temp
mv $SRC_DIR/OWNERS temp
mv $SRC_DIR/.gitee temp
mv $SRC_DIR/说明.md temp

diff -Nur '--exclude=.git' apex $SRC_DIR > npu.patch
