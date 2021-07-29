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

CUR_DIR=$(dirname $(readlink -f $0))
ROOT_DIR=$CUR_DIR/..
APEX_DIR=$ROOT_DIR/apex

function main()
{
    cd $ROOT_DIR || exit

    # patch
    cp $ROOT_DIR/patch/npu.patch $APEX_DIR
    cd $APEX_DIR || exit
    patch -p1 <npu.patch
    rm -rf npu.patch
    cp -r $ROOT_DIR/src/* $APEX_DIR
    cp -r $ROOT_DIR/tests/* $APEX_DIR/tests
}

main "$@"

