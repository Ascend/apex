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

print_banner() {
  printf "\n\n\n\e[30m\e[42m$1\e[0m\n\n\n\n"
}

print_banner "Distributed status:  $1"

echo $2
DATADIR=$2

if [ "$1" == "single_npu" ]
then
  BASE_CMD="python main_amp.py -a resnet50 --b 128 --workers 4 --deterministic --prints-to-process 5 "
fi
if [ $# == 3 ]
then
  BASE_CMD="${BASE_CMD} --npu $3"
fi

combine_grads=(
""
"--combine-grad"
)

npu_fused_sgds=(
""
"--npu-fused-sgd"
)

loss_scales=(
"--loss-scale dynamic"
"--loss-scale 1024.0"
)

opt_levels=(
"O1"
"O2"
)

rm True*
rm False*

set -e

for opt_level in "${opt_levels[@]}"
do
  for loss_scale in "${loss_scales[@]}"
  do
    for combine_grad in "${combine_grads[@]}"
    do
      for npu_fused_sgd in "${npu_fused_sgds[@]}"
      do
        print_banner "${BASE_CMD} --opt-level ${opt_level} ${loss_scale} ${combine_grad} ${npu_fused_sgd} $DATADIR"
        set -x
        ${BASE_CMD} --opt-level ${opt_level} ${loss_scale} ${combine_grad} ${npu_fused_sgd} $DATADIR
        set +x
      done
    done
  done
done

for opt_level in "${opt_levels[@]}"
do
  for loss_scale in "${loss_scales[@]}"
  do
    echo "compare_npu.py --opt-level ${opt_level} ${loss_scale}"
    set -x
    python compare_npu.py --opt-level ${opt_level} ${loss_scale}
    set +x
  done
done
