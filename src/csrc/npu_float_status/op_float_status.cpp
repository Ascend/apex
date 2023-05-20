/*
 * Copyright (c) 2020, Huawei Technologies.All rights reserved.
 * 
 * Licensed under the BSD 3-Clause License  (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * https://opensource.org/licenses/BSD-3-Clause
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cstdint>
#include <iostream>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <torch/extension.h>
#include <torch/csrc/utils/tensor_flatten.h>
#include "op_runner.h"
#include "common.h"

OperatorDesc CreateFloatStatusOpDesc(const std::string opType)
{
    std::vector<int64_t> shape{FLOAT_STATUS_OP_TENSOR_DIMS_SIZE};
    aclDataType dataType = ACL_FLOAT;
    aclFormat format = ACL_FORMAT_ND;
    OperatorDesc opDesc(opType);
    if ((opType == OP_TYPE_NPU_GET_FLOAT_STATUS) ||
        (opType == OP_TYPE_NPU_CLEAR_FLOAT_STATUS)) {
        opDesc.AddInputTensorDesc(dataType, shape.size(), shape.data(), format);
    }
    opDesc.AddOutputTensorDesc(dataType, shape.size(), shape.data(), format);
    return opDesc;
}

/* This function is used for linking torch/acl .so files */
at::Tensor TestFlatten(std::vector<at::Tensor> tensors)
{
  return torch::utils::flatten_dense_tensors(tensors);
}

PYBIND11_MODULE(npu_float_status, m) {
    m.doc() = "float status op";
    m.def("TestFlatten", &TestFlatten, "Test flatten");
}