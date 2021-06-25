/*
 * Copyright (c) 2021, Huawei Technologies.All rights reserved.
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

#include <torch/extension.h>
#include <vector>
#include <iostream>


void change_data_ptr(at::Tensor des, at::Tensor src, int offset)
{
  if (src.scalar_type() == at::ScalarType::Half) {
    at::Half* data_ptr = static_cast<at::Half*>(src.storage().data_ptr().get()) + offset;
    at::DataPtr aim_data_ptr = at::DataPtr(data_ptr, des.storage().device());
    des.storage().set_data_ptr(std::move(aim_data_ptr));
  } else {
    float* data_ptr = static_cast<float*>(src.storage().data_ptr().get()) + offset;
    at::DataPtr aim_data_ptr = at::DataPtr(data_ptr, des.storage().device());
    des.storage().set_data_ptr(std::move(aim_data_ptr));
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("change_data_ptr", &change_data_ptr, "Change data ptr");
}
